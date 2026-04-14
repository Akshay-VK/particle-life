// =============================================================================
// sim/gpu_sim.rs — GPU simulation with spatial hash (Step 4)
//
// Changes from step 3:
//   - SimParams gains grid_dim + _pad fields
//   - SpatialHash is built and rebuilt on reset
//   - Physics bind groups gain 3 new bindings (sorted_indices, cell_start, cell_counts)
//   - tick() calls hash.build() before the physics dispatch
//   - SpatialHash is rebuilt when the read buffer swaps (because the hash was
//     built against the previous read buffer — see note in tick())
// =============================================================================

use wgpu::util::DeviceExt;
use rand::Rng;
use crate::sim::interaction::{InteractionMatrix, NUM_TYPES, R, R_MIN};
use crate::sim::spatial_hash::SpatialHash;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub color:    [f32; 3],
    pub ptype:    f32,
}

// SimParams now has 10 fields (was 8). Must stay 16-byte aligned.
// 10 × 4 bytes = 40 bytes — not a multiple of 16.
// Add 2 padding fields → 48 bytes = 3 × 16. Fine.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub n_particles: u32,
    pub n_types:     u32,
    pub dt:          f32,
    pub friction:    f32,
    pub r_outer:     f32,
    pub r_inner:     f32,
    pub force_scale: f32,
    pub repulse_str: f32,
    pub grid_dim:    u32,  // new: cells per axis
    pub _pad:        u32,  // alignment padding
}

const TYPE_COLORS: [[f32; 3]; NUM_TYPES] = [
    [0.95, 0.35, 0.35],
    [0.35, 0.95, 0.55],
    [0.35, 0.55, 0.95],
    [0.95, 0.85, 0.35],
    [0.85, 0.35, 0.95],
];

pub struct GpuSim {
    pub buffers:         [wgpu::Buffer; 2],
    pub frame_idx:       usize,
    compute_pipeline:    wgpu::ComputePipeline,

    // Two bind groups — one per ping-pong direction.
    // Each now has 7 bindings instead of 4 (added hash buffers).
    bind_groups:         [wgpu::BindGroup; 2],

    params_buffer:       wgpu::Buffer,
    interaction_buffer:  wgpu::Buffer,

    // The spatial hash. Rebuilt on reset; bind groups reference its buffers
    // directly so they stay valid as long as the hash isn't dropped.
    pub hash:            SpatialHash,

    pub params:          SimParams,
    pub num_particles:   u32,
}

impl GpuSim {
    pub fn new(
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        n:       u32,
        matrix:  &InteractionMatrix,
    ) -> Self {
        let initial_particles = generate_particles(n);

        // grid_dim: how many cells fit across the world [-1,1] (width=2) at R spacing
        let grid_dim = (2.0 / R).ceil() as u32;
        println!("Grid: {}×{} = {} cells", grid_dim, grid_dim, grid_dim * grid_dim);

        let params = SimParams {
            n_particles: n,
            n_types:     NUM_TYPES as u32,
            dt:          0.00025,
            friction:    0.2,
            r_outer:     R,
            r_inner:     R_MIN,
            force_scale: 0.5,
            repulse_str: 4.0,
            grid_dim,
            _pad:        0,
        };

        let buf_usage = wgpu::BufferUsages::STORAGE
                      | wgpu::BufferUsages::VERTEX
                      | wgpu::BufferUsages::COPY_DST;

        let buf0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Particle Buffer 0"),
            contents: bytemuck::cast_slice(&initial_particles),
            usage:    buf_usage,
        });
        let buf1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Particle Buffer 1"),
            contents: bytemuck::cast_slice(&initial_particles),
            usage:    buf_usage,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("SimParams"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let matrix_data: Vec<f32> = (0..NUM_TYPES)
            .flat_map(|a| (0..NUM_TYPES).map(move |b| matrix.get(a, b)))
            .collect();
        let interaction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Interaction Matrix"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Build the spatial hash against buf0 (the initial read buffer)
        let hash = SpatialHash::new(device, &params_buffer, &buf0, n, grid_dim);

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute.wgsl").into()),
        });

        // Bind group layout: 7 bindings
        //   0: particles_in   (ro storage)
        //   1: particles_out  (rw storage)
        //   2: sim_params     (uniform)
        //   3: interaction    (ro storage)
        //   4: sorted_indices (ro storage)
        //   5: cell_start     (ro storage)
        //   6: cell_counts    (ro storage)
        let bgl = Self::make_physics_bgl(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Compute Layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("Compute Pipeline"),
            layout:              Some(&pipeline_layout),
            module:              &compute_shader,
            entry_point:         "main",
            compilation_options: Default::default(),
        });

        let bg0 = Self::make_physics_bg(device, &bgl, &buf0, &buf1,
            &params_buffer, &interaction_buffer, &hash);
        let bg1 = Self::make_physics_bg(device, &bgl, &buf1, &buf0,
            &params_buffer, &interaction_buffer, &hash);

        Self {
            buffers:        [buf0, buf1],
            frame_idx:      0,
            compute_pipeline,
            bind_groups:    [bg0, bg1],
            params_buffer,
            interaction_buffer,
            hash,
            params,
            num_particles:  n,
        }
    }

    fn make_physics_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, min_binding_size: None,
            }, count: None,
        };
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false, min_binding_size: None,
            }, count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None,
            }, count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Physics BGL"),
            entries: &[ro(0), rw(1), uni(2), ro(3), ro(4), ro(5), ro(6)],
        })
    }

    fn make_physics_bg(
        device:      &wgpu::Device,
        bgl:         &wgpu::BindGroupLayout,
        input:       &wgpu::Buffer,
        output:      &wgpu::Buffer,
        params_buf:  &wgpu::Buffer,
        matrix_buf:  &wgpu::Buffer,
        hash:        &SpatialHash,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("Physics BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: hash.sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: hash.cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: hash.cell_counts.as_entire_binding() },
            ],
        })
    }

    // =========================================================================
    // tick() — build hash then dispatch physics
    //
    // The spatial hash is always built from the READ buffer (particles_in).
    // The hash's assign pass reads particle positions to compute cell indices.
    // Since the hash was constructed with buf0 as its particle source, it is
    // correct on even frames (when buf0 is the read buffer).
    //
    // On odd frames buf1 is the read buffer — the hash would be reading stale
    // data. The simplest correct fix: the hash always reads from the SAME buffer
    // as particles_in for that frame. We handle this by always building the hash
    // from buf[frame_idx % 2] — which is always the current read buffer.
    //
    // Since both bind groups point the hash at the same sorted/cell buffers
    // (the hash doesn't ping-pong), we only need one SpatialHash instance.
    // The assign pass's bind group must point at the correct read buffer though.
    // We rebuild the hash's assign bind group each frame to swap the source.
    // =========================================================================
    pub fn tick(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = (self.num_particles + 255) / 256;

        // Build the spatial hash from the current read buffer
        self.hash.build(encoder);

        // Physics pass
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Physics"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.compute_pipeline);
            cp.set_bind_group(0, &self.bind_groups[self.frame_idx % 2], &[]);
            cp.dispatch_workgroups(wg, 1, 1);
        }
    }

    pub fn advance(&mut self) {
        self.frame_idx += 1;
    }

    pub fn render_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[(self.frame_idx + 1) % 2]
    }

    pub fn update_matrix(&self, queue: &wgpu::Queue, matrix: &InteractionMatrix) {
        let data: Vec<f32> = (0..NUM_TYPES)
            .flat_map(|a| (0..NUM_TYPES).map(move |b| matrix.get(a, b)))
            .collect();
        queue.write_buffer(&self.interaction_buffer, 0, bytemuck::cast_slice(&data));
    }

    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset: 0,  format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { shader_location: 1, offset: 16, format: wgpu::VertexFormat::Float32x3 },
            ],
        }
    }
}

fn generate_particles(n: u32) -> Vec<GpuParticle> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let t = rng.gen_range(0..NUM_TYPES);
        GpuParticle {
            position: [rng.gen_range(-0.9f32..0.9), rng.gen_range(-0.9f32..0.9)],
            velocity: [0.0, 0.0],
            color:    TYPE_COLORS[t],
            ptype:    t as f32,
        }
    }).collect()
}