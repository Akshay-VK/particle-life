// =============================================================================
// sim/gpu_sim.rs — Step 5: internal state, InitConfig, state_transfer matrix
// =============================================================================

use wgpu::util::DeviceExt;
use rand::Rng;
use crate::sim::interaction::{InteractionMatrix, NUM_TYPES, R, R_MIN};
use crate::sim::spatial_hash::SpatialHash;

// =============================================================================
// GpuParticle — 48 bytes, must match compute.wgsl EXACTLY
//
// Rust layout (all fields 4-byte aligned via #[repr(C)]):
//   offset  0: position    [f32;2]   8 bytes
//   offset  8: velocity    [f32;2]   8 bytes
//   offset 16: ptype       f32       4 bytes
//   offset 20: state       f32       4 bytes
//   offset 24: _pad_color  [f32;2]   8 bytes  ← pads to align vec3 to offset 32
//   offset 32: color       [f32;3]  12 bytes
//   offset 44: _pad_end    f32       4 bytes
//   total: 48 bytes
//
// WGSL vec3<f32> has align=16, so color lands at offset 32 in WGSL too.
// The layouts are identical.
// =============================================================================
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position:   [f32; 2],
    pub velocity:   [f32; 2],
    pub ptype:      f32,
    pub state:      f32,
    pub _pad_color: [f32; 2],  // DO NOT REMOVE — keeps color at offset 32
    pub color:      [f32; 3],
    pub _pad_end:   f32,       // DO NOT REMOVE — keeps total at 48 bytes
}

// =============================================================================
// SimParams — 48 bytes (12 fields × 4 bytes = 48, multiple of 16)
// =============================================================================
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub n_particles:          u32,
    pub n_types:              u32,
    pub dt:                   f32,
    pub friction:             f32,
    pub r_outer:              f32,
    pub r_inner:              f32,
    pub force_scale:          f32,
    pub repulse_str:          f32,
    pub grid_dim:             u32,
    pub state_decay:          f32,
    pub state_transfer_scale: f32,
    pub _pad:                 u32,
}

// =============================================================================
// InitConfig — controls how particles are initially positioned
// =============================================================================
pub enum InitConfig {
    /// Uniformly random positions across the world
    Random,
    /// N tight clusters, each seeded with all types
    Clustered { n_clusters: u32 },
    /// Types arranged in concentric rings
    Rings,
    /// Evenly spaced grid, types assigned by grid cell
    Grid,
}

const TYPE_COLORS: [[f32; 3]; NUM_TYPES] = [
    [0.95, 0.35, 0.35],
    [0.35, 0.95, 0.55],
    [0.35, 0.55, 0.95],
    [0.95, 0.85, 0.35],
    [0.85, 0.35, 0.95],
];

// =============================================================================
// GpuSim
// =============================================================================
pub struct GpuSim {
    pub buffers:          [wgpu::Buffer; 2],
    pub frame_idx:        usize,
    compute_pipeline:     wgpu::ComputePipeline,

    // Two bind groups — one per ping-pong direction (7 bindings each + state_transfer = 8)
    bind_groups:          [wgpu::BindGroup; 2],

    params_buffer:        wgpu::Buffer,
    interaction_buffer:   wgpu::Buffer,
    state_transfer_buf:   wgpu::Buffer,

    pub hash:             SpatialHash,
    // We store two assign bind groups so the hash always reads
    // from the correct (current read) particle buffer each frame.
    // hash_assign_bg[0] reads from buf[0], hash_assign_bg[1] reads from buf[1].
    hash_assign_bgl:      wgpu::BindGroupLayout,
    hash_assign_bgs:      [wgpu::BindGroup; 2],

    pub params:           SimParams,
    pub num_particles:    u32,
}

impl GpuSim {
    pub fn new(
        device:  &wgpu::Device,
        _queue:  &wgpu::Queue,
        n:       u32,
        matrix:  &InteractionMatrix,
        config:  InitConfig,
    ) -> Self {
        let initial_particles = generate_particles(n, &config);
        let grid_dim = (2.0 / R).ceil() as u32;
        println!("Grid: {}×{} = {} cells", grid_dim, grid_dim, grid_dim * grid_dim);

        let params = SimParams {
            n_particles:          n,
            n_types:              NUM_TYPES as u32,
            dt:                   0.0001,
            friction:             0.2,
            r_outer:              R,
            r_inner:              R_MIN,
            force_scale:          0.5,
            repulse_str:          4.0,
            grid_dim,
            state_decay:          0.01,
            state_transfer_scale: 1.0,
            _pad:                 0,
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

        // Interaction matrix (force strengths)
        let matrix_data: Vec<f32> = (0..NUM_TYPES)
            .flat_map(|a| (0..NUM_TYPES).map(move |b| matrix.get(a, b)))
            .collect();
        let interaction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Interaction Matrix"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // State transfer matrix — separate N×N table controlling how state
        // propagates between type pairs. Initialised to small random values.
        let state_data = generate_state_transfer_matrix();
        let state_transfer_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("State Transfer"),
            contents: bytemuck::cast_slice(&state_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Build the spatial hash. We pass buf0 initially; the assign BGs
        // are rebuilt below to cover both ping-pong directions.
        let hash = SpatialHash::new(device, &params_buffer, &buf0, n, grid_dim);

        // Build per-frame assign bind groups so the hash always reads
        // from the current read buffer (not always buf0).
        let hash_assign_bgl = SpatialHash::assign_bgl(device);
        let hash_assign_bg0 = SpatialHash::make_assign_bg(
            device, &hash_assign_bgl, &buf0,
            &hash.particle_cells, &hash.sorted_indices, &params_buffer,
        );
        let hash_assign_bg1 = SpatialHash::make_assign_bg(
            device, &hash_assign_bgl, &buf1,
            &hash.particle_cells, &hash.sorted_indices, &params_buffer,
        );

        // Physics pipeline
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute.wgsl").into()),
        });

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
            &params_buffer, &interaction_buffer, &state_transfer_buf, &hash);
        let bg1 = Self::make_physics_bg(device, &bgl, &buf1, &buf0,
            &params_buffer, &interaction_buffer, &state_transfer_buf, &hash);

        // Suppress unused variable warning for queue

        Self {
            buffers:          [buf0, buf1],
            frame_idx:        0,
            compute_pipeline,
            bind_groups:      [bg0, bg1],
            params_buffer,
            interaction_buffer,
            state_transfer_buf,
            hash,
            hash_assign_bgl,
            hash_assign_bgs:  [hash_assign_bg0, hash_assign_bg1],
            params,
            num_particles:    n,
        }
    }

    fn make_physics_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let ro  = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        let rw  = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Physics BGL"),
            // 0:in 1:out 2:params 3:interaction 4:sorted 5:cell_start 6:cell_counts 7:state_transfer
            entries: &[ro(0), rw(1), uni(2), ro(3), ro(4), ro(5), ro(6), ro(7)],
        })
    }

    fn make_physics_bg(
        device:            &wgpu::Device,
        bgl:               &wgpu::BindGroupLayout,
        input:             &wgpu::Buffer,
        output:            &wgpu::Buffer,
        params_buf:        &wgpu::Buffer,
        matrix_buf:        &wgpu::Buffer,
        state_transfer:    &wgpu::Buffer,
        hash:              &SpatialHash,
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
                wgpu::BindGroupEntry { binding: 7, resource: state_transfer.as_entire_binding() },
            ],
        })
    }

    pub fn tick(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = (self.num_particles + 255) / 256;

        // Use the assign bind group that reads from the CURRENT read buffer.
        // frame_idx % 2 == 0 → read from buf[0] → use assign_bg[0]
        // frame_idx % 2 == 1 → read from buf[1] → use assign_bg[1]
        self.hash.build_with_assign_bg(encoder, &self.hash_assign_bgs[self.frame_idx % 2]);

        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Physics"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.compute_pipeline);
            cp.set_bind_group(0, &self.bind_groups[self.frame_idx % 2], &[]);
            cp.dispatch_workgroups(wg, 1, 1);
        }
    }

    pub fn advance(&mut self) { self.frame_idx += 1; }

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

    /// Randomise the state transfer matrix and upload it
    pub fn randomise_state_transfer(&self, queue: &wgpu::Queue) {
        let data = generate_state_transfer_matrix();
        queue.write_buffer(&self.state_transfer_buf, 0, bytemuck::cast_slice(&data));
        println!("State transfer matrix randomised");
    }

    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress, // 48 bytes
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position at offset 0
                wgpu::VertexAttribute { shader_location: 0, offset: 0,  format: wgpu::VertexFormat::Float32x2 },
                // color at offset 32 (skip velocity+ptype+state+_pad_color)
                wgpu::VertexAttribute { shader_location: 1, offset: 32, format: wgpu::VertexFormat::Float32x3 },
                // state at offset 20
                wgpu::VertexAttribute { shader_location: 2, offset: 20, format: wgpu::VertexFormat::Float32 },
            ],
        }
    }
}

// =============================================================================
// Particle generation
// =============================================================================
fn generate_particles(n: u32, config: &InitConfig) -> Vec<GpuParticle> {
    let mut rng = rand::thread_rng();

    match config {
        InitConfig::Random => {
            (0..n).map(|_| {
                let t = rng.gen_range(0..NUM_TYPES);
                make_particle(
                    rng.gen_range(-0.9f32..0.9),
                    rng.gen_range(-0.9f32..0.9),
                    t, 0.0,
                )
            }).collect()
        }

        InitConfig::Clustered { n_clusters } => {
            // Evenly distribute cluster centres, then scatter particles around them
            let nc = *n_clusters as usize;
            let cluster_centers: Vec<[f32; 2]> = (0..nc).map(|i| {
                let angle = (i as f32 / nc as f32) * std::f32::consts::TAU;
                [angle.cos() * 0.6, angle.sin() * 0.6]
            }).collect();
            (0..n).map(|i| {
                let c = i as usize % nc;
                let t = rng.gen_range(0..NUM_TYPES);
                let spread = 0.15f32;
                make_particle(
                    cluster_centers[c][0] + rng.gen_range(-spread..spread),
                    cluster_centers[c][1] + rng.gen_range(-spread..spread),
                    t, 0.0,
                )
            }).collect()
        }

        InitConfig::Rings => {
            // Each type gets its own ring radius
            (0..n).map(|i| {
                let t = i as usize % NUM_TYPES;
                let radius = 0.2 + (t as f32 / NUM_TYPES as f32) * 0.7;
                let angle  = rng.gen_range(0.0f32..std::f32::consts::TAU);
                make_particle(
                    angle.cos() * radius,
                    angle.sin() * radius,
                    t, 0.0,
                )
            }).collect()
        }

        InitConfig::Grid => {
            // Evenly spaced grid; type assigned by column index
            let side = (n as f32).sqrt().ceil() as usize;
            (0..n).map(|i| {
                let row = i as usize / side;
                let col = i as usize % side;
                let x = -0.9 + (col as f32 / (side - 1) as f32) * 1.8;
                let y = -0.9 + (row as f32 / (side - 1) as f32) * 1.8;
                make_particle(x, y, col % NUM_TYPES, 0.0)
            }).collect()
        }
    }
}

fn make_particle(x: f32, y: f32, t: usize, state: f32) -> GpuParticle {
    GpuParticle {
        position:   [x, y],
        velocity:   [0.0, 0.0],
        ptype:      t as f32,
        state,
        _pad_color: [0.0, 0.0],
        color:      TYPE_COLORS[t],
        _pad_end:   0.0,
    }
}

fn generate_state_transfer_matrix() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    // Small values: state transfer is subtle by default.
    // Mix of positive (amplify) and near-zero (ignore).
    (0..NUM_TYPES * NUM_TYPES)
        .map(|_| rng.gen_range(-0.3f32..0.8))
        .collect()
}