// =============================================================================
// sim/gpu_sim.rs — GPU-resident simulation (Step 3)
//
// This replaces simulation.rs. The key difference:
//   Step 2: CPU owns Vec<SimParticle>, uploads positions every frame
//   Step 3: GPU owns two wgpu::Buffers, CPU only uploads params when they change
//
// WHAT THIS FILE OWNS:
//   - Two particle storage buffers (ping-pong)
//   - The compute pipeline (runs the physics shader)
//   - Bind groups for each ping-pong direction (A→B and B→A)
//   - SimParams and interaction matrix uniform/storage buffers
//
// WHAT THE CPU DOES EACH FRAME:
//   1. Call tick() — records and submits a compute dispatch
//   2. Call read_buffer() — returns which buffer was just written (for render)
//   That's it. No data moves between CPU and GPU per frame.
// =============================================================================

use wgpu::util::DeviceExt;
use rand::Rng;
use crate::sim::interaction::{InteractionMatrix, NUM_TYPES, R, R_MIN};

// =============================================================================
// GpuParticle — the layout that lives inside both storage buffers
//
// IMPORTANT: This must match the struct GpuParticle in compute.wgsl exactly.
//
// Fields:
//   position  vec2<f32>  bytes 0-7
//   velocity  vec2<f32>  bytes 8-15
//   color     vec3<f32>  bytes 16-27
//   ptype     f32        bytes 28-31   (particle type stored as float)
//
// Total: 32 bytes. Naturally 16-byte aligned (vec2 is 8, vec3 padded to 12+4).
//
// The render shader reads position (@location 0) and color (@location 1).
// The vertex layout in GpuSim::vertex_layout() tells wgpu their byte offsets.
// =============================================================================
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],   // bytes  0-7
    pub velocity: [f32; 2],   // bytes  8-15
    pub color:    [f32; 3],   // bytes 16-27
    pub ptype:    f32,         // bytes 28-31
}

// =============================================================================
// SimParams — uploaded to a uniform buffer once per frame (or when changed)
//
// Must be 16-byte aligned for uniform buffer rules.
// 8 × f32/u32 = 32 bytes — fine.
// =============================================================================
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
}

// One color per type — same palette as before
const TYPE_COLORS: [[f32; 3]; NUM_TYPES] = [
    [0.95, 0.35, 0.35],
    [0.35, 0.95, 0.55],
    [0.35, 0.55, 0.95],
    [0.95, 0.85, 0.35],
    [0.85, 0.35, 0.95],
];

// =============================================================================
// GpuSim — owns all GPU resources for the simulation
// =============================================================================
pub struct GpuSim {
    // ---- ping-pong buffers ---------------------------------------------------
    // buf[0] and buf[1] alternate roles each frame.
    // On even frames: read from buf[0], write to buf[1]
    // On odd  frames: read from buf[1], write to buf[0]
    pub buffers:        [wgpu::Buffer; 2],
    pub frame_idx:      usize,             // which frame we're on (even/odd)

    // ---- compute pipeline ---------------------------------------------------
    compute_pipeline:   wgpu::ComputePipeline,

    // ---- bind groups --------------------------------------------------------
    // bind_groups[0]: buf[0] as input,  buf[1] as output
    // bind_groups[1]: buf[1] as input,  buf[0] as output
    // We swap which one we use each frame to match the ping-pong.
    bind_groups:        [wgpu::BindGroup; 2],

    // ---- param buffers ------------------------------------------------------
    params_buffer:      wgpu::Buffer,     // SimParams uniform
    interaction_buffer: wgpu::Buffer,     // N×N matrix storage

    // ---- cached params ------------------------------------------------------
    pub params:         SimParams,
    pub num_particles:  u32,
}

impl GpuSim {
    // =========================================================================
    // new() — build everything from scratch
    //
    // Takes the wgpu Device and Queue, and an InteractionMatrix.
    // Generates initial particle state on CPU, uploads once, then everything
    // runs on GPU.
    // =========================================================================
    pub fn new(
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        n:       u32,
        matrix:  &InteractionMatrix,
    ) -> Self {
        // ---------------------------------------------------------------------
        // Generate initial particle data on CPU
        // ---------------------------------------------------------------------
        let initial_particles = generate_particles(n);

        let params = SimParams {
            n_particles: n,
            n_types:     NUM_TYPES as u32,
            dt:          0.01,
            friction:    0.2,
            r_outer:     R,
            r_inner:     R_MIN,
            force_scale: 0.5,
            repulse_str: 4.0,
        };

        // ---------------------------------------------------------------------
        // Storage buffers (ping-pong pair)
        //
        // STORAGE:    accessible from compute shaders as storage buffers
        // VERTEX:     also usable as vertex buffer for the render pass
        // COPY_DST:   allows initial upload via create_buffer_init
        //
        // Both buffers get the same initial data. On the very first frame,
        // buf[0] is read and buf[1] is written — buf[1] gets the first update.
        // ---------------------------------------------------------------------
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

        // ---------------------------------------------------------------------
        // SimParams uniform buffer
        //
        // UNIFORM:    accessible as var<uniform> in the shader
        // COPY_DST:   lets us update dt/friction/etc. without rebuilding
        // ---------------------------------------------------------------------
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("SimParams"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---------------------------------------------------------------------
        // Interaction matrix storage buffer
        //
        // Too large for a uniform buffer on some hardware (max uniform = 64KB,
        // a 64-type matrix would be 64*64*4 = 16KB — fine now, but storage
        // is safer and has no size limit).
        // ---------------------------------------------------------------------
        let matrix_data: Vec<f32> = (0..NUM_TYPES)
            .flat_map(|a| (0..NUM_TYPES).map(move |b| matrix.get(a, b)))
            .collect();

        let interaction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Interaction Matrix"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ---------------------------------------------------------------------
        // Compile the compute shader
        // ---------------------------------------------------------------------
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compute.wgsl").into()
            ),
        });

        // ---------------------------------------------------------------------
        // Bind group layout
        //
        // Describes the TYPES of resources the shader expects at each binding.
        // The actual buffers are specified in the bind groups below.
        // We reuse this layout for both bind groups (A→B and B→A).
        // ---------------------------------------------------------------------
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Compute BGL"),
            entries: &[
                // binding 0: particles_in  (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 1: particles_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 2: sim_params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 3: interaction matrix (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        // ---------------------------------------------------------------------
        // Compute pipeline
        //
        // Unlike the render pipeline, a compute pipeline only has one stage.
        // The pipeline layout references our bind group layout.
        // ---------------------------------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Compute Pipeline Layout"),
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

        // ---------------------------------------------------------------------
        // Bind groups (two of them — one per ping-pong direction)
        //
        // bind_groups[0]: buf0 = input,  buf1 = output  (used on even frames)
        // bind_groups[1]: buf1 = input,  buf0 = output  (used on odd frames)
        // ---------------------------------------------------------------------
        let make_bind_group = |input: &wgpu::Buffer, output: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("Compute Bind Group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: interaction_buffer.as_entire_binding() },
                ],
            })
        };

        let bg0 = make_bind_group(&buf0, &buf1); // even frames: 0→1
        let bg1 = make_bind_group(&buf1, &buf0); // odd  frames: 1→0

        Self {
            buffers:            [buf0, buf1],
            frame_idx:          0,
            compute_pipeline,
            bind_groups:        [bg0, bg1],
            params_buffer,
            interaction_buffer,
            params,
            num_particles:      n,
        }
    }

    // =========================================================================
    // tick() — dispatch one frame of physics on the GPU
    //
    // Records a compute pass into the encoder. The caller submits the encoder
    // together with the render pass in the same queue.submit() call —
    // this ensures the compute finishes before the render reads the buffer.
    //
    // The workgroup dispatch:
    //   We need one thread per particle.
    //   Workgroup size is 256 (set in the shader).
    //   So we dispatch ceil(n / 256) workgroups.
    //   Threads beyond n_particles return early (guard in shader).
    // =========================================================================
    pub fn tick(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (self.num_particles + 255) / 256;

        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { label: Some("Physics Pass"), timestamp_writes: None }
        );

        compute_pass.set_pipeline(&self.compute_pipeline);

        // Use the bind group for this frame's ping-pong direction
        // even frame → bind_groups[0] (reads buf0, writes buf1)
        // odd  frame → bind_groups[1] (reads buf1, writes buf0)
        compute_pass.set_bind_group(0, &self.bind_groups[self.frame_idx % 2], &[]);

        compute_pass.dispatch_workgroups(workgroups, 1, 1);
        // The pass finalises when it drops at the end of this scope
    }

    // =========================================================================
    // advance() — flip the ping-pong index after the frame is submitted
    //
    // Called from main.rs after queue.submit(). This tells us which buffer
    // was WRITTEN this frame (and should be READ by the render pass next frame).
    // =========================================================================
    pub fn advance(&mut self) {
        self.frame_idx += 1;
    }

    // =========================================================================
    // render_buffer() — returns the buffer that was just written
    //
    // The render pass reads from this buffer as a vertex buffer.
    // After tick() writes to buf[(frame_idx+1)%2], that's our fresh data.
    // =========================================================================
    pub fn render_buffer(&self) -> &wgpu::Buffer {
        // tick() reads from frame_idx%2 and writes to (frame_idx+1)%2
        &self.buffers[(self.frame_idx + 1) % 2]
    }

    // =========================================================================
    // update_matrix() — re-upload the interaction matrix after R keypress
    // =========================================================================
    pub fn update_matrix(&self, queue: &wgpu::Queue, matrix: &InteractionMatrix) {
        let matrix_data: Vec<f32> = (0..NUM_TYPES)
            .flat_map(|a| (0..NUM_TYPES).map(move |b| matrix.get(a, b)))
            .collect();
        queue.write_buffer(&self.interaction_buffer, 0, bytemuck::cast_slice(&matrix_data));
    }

    // =========================================================================
    // update_params() — re-upload SimParams (e.g. after friction change)
    // =========================================================================
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    // =========================================================================
    // vertex_layout() — describes GpuParticle to the render pipeline
    //
    // The render shader only uses position (@location 0) and color (@location 1).
    // We describe all fields but only expose position and color as attributes.
    // The shader ignores velocity and ptype — they're just stride padding to it.
    // =========================================================================
    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress, // 32 bytes
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: float32x2 at offset 0
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset:          0,
                    format:          wgpu::VertexFormat::Float32x2,
                },
                // color: float32x3 at offset 16 (skip velocity at bytes 8-15)
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset:          16,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // ptype at offset 28 is NOT listed — render shader doesn't use it
            ],
        }
    }
}

// =============================================================================
// generate_particles() — CPU-side initial state, uploaded once
// =============================================================================
fn generate_particles(n: u32) -> Vec<GpuParticle> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let t = rng.gen_range(0..NUM_TYPES);
        GpuParticle {
            position: [rng.gen_range(-0.5f32..0.5), rng.gen_range(-0.5f32..0.5)],
            velocity: [0.0, 0.0],
            color:    TYPE_COLORS[t],
            ptype:    t as f32,
        }
    }).collect()
}