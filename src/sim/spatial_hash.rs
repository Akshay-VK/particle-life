// =============================================================================
// sim/spatial_hash.rs — GPU spatial hash builder
//
// Changes from step 4:
//   - assign_bg is no longer stored internally; instead the BGL and a
//     make_assign_bg() constructor are public so GpuSim can build two assign
//     bind groups (one per ping-pong buffer) and pass the correct one each frame.
//   - build() is replaced by build_with_assign_bg() which takes the BG externally.
//   - particle_cells and sorted_indices exposed as public so GpuSim can
//     build the assign bind groups.
//   - cell_start has COPY_SRC (already fixed in previous step).
//   - SimParams struct updated to match new layout.
// =============================================================================

use wgpu::util::DeviceExt;

pub struct SpatialHash {
    // ---- buffers exposed to physics shader ----------------------------------
    pub sorted_indices: wgpu::Buffer,
    pub cell_start:     wgpu::Buffer,
    pub cell_counts:    wgpu::Buffer,

    // ---- exposed for assign bind group construction in GpuSim ---------------
    pub particle_cells: wgpu::Buffer,

    // ---- internal -----------------------------------------------------------
    cell_cursor:    wgpu::Buffer,

    // ---- pipelines ----------------------------------------------------------
    // assign_pipeline is now pub so GpuSim can use the same pipeline
    // with externally-supplied bind groups
    pub assign_pipeline:  wgpu::ComputePipeline,
    count_pipeline:   wgpu::ComputePipeline,
    prefix_pipeline:  wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    // ---- bind groups (all except assign, which is managed by GpuSim) --------
    count_bg:   wgpu::BindGroup,
    prefix_bg:  wgpu::BindGroup,
    scatter_bg: wgpu::BindGroup,

    pub grid_dim:   u32,
    pub n_cells:    u32,
    n_particles:    u32,
}

impl SpatialHash {
    pub fn new(
        device:       &wgpu::Device,
        params_buf:   &wgpu::Buffer,
        particle_buf: &wgpu::Buffer,  // only used for initial assign BG (unused now)
        n_particles:  u32,
        grid_dim:     u32,
    ) -> Self {
        let n_cells = grid_dim * grid_dim;

        let particle_cells = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Cells"), size: (n_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let sorted_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Indices"), size: (n_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let cell_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counts"), size: (n_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_start = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Start"), size: (n_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cell_cursor = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Cursor"), size: (n_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- helpers ----
        let ro  = |b: u32| bgle(b, false);
        let rw  = |b: u32| bgle(b, true);
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None }, count: None,
        };

        let make_pipeline = |src: &str, label: &str, bgl: &wgpu::BindGroupLayout| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&layout), module: &shader,
                entry_point: "main", compilation_options: Default::default(),
            })
        };

        // Pass 1 — assign (BGL is public, BG is built externally by GpuSim)
        let assign_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Assign BGL"),
            entries: &[ro(0), rw(1), rw(2), uni(3)],
        });
        let assign_pipeline = make_pipeline(
            include_str!("../shaders/hash_assign.wgsl"), "Assign Pipeline", &assign_bgl
        );
        // Build a placeholder assign BG pointing at particle_buf so the struct
        // compiles — it will never be used; GpuSim supplies the real ones.
        let _unused_assign_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Assign BG (unused)"), layout: &assign_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 2 — count
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Count BGL"), entries: &[ro(0), rw(1), uni(2)],
        });
        let count_pipeline = make_pipeline(
            include_str!("../shaders/hash_count.wgsl"), "Count Pipeline", &count_bgl
        );
        let count_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Count BG"), layout: &count_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 3 — prefix
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Prefix BGL"), entries: &[ro(0), rw(1), uni(2)],
        });
        let prefix_pipeline = make_pipeline(
            include_str!("../shaders/hash_prefix.wgsl"), "Prefix Pipeline", &prefix_bgl
        );
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prefix BG"), layout: &prefix_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 4 — scatter
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scatter BGL"), entries: &[ro(0), rw(1), rw(2), uni(3)],
        });
        let scatter_pipeline = make_pipeline(
            include_str!("../shaders/hash_scatter.wgsl"), "Scatter Pipeline", &scatter_bgl
        );
        let scatter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter BG"), layout: &scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_cursor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        Self {
            sorted_indices, cell_start, cell_counts,
            particle_cells, cell_cursor,
            assign_pipeline, count_pipeline, prefix_pipeline, scatter_pipeline,
            count_bg, prefix_bg, scatter_bg,
            grid_dim, n_cells, n_particles,
        }
    }

    // =========================================================================
    // Public helpers for GpuSim to build assign bind groups externally
    // =========================================================================

    /// Returns a clone of the assign BGL (used by GpuSim to build per-buffer BGs)
    pub fn assign_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Assign BGL (ext)"),
            entries: &[
                bgle(0, false), bgle(1, true), bgle(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        })
    }

    /// Build one assign bind group pointing at a specific particle buffer
    pub fn make_assign_bg(
        device:         &wgpu::Device,
        bgl:            &wgpu::BindGroupLayout,
        particle_buf:   &wgpu::Buffer,
        particle_cells: &wgpu::Buffer,
        sorted_indices: &wgpu::Buffer,
        params_buf:     &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Assign BG"), layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        })
    }

    /// Return a reference to sorted_indices for use in assign BG construction
    pub fn sorted_indices_buf(&self) -> &wgpu::Buffer {
        &self.sorted_indices
    }

    // =========================================================================
    // build_with_assign_bg — record all 4 hash passes
    // =========================================================================
    pub fn build_with_assign_bg(
        &self,
        encoder:   &mut wgpu::CommandEncoder,
        assign_bg: &wgpu::BindGroup,
    ) {
        let wg = (self.n_particles + 255) / 256;

        encoder.clear_buffer(&self.cell_counts, 0, None);

        { // Pass 1
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Assign"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.assign_pipeline);
            cp.set_bind_group(0, assign_bg, &[]);
            cp.dispatch_workgroups(wg, 1, 1);
        }
        { // Pass 2
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Count"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.count_pipeline);
            cp.set_bind_group(0, &self.count_bg, &[]);
            cp.dispatch_workgroups(wg, 1, 1);
        }
        { // Pass 3
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Prefix"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.prefix_pipeline);
            cp.set_bind_group(0, &self.prefix_bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.cell_start, 0, &self.cell_cursor, 0, (self.n_cells * 4) as u64,
        );

        { // Pass 4
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Scatter"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.scatter_pipeline);
            cp.set_bind_group(0, &self.scatter_bg, &[]);
            cp.dispatch_workgroups(wg, 1, 1);
        }
    }
}

// Helper: make a storage bind group layout entry
fn bgle(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: !read_write },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}