// =============================================================================
// sim/spatial_hash.rs — GPU spatial hash builder
//
// Owns the four buffers and four compute pipelines that build the hash each
// frame. GpuSim calls build() once per frame before the physics pass.
//
// BUFFERS:
//   particle_cells   [n]         which cell each particle is in
//   sorted_indices   [n]         particle indices sorted by cell
//   cell_counts      [grid_dim²] how many particles per cell
//   cell_start       [grid_dim²] where each cell starts in sorted_indices
//   cell_cursor      [grid_dim²] temporary write cursors for scatter pass
//
// PASSES (in order):
//   1. assign  — fill particle_cells[], identity-init sorted_indices[]
//   2. count   — atomicAdd into cell_counts[]
//   3. prefix  — exclusive prefix sum → cell_start[]
//   4. scatter — place particle indices into sorted_indices[] by cell
// =============================================================================

use wgpu::util::DeviceExt;

pub struct SpatialHash {
    // ---- buffers exposed to physics shader ----------------------------------
    pub sorted_indices: wgpu::Buffer,  // [n_particles]
    pub cell_start:     wgpu::Buffer,  // [grid_dim²]
    pub cell_counts:    wgpu::Buffer,  // [grid_dim²]  (also read by physics)

    // ---- internal buffers ---------------------------------------------------
    particle_cells: wgpu::Buffer,  // [n_particles]  cell index per particle
    cell_cursor:    wgpu::Buffer,  // [grid_dim²]    scatter write cursors

    // ---- pipelines ----------------------------------------------------------
    assign_pipeline:  wgpu::ComputePipeline,
    count_pipeline:   wgpu::ComputePipeline,
    prefix_pipeline:  wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    // ---- bind groups --------------------------------------------------------
    assign_bg:  wgpu::BindGroup,
    count_bg:   wgpu::BindGroup,
    prefix_bg:  wgpu::BindGroup,
    scatter_bg: wgpu::BindGroup,

    // ---- dimensions ---------------------------------------------------------
    pub grid_dim:    u32,
    pub n_cells:     u32,
    n_particles:     u32,
}

impl SpatialHash {
    pub fn new(
        device:       &wgpu::Device,
        params_buf:   &wgpu::Buffer,  // the shared SimParams uniform buffer
        particle_buf: &wgpu::Buffer,  // the "read" particle buffer for this frame
        n_particles:  u32,
        grid_dim:     u32,
    ) -> Self {
        let n_cells = grid_dim * grid_dim;

        // ------------------------------------------------------------------
        // Allocate buffers
        // ------------------------------------------------------------------

        // particle_cells[i] = flat cell index for particle i
        let particle_cells = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Particle Cells"),
            size:               (n_particles * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // sorted_indices[k] = particle index of the k-th particle in sorted order
        let sorted_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Sorted Indices"),
            size:               (n_particles * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // cell_counts[c] = number of particles in cell c
        let cell_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Cell Counts"),
            size:               (n_cells * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // cell_start[c] = index into sorted_indices[] where cell c begins
        let cell_start = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Cell Start"),
            size:               (n_cells * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // cell_cursor[c] = current write head for cell c during scatter
        // Initialised from cell_start each frame (done via COPY in build())
        let cell_cursor = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Cell Cursor"),
            size:               (n_cells * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ------------------------------------------------------------------
        // Helper: build a bind group layout from a list of (binding, type) pairs
        // ------------------------------------------------------------------
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size:   None,
            },
            count: None,
        };

        // ------------------------------------------------------------------
        // Compile shaders and build pipelines
        // ------------------------------------------------------------------
        let make_pipeline = |device: &wgpu::Device,
                              src: &str,
                              label: &str,
                              bgl: &wgpu::BindGroupLayout| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some(label),
                bind_group_layouts:   &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(label),
                layout:              Some(&layout),
                module:              &shader,
                entry_point:         "main",
                compilation_options: Default::default(),
            })
        };

        // Pass 1 — assign
        let assign_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Assign BGL"),
            entries: &[storage_ro(0), storage_rw(1), storage_rw(2), uniform(3)],
        });
        let assign_pipeline = make_pipeline(
            device,
            include_str!("../shaders/hash_assign.wgsl"),
            "Assign Pipeline",
            &assign_bgl,
        );
        let assign_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Assign BG"),
            layout:  &assign_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 2 — count
        let count_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Count BGL"),
            entries: &[storage_ro(0), storage_rw(1), uniform(2)],
        });
        let count_pipeline = make_pipeline(
            device,
            include_str!("../shaders/hash_count.wgsl"),
            "Count Pipeline",
            &count_bgl,
        );
        let count_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Count BG"),
            layout:  &count_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 3 — prefix sum
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Prefix BGL"),
            entries: &[storage_ro(0), storage_rw(1), uniform(2)],
        });
        let prefix_pipeline = make_pipeline(
            device,
            include_str!("../shaders/hash_prefix.wgsl"),
            "Prefix Pipeline",
            &prefix_bgl,
        );
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Prefix BG"),
            layout:  &prefix_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        // Pass 4 — scatter
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Scatter BGL"),
            entries: &[storage_ro(0), storage_rw(1), storage_rw(2), uniform(3)],
        });
        let scatter_pipeline = make_pipeline(
            device,
            include_str!("../shaders/hash_scatter.wgsl"),
            "Scatter Pipeline",
            &scatter_bgl,
        );
        let scatter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Scatter BG"),
            layout:  &scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_cursor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        Self {
            sorted_indices,
            cell_start,
            cell_counts,
            particle_cells,
            cell_cursor,
            assign_pipeline,
            count_pipeline,
            prefix_pipeline,
            scatter_pipeline,
            assign_bg,
            count_bg,
            prefix_bg,
            scatter_bg,
            grid_dim,
            n_cells,
            n_particles,
        }
    }

    // =========================================================================
    // build() — record all four hash passes into the encoder
    //
    // Must be called BEFORE the physics pass each frame.
    // Each pass is a separate compute pass — wgpu guarantees they execute
    // in order within a single encoder submission.
    //
    // Between frames we must zero cell_counts[] so stale counts don't
    // accumulate. We use encoder.clear_buffer() for this — it's a GPU-side
    // memset, much faster than re-uploading from CPU.
    // =========================================================================
    pub fn build(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_particles = (self.n_particles + 255) / 256;

        // Zero cell_counts[] — stale values from the last frame would corrupt
        // the count pass (atomicAdd would accumulate across frames)
        encoder.clear_buffer(&self.cell_counts, 0, None);

        // Pass 1: assign each particle to a cell
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Assign"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.assign_pipeline);
            cp.set_bind_group(0, &self.assign_bg, &[]);
            cp.dispatch_workgroups(wg_particles, 1, 1);
        }

        // Pass 2: count particles per cell (atomicAdd)
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Count"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.count_pipeline);
            cp.set_bind_group(0, &self.count_bg, &[]);
            cp.dispatch_workgroups(wg_particles, 1, 1);
        }

        // Pass 3: prefix sum → cell_start[]
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Prefix"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.prefix_pipeline);
            cp.set_bind_group(0, &self.prefix_bg, &[]);
            cp.dispatch_workgroups(1, 1, 1); // single thread sequential scan
        }

        // Copy cell_start[] into cell_cursor[] so scatter has fresh write heads
        // cell_cursor is initialised to cell_start values — each scatter thread
        // atomically increments its cell's cursor to claim a unique slot.
        encoder.copy_buffer_to_buffer(
            &self.cell_start, 0,
            &self.cell_cursor, 0,
            (self.n_cells * 4) as u64,
        );

        // Pass 4: scatter particle indices into sorted order
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("Hash Scatter"), timestamp_writes: None }
            );
            cp.set_pipeline(&self.scatter_pipeline);
            cp.set_bind_group(0, &self.scatter_bg, &[]);
            cp.dispatch_workgroups(wg_particles, 1, 1);
        }
    }
}