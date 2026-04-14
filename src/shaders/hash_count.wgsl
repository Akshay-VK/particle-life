// =============================================================================
// hash_count.wgsl — Pass 2 of 4: count particles per cell
//
// One thread per particle. Atomically increments a counter for the cell that
// particle i belongs to. After this pass, cell_counts[c] = number of particles
// in cell c.
//
// We use atomicAdd because multiple threads may map to the same cell
// simultaneously — without atomics, counts would be corrupted.
// =============================================================================

struct SimParams {
    n_particles: u32,
    n_types:     u32,
    dt:          f32,
    friction:    f32,
    r_outer:     f32,
    r_inner:     f32,
    force_scale: f32,
    repulse_str: f32,
    grid_dim:    u32,
    _pad:        u32,
}

@group(0) @binding(0) var<storage, read>       particle_cells: array<u32>;
@group(0) @binding(1) var<storage, read_write>  cell_counts:    array<atomic<u32>>;
@group(0) @binding(2) var<uniform>              params:         SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles { return; }

    let cell = particle_cells[i];
    atomicAdd(&cell_counts[cell], 1u);
}
