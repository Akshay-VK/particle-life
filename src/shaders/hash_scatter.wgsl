// =============================================================================
// hash_scatter.wgsl — Pass 4 of 4: scatter particles into sorted order
//
// Uses the cell_start table built by the prefix sum to place each particle
// index into the correct position in sorted_indices[].
//
// After this pass, sorted_indices[] contains all particle indices grouped by
// cell: all particles in cell 0 first, then cell 1, etc.
//
// We use atomicAdd on a temporary cell_cursor[] array to safely claim a slot
// in sorted_indices[] without two threads writing to the same index.
//
// cell_cursor[] starts as a copy of cell_start[] (set on CPU before dispatch),
// and each particle atomically claims the next available slot in its cell.
//
// Example for cell 2 which starts at index 5 and has 3 particles:
//   Thread A: atomicAdd(&cell_cursor[2], 1) → returns 5, writes its idx to sorted[5]
//   Thread B: atomicAdd(&cell_cursor[2], 1) → returns 6, writes its idx to sorted[6]
//   Thread C: atomicAdd(&cell_cursor[2], 1) → returns 7, writes its idx to sorted[7]
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

@group(0) @binding(0) var<storage, read>       particle_cells:  array<u32>;
@group(0) @binding(1) var<storage, read_write>  sorted_indices:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  cell_cursor:     array<atomic<u32>>;
@group(0) @binding(3) var<uniform>              params:          SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles { return; }

    let cell = particle_cells[i];

    // Atomically claim the next free slot in this cell's region of sorted_indices
    let slot = atomicAdd(&cell_cursor[cell], 1u);

    // Place particle i at that slot
    sorted_indices[slot] = i;
}
