// =============================================================================
// hash_prefix.wgsl — Pass 3 of 4: exclusive prefix sum over cell_counts
//
// Transforms cell_counts[] from "how many particles in each cell" into
// "where in the sorted array does each cell's particles start".
//
// Example:
//   cell_counts before: [3, 0, 2, 5, 1]
//   cell_start  after:  [0, 3, 3, 5, 10]   ← exclusive prefix sum
//
// This is a single-threaded sequential prefix sum — one thread does all cells.
// It's not the fastest approach (a parallel scan would be better for huge grids)
// but it's simple, correct, and the grid is small enough that it's not a
// bottleneck: at R=0.05 the grid is 40×40=1600 cells, trivial for one thread.
//
// For step 5+ if you increase NUM_TYPES and shrink R dramatically (much finer
// grid), swap this for a parallel work-efficient scan.
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

@group(0) @binding(0) var<storage, read>       cell_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write>  cell_start:  array<u32>;
@group(0) @binding(2) var<uniform>              params:      SimParams;

@compute @workgroup_size(1)
fn main() {
    // Total number of grid cells
    let n_cells = params.grid_dim * params.grid_dim;

    var running_sum: u32 = 0u;
    for (var c: u32 = 0u; c < n_cells; c++) {
        cell_start[c]  = running_sum;
        running_sum   += cell_counts[c];
    }
    // After the loop, running_sum == n_particles (all particles accounted for)
}
