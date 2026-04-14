// =============================================================================
// hash_assign.wgsl — Pass 1 of 4: assign each particle to a grid cell
//
// One thread per particle. Computes which cell the particle belongs to and
// writes it into particle_cells[]. Also writes the particle's own index into
// particle_indices[] as the unsorted starting point for the sort passes.
//
// The grid:
//   World space is [-1, 1] × [-1, 1] = 2 units wide/tall.
//   Cell size = r_outer (the interaction radius).
//   grid_dim = ceil(2.0 / cell_size) cells per axis.
//   Cell (cx, cy) maps to flat index: cy * grid_dim + cx
//
// We use a flat 1D cell index throughout — easier to sort than a 2D coord.
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
    grid_dim:    u32,   // cells per axis  (NEW in step 4)
    _pad:        u32,   // keep 16-byte alignment
}

struct GpuParticle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color:    vec3<f32>,
    ptype:    f32,
}

@group(0) @binding(0) var<storage, read>       particles:        array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write>  particle_cells:   array<u32>; // cell index per particle
@group(0) @binding(2) var<storage, read_write>  particle_indices: array<u32>; // [0,1,2,...,n-1] initially
@group(0) @binding(3) var<uniform>              params:           SimParams;

// Convert a world-space position to a flat cell index.
// Clamps to valid range so boundary particles don't overflow.
fn cell_index(pos: vec2<f32>) -> u32 {
    let cell_size = params.r_outer;
    let gd        = params.grid_dim;

    // Map [-1,1] → [0, grid_dim)
    // Add 1.0 to shift from [-1,1] to [0,2], then divide by cell_size
    let cx = u32(clamp((pos.x + 1.0) / cell_size, 0.0, f32(gd - 1u)));
    let cy = u32(clamp((pos.y + 1.0) / cell_size, 0.0, f32(gd - 1u)));

    return cy * gd + cx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles { return; }

    particle_cells[i]   = cell_index(particles[i].position);
    particle_indices[i] = i; // identity mapping — sort pass will reorder this
}
