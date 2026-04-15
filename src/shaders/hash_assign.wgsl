// =============================================================================
// hash_assign.wgsl — Pass 1: assign each particle to a grid cell
// Updated for new GpuParticle layout (48 bytes, state field added)
// =============================================================================

struct SimParams {
    n_particles:          u32,
    n_types:              u32,
    dt:                   f32,
    friction:             f32,
    r_outer:              f32,
    r_inner:              f32,
    force_scale:          f32,
    repulse_str:          f32,
    grid_dim:             u32,
    state_decay:          f32,
    state_transfer_scale: f32,
    _pad:                 u32,
}

struct GpuParticle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    ptype:    f32,
    state:    f32,
    color:    vec3<f32>,
    _pad:     f32,
}

@group(0) @binding(0) var<storage, read>       particles:        array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write>  particle_cells:   array<u32>;
@group(0) @binding(2) var<storage, read_write>  particle_indices: array<u32>;
@group(0) @binding(3) var<uniform>              params:           SimParams;

fn cell_index(pos: vec2<f32>) -> u32 {
    let cell_size = params.r_outer;
    let gd        = params.grid_dim;
    let cx = u32(clamp((pos.x + 1.0) / cell_size, 0.0, f32(gd - 1u)));
    let cy = u32(clamp((pos.y + 1.0) / cell_size, 0.0, f32(gd - 1u)));
    return cy * gd + cx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles { return; }
    particle_cells[i]   = cell_index(particles[i].position);
    particle_indices[i] = i;
}
