// =============================================================================
// compute.wgsl — Physics + internal state update
//
// Changes in this version:
//   1. GpuParticle gains `state: f32` and layout is now 48 bytes
//   2. SimParams gains state_decay and state_transfer_scale
//   3. Cell neighbourhood lookup is now TOROIDAL — fixes edge particle bug
//   4. State is updated: each particle's state is influenced by neighbours
//      and decays toward 0 over time
// =============================================================================

// -----------------------------------------------------------------------------
// GpuParticle — 48 bytes, must match Rust GpuParticle exactly
//
// WGSL layout (vec3 has align=16):
//   offset  0: position    vec2<f32>   (8 bytes)
//   offset  8: velocity    vec2<f32>   (8 bytes)
//   offset 16: ptype       f32         (4 bytes)
//   offset 20: state       f32         (4 bytes)
//   offset 24: [implicit padding 8 bytes to align vec3 to 32]
//   offset 32: color       vec3<f32>   (12 bytes)
//   offset 44: _pad        f32         (4 bytes)
//   total: 48 bytes
// -----------------------------------------------------------------------------
struct GpuParticle {
    position: vec2<f32>,  // offset 0
    velocity: vec2<f32>,  // offset 8
    ptype:    f32,         // offset 16
    state:    f32,         // offset 20
    // 8 bytes implicit WGSL padding here (vec3 must start at multiple of 16)
    color:    vec3<f32>,  // offset 32
    _pad:     f32,         // offset 44
}

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
    state_decay:          f32,  // how fast state returns to 0 each frame
    state_transfer_scale: f32,  // global multiplier on state transfer
    _pad:                 u32,
}

@group(0) @binding(0) var<storage, read>       particles_in:     array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write>  particles_out:    array<GpuParticle>;
@group(0) @binding(2) var<uniform>              sim_params:       SimParams;
@group(0) @binding(3) var<storage, read>        interaction:      array<f32>;
@group(0) @binding(4) var<storage, read>        sorted_indices:   array<u32>;
@group(0) @binding(5) var<storage, read>        cell_start:       array<u32>;
@group(0) @binding(6) var<storage, read>        cell_counts:      array<u32>;
@group(0) @binding(7) var<storage, read>        state_transfer:   array<f32>;

// -----------------------------------------------------------------------------
// force_curve — piecewise force between two types at a distance
// -----------------------------------------------------------------------------
fn force_curve(type_a: u32, type_b: u32, dist: f32) -> f32 {
    let r     = sim_params.r_outer;
    let r_min = sim_params.r_inner;
    if dist <= 0.0 || dist >= r { return 0.0; }
    if dist < r_min {
        return -1.0 * sim_params.repulse_str * (dist / r_min - 1.0);
    }
    let norm = (dist - r_min) / (r - r_min);
    let tent = 1.0 - abs(2.0 * norm - 1.0);
    let idx  = type_a * sim_params.n_types + type_b;
    return tent * interaction[idx] * sim_params.force_scale;
}

// -----------------------------------------------------------------------------
// world_to_cell — convert world position to integer cell coordinates
// -----------------------------------------------------------------------------
fn world_to_cell(pos: vec2<f32>) -> vec2<i32> {
    let cell_size = sim_params.r_outer;
    let cx = i32((pos.x + 1.0) / cell_size);
    let cy = i32((pos.y + 1.0) / cell_size);
    return vec2<i32>(cx, cy);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim_params.n_particles { return; }

    var p       = particles_in[idx];
    let type_a  = u32(p.ptype);
    let gd      = i32(sim_params.grid_dim);
    let my_cell = world_to_cell(p.position);

    var force         = vec2<f32>(0.0, 0.0);
    var state_delta   = 0.0f;

    // -------------------------------------------------------------------------
    // 3×3 TOROIDAL neighbourhood loop
    //
    // FIX for edge particle bug: instead of skipping out-of-bounds cells,
    // we wrap them toroidally. A particle at the left edge now correctly
    // sees particles near the right edge as neighbours.
    //
    // wx, wy = wrapped cell coordinates using modular arithmetic
    // -------------------------------------------------------------------------
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            // Toroidal wrap: ((x % gd) + gd) % gd handles negative x correctly
            let wx = ((my_cell.x + dx) % gd + gd) % gd;
            let wy = ((my_cell.y + dy) % gd + gd) % gd;

            let cell  = u32(wy * gd + wx);
            let start = cell_start[cell];
            let count = cell_counts[cell];

            for (var k: u32 = 0u; k < count; k++) {
                let j = sorted_indices[start + k];
                if j == idx { continue; }

                let q      = particles_in[j];
                let type_b = u32(q.ptype);

                // Toroidal shortest-path distance for force
                var diff = p.position - q.position;
                if diff.x >  1.0 { diff.x -= 2.0; }
                if diff.x < -1.0 { diff.x += 2.0; }
                if diff.y >  1.0 { diff.y -= 2.0; }
                if diff.y < -1.0 { diff.y += 2.0; }

                let dist = length(diff);
                if dist <= 0.0 || dist >= sim_params.r_outer { continue; }

                // Force contribution
                let f_mag = force_curve(type_a, type_b, dist);
                force += (diff / dist) * f_mag;

                // State transfer contribution
                // Neighbours within interaction range transfer state proportionally
                // to how close they are (tent function weight) and the transfer matrix
                let norm   = (dist - sim_params.r_inner) / (sim_params.r_outer - sim_params.r_inner);
                let weight = clamp(1.0 - abs(2.0 * norm - 1.0), 0.0, 1.0);
                let tidx   = type_a * sim_params.n_types + type_b;
                state_delta += weight * q.state * state_transfer[tidx] * sim_params.state_transfer_scale;
            }
        }
    }

    // Physics integration
    p.velocity += force * sim_params.dt;
    p.velocity *= (1.0 - sim_params.friction);
    p.position += p.velocity;

    // Toroidal position wrap
    if p.position.x >  1.0 { p.position.x -= 2.0; }
    if p.position.x < -1.0 { p.position.x += 2.0; }
    if p.position.y >  1.0 { p.position.y -= 2.0; }
    if p.position.y < -1.0 { p.position.y += 2.0; }

    // State update:
    //   1. Accumulate transfers from neighbours
    //   2. Decay toward 0 (state_decay fraction lost per frame)
    //   3. Clamp to [0, 1]
    p.state = clamp(
        p.state + state_delta * sim_params.dt - p.state * sim_params.state_decay,
        0.0,
        1.0
    );

    particles_out[idx] = p;
}
