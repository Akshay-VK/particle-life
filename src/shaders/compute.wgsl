// =============================================================================
// compute.wgsl — Physics pass (Step 4: spatial hash version)
//
// Changes from step 3:
//   - Reads sorted_indices[] and cell_start[] from the spatial hash
//   - Inner loop now iterates only over particles in the 3×3 neighbourhood
//     (~constant iterations) instead of all n particles (O(n) → O(1) per particle)
//   - SimParams has two new fields: grid_dim and _pad
// =============================================================================

struct GpuParticle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color:    vec3<f32>,
    ptype:    f32,
}

struct SimParams {
    n_particles: u32,
    n_types:     u32,
    dt:          f32,
    friction:    f32,
    r_outer:     f32,
    r_inner:     f32,
    force_scale: f32,
    repulse_str: f32,
    grid_dim:    u32,   // cells per axis
    _pad:        u32,
}

@group(0) @binding(0) var<storage, read>       particles_in:   array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write>  particles_out:  array<GpuParticle>;
@group(0) @binding(2) var<uniform>              sim_params:     SimParams;
@group(0) @binding(3) var<storage, read>        interaction:    array<f32>;
@group(0) @binding(4) var<storage, read>        sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read>        cell_start:     array<u32>;
@group(0) @binding(6) var<storage, read>        cell_counts:    array<u32>;

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

// Convert world position to integer cell coordinates (cx, cy)
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

    var p      = particles_in[idx];
    let type_a = u32(p.ptype);
    let gd     = i32(sim_params.grid_dim);
    let my_cell = world_to_cell(p.position);

    var force = vec2<f32>(0.0, 0.0);

    // -------------------------------------------------------------------------
    // 3×3 neighbourhood loop
    //
    // We check the cell the particle is in plus all 8 surrounding cells.
    // Any particle further than one cell away is guaranteed to be beyond
    // r_outer, so we can safely skip it.
    //
    // For each neighbour cell:
    //   1. Clamp to valid grid range (no toroidal grid wrap for the hash —
    //      particles near the edge simply have fewer neighbours, which is fine)
    //   2. Look up where that cell's particles start in sorted_indices[]
    //   3. Iterate over those particles and accumulate forces
    // -------------------------------------------------------------------------
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            var nx = my_cell.x + dx;
            var ny = my_cell.y + dy;

            // fixed — wraps around
            nx = ((nx % gd) + gd) % gd;
            ny = ((ny % gd) + gd) % gd;
            // let cell = u32(wy * gd + wx);

            let cell    = u32(ny * gd + nx);
            let start   = cell_start[cell];
            let count   = cell_counts[cell];

            for (var k: u32 = 0u; k < count; k++) {
                let j = sorted_indices[start + k];
                if j == idx { continue; }

                let q      = particles_in[j];
                let type_b = u32(q.ptype);

                var diff = p.position - q.position;

                // Toroidal wrap for force calculation
                if diff.x >  1.0 { diff.x -= 2.0; }
                if diff.x < -1.0 { diff.x += 2.0; }
                if diff.y >  1.0 { diff.y -= 2.0; }
                if diff.y < -1.0 { diff.y += 2.0; }

                let dist = length(diff);
                if dist <= 0.0 || dist >= sim_params.r_outer { continue; }

                let f_mag = force_curve(type_a, type_b, dist);
                force    += (diff / dist) * f_mag;
            }
        }
    }

    p.velocity += force * sim_params.dt;
    p.velocity *= (1.0 - sim_params.friction);
    p.position += p.velocity;

    if p.position.x >  1.0 { p.position.x -= 2.0; }
    if p.position.x < -1.0 { p.position.x += 2.0; }
    if p.position.y >  1.0 { p.position.y -= 2.0; }
    if p.position.y < -1.0 { p.position.y += 2.0; }

    particles_out[idx] = p;
}
