// =============================================================================
// shaders/compute.wgsl — GPU physics compute shader
//
// This replaces the CPU tick() loop entirely. One GPU thread runs per particle.
// All threads run simultaneously — that's the speedup.
//
// READING ORDER:
//   1. GpuParticle struct    — particle layout in both buffers
//   2. SimParams struct      — per-frame constants from CPU
//   3. Bindings              — the four buffers this shader sees
//   4. force_curve()         — identical logic to interaction.rs on CPU
//   5. main()                — the actual physics per particle
// =============================================================================


// =============================================================================
// 1. GpuParticle — must match GpuParticle in sim/gpu_sim.rs exactly
//
// Layout (32 bytes, 16-byte aligned):
//   offset 0:  position  vec2<f32>   (8 bytes)
//   offset 8:  velocity  vec2<f32>   (8 bytes)
//   offset 16: color     vec3<f32>   (12 bytes)
//   offset 28: ptype     f32         (4 bytes, stores integer type as float)
// =============================================================================
struct GpuParticle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color:    vec3<f32>,
    ptype:    f32,
}

// =============================================================================
// 2. SimParams — uploaded from CPU as a uniform each frame
// =============================================================================
struct SimParams {
    n_particles:  u32,
    n_types:      u32,
    dt:           f32,
    friction:     f32,
    r_outer:      f32,
    r_inner:      f32,
    force_scale:  f32,
    repulse_str:  f32,
}

// =============================================================================
// 3. Bindings
// =============================================================================
@group(0) @binding(0) var<storage, read>       particles_in:  array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write>  particles_out: array<GpuParticle>;
@group(0) @binding(2) var<uniform>              sim_params:    SimParams;
@group(0) @binding(3) var<storage, read>        interaction:   array<f32>;

// =============================================================================
// 4. Force curve
// =============================================================================
fn force_curve(type_a: u32, type_b: u32, dist: f32) -> f32 {
    let r     = sim_params.r_outer;
    let r_min = sim_params.r_inner;

    if dist <= 0.0 || dist >= r {
        return 0.0;
    }

    if dist < r_min {
        return -1.0 * sim_params.repulse_str * (dist / r_min - 1.0);
    }

    let norm = (dist - r_min) / (r - r_min);
    let tent = 1.0 - abs(2.0 * norm - 1.0);

    let matrix_idx = type_a * sim_params.n_types + type_b;
    let matrix_val = interaction[matrix_idx];

    return tent * matrix_val * sim_params.force_scale;
}

// =============================================================================
// 5. Compute entry point
// =============================================================================
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if idx >= sim_params.n_particles { return; }

    var p = particles_in[idx];
    let type_a = u32(p.ptype);

    var force = vec2<f32>(0.0, 0.0);

    for (var j: u32 = 0u; j < sim_params.n_particles; j++) {
        if j == idx { continue; }

        let q = particles_in[j];
        let type_b = u32(q.ptype);

        var diff = p.position - q.position;

        if diff.x >  1.0 { diff.x -= 2.0; }
        if diff.x < -1.0 { diff.x += 2.0; }
        if diff.y >  1.0 { diff.y -= 2.0; }
        if diff.y < -1.0 { diff.y += 2.0; }

        let dist = length(diff);

        if dist <= 0.0 || dist >= sim_params.r_outer { continue; }

        let f_mag = force_curve(type_a, type_b, dist);
        force += (diff / dist) * f_mag;
    }

    p.velocity += force * sim_params.dt;
    p.velocity *= (1.0 - sim_params.friction);
    p.position += p.velocity * sim_params.dt;

    if p.position.x >  1.0 { p.position.x -= 2.0; }
    if p.position.x < -1.0 { p.position.x += 2.0; }
    if p.position.y >  1.0 { p.position.y -= 2.0; }
    if p.position.y < -1.0 { p.position.y += 2.0; }

    particles_out[idx] = p;
}