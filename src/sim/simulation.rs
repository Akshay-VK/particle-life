// =============================================================================
// sim/simulation.rs — CPU-side particle physics
//
// This owns all particle state (positions, velocities, types) and ticks the
// physics each frame. After each tick, it provides a slice of GPU-ready vertex
// data that gets uploaded to the GPU buffer via queue.write_buffer().
//
// WHY A SEPARATE CPU STRUCT?
// ---------------------------
// The GPU vertex buffer only needs position + color. The simulation also needs
// velocity and type, which the GPU doesn't care about. Keeping them separate
// avoids uploading unnecessary data every frame.
//
// CPU STRUCT:  SimParticle  { position, velocity, particle_type }  ← lives here
// GPU STRUCT:  GpuParticle  { position, color, _pad }              ← lives in renderer.rs
//
// Each tick:
//   1. update() computes new positions/velocities from interaction forces
//   2. gpu_particles() converts SimParticles → GpuParticles for upload
// =============================================================================

use rand::Rng;
use crate::sim::interaction::{InteractionMatrix, NUM_TYPES, R};

// =============================================================================
// SimParticle — full CPU-side particle state
//
// NOT uploaded to GPU. Positions are in "world space": [-1, 1] on both axes.
// We keep world space == clip space for now (simplest case).
// In a later step we'll add a camera/zoom and a proper world→clip transform.
// =============================================================================
#[derive(Clone)]
pub struct SimParticle {
    pub position:      [f32; 2],
    pub velocity:      [f32; 2],
    pub particle_type: usize,     // index into NUM_TYPES
}

// =============================================================================
// GpuParticle — what we upload to the vertex buffer each frame
//
// Must match the Particle struct in renderer.rs exactly (same layout).
// Defined here too so simulation.rs can produce it without depending on wgpu.
// =============================================================================
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],
    pub color:    [f32; 3],
    pub _pad:     f32,
}

// One color per particle type — matches the colors in renderer.rs step 1
const TYPE_COLORS: [[f32; 3]; NUM_TYPES] = [
    [0.95, 0.35, 0.35], // red
    [0.35, 0.95, 0.55], // green
    [0.35, 0.55, 0.95], // blue
    [0.95, 0.85, 0.35], // yellow
    [0.85, 0.35, 0.95], // purple
];

// =============================================================================
// Simulation — owns all particle state + interaction rules
// =============================================================================
pub struct Simulation {
    pub particles:   Vec<SimParticle>,
    pub matrix:      InteractionMatrix,

    // Physics constants — tweak these to change the feel of the simulation
    pub friction:    f32,  // velocity damping per frame. 0=no friction, 1=instant stop
    pub dt:          f32,  // timestep. Larger = faster but less stable
}

impl Simulation {
    // =========================================================================
    // Create a new simulation with `n` randomly placed particles.
    // =========================================================================
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Spawn particles clustered near the centre so they interact immediately
        let particles = (0..n).map(|_| {
            SimParticle {
                position: [
                    rng.gen_range(-0.5f32..0.5),
                    rng.gen_range(-0.5f32..0.5),
                ],
                velocity:      [0.0, 0.0],
                particle_type: rng.gen_range(0..NUM_TYPES),
            }
        }).collect();

        let matrix = InteractionMatrix::random();
        matrix.print(); // print to terminal so we know what rules are in play

        Self {
            particles,
            matrix,
            friction: 0.05,  // 5% velocity loss per frame
            dt:       0.001, // small timestep for stability
        }
    }

    // =========================================================================
    // PHYSICS TICK — called once per frame from main.rs
    //
    // For every particle i, we loop over every other particle j and accumulate
    // force contributions. This is O(n²) — fine for 5k particles, too slow for
    // 100k+. The spatial hash (step 4) will fix that.
    //
    // Force sign convention:
    //   force > 0 → push i away from j  (repulsion)
    //   force < 0 → pull i toward j     (attraction)
    //
    // We apply the force along the normalised j→i direction vector.
    // =========================================================================
    pub fn tick(&mut self) {
        let n = self.particles.len();

        // ----------------------------------------------------------------
        // We need to read all particles while writing to them.
        // Rust won't let us borrow self.particles mutably and immutably at
        // the same time, so we snapshot positions/types into a read-only Vec.
        // This is a small allocation but negligible at <10k particles.
        // ----------------------------------------------------------------
        let snapshot: Vec<([f32; 2], usize)> = self.particles
            .iter()
            .map(|p| (p.position, p.particle_type))
            .collect();

        for i in 0..n {
            let (pos_i, type_i) = snapshot[i];
            let mut fx = 0.0f32;
            let mut fy = 0.0f32;

            for j in 0..n {
                if i == j { continue; }

                let (pos_j, type_j) = snapshot[j];

                // Vector from j to i
                let mut dx = pos_i[0] - pos_j[0];
                let mut dy = pos_i[1] - pos_j[1];

                // Wrap-around distance (toroidal world).
                // Without this, particles at opposite edges feel a huge force
                // pulling them toward each other across the world, which looks wrong.
                // Instead we find the shortest path, wrapping around the boundary.
                if dx >  1.0 { dx -= 2.0; }
                if dx < -1.0 { dx += 2.0; }
                if dy >  1.0 { dy -= 2.0; }
                if dy < -1.0 { dy += 2.0; }

                let dist = (dx * dx + dy * dy).sqrt();

                // Skip if outside interaction radius (also avoids divide-by-zero)
                if dist == 0.0 || dist >= R {
                    continue;
                }

                // Get force magnitude from the interaction curve
                let f = self.matrix.force(type_i, type_j, dist);

                // Apply along normalised j→i direction
                // (positive f = push away from j = add to the j→i direction)
                fx += f * (dx / dist);
                fy += f * (dy / dist);
            }

            // Integrate: velocity += force * dt
            let p = &mut self.particles[i];
            p.velocity[0] += fx * self.dt;
            p.velocity[1] += fy * self.dt;

            // Friction: dampen velocity slightly each frame.
            // This prevents particles from accelerating forever.
            // Without friction the system becomes chaotic very quickly.
            p.velocity[0] *= 1.0 - self.friction;
            p.velocity[1] *= 1.0 - self.friction;

            // Integrate: position += velocity * dt
            p.position[0] += p.velocity[0];
            p.position[1] += p.velocity[1];

            // Wrap position around world boundaries (toroidal)
            // Keeps particles from flying off screen permanently
            if p.position[0] >  1.0 { p.position[0] -= 2.0; }
            if p.position[0] < -1.0 { p.position[0] += 2.0; }
            if p.position[1] >  1.0 { p.position[1] -= 2.0; }
            if p.position[1] < -1.0 { p.position[1] += 2.0; }
        }
    }

    // =========================================================================
    // Produce GPU-ready vertex data.
    //
    // Called after tick(). Converts SimParticles → GpuParticles (position+color).
    // The result is passed to queue.write_buffer() in the renderer.
    // =========================================================================
    pub fn gpu_particles(&self) -> Vec<GpuParticle> {
        self.particles.iter().map(|p| GpuParticle {
            position: p.position,
            color:    TYPE_COLORS[p.particle_type],
            _pad:     0.0,
        }).collect()
    }

    // =========================================================================
    // Randomise the interaction matrix and print the new rules.
    // Called when the user presses R (wired up in main.rs next).
    // =========================================================================
    pub fn randomise_matrix(&mut self) {
        self.matrix = InteractionMatrix::random();
        self.matrix.print();
    }
}