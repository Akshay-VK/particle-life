// =============================================================================
// sim/interaction.rs — Interaction matrix + force curve
//
// The interaction matrix is the heart of particle life.
// It's an N×N table where entry [a][b] is a single float:
//
//   positive → type-a particles are attracted to type-b particles
//   negative → type-a particles are repelled from type-b particles
//
// The relationship is NOT symmetric by default: [a][b] ≠ [b][a].
// That asymmetry is what produces interesting chasing, fleeing, clustering.
//
// THE FORCE CURVE
// ---------------
// We don't use a simple linear force. Real particle life uses a piecewise
// curve that has two zones:
//
//   Zone 1: 0 < dist < R_MIN   → always repulsive (collision avoidance)
//   Zone 2: R_MIN < dist < R   → follows the matrix value (attract or repel)
//
//   force
//     |        /\
//     |       /  \
//   --+------/----\--------  dist
//     |  R_MIN    R
//     |\
//     | (short-range repulsion, keeps particles from collapsing to a point)
//
// Without the short-range repulsion, all attracted particles would collapse
// into a single point and stay there — boring.
// =============================================================================

use rand::Rng;

pub const NUM_TYPES: usize = 5;

// Half the force curve's outer radius, in world units.
// Particles don't interact beyond R.
pub const R:     f32 = 0.1;

// Inner radius: below this distance, repulsion kicks in regardless of matrix.
pub const R_MIN: f32 = 0.035;

// =============================================================================
// InteractionMatrix
//
// Stores the N×N attraction/repulsion values and exposes:
//   - get(a, b)     → the scalar for that pair
//   - force(a, b, dist) → the actual force magnitude at a given distance
//   - randomise()   → fill with random values in [-1, 1]
// =============================================================================
pub struct InteractionMatrix {
    // Flat row-major storage: entry for types (a, b) lives at index a*NUM_TYPES + b
    values: Vec<f32>,
}

impl InteractionMatrix {
    // Build a random matrix — good starting point for exploration
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let values = (0..NUM_TYPES * NUM_TYPES)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        Self { values }
    }

    // Preset: a loosely social matrix where most types weakly attract each other.
    // Good for seeing early clustering behaviour.
    pub fn social() -> Self {
        let mut m = Self::random();
        // Bias all values slightly positive (weakly attractive overall)
        for v in &mut m.values {
            *v = (*v * 0.5) + 0.3;
        }
        m
    }

    // Get the raw attraction scalar for a pair of types
    #[inline]
    pub fn get(&self, type_a: usize, type_b: usize) -> f32 {
        self.values[type_a * NUM_TYPES + type_b]
    }

    // Set a value (used later for UI tweaking)
    #[inline]
    pub fn set(&mut self, type_a: usize, type_b: usize, value: f32) {
        self.values[type_a * NUM_TYPES + type_b] = value;
    }

    // =========================================================================
    // THE FORCE FUNCTION
    //
    // Given two particles of types a and b at distance `dist`, returns the
    // scalar force magnitude that should be applied along the vector from b→a.
    //
    // Positive return → push a away from b (repulsion)
    // Negative return → pull a toward  b (attraction)
    //
    // The piecewise curve:
    //
    //   dist < R_MIN:
    //     Linear ramp from -MAX_REPULSE at dist=0 to 0 at dist=R_MIN
    //     (always repulsive — prevents collapse)
    //
    //   R_MIN <= dist <= R:
    //     Tent function peaking at (R_MIN + R) / 2
    //     Scaled by the matrix value — positive = attract, negative = repel
    //
    //   dist > R:
    //     Zero (no interaction beyond the outer radius)
    // =========================================================================
    pub fn force(&self, type_a: usize, type_b: usize, dist: f32) -> f32 {
        if dist >= R || dist <= 0.0 {
            return 0.0;
        }

        if dist < R_MIN {
            // Short-range repulsion: ramps from -1 at dist=0 to 0 at dist=R_MIN
            // We multiply by a large constant so it actually overcomes attraction
            let repulse_strength = 2.0;
            return -1.0 * repulse_strength * (dist / R_MIN - 1.0);
            //                                        ^^^
            // At dist=0:     0/R_MIN - 1 = -1  → force = -repulse_strength (strong push)
            // At dist=R_MIN: 1       - 1 =  0  → force = 0 (transitions to zone 2)
        }

        // Zone 2: tent function
        // Normalise dist into [0, 1] range within the outer zone
        let norm = (dist - R_MIN) / (R - R_MIN); // 0 at R_MIN, 1 at R

        // Tent: rises from 0 to peak at norm=0.5, falls back to 0 at norm=1
        // This gives a smooth force that fades out at the boundary (avoids jarring)
        let tent = 1.0 - (2.0 * norm - 1.0).abs();

        // Scale by the matrix value and a global force multiplier
        let force_scale = 0.5;
        tent * self.get(type_a, type_b) * force_scale
        // Note: negative matrix value → negative force → attraction (pulled toward b)
        //       positive matrix value → positive force → repulsion (pushed from b)
        // The force is applied along the b→a direction vector in simulation.rs,
        // so the sign convention feels natural there.
    }

    // Construct from a flat Vec (used by the HTTP bridge)
    pub fn from_values(values: Vec<f32>) -> Self {
        Self { values }
    }

    // Expose raw values for the HTTP bridge
    pub fn raw_values(&self) -> &[f32] {
        &self.values
    }

    // Print the matrix — useful for debugging in the terminal
    pub fn print(&self) {
        println!("Interaction matrix ({NUM_TYPES}×{NUM_TYPES}):");
        for a in 0..NUM_TYPES {
            print!("  [");
            for b in 0..NUM_TYPES {
                print!("{:+.2} ", self.get(a, b));
            }
            println!("]");
        }
    }
}