# Particle Life — Step-by-Step Build

GPU-accelerated particle simulation in Rust + wgpu.

---

## Run

```bash
cargo run --release
```

`--release` is critical — debug mode is 10-20x slower.

---

## Controls

| Key    | Action |
|--------|--------|
| `R`    | Randomise interaction matrix (new behaviours instantly) |
| `Space`| Scatter particles to new random positions |
| `Esc`  | Quit |

---

## File map — read in this order

```
src/
├── main.rs                      ← event loop, sim+render coordination
├── sim/
│   ├── mod.rs                   ← re-exports
│   ├── interaction.rs           ← InteractionMatrix + force curve
│   └── simulation.rs            ← SimParticle, GpuParticle, physics tick
└── render/
    ├── mod.rs                   ← re-exports
    ├── renderer.rs              ← wgpu pipeline, update(), render()
    └── shaders/
        └── render.wgsl          ← vertex + fragment shaders
```

---

## What each step adds

| Step | What's added | Particle limit |
|------|-------------|----------------|
| 1 ✓  | Window, wgpu, static dots | N/A |
| **2 ✓** | **CPU physics, interaction matrix, forces** | **~8,000** |
| 3    | GPU compute shader (physics on GPU) | ~500,000 |
| 4    | GPU spatial hash (O(n) neighbour lookup) | 10,000,000+ |
| 5    | Internal particle state, reactions, UI | — |

---

## Key concepts in this step

### Two particle structs

```
SimParticle  { position, velocity, type }   ← CPU only, full state
GpuParticle  { position, color, _pad }      ← uploaded to GPU each frame
```

Keeping them separate means we only upload what the GPU actually needs.

### The force curve

Two zones per particle pair:

```
dist < R_MIN  →  always repulsive (prevents collapse into a point)
R_MIN to R    →  tent function × matrix value (attract or repel)
dist > R      →  zero (particles are invisible to each other)
```

### Toroidal world

Particles wrap around the edges so the world has no walls. When computing
distance between two particles, we take the *shortest* path, which may
cross a boundary. Without this, particles near opposite edges would feel
a huge artificial force pulling them toward each other.

### The main loop order

```
sim.tick()          — advance physics  (CPU, O(n²))
sim.gpu_particles() — convert to upload format
renderer.update()   — queue.write_buffer() → GPU
renderer.render()   — GPU draws the updated buffer
```

### Why O(n²) is the bottleneck

Each of the n particles checks every other particle = n² distance checks
per frame. At 5000 particles that's 25 million checks at ~60fps.
Step 3 (GPU compute) parallelises the checks. Step 4 (spatial hash) reduces
the count from n² to ~n by only checking nearby cells.

---

## Tuning suggestions

In `sim/simulation.rs`, try changing:
- `friction`: 0.01 = very fluid, 0.15 = sluggish
- `dt`: 0.0005 = slow/stable, 0.002 = fast/chaotic

In `sim/interaction.rs`, try changing:
- `R`: 0.05 = tight clusters, 0.2 = loose long-range interactions
- `R_MIN`: smaller = particles overlap more before repelling
- `force_scale` in the force() function: higher = more energetic