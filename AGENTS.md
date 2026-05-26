# AGENTS.md — particle-life v2

## Codebase reality vs README

The README describes step 2 (CPU physics, `sim/simulation.rs`). The codebase is actually at **step 5** — all physics run on GPU via compute shaders. The `sim/simulation.rs` file no longer exists; replaced by `sim/gpu_sim.rs` + `sim/spatial_hash.rs`.

## Controls

| Key | Action |
|-----|--------|
| R | Randomise interaction (force) matrix |
| T | Randomise state transfer matrix |
| V | Toggle view mode: species colour / state greyscale |
| `[` `]` | Decrease / increase state decay |
| `-` `=` | Decrease / increase state transfer scale |
| 1 / Space | Reset: random placement |
| 2 | Reset: 8 clusters |
| 3 | Reset: concentric rings (one per type) |
| 4 | Reset: grid |
| C | Reset camera |
| Scroll | Zoom toward cursor |
| RMB drag | Pan |
| Esc | Quit |

## Rust edition

`edition = "2024"` in Cargo.toml — notably newer than the common 2021 default.

## Constraints

- No tests exist anywhere in the repo
- No CI, no lint/format config, no pre-commit hooks
- wgpu surface config uses `AutoNoVsync` + `desired_maximum_frame_latency: 2`
- `env_logger::init()` called in `main()` — set `RUST_LOG=warn` to see wgpu diagnostics
- `pollster::block_on()` for async wgpu init (no tokio/async runtime)
- Only dependency for random numbers is `rand 0.8`


# Particle Life Simulation — Complete Agent Context

## 1. Project Overview

A GPU-accelerated **Particle Life simulation** written in Rust. Particles belong to species, exert attraction/repulsion forces on each other (governed by an interaction matrix), and carry internal state that evolves through a state transfer matrix. The long-term vision is a rich emergent ecosystem simulator with environmental fields, evolutionary dynamics, and complex multi-layer interactions.

**Design philosophy:** Correctness over speed of delivery. Every implementation step is verified to be visually/logically correct before moving on. Code is heavily commented.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Rust (stable) |
| GPU API | wgpu 0.20 (WebGPU backend) |
| Shader language | WGSL |
| Windowing | winit |
| Buffer casting | bytemuck |
| UI (deferred) | egui — NOT YET integrated |

**Why this stack:** Rust + wgpu + WGSL was deliberately chosen as the optimal fit for the long-term goal of large-scale GPU-accelerated simulation. egui integration is deferred until the parameter set stabilises, to avoid rework.

---

## 3. Project File Structure

```
src/
  main.rs                    # Entry point: winit event loop, state init, frame dispatch
  render/
    mod.rs                   # Re-exports
    renderer.rs              # wgpu render pipeline, vertex buffer, draw calls
  sim/
    mod.rs                   # Re-exports, top-level sim types
    interaction.rs           # InteractionMatrix, StateTransferMatrix, force curve helpers
    gpu_sim.rs               # GpuSim: owns all GPU buffers, runs compute passes each frame
    spatial_hash.rs          # SpatialHash: four-pass GPU counting sort
  shaders/
    render.wgsl              # Vertex + fragment shader for particle rendering
    compute.wgsl             # Main physics compute shader (forces, integration)
    hash_assign.wgsl         # Spatial hash pass 1: assign particles to cells
    hash_count.wgsl          # Spatial hash pass 2: count particles per cell
    hash_prefix.wgsl         # Spatial hash pass 3: prefix sum (exclusive scan)
    hash_scatter.wgsl        # Spatial hash pass 4: scatter particles into sorted order
```

---

## 4. Core Data Structures

### 4.1 GpuParticle (Rust + WGSL)

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],      // offset  0
    pub velocity: [f32; 2],      // offset  8
    pub ptype:    f32,            // offset 16  (species index)
    pub state:    f32,            // offset 20  (0.0–1.0)
    pub _pad_color: [f32; 2],    // offset 24  ← pushes color to align=16
    pub color:    [f32; 3],      // offset 32  (species RGB)
    pub _pad_end: f32,           // offset 44  ← total must be 48 (multiple of 16)
}
// Total: 48 bytes
```

**WGSL alignment rule:** `vec3<f32>` in WGSL has 16-byte alignment. The Rust struct explicitly places `_pad_color` at offset 24 so `color` lands at offset 32 (a 16-byte boundary). The corresponding WGSL struct in every shader is:

```wgsl
struct GpuParticle {
    position: vec2<f32>,  // offset 0
    velocity: vec2<f32>,  // offset 8
    ptype:    f32,         // offset 16
    state:    f32,         // offset 20
    // implicit 8-byte WGSL padding for vec3 alignment
    color:    vec3<f32>,  // offset 32
    _pad:     f32,         // offset 44
}
```

**Do NOT change this layout without updating the WGSL struct in EVERY shader** (`compute.wgsl`, `hash_assign.wgsl`, and any new shader that reads `GpuParticle`).

### 4.2 SimParams (Uniform buffer, shared across all passes)

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub n_particles:          u32,
    pub n_types:              u32,
    pub dt:                   f32,
    pub friction:             f32,
    pub r_outer:              f32,
    pub r_inner:              f32,
    pub force_scale:          f32,
    pub repulse_str:          f32,
    pub grid_dim:             u32,
    pub state_decay:          f32,
    pub state_transfer_scale: f32,
    pub _pad:                 u32,  // pad to 48 bytes (multiple of 16)
}
// Total: 48 bytes
```

This is a single uniform buffer bound to every compute pass via the same bind group slot.

**Important:** Some hash shaders (`hash_count.wgsl`, `hash_prefix.wgsl`, `hash_scatter.wgsl`) declare a shorter SimParams struct (only up to `grid_dim` + `_pad`). This works because those shaders only read fields up to `grid_dim` and wgpu allows a larger buffer than the shader declares. If you add fields before `grid_dim`, you must update all shaders.

### 4.3 InteractionMatrix

A `num_species × num_species` matrix of `f32` values in `[-1.0, 1.0]`.  
`matrix[a][b]` = force that species `b` exerts on species `a`.  
Positive = attraction, negative = repulsion.  
Stored flat: `interactions: Vec<f32>` of length `num_species²`.  
Uploaded to GPU as a storage buffer.

### 4.4 StateTransferMatrix

A `num_species × num_species` matrix of `f32` values.  
`matrix[a][b]` = how much a particle of species `a` shifts its internal state toward 1.0 when near a particle of species `b`.  
Used in `compute.wgsl` to update `particle.state` each frame.

---

## 5. GPU Architecture

### 5.1 Ping-Pong Double Buffer

Two GPU storage buffers for particles: `particle_buf_a` and `particle_buf_b`.  
Each frame, the compute shader reads from one and writes to the other, then they are swapped.  
This avoids read/write hazards since WGSL compute shaders cannot read and write the same buffer safely.

**Implementation in GpuSim:**
- `frame_idx: usize` tracks which buffer is the current "read" buffer (`frame_idx % 2` → 0 = A, 1 = B).
- Incremented after each frame via `advance()`: `self.frame_idx += 1;`
- The render pass always reads from the opposite buffer (the one just written by compute).

### 5.2 Spatial Hash (Four-Pass GPU Counting Sort)

Enables O(1) neighbour lookup instead of O(N²) all-pairs. Critical for scaling to hundreds of thousands of particles.

**The world is divided into a grid.** Each cell has side length = `max_force_radius`, so only the 3×3 neighbourhood of a particle's cell needs to be checked.

**Four passes, each a separate compute shader dispatch:**

**Pass 1 — hash_assign.wgsl**
Each particle computes its cell index: `cell = floor(pos / cell_size)`. Writes `(particle_index, cell_index)` pairs into a `cell_ids` buffer and a `particle_ids` buffer.

**Pass 2 — hash_count.wgsl**
Atomically increments a counter for each cell: `cell_counts[cell_id] += 1`.

**Pass 3 — hash_prefix.wgsl**
Computes an exclusive prefix sum over `cell_counts` → `cell_start`. After this pass, `cell_start[c]` = the index in the sorted output where cell `c`'s particles begin.

**Pass 4 — hash_scatter.wgsl**
Each particle looks up its cell's start index, atomically claims a slot (`atomicAdd(cell_start[cell], 1)`), and writes itself into `sorted_particles[slot]`. After scatter, particles are grouped by cell.

**Why the `cell_start` buffer needs `COPY_SRC`:** The prefix sum result is copied (via `copy_buffer_to_buffer`) into a `cell_cursor` buffer for the scatter pass, where `atomicAdd` claims slots. Without `COPY_SRC`, the buffer-to-buffer copy silently fails.

**The assign bind group is externalised and double-buffered.**  
The hash assign shader must read from the *current frame's* particle buffer (the ping-pong read side). Because the read side swaps each frame, two separate bind groups (`assign_bg_a`, `assign_bg_b`) are pre-built — one pointing at buffer A, one at buffer B — and the correct one is selected each frame. This was a critical bug fix: using a single stale bind group caused the spatial hash to operate on last frame's data.

### 5.3 Bind Group Layout Summary

| Pass | Bind Group 0 | Bind Group 1 |
|---|---|---|
| hash_assign | SimParams (uniform) | particles_read (storage, read-only) |
| hash_count | SimParams (uniform) | cell_ids (storage), cell_counts (storage) |
| hash_prefix | SimParams (uniform) | cell_counts (storage, r/w) → cell_start |
| hash_scatter | SimParams (uniform) | particle_ids, cell_ids, cell_start, sorted_particles |
| compute | SimParams (uniform) | particles_read, particles_write, sorted, cell_start, interaction_matrix, state_transfer_matrix |
| render | — | particles_read (vertex buffer) |

---

## 6. Physics

### 6.1 Force Curve

For two particles at distance `d`, with interaction strength `s = matrix[species_a][species_b]`:

```
if d < r_inner:
    force = -repulse_str * (1 - d/r_inner)   // always repel
elif d < r_outer:
    // tent function: peaks at midpoint between r_inner and r_outer
    t = (d - r_inner) / (r_outer - r_inner)
    tent = 1 - |2t - 1|
    force = tent * s * force_scale
else:
    force = 0
```

The force is applied along the unit vector from particle B to particle A.  
**The repulsion term has a negative sign** — this was a bug fix. Without it, particles at very close range would accelerate toward each other instead of bouncing, causing explosive clustering.

The force curve in `interaction.rs` uses constants (`force_scale: 0.5`, `repulse_str: 2.0`). The GPU version in `compute.wgsl` reads these from the `SimParams` uniform buffer (`repulse_str: 4.0`, `force_scale: 0.5`). The CPU source is not actually used at runtime — all physics run on GPU — but the interaction.rs function is retained for reference/documentation.

### 6.2 Integration

```wgsl
// In compute.wgsl
velocity = velocity + force * dt;
velocity = velocity * (1.0 - friction);   // friction = fraction lost per frame
position = position + velocity;
// Wrap around world boundary (toroidal):
if position.x >  1.0 { position.x -= 2.0; }
if position.x < -1.0 { position.x += 2.0; }
if position.y >  1.0 { position.y -= 2.0; }
if position.y < -1.0 { position.y += 2.0; }
```

`friction` is the fraction of velocity lost each frame (e.g. `friction: 0.2` → 20% loss per frame, keeping 80%). It's a damping factor, not a physical coefficient. World coordinates are in `[-1, 1]²`.

### 6.3 State Update

Each frame, for each particle A:
```
state_delta = 0
for each nearby particle B:
    state_delta += state_transfer_matrix[species_A][species_B] * weight(d)
particle.state = clamp(particle.state + state_delta * dt, 0.0, 1.0)
```

`state` is a single `f32` in [0, 1] per particle representing an internal "energy" or "activation" level.

---

## 7. Rendering

### 7.1 Pipeline

Particles are rendered as point sprites. The render pipeline:
- Vertex shader reads `GpuParticle` directly from the storage buffer via standard vertex attributes (`@location` bindings matching `GpuSim::vertex_layout()`).
- Outputs `position` (clip space) and `color`.
- Fragment shader outputs the color — previously used `@builtin(point_coord)` for circular masking, but this was **removed** in wgpu 0.20 (breaking change). The current implementation renders square points or uses an alternative masking approach.

### 7.2 Camera

- **Zoom:** Mouse scroll wheel → scale factor applied to projection matrix.
- **Pan:** Click and drag → translation offset in world space.
- **Aspect ratio correction:** Projection matrix accounts for window aspect ratio so the simulation world is not distorted on non-square windows.

### 7.3 Dual View Mode (V key toggle)

- **Species colour mode:** Each particle is coloured by its species RGB.
- **Greyscale state mode:** Each particle is rendered as greyscale where brightness = `particle.state`. Useful for visualising internal state dynamics.

---

## 8. Initialisation Configs

Four `InitConfig` variants for particle placement at startup:

```rust
pub enum InitConfig {
    Random,                    // Uniform random positions, random species assignment
    Clustered { n_clusters: u32 },  // Species grouped into spatial clusters
    Rings,                     // Species arranged in concentric rings
    Grid,                      // Species tiled in a regular grid pattern
}
```

---

## 9. Known Bugs Fixed (Historical — Do Not Reintroduce)

**Bug 1: Missing `COPY_SRC` on `cell_start` buffer**  
`cell_start` must have `BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC`. Without `COPY_SRC`, certain wgpu validation layers panic or bind group creation silently fails.

**Bug 2: `@builtin(point_coord)` removed in wgpu 0.20**  
wgpu 0.20 removed this builtin. The fragment shader must not reference it. Circular point rendering must use an alternative method (e.g. a unit quad per particle via instancing, or skip circular masking entirely).

**Bug 3: Missing negative sign in repulsion**  
The repulsion branch of the force curve was missing a `-` sign, causing attraction instead of repulsion at close range. Symptom: particles collapse into singularities.

**Bug 4: Edge particle clustering from non-toroidal cell lookup**  
The 3×3 cell neighbourhood search was not wrapping at world edges. Particles near the boundary had fewer neighbours counted, causing incorrect force imbalance and edge clustering. Fixed by wrapping cell indices with modulo arithmetic in the neighbourhood loop.

**Bug 5: Stale assign bind group**  
The spatial hash assign pass was using a single static bind group pointing at buffer A regardless of ping-pong state. Fixed by creating two bind groups (`assign_bg_a`, `assign_bg_b`) and selecting the correct one based on `self.frame_idx % 2` each frame.

---

## 10. Architecture Decisions & Rationale

| Decision | Rationale |
|---|---|
| Environmental fields over 3D | Fields add richer behavioural complexity. Adding a Z axis mainly adds geometry, not interesting dynamics. |
| Structural constraints on interaction matrices first | More interpretable and controllable than jumping straight to CMA-ES or MAP-Elites. |
| egui deferred until parameter set stabilises | Avoids building UI around a moving target. Keyboard controls are sufficient for exploration phase. |
| WGSL over GLSL/HLSL | Native wgpu shading language; no translation layer. |
| bytemuck for buffer casting | Zero-cost safe transmutation of Rust structs to `&[u8]` for GPU upload. |
| Ping-pong buffers | Avoids GPU read/write hazard on particle buffer during compute. |
| Externalised double-buffered assign bind group | Only correct solution to the stale-buffer problem in the spatial hash. |

---

## 11. Planned / On The Horizon

These are NOT yet implemented. Do not assume they exist in the codebase:

- **Environmental fields:** Scalar or vector fields on a grid that particles read and/or write. Higher leverage than 3D geometry for emergent complexity.
- **Structured interaction matrix control:** Constraint templates (e.g. predator-prey, mutualism, competition topologies) before introducing search algorithms.
- **Search/optimisation for matrices:** CMA-ES, MAP-Elites, novelty search — long-term, after structural constraints.
- **egui UI:** Sliders for `dt`, `friction`, interaction matrix editing, species counts. Deferred.

---

## 12. Crate Dependencies (Cargo.toml)

```toml
[dependencies]
wgpu = "0.20"
winit = { version = "...", features = ["..."] }
bytemuck = { version = "...", features = ["derive"] }
# egui and egui-wgpu: NOT YET added
```

Exact versions: check `Cargo.lock`. The wgpu 0.20 API is the baseline; do not silently upgrade to 0.21+ without auditing breaking changes.

---

## 13. WGSL Conventions Used In This Codebase

- Workgroup size: `@compute @workgroup_size(256)` for most passes (compute, assign, count, scatter). The prefix sum pass is single-threaded (`@workgroup_size(1)`).
- Particle index: `let idx = global_invocation_id.x;` — always guard with `if idx >= num_particles { return; }`.
- Atomics: `atomicAdd` used in hash_count and hash_scatter. Buffer must be declared `var<storage, read_write>` with `atomic<u32>` element type.
- World wrap: Conditional `if` checks for `pos > 1.0` / `pos < -1.0`, adjusting by ±2.0. World is `[-1, 1]²`.

- Storage buffer access for particles: `var<storage, read>` for source, `var<storage, read_write>` for destination.

---

## 14. How To Build & Run

```bash
cargo run --release
```

Debug build is too slow for the simulation to run at useful framerates. Always use `--release`.

---

## 15. Agent Instructions

When working on this codebase:

1. **Never change `GpuParticle` layout** without updating the WGSL struct in ALL shaders.
2. **Never remove buffer usages flags** — wgpu is strict; missing flags cause runtime panics, not compile errors.
3. **Ping-pong state** is tracked in `GpuSim::frame_idx: usize`. Any new compute pass that reads particles must use the correct buffer (`if self.frame_idx % 2 == 0 { buf_a } else { buf_b }`).
4. **Assign bind group must stay double-buffered.** Do not collapse it back to a single bind group.
5. **wgpu 0.20 API only.** Do not use deprecated or removed APIs (e.g. `point_coord` builtin).
6. **Spatial hash cell indices must wrap** at world edges. Always use `cell_x % grid_dim` and `cell_y % grid_dim` in neighbourhood loops.
7. **SimParams is a shared uniform.** If you add fields, update the Rust struct, the WGSL struct in every shader, and re-pad to a 16-byte multiple.
8. **`bytemuck::Pod`** requires no padding bytes — all padding must be explicit named fields.
9. **Physics correctness:** repulsion is always negative (pushing apart). Attraction is positive. The force curve is a tent shape in the `[r_inner, r_outer]` range.
10. **egui is not in the project yet.** Do not add it unless explicitly asked.
