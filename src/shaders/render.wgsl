// =============================================================================
// render.wgsl
//
// Two view modes, toggled with V key:
//
//   mode 0 — SPECIES view (default)
//     Particles coloured by type. State tint removed — it was too subtle
//     and muddied the colours. This view is for watching force dynamics.
//
//   mode 1 — STATE view
//     All particles rendered as black→grey→white based purely on state value.
//     Type colour is ignored entirely. This makes state patterns clearly
//     visible: dark = inactive (0), bright white = fully excited (1).
//     This view is reserved for state-related layers going forward.
//
// RenderParams layout (32 bytes):
//   offset  0: aspect    f32
//   offset  4: zoom      f32
//   offset  8: pan_x     f32
//   offset 12: pan_y     f32
//   offset 16: view_mode u32   (0 = species, 1 = state)
//   offset 20: _pad0     u32
//   offset 24: _pad1     u32
//   offset 28: _pad2     u32
// =============================================================================

struct RenderParams {
    aspect:    f32,
    zoom:      f32,
    pan_x:     f32,
    pan_y:     f32,
    view_mode: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

@group(0) @binding(0) var<uniform> rp: RenderParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:          vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) color:    vec3<f32>,
    @location(2) state:    f32,
) -> VertexOutput {
    var out: VertexOutput;

    // Camera transform: pan then zoom, then aspect correction
    let world = (position + vec2<f32>(rp.pan_x, rp.pan_y)) * rp.zoom;
    out.clip_position = vec4<f32>(world.x / rp.aspect, world.y, 0.0, 1.0);

    if rp.view_mode == 0u {
        // SPECIES view: pure type colour, no state tint
        out.color = color;
    } else {
        // STATE view: black (0) → white (1), ignores type colour entirely
        out.color = vec3<f32>(state, state, state);
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
