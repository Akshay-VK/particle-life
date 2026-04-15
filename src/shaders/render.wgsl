// =============================================================================
// render.wgsl — camera (zoom + pan) + aspect ratio + state colour tint
//
// RenderParams layout (32 bytes, 16-byte aligned):
//   aspect  f32  — width/height
//   zoom    f32  — 1.0 = default, >1 = zoomed in
//   pan_x   f32  — world-space pan offset X
//   pan_y   f32  — world-space pan offset Y
//
// State colour tint:
//   Each particle's state (0..1) shifts its colour toward white.
//   This gives a live visual of which particles are "excited" without
//   needing a separate render pass.
// =============================================================================

struct RenderParams {
    aspect: f32,
    zoom:   f32,
    pan_x:  f32,
    pan_y:  f32,
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

    // Apply camera: translate by pan, then scale by zoom
    // pan is in world space (before zoom), so panning speed is consistent
    let world  = (position + vec2<f32>(rp.pan_x, rp.pan_y)) * rp.zoom;

    // Aspect correction: squeeze X so the square world fills a rectangle screen
    out.clip_position = vec4<f32>(world.x / rp.aspect, world.y, 0.0, 1.0);

    // Tint colour toward white based on state (0=original, 1=white)
    out.color = mix(color, vec3<f32>(1.0, 1.0, 1.0), state * 0.6);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
