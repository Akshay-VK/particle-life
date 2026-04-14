// =============================================================================
// render.wgsl — aspect ratio corrected
//
// The problem: particle positions are in [-1, 1] on both axes (clip space).
// Clip space maps directly to the screen rectangle, so on a 1280x720 window
// one clip-space unit is 720px tall but only ~562px wide. A particle that
// moves 0.1 units right appears to move less than one moving 0.1 units up —
// circles look like ellipses.
//
// The fix: store the simulation in a square world [-1,1]x[-1,1], then in the
// vertex shader scale X by (height/width) before passing to clip space. This
// "squeezes" the X axis so that equal world-space distances look equal on screen.
//
// aspect = width / height
// corrected_x = world_x / aspect   (shrink X to compensate for wide screen)
// =============================================================================

struct RenderParams {
    aspect: f32,   // width / height, updated on every resize
    _pad0:  f32,   // uniforms must be 16-byte aligned — pad to 16 bytes
    _pad1:  f32,
    _pad2:  f32,
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
) -> VertexOutput {
    var out: VertexOutput;

    // Divide x by aspect ratio to map the square world onto the rectangular screen.
    // On a 16:9 window (aspect=1.777), x is compressed by ~56% so circles stay round.
    let corrected = vec2<f32>(position.x / rp.aspect, position.y);

    out.clip_position = vec4<f32>(corrected, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
