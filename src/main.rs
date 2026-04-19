// =============================================================================
// main.rs
//
// Controls:
//   Scroll wheel      — zoom toward cursor
//   Right mouse drag  — pan
//   C                 — reset camera
//   R                 — randomise interaction (force) matrix
//   T                 — randomise state transfer matrix
//   [ ]               — state decay down / up
//   - =               — state transfer scale down / up
//   1                 — reset: Random placement
//   2                 — reset: Clustered (8 clusters)
//   3                 — reset: Rings (one ring per type)
//   4                 — reset: Grid
//   Space             — reset: Random (same as 1)
//   Escape            — quit
// =============================================================================

mod render;
mod sim;

use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

const NUM_PARTICLES: u32 = 20_000;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Event loop failed");
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title(
                "Particle Life  |  Scroll=zoom  RMB=pan  C=cam  V=view  R=rules  T=state  []=decay  -/==transfer  1-4=init"
            )
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
            .build(&event_loop)
            .expect("Window failed")
    );

    let mut renderer = pollster::block_on(render::Renderer::new(window.clone(), NUM_PARTICLES));

    let matrix = sim::InteractionMatrix::random();
    matrix.print();
    let mut gpu_sim = sim::GpuSim::new(
        &renderer.device,
        &renderer.queue,
        NUM_PARTICLES,
        &matrix,
        sim::InitConfig::Random,
    );

    // ---- input state --------------------------------------------------------
    let mut cursor_pos = (0.0f32, 0.0f32); // last known cursor in pixels
    let mut last_drag = (0.0f32, 0.0f32); // cursor position when RMB went down
    let mut rmb_down = false;

    // ---- fps counter --------------------------------------------------------
    let mut frames = 0u32;
    let mut fps_timer = std::time::Instant::now();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::AboutToWait => window.request_redraw(),

                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(s) => renderer.resize(s),

                    // ------------------------------------------------------------------
                    // Main render loop
                    // ------------------------------------------------------------------
                    WindowEvent::RedrawRequested => {
                        renderer.render(&mut gpu_sim);

                        frames += 1;
                        if fps_timer.elapsed().as_secs_f32() >= 1.0 {
                            println!(
                                "FPS: {:3}  |  decay: {:.4}  transfer_scale: {:.2}",
                                frames,
                                gpu_sim.params.state_decay,
                                gpu_sim.params.state_transfer_scale,
                            );
                            frames = 0;
                            fps_timer = std::time::Instant::now();
                        }
                    }

                    // ------------------------------------------------------------------
                    // Scroll wheel — zoom toward cursor position
                    // ------------------------------------------------------------------
                    WindowEvent::MouseWheel { delta, .. } => {
                        let lines = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                        };
                        renderer.zoom_at(lines, cursor_pos.0, cursor_pos.1);
                    }

                    // ------------------------------------------------------------------
                    // Cursor tracking — used for zoom pivot and pan delta
                    // ------------------------------------------------------------------
                    WindowEvent::CursorMoved { position, .. } => {
                        let nx = position.x as f32;
                        let ny = position.y as f32;
                        if rmb_down {
                            let dx = nx - last_drag.0;
                            let dy = ny - last_drag.1;
                            renderer.pan(dx, dy);
                        }
                        last_drag = (nx, ny);
                        cursor_pos = (nx, ny);
                    }

                    // ------------------------------------------------------------------
                    // Right mouse button — pan
                    // ------------------------------------------------------------------
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Right,
                        ..
                    } => {
                        rmb_down = state == ElementState::Pressed;
                    }

                    // ------------------------------------------------------------------
                    // Keyboard
                    // ------------------------------------------------------------------
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match code {
                        // R — new force rules, keep positions
                        KeyCode::KeyR => {
                            let m = sim::InteractionMatrix::random();
                            m.print();
                            gpu_sim.update_matrix(&renderer.queue, &m);
                        }

                        // T — new state transfer matrix
                        KeyCode::KeyT => {
                            gpu_sim.randomise_state_transfer(&renderer.queue);
                        }

                        // C — reset camera to default
                        KeyCode::KeyC => {
                            renderer.reset_camera();
                            println!("Camera reset");
                        }

                        // [ — state decay down
                        KeyCode::BracketLeft => {
                            gpu_sim.params.state_decay =
                                (gpu_sim.params.state_decay - 0.005).max(0.0);
                            gpu_sim.update_params(&renderer.queue);
                            println!("State decay: {:.4}", gpu_sim.params.state_decay);
                        }

                        // ] — state decay up
                        KeyCode::BracketRight => {
                            gpu_sim.params.state_decay =
                                (gpu_sim.params.state_decay + 0.005).min(0.5);
                            gpu_sim.update_params(&renderer.queue);
                            println!("State decay: {:.4}", gpu_sim.params.state_decay);
                        }

                        // - — state transfer scale down
                        KeyCode::Minus => {
                            gpu_sim.params.state_transfer_scale =
                                (gpu_sim.params.state_transfer_scale - 0.1).max(0.0);
                            gpu_sim.update_params(&renderer.queue);
                            println!("Transfer scale: {:.2}", gpu_sim.params.state_transfer_scale);
                        }

                        // = — state transfer scale up
                        KeyCode::Equal => {
                            gpu_sim.params.state_transfer_scale =
                                (gpu_sim.params.state_transfer_scale + 0.1).min(5.0);
                            gpu_sim.update_params(&renderer.queue);
                            println!("Transfer scale: {:.2}", gpu_sim.params.state_transfer_scale);
                        }

                        // 1-4 — init configs
                        KeyCode::Digit1 | KeyCode::Space => do_reset(
                            &mut gpu_sim,
                            &renderer,
                            NUM_PARTICLES,
                            sim::InitConfig::Random,
                        ),
                        KeyCode::Digit2 => do_reset(
                            &mut gpu_sim,
                            &renderer,
                            NUM_PARTICLES,
                            sim::InitConfig::Clustered { n_clusters: 8 },
                        ),
                        KeyCode::Digit3 => do_reset(
                            &mut gpu_sim,
                            &renderer,
                            NUM_PARTICLES,
                            sim::InitConfig::Rings,
                        ),
                        KeyCode::Digit4 => do_reset(
                            &mut gpu_sim,
                            &renderer,
                            NUM_PARTICLES,
                            sim::InitConfig::Grid,
                        ),

                        // V — toggle view mode (Species / State)
                        KeyCode::KeyV => renderer.toggle_view(),

                        KeyCode::Escape => elwt.exit(),
                        _ => {}
                    },

                    WindowEvent::CloseRequested => elwt.exit(),
                    _ => {}
                },
                _ => {}
            }
        })
        .expect("Event loop error");
}

fn do_reset(
    gpu_sim: &mut sim::GpuSim,
    renderer: &render::Renderer,
    n: u32,
    config: sim::InitConfig,
) {
    let m = sim::InteractionMatrix::random();
    m.print();
    *gpu_sim = sim::GpuSim::new(&renderer.device, &renderer.queue, n, &m, config);
    println!("Reset complete");
}
