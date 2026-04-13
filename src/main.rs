// =============================================================================
// main.rs  (Step 2)
//
// Changes from Step 1:
//   - `mod sim` added
//   - Simulation::new() called before renderer (renderer needs initial GPU data)
//   - sim.tick() called every frame before renderer.render()
//   - renderer.update() uploads fresh positions to GPU after each tick
//   - Keyboard: R = randomise matrix, Space = reset positions
// =============================================================================

mod render;
mod sim;

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

// How many particles to simulate. At 5000, CPU O(n²) runs comfortably at 60fps.
// Push to ~8000 and you'll start feeling it — that's the motivation for step 3.
const NUM_PARTICLES: usize = 1_000;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Event loop failed");
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Particle Life  |  R = new rules  |  Space = reset")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
            .build(&event_loop)
            .expect("Window creation failed")
    );

    // -------------------------------------------------------------------------
    // Create the simulation FIRST.
    //
    // The renderer needs the initial GPU particle data to size its buffer.
    // So we tick the simulation once (just to generate positions), convert
    // to GpuParticles, then hand those to the renderer constructor.
    // -------------------------------------------------------------------------
    let mut sim = sim::Simulation::new(NUM_PARTICLES);
    let initial_gpu = sim.gpu_particles();

    let mut renderer = pollster::block_on(
        render::Renderer::new(window.clone(), &initial_gpu)
    );

    // fps tracking — prints to terminal every second
    let mut frame_count  = 0u32;
    let mut last_fps_time = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        match event {

            Event::AboutToWait => {
                window.request_redraw();
            }

            Event::WindowEvent { event, .. } => match event {

                WindowEvent::Resized(s) => renderer.resize(s),

                WindowEvent::RedrawRequested => {
                    // ---------------------------------------------------------
                    // MAIN LOOP — runs every frame
                    //
                    // Order matters:
                    //   1. tick()         — advance physics (CPU)
                    //   2. gpu_particles() — convert to upload format
                    //   3. update()       — write positions to GPU buffer
                    //   4. render()       — GPU draws the updated buffer
                    // ---------------------------------------------------------
                    sim.tick();
                    let gpu_data = sim.gpu_particles();
                    renderer.update(&gpu_data);
                    renderer.render();

                    // Print fps once per second
                    frame_count += 1;
                    let elapsed = last_fps_time.elapsed();
                    if elapsed.as_secs_f32() >= 1.0 {
                        println!("FPS: {}", frame_count);
                        frame_count  = 0;
                        last_fps_time = std::time::Instant::now();
                    }
                }

                // R — randomise the interaction matrix
                // You'll see the particles reorganise into new patterns
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyR),
                        state: winit::event::ElementState::Pressed, ..
                    }, ..
                } => {
                    println!("--- Randomising interaction matrix ---");
                    sim.randomise_matrix();
                }

                // Space — scatter particles back to random positions
                // Good for seeing the system settle from scratch with current rules
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        state: winit::event::ElementState::Pressed, ..
                    }, ..
                } => {
                    println!("--- Resetting particle positions ---");
                    sim = sim::Simulation::new(NUM_PARTICLES);
                    // Note: matrix is also re-randomised here because Simulation::new
                    // always generates a fresh one. Later we can add reset_positions()
                    // that keeps the current matrix but re-scatters particles.
                }

                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape), ..
                    }, ..
                } => elwt.exit(),

                _ => {}
            },

            _ => {}
        }
    }).expect("Event loop error");
}