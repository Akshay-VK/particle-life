// =============================================================================
// main.rs  (Step 3)
//
// Much simpler than Step 2 — the CPU no longer ticks physics or uploads data.
// The main loop just calls renderer.render(sim) which dispatches both the
// compute and render passes in one encoder submission.
//
// Changes from Step 2:
//   - No more sim.tick() / sim.gpu_particles() / renderer.update() calls
//   - GpuSim::new() takes the device + queue (it creates its own GPU resources)
//   - Renderer::new() no longer needs initial particle data
//   - R key: randomise matrix + upload via sim.update_matrix()
//   - Space: rebuild GpuSim entirely (re-scatters particles, new matrix)
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

// Raise this now that physics runs on the GPU.
// Try 50_000 to start. On a modern GPU, 100_000+ should be smooth at O(n²).
// Step 4 (spatial hash) will push this to millions.
const NUM_PARTICLES: u32 = 15_000;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Event loop failed");
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Particle Life  |  R = new rules  |  Space = reset")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32))
            .build(&event_loop)
            .expect("Window failed")
    );

    // -------------------------------------------------------------------------
    // Init renderer first — we need the device to build GpuSim.
    // -------------------------------------------------------------------------
    let mut renderer = pollster::block_on(
        render::Renderer::new(window.clone(), NUM_PARTICLES)
    );

    // -------------------------------------------------------------------------
    // Build the GPU simulation using the renderer's device + queue.
    // GpuSim allocates its buffers and pipelines here.
    // -------------------------------------------------------------------------
    let matrix = sim::InteractionMatrix::random();
    matrix.print();

    let mut gpu_sim = sim::GpuSim::new(
        &renderer.device,
        &renderer.queue,
        NUM_PARTICLES,
        &matrix,
    );

    // fps counter
    let mut frames    = 0u32;
    let mut fps_timer = std::time::Instant::now();

    event_loop.run(move |event, elwt| {
        match event {

            Event::AboutToWait => window.request_redraw(),

            Event::WindowEvent { event, .. } => match event {

                WindowEvent::Resized(s) => renderer.resize(s),

                WindowEvent::RedrawRequested => {
                    // ---------------------------------------------------------
                    // THE ENTIRE MAIN LOOP IS NOW ONE LINE.
                    //
                    // renderer.render() internally:
                    //   1. Acquires the swap chain texture
                    //   2. Creates a command encoder
                    //   3. Records compute pass (physics via gpu_sim.tick())
                    //   4. Records render pass (draws gpu_sim.render_buffer())
                    //   5. Submits both passes together
                    //   6. Presents the frame
                    //   7. Calls gpu_sim.advance() to flip ping-pong
                    // ---------------------------------------------------------
                    renderer.render(&mut gpu_sim);

                    frames += 1;
                    if fps_timer.elapsed().as_secs_f32() >= 1.0 {
                        println!("FPS: {}  |  Particles: {}", frames, NUM_PARTICLES);
                        frames    = 0;
                        fps_timer = std::time::Instant::now();
                    }
                }

                // R — new interaction rules, keep particle positions
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyR),
                        state: winit::event::ElementState::Pressed, ..
                    }, ..
                } => {
                    let new_matrix = sim::InteractionMatrix::random();
                    new_matrix.print();
                    gpu_sim.update_matrix(&renderer.queue, &new_matrix);
                }

                // Space — full reset: new positions + new rules
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        state: winit::event::ElementState::Pressed, ..
                    }, ..
                } => {
                    println!("--- Full reset ---");
                    let new_matrix = sim::InteractionMatrix::random();
                    new_matrix.print();
                    gpu_sim = sim::GpuSim::new(
                        &renderer.device,
                        &renderer.queue,
                        NUM_PARTICLES,
                        &new_matrix,
                    );
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