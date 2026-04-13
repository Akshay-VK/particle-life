// =============================================================================
// render/renderer.rs — wgpu setup and particle rendering  (Step 2)
//
// Changes from Step 1:
//   - Particle struct removed; we now use sim::GpuParticle (identical layout)
//   - particle_buffer is now COPY_DST so we can re-upload positions every frame
//   - update() added: takes fresh GpuParticle data and writes it to the buffer
//   - generate_particles() removed; the Simulation owns initial state now
// =============================================================================

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

// GpuParticle lives in sim::simulation — same memory layout as the old Particle.
// We import it here so the renderer stays decoupled from physics details.
use crate::sim::simulation::GpuParticle;

impl GpuParticle {
    // Vertex buffer layout — must match @location bindings in render.wgsl
    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2, // position
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // color
                },
            ],
        }
    }
}

pub struct Renderer {
    surface:         wgpu::Surface<'static>,
    device:          wgpu::Device,
    queue:           wgpu::Queue,
    config:          wgpu::SurfaceConfiguration,
    size:            PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    particle_buffer: wgpu::Buffer,
    num_particles:   u32,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, initial_particles: &[GpuParticle]) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        println!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to get device");

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  wgpu::PresentMode::AutoNoVsync,
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Pipeline Layout"),
            bind_group_layouts:   &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         "vs_main",
                buffers:             &[GpuParticle::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
        });

        // -----------------------------------------------------------------------
        // Particle buffer — seeded with initial_particles from the Simulation.
        //
        // VERTEX:   read by the vertex shader each frame
        // COPY_DST: lets queue.write_buffer() overwrite it with new positions
        //
        // The buffer is sized for the initial particle count and never resized.
        // Changing particle count requires rebuilding the buffer (not done here).
        // -----------------------------------------------------------------------
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Particle Buffer"),
            contents: bytemuck::cast_slice(initial_particles),
            usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            particle_buffer,
            num_particles: initial_particles.len() as u32,
        }
    }

    // =========================================================================
    // UPDATE — upload fresh particle positions to the GPU buffer.
    //
    // Called every frame after Simulation::tick().
    //
    // queue.write_buffer() is the standard way to push CPU data to the GPU
    // for a buffer flagged COPY_DST. It enqueues a copy that runs before the
    // next render pass sees the buffer — so the draw call always gets fresh data.
    //
    // In step 3 we replace this entire function: positions will be updated
    // by a compute shader directly on the GPU, so nothing needs to come back
    // to the CPU at all.
    // =========================================================================
    pub fn update(&self, particles: &[GpuParticle]) {
        self.queue.write_buffer(
            &self.particle_buffer,
            0,                              // byte offset into the buffer
            bytemuck::cast_slice(particles) // &[GpuParticle] → &[u8]
        );
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size          = new_size;
            self.config.width  = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t)                             => t,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e)                            => panic!("Surface error: {e:?}"),
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") }
        );

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });

            rp.set_pipeline(&self.render_pipeline);
            rp.set_vertex_buffer(0, self.particle_buffer.slice(..));
            rp.draw(0..self.num_particles, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}