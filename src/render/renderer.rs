// =============================================================================
// render/renderer.rs — with aspect ratio correction
//
// New additions vs step 3:
//   - RenderParams uniform buffer holding the aspect ratio
//   - A bind group layout + bind group for the render pipeline
//   - aspect_buffer updated every time resize() is called
// =============================================================================

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;
use crate::sim::gpu_sim::GpuSim;

// Matches the RenderParams struct in render.wgsl.
// Must be 16-byte aligned — pad to 4 floats.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    aspect: f32,
    _pad:   [f32; 3],
}

pub struct Renderer {
    pub surface:      wgpu::Surface<'static>,
    pub device:       wgpu::Device,
    pub queue:        wgpu::Queue,
    config:           wgpu::SurfaceConfiguration,
    size:             PhysicalSize<u32>,
    render_pipeline:  wgpu::RenderPipeline,
    pub num_particles: u32,

    // aspect ratio uniform
    aspect_buffer:    wgpu::Buffer,
    render_bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, num_particles: u32) -> Self {
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

        // ---------------------------------------------------------------------
        // Aspect ratio uniform buffer
        // Updated every time the window is resized.
        // ---------------------------------------------------------------------
        let initial_aspect = size.width as f32 / size.height as f32;
        let aspect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Aspect Buffer"),
            contents: bytemuck::bytes_of(&RenderParams {
                aspect: initial_aspect,
                _pad:   [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---------------------------------------------------------------------
        // Bind group layout for the render pipeline
        // One uniform buffer at binding 0 — the RenderParams.
        // ---------------------------------------------------------------------
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Render BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Render Bind Group"),
            layout:  &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: aspect_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Render Pipeline Layout"),
            bind_group_layouts:   &[&render_bgl],  // <-- bind group now included
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         "vs_main",
                buffers:             &[GpuSim::vertex_layout()],
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            num_particles,
            aspect_buffer,
            render_bind_group,
        }
    }

    // =========================================================================
    // resize() — rebuilds swap chain AND updates the aspect ratio uniform
    // =========================================================================
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size          = new_size;
            self.config.width  = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Push the new aspect ratio to the GPU uniform buffer
            let aspect = new_size.width as f32 / new_size.height as f32;
            self.queue.write_buffer(
                &self.aspect_buffer,
                0,
                bytemuck::bytes_of(&RenderParams { aspect, _pad: [0.0; 3] }),
            );
        }
    }

    pub fn render(&mut self, sim: &mut crate::sim::GpuSim) {
        let output = match self.surface.get_current_texture() {
            Ok(t)                             => t,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e)                            => panic!("Surface error: {e:?}"),
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") }
        );

        sim.tick(&mut encoder);

        let particle_buffer = sim.render_buffer();

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.08, a: 1.0
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });

            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.render_bind_group, &[]); // aspect ratio
            rp.set_vertex_buffer(0, particle_buffer.slice(..));
            rp.draw(0..self.num_particles, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        sim.advance();
    }
}