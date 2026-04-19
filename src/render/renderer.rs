// =============================================================================
// render/renderer.rs
// =============================================================================

use crate::sim::gpu_sim::GpuSim;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

// Must match RenderParams in render.wgsl exactly — 32 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    aspect: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    view_mode: u32, // 0 = species, 1 = state
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    pub num_particles: u32,

    render_params: RenderParams,
    render_buf: wgpu::Buffer,
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
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        println!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to get device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let render_params = RenderParams {
            aspect: size.width as f32 / size.height as f32,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            view_mode: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let render_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: render_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[GpuSim::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            num_particles,
            render_params,
            render_buf,
            render_bind_group,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.render_params.aspect = new_size.width as f32 / new_size.height as f32;
            self.upload_render_params();
        }
    }

    // =========================================================================
    // Camera
    // =========================================================================

    pub fn zoom_at(&mut self, delta: f32, cursor_x: f32, cursor_y: f32) {
        let factor = if delta > 0.0 { 1.15f32 } else { 1.0 / 1.15 };

        let w = self.size.width as f32;
        let h = self.size.height as f32;
        let ndc_x = (cursor_x / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor_y / h) * 2.0;

        // World point under cursor before zoom
        let world_x =
            ndc_x * self.render_params.aspect / self.render_params.zoom - self.render_params.pan_x;
        let world_y = ndc_y / self.render_params.zoom - self.render_params.pan_y;

        self.render_params.zoom = (self.render_params.zoom * factor).clamp(0.1, 50.0);

        // Recompute pan so that world point stays under cursor
        self.render_params.pan_x =
            ndc_x * self.render_params.aspect / self.render_params.zoom - world_x;
        self.render_params.pan_y = ndc_y / self.render_params.zoom - world_y;

        self.upload_render_params();
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let w = self.size.width as f32;
        let h = self.size.height as f32;
        self.render_params.pan_x +=
            dx * 2.0 * self.render_params.aspect / (w * self.render_params.zoom);
        self.render_params.pan_y -= dy * 2.0 / (h * self.render_params.zoom);
        self.upload_render_params();
    }

    pub fn reset_camera(&mut self) {
        self.render_params.zoom = 1.0;
        self.render_params.pan_x = 0.0;
        self.render_params.pan_y = 0.0;
        self.upload_render_params();
    }

    // =========================================================================
    // View mode toggle
    // =========================================================================

    pub fn toggle_view(&mut self) {
        self.render_params.view_mode = if self.render_params.view_mode == 0 {
            1
        } else {
            0
        };
        let name = if self.render_params.view_mode == 0 {
            "SPECIES"
        } else {
            "STATE"
        };
        println!("View: {name}");
        self.upload_render_params();
    }

    // =========================================================================

    fn upload_render_params(&self) {
        self.queue
            .write_buffer(&self.render_buf, 0, bytemuck::bytes_of(&self.render_params));
    }

    pub fn render(&mut self, sim: &mut crate::sim::GpuSim) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => panic!("Surface error: {e:?}"),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        sim.tick(&mut encoder);

        let particle_buffer = sim.render_buffer();
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.render_bind_group, &[]);
            rp.set_vertex_buffer(0, particle_buffer.slice(..));
            rp.draw(0..self.num_particles, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        sim.advance();
    }
}
