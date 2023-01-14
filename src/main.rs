use nannou::prelude::*;
use nannou_egui::{self, egui, Egui};

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    settings: Settings,
    atoms: Vec<Atom>,
    egui: Egui,
}
fn init(app: &App, _window: window::Id) -> Model {
    let win = app.window(_window).unwrap();
    let egui = Egui::from_window(&win);

    let (r_min, r_max) = (0.2, 50.0);
    let friction = 0.2;

    const num: usize = 1200;
    const num_t: usize = 2;
    let mut rel = Relation::new(num_t);
    rel.set(0.03, 0 as usize, 0 as usize);
    rel.set(0.015, 1 as usize, 0 as usize);
    rel.set(-0.02, 0 as usize, 1 as usize);
    rel.set(-0.02, 1 as usize, 1 as usize);

    let settings = Settings {
        r_min,
        r_max,
        friction,
        num,
        num_t,
        rel,
        pn: num,
        pnt: num_t,
    };
    let mut atoms: Vec<Atom> = vec![Atom::default(); num];
    let (bx, by) = app.window_rect().w_h();
    for i in 0..atoms.len() {
        atoms[i].pos.x = random_range(-bx / 2.0, bx / 2.0);
        atoms[i].pos.y = random_range(-by / 2.0, by / 2.0);
        atoms[i].t = (i % num_t) as usize;
    }
    Model {
        _window,
        settings,
        atoms,
        egui,
    }
}
fn restart(app: &App, _window: window::Id, n: usize, n_t: usize) -> Model {
    let m = init(&app, _window);
    let num: usize = n;
    let num_t: usize = n_t;

    let rel = Relation::new(num_t);

    let settings = Settings {
        r_min: m.settings.r_min,
        r_max: m.settings.r_max,
        friction: m.settings.friction,
        num,
        num_t,
        rel,
        pn: num,
        pnt: num_t,
    };
    let mut atoms: Vec<Atom> = vec![Atom::default(); n];
    let (bx, by) = app.window_rect().w_h();
    for i in 0..atoms.len() {
        atoms[i].pos.x = random_range(-bx / 2.0, bx / 2.0);
        atoms[i].pos.y = random_range(-by / 2.0, by / 2.0);
        atoms[i].t = (i % num_t) as usize;
    }
    let e = Egui::from_window(&(app.window(_window).unwrap()));
    Model {
        _window,
        settings,
        atoms,
        egui: e,
    }
}
fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .title("Particle life")
        .size(640, 320)
        .raw_event(raw_events)
        .event(events)
        .view(view)
        .build()
        .unwrap();

    init(&app, _window)
}

fn update(_app: &App, model: &mut Model, update: Update) {
    //-------------------EGUI-------------------
    let mut egui = &mut model.egui;
    //let set = &mut model.settings;
    egui.set_elapsed_time(update.since_start);

    let c = egui.begin_frame();

    egui::Window::new("Settings for particle life: ").show(&c, |ui| {
        ui.label("Friction: ");
        ui.add(egui::Slider::new(&mut model.settings.friction, 0.01..=0.75));

        ui.label("Minumum distance:");
        ui.add(egui::Slider::new(&mut model.settings.r_min, 0.01..=0.9));

        ui.label("Maximum distance:");
        ui.add(egui::Slider::new(&mut model.settings.r_max, 5.0..=200.0));

        ui.label("Species:");
        ui.add(egui::Slider::new(
            &mut model.settings.pnt,
            1 as usize..=5 as usize,
        ));

        ui.label("Number of particles:");
        ui.add(egui::Slider::new(
            &mut model.settings.pn,
            500 as usize..=5000 as usize,
        ));

        //ui.add(
        egui::Grid::new("Atomic relations:").show(ui, |ui| {
            ui.label("");
            for i in 0..model.settings.rel.table.len() {
                ui.label(format!("{}", i));
            }
            ui.end_row();
            for i in 0..model.settings.rel.table.len() {
                ui.label(format!("{}", i));
                for j in 0..model.settings.rel.table[i].len() {
                    ui.add(egui::Slider::new(
                        &mut model.settings.rel.table[i][j],
                        -0.5..=0.5,
                    ));
                }
                ui.end_row();
            }
        })
        //);

        //

        //drop(c);
    });
    //------------------------------------------
    if model.settings.pn != model.settings.num || model.settings.pnt != model.settings.num_t {
        let mut dmod = restart(_app, model._window, model.settings.pn, model.settings.pnt);
        model._window = dmod._window;
        model.atoms = dmod.atoms;
        //egui = &mut dmod.egui;
        model.settings = dmod.settings;
    }

    for i in 0..model.atoms.len() {
        let mut f = Vec2::ZERO;
        for j in 0..model.atoms.len() {
            if i == j {
                continue;
            }
            f = f + model.atoms[i].get_force(&model.atoms[j], &model.settings);
        }
        //println!("{}", f);
        model.atoms[i].apply_forces(f, &model.settings);
        model.atoms[i].update();
    }
}
fn events(_app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        KeyPressed(_k) => {
            //
        }
        _ => {}
    }
}
fn raw_events(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(DARKSLATEGRAY);
    //draw.ellipse().color(STEELBLUE);
    for i in &model.atoms {
        i.draw(&draw);
    }
    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
struct Settings {
    r_min: f32,
    r_max: f32,
    friction: f32,
    num: usize,
    num_t: usize,
    rel: Relation,
    pn: usize,
    pnt: usize,
}
#[derive(Copy, Clone, Debug)]
struct Atom {
    pos: Vec2,
    vel: Vec2,
    t: usize,
}
impl Atom {
    fn new(pos: Vec2, vel: Vec2, t: usize) -> Self {
        Self { pos, vel, t }
    }
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            t: 0 as usize,
        }
    }
    fn get_force(&self, p: &Atom, s: &Settings) -> Vec2 {
        let rel = &s.rel;

        let delta = p.pos - self.pos;
        let mut d = delta.length();
        if d > s.r_max {
            Vec2::ZERO
        } else {
            d = d / s.r_max;
            if d < s.r_min {
                delta * ((d / s.r_min) - 1.0)
            } else {
                let avg = (s.r_min + s.r_max) / 2.0;
                if d > avg {
                    d = d - avg + s.r_min;
                }
                let g = rel.get(self.t, p.t);
                delta * s.r_max * (g * (d - s.r_min) / (avg - s.r_min))
            }
        }
    }
    fn apply_forces(&mut self, f: Vec2, s: &Settings) {
        self.vel = (self.vel + f) * s.friction;
    }
    fn update(&mut self) {
        self.pos = self.pos + self.vel;
    }
    fn draw(&self, d: &Draw) {
        let col = get_col(self.t);
        d.ellipse()
            .color(col)
            .x_y(self.pos.x, self.pos.y)
            .w_h(5.0, 5.0);
    }
}
struct Relation {
    table: Vec<Vec<f32>>,
}
impl Relation {
    fn new(s: usize) -> Self {
        Self {
            table: vec![vec![0.0; s]; s],
        }
    }
    fn get(&self, c: usize, r: usize) -> f32 {
        self.table[c][r]
    }
    fn set(&mut self, v: f32, c: usize, r: usize) {
        self.table[c][r] = v;
    }
}
fn get_col(t: usize) -> Rgb<u8> {
    match t {
        0 => SALMON,
        1 => SEAGREEN,
        2 => PLUM,
        3 => INDIANRED,
        4 => KHAKI,
        5 => LIGHTCORAL,
        6 => LIGHTPINK,
        _ => SEASHELL,
    }
}
