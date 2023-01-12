use nannou::prelude::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    settings: Settings,
    atoms: Vec<Atom>,
}

fn model(app: &App) -> Model {
    let _window = app.new_window().view(view).build().unwrap();

    let (r_min, r_max) = (0.3, 100.0);
    let friction = 0.5;
    const num: usize = 100;
    let settings = Settings {
        r_min,
        r_max,
        friction,
        num,
    };
    let mut atoms: Vec<Atom> = vec![Atom::default(); num];
    let (bx, by) = app.window_rect().w_h();
    for mut i in 0..atoms.len() {
        atoms[i].pos.x = random_range(-bx / 2.0, bx / 2.0);
        atoms[i].pos.y = random_range(-by / 2.0, by / 2.0);
    }

    Model {
        _window,
        settings,
        atoms,
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(DARKSLATEGRAY);
    //draw.ellipse().color(STEELBLUE);
    for i in &model.atoms {
        i.draw(&draw);
    }
    draw.to_frame(app, &frame).unwrap();
}
struct Settings {
    r_min: f32,
    r_max: f32,
    friction: f32,
    num: usize,
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
    fn get_force(&self, p: &Atom, s: &Settings, rel: &Relation) -> Vec2 {
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
        _ => SEASHELL,
    }
}
