use axum::{
    extract::State,
    http::{Method, StatusCode, header},
    routing::{get, options, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::RwLock;

#[derive(Clone, Serialize, Deserialize)]
pub struct SharedState {
    pub num_particles: u32,
    pub num_species: u32,
    pub dt: f32,
    pub max_force_radius: f32,
    pub min_distance: f32,
    pub friction: f32,
    pub world_size: f32,
    pub interaction_matrix: Vec<f32>,
    pub state_transfer_matrix: Vec<f32>,
    pub species_colors: Vec<[f32; 3]>,
    pub paused: bool,
    pub speed_multiplier: f32,
    pub reset_requested: bool,
    pub randomise_matrix_requested: bool,
}

pub struct AppState {
    pub state: RwLock<SharedState>,
    pub dirty: AtomicBool,
}

pub async fn run_server(app_state: Arc<AppState>) {
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/state", get(get_state))
        .route("/params", post(post_params))
        .route("/*_", options(handle_options))
        .layer(cors)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind server");
    axum::serve(listener, app)
        .await
        .expect("Server error");
}

async fn get_state(state: State<Arc<AppState>>) -> Json<SharedState> {
    let guard = state.state.read().await;
    Json(guard.clone())
}

async fn post_params(
    state: State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> StatusCode {
    let mut guard = state.state.write().await;

    if let Some(v) = body.get("num_particles").and_then(|v| v.as_u64()) {
        guard.num_particles = v as u32;
    }
    if let Some(v) = body.get("num_species").and_then(|v| v.as_u64()) {
        guard.num_species = v as u32;
    }
    if let Some(v) = body.get("dt").and_then(|v| v.as_f64()) {
        guard.dt = v as f32;
    }
    if let Some(v) = body.get("max_force_radius").and_then(|v| v.as_f64()) {
        guard.max_force_radius = v as f32;
    }
    if let Some(v) = body.get("min_distance").and_then(|v| v.as_f64()) {
        guard.min_distance = v as f32;
    }
    if let Some(v) = body.get("friction").and_then(|v| v.as_f64()) {
        guard.friction = v as f32;
    }
    if let Some(v) = body.get("world_size").and_then(|v| v.as_f64()) {
        guard.world_size = v as f32;
    }
    if let Some(v) = body.get("speed_multiplier").and_then(|v| v.as_f64()) {
        guard.speed_multiplier = v as f32;
    }
    if let Some(v) = body.get("paused").and_then(|v| v.as_bool()) {
        guard.paused = v;
    }
    if let Some(v) = body.get("reset_requested").and_then(|v| v.as_bool()) {
        guard.reset_requested = v;
    }
    if let Some(v) = body.get("randomise_matrix_requested").and_then(|v| v.as_bool()) {
        guard.randomise_matrix_requested = v;
    }
    if let Some(arr) = body.get("interaction_matrix").and_then(|v| v.as_array()) {
        guard.interaction_matrix = arr
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
    }
    if let Some(arr) = body.get("state_transfer_matrix").and_then(|v| v.as_array()) {
        guard.state_transfer_matrix = arr
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
    }
    if let Some(arr) = body.get("species_colors").and_then(|v| v.as_array()) {
        guard.species_colors = arr
            .iter()
            .filter_map(|v| {
                v.as_array().and_then(|c| {
                    if c.len() == 3 {
                        Some([
                            c[0].as_f64()? as f32,
                            c[1].as_f64()? as f32,
                            c[2].as_f64()? as f32,
                        ])
                    } else {
                        None
                    }
                })
            })
            .collect();
    }

    state.dirty.store(true, Ordering::Release);
    StatusCode::OK
}

async fn handle_options() -> StatusCode {
    StatusCode::OK
}
