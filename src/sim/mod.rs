// sim/mod.rs
pub mod interaction;
pub mod gpu_sim;
pub mod spatial_hash;

pub use interaction::InteractionMatrix;
pub use gpu_sim::GpuSim;
pub use spatial_hash::SpatialHash;