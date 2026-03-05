use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;

// Use Wgpu with default settings (f32, i32, BestAvailable/Auto)
/// Primary inference backend (Wgpu, auto-selects best available GPU API).
pub type WorkerBackend = Wgpu;
/// Autodiff wrapper around [`WorkerBackend`] for training/gradient workloads.
pub type WorkerAutodiffBackend = Autodiff<WorkerBackend>;
