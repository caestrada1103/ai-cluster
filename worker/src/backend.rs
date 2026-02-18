use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;

// Use Wgpu with default settings (f32, i32, BestAvailable/Auto)
pub type WorkerBackend = Wgpu;
pub type WorkerAutodiffBackend = Autodiff<WorkerBackend>;
