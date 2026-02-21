//! GPU management for AI inference workers
//!
//! Provides a unified interface for GPU detection, memory tracking,
//! and device operations. Uses Burn's wgpu backend by default for
//! automatic GPU detection across NVIDIA, AMD, and Intel.

use std::collections::HashMap;
use std::sync::Arc;


use dashmap::DashMap;
use tokio::sync::Semaphore;
use tracing::{info, warn, debug};
use std::process::Command;

use crate::cluster::GpuInfo;
use crate::error::WorkerError;

/// The active backend type, selected at compile time via features.
///
/// Default is Wgpu which auto-detects GPU vendor via Vulkan/DX12/Metal.
/// Users can opt into native CUDA or ROCm for maximum performance.
#[cfg(feature = "wgpu")]
pub type ActiveBackend = burn::backend::Wgpu;

#[cfg(all(feature = "cuda", not(feature = "wgpu")))]
pub type ActiveBackend = burn::backend::Cuda;

#[cfg(all(feature = "rocm", not(feature = "wgpu"), not(feature = "cuda")))]
pub type ActiveBackend = burn::backend::Rocm;

#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "cuda"), not(feature = "rocm")))]
pub type ActiveBackend = burn::backend::NdArray;

/// GPU device information
#[derive(Debug, Clone)]
pub struct GPUDevice {
    /// Device index
    pub id: usize,

    /// Device name/model
    pub name: String,

    /// Total VRAM in bytes
    pub total_memory: u64,

    /// Available VRAM in bytes
    pub available_memory: u64,

    /// Current utilization (0-100)
    pub utilization: f32,

    /// Temperature in Celsius
    pub temperature: f32,

    /// Power usage in watts
    pub power_usage: u32,

    /// Device capabilities
    pub capabilities: Vec<String>,

    /// Whether device supports peer-to-peer access
    pub supports_p2p: bool,
}

/// GPU memory allocation tracking
struct MemoryAllocation {
    /// Size in bytes
    size: u64,

    /// Allocation timestamp
    _timestamp: std::time::Instant,

    /// Owner (model name or request ID)
    owner: String,
}

/// GPU Manager — handles device detection and memory tracking
pub struct GPUManager {
    /// Available GPU devices
    devices: Vec<GPUDevice>,

    /// Device index mapping (original GPU ID → index in devices vec)
    device_map: HashMap<usize, usize>,

    /// Memory allocations per device
    allocations: Arc<DashMap<usize, Vec<MemoryAllocation>>>,

    /// Memory locks per device (for concurrent access)
    memory_locks: Vec<Arc<Semaphore>>,

    /// Whether peer-to-peer is enabled
    _p2p_enabled: bool,
}

impl GPUManager {
    /// Create a new GPU manager for the specified device IDs.
    ///
    /// If `gpu_ids` is empty, all detected GPUs are used.
    pub async fn new(gpu_ids: &[usize]) -> Result<Self, WorkerError> {
        info!("Initializing GPU manager with devices: {:?}", gpu_ids);

        let mut devices = Vec::new();
        let mut device_map = HashMap::new();
        let mut memory_locks = Vec::new();

        // Detect available devices
        let available_devices = Self::detect_devices().await;

        if available_devices.is_empty() {
            return Err(WorkerError::NoGpusFound);
        }

        info!("Detected {} GPU device(s)", available_devices.len());

        // Use all devices if none specified
        let ids_to_use: Vec<usize> = if gpu_ids.is_empty() {
            (0..available_devices.len()).collect()
        } else {
            gpu_ids.to_vec()
        };

        for (idx, &gpu_id) in ids_to_use.iter().enumerate() {
            if gpu_id >= available_devices.len() {
                warn!("GPU {} not available, skipping", gpu_id);
                continue;
            }

            let mut device = available_devices[gpu_id].clone();
            device.id = idx;

            info!(
                "Initialized GPU {}: {} ({}MB total, {}MB free)",
                idx,
                device.name,
                device.total_memory / 1024 / 1024,
                device.available_memory / 1024 / 1024,
            );

            devices.push(device);
            device_map.insert(gpu_id, idx);
            memory_locks.push(Arc::new(Semaphore::new(1)));
        }

        if devices.is_empty() {
            return Err(WorkerError::NoGpusFound);
        }

        Ok(Self {
            devices,
            device_map,
            allocations: Arc::new(DashMap::new()),
            memory_locks,
            _p2p_enabled: false,
        })
    }

    /// Detect available GPU devices using wgpu adapter enumeration.
    /// Deduplicates multiple backends (Vulkan/DX12/GL) for the same physical card.
    async fn detect_devices() -> Vec<GPUDevice> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default() & !wgpu::InstanceFlags::VALIDATION & !wgpu::InstanceFlags::DEBUG,
            ..Default::default()
        });
        
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let mut devices = Vec::new();
        let mut seen_hardware = std::collections::HashSet::new();

        for adapter in adapters {
            let info = adapter.get_info();
            
            // Skip software/CPU renderers if we have hardware options, but keep as fallback
            if info.device_type == wgpu::DeviceType::Cpu && !devices.is_empty() {
                continue;
            }

            // Create a unique key for the physical hardware to avoid double-counting 
            // (e.g. same card via Vulkan and DX12)
            let hardware_id = format!("{}-{}-{:?}", info.name, info.vendor, info.device_type);
            if seen_hardware.contains(&hardware_id) {
                continue;
            }
            seen_hardware.insert(hardware_id);

            let idx = devices.len();
            let name = info.name.clone();
            let mut total_memory = Self::estimate_total_memory(); 
            
            // Try to get precise VRAM for NVIDIA via nvidia-smi
            if info.name.to_lowercase().contains("nvidia") {
                if let Some(vram) = Self::try_detect_nvidia_memory() {
                    total_memory = vram;
                }
            }

            debug!(
                "Detected {} adapter {}: {} ({:?}) - VRAM: {}MB",
                info.backend, idx, name, info.device_type, total_memory / 1024 / 1024
            );

            devices.push(GPUDevice {
                id: idx,
                name: format!("{} ({})", name, info.backend),
                total_memory,
                available_memory: total_memory,
                utilization: 0.0,
                temperature: 0.0,
                power_usage: 0,
                capabilities: vec!["fp32".to_string()],
                supports_p2p: false,
            });
        }

        if devices.is_empty() {
             devices.push(GPUDevice {
                id: 0,
                name: "CPU Fallback".to_string(),
                total_memory: 0,
                available_memory: 0,
                utilization: 0.0,
                temperature: 0.0,
                power_usage: 0,
                capabilities: vec!["fp32".to_string()],
                supports_p2p: false,
            });
        }

        devices
    }

    /// Try to run nvidia-smi to get total VRAM
    fn try_detect_nvidia_memory() -> Option<u64> {
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
            .ok()?;
            
        if output.status.success() {
            let val_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Ok(mb) = val_str.parse::<u64>() {
                return Some(mb * 1024 * 1024);
            }
        }
        None
    }

    /// Try to detect the GPU name from the environment
    fn detect_gpu_name() -> String {
        // On Windows, we can try to read GPU info from environment or WMI
        // For now, report based on active backend
        #[cfg(feature = "wgpu")]
        {
            return "GPU (wgpu auto-detected)".to_string();
        }
        #[cfg(feature = "cuda")]
        {
            return "NVIDIA GPU (CUDA)".to_string();
        }
        #[cfg(feature = "rocm")]
        {
            return "AMD GPU (ROCm)".to_string();
        }
        #[allow(unreachable_code)]
        "Unknown GPU".to_string()
    }

    /// Estimate total GPU memory
    /// Since wgpu cannot query VRAM size across all vendors easily yet,
    /// we allow overriding it via environment variable, defaulting to 8GB.
    fn estimate_total_memory() -> u64 {
        let gb = std::env::var("GPU_VRAM_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(8);
        
        gb * 1024 * 1024 * 1024
    }

    /// Get number of GPU devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device by internal index
    pub fn get_device(&self, idx: usize) -> Option<&GPUDevice> {
        self.devices.get(idx)
    }

    /// Get device by original GPU ID
    pub fn get_device_by_original_id(&self, original_id: usize) -> Option<&GPUDevice> {
        self.device_map
            .get(&original_id)
            .and_then(|&idx| self.devices.get(idx))
    }

    /// Get all GPU info for status reporting (gRPC)
    pub async fn get_all_gpu_info(&self) -> Vec<GpuInfo> {
        let mut infos = Vec::new();

        for device in &self.devices {
            let available = self.get_available_memory(device.id).await;

            infos.push(GpuInfo {
                id: device.id as i32,
                name: device.name.clone(),
                total_memory: device.total_memory,
                available_memory: available,
                utilization: device.utilization,
                temperature: device.temperature,
                power_usage: device.power_usage,
                capabilities: device.capabilities.clone(),
            });
        }

        infos
    }

    /// Get available memory for a device (total minus tracked allocations)
    pub async fn get_available_memory(&self, device_id: usize) -> u64 {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return 0,
        };

        let mut available = device.available_memory;

        if let Some(allocations) = self.allocations.get(&device_id) {
            for alloc in allocations.value() {
                available = available.saturating_sub(alloc.size);
            }
        }

        available
    }

    /// Allocate memory on a device (tracking only — actual GPU alloc via Burn)
    pub async fn allocate_memory(
        &self,
        device_id: usize,
        size: u64,
        owner: String,
    ) -> Result<(), WorkerError> {
        let _permit = self.memory_locks[device_id]
            .acquire()
            .await
            .map_err(|e| WorkerError::Resource(format!("Failed to acquire memory lock: {}", e)))?;

        let available = self.get_available_memory(device_id).await;
        if available < size {
            return Err(WorkerError::OutOfMemory {
                requested: size as usize,
                available: available as usize,
                device: device_id,
            });
        }

        let mut allocations = self.allocations.entry(device_id).or_default();
        allocations.push(MemoryAllocation {
            size,
            _timestamp: std::time::Instant::now(),
            owner,
        });

        debug!("Allocated {} bytes on GPU {}", size, device_id);
        Ok(())
    }

    /// Free memory allocated to an owner
    pub async fn free_memory(&self, owner: &str) {
        for mut entry in self.allocations.iter_mut() {
            let device_id = *entry.key();
            let allocations = entry.value_mut();
            allocations.retain(|alloc| alloc.owner != owner);
            debug!("Freed memory for {} on GPU {}", owner, device_id);
        }
    }

    /// Get memory usage statistics for a device: (total, available, num_allocations)
    pub async fn get_memory_stats(&self, device_id: usize) -> (u64, u64, usize) {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return (0, 0, 0),
        };

        let allocated: u64 = self
            .allocations
            .get(&device_id)
            .map(|alloc| alloc.value().iter().map(|a| a.size).sum())
            .unwrap_or(0);

        let available = device.available_memory.saturating_sub(allocated);
        let count = self
            .allocations
            .get(&device_id)
            .map(|alloc| alloc.value().len())
            .unwrap_or(0);

        (device.total_memory, available, count)
    }

    /// Check if all GPUs are healthy
    pub async fn is_healthy(&self) -> bool {
        for device in &self.devices {
            let available = self.get_available_memory(device.id).await;
            if available == 0 && device.total_memory > 0 {
                return false;
            }
            if device.temperature > 100.0 {
                return false;
            }
        }
        true
    }

    /// Get system memory information: (free, total)
    pub async fn system_memory(&self) -> (u64, u64) {
        // Fallback — real system memory detection can be added later
        (0, 0)
    }
}