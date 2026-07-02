//! GPU management for AI inference workers
//!
//! Provides a unified interface for GPU detection, memory tracking,
//! and device operations. Uses Burn's wgpu backend by default for
//! automatic GPU detection across NVIDIA, AMD, and Intel.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};


use dashmap::DashMap;
use tokio::sync::Semaphore;
use tracing::{info, warn, debug};
use std::process::Command;

use crate::cluster::GpuInfo;
use crate::error::WorkerError;

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

}

/// GPU memory allocation tracking
struct MemoryAllocation {
    /// Owner tag (model name) — used to free the reservation on unload.
    tag: String,
    /// Reserved bytes.
    size: u64,
    /// Allocation timestamp
    _timestamp: std::time::Instant,
}

/// GPU Manager — handles device detection and memory tracking
pub struct GPUManager {
    /// Available GPU devices
    devices: Vec<GPUDevice>,

    /// Memory allocations per device
    allocations: Arc<DashMap<usize, Vec<MemoryAllocation>>>,

    /// Running sum of allocated bytes per device — updated atomically on alloc/free
    /// to avoid O(n) iteration in `get_available_memory`.
    used_bytes: Arc<Vec<AtomicU64>>,

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
            memory_locks.push(Arc::new(Semaphore::new(1)));
        }

        if devices.is_empty() {
            return Err(WorkerError::NoGpusFound);
        }

        let num_devices = devices.len();
        Ok(Self {
            devices,
            allocations: Arc::new(DashMap::new()),
            used_bytes: Arc::new((0..num_devices).map(|_| AtomicU64::new(0)).collect()),
            memory_locks,
            _p2p_enabled: false,
        })
    }

    /// Detect available GPU devices using wgpu adapter enumeration.
    /// Deduplicates multiple backends (Vulkan/DX12/GL) for the same physical card.
    async fn detect_devices() -> Vec<GPUDevice> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default()
                .difference(wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG),
            ..Default::default()
        });
        
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let mut devices: Vec<(GPUDevice, bool)> = Vec::new();
        let mut seen_hardware = std::collections::HashSet::new();

        for adapter in adapters {
            let info = adapter.get_info();

            let is_cpu = info.device_type == wgpu::DeviceType::Cpu;

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
            let lname = info.name.to_lowercase();
            if lname.contains("nvidia") {
                if let Some(vram) = Self::try_detect_nvidia_memory(idx) {
                    total_memory = vram;
                }
            } else if lname.contains("amd") || lname.contains("radeon") {
                if let Some(vram) = Self::try_detect_amd_memory(idx) {
                    total_memory = vram;
                }
            }

            debug!(
                "Detected {} adapter {}: {} ({:?}) - VRAM: {}MB",
                info.backend, idx, name, info.device_type, total_memory / 1024 / 1024
            );

            devices.push((
                GPUDevice {
                    id: idx,
                    name: format!("{} ({})", name, info.backend),
                    total_memory,
                    available_memory: total_memory,
                    utilization: 0.0,
                    temperature: 0.0,
                    power_usage: 0,
                    capabilities: vec!["fp32".to_string()],
                },
                is_cpu,
            ));
        }

        // Prefer hardware adapters: drop CPU/software renderers whenever any real GPU exists,
        // regardless of enumeration order (llvmpipe often enumerates first).
        if devices.iter().any(|(_, is_cpu)| !is_cpu) {
            devices.retain(|(_, is_cpu)| !is_cpu);
        }
        let mut devices: Vec<GPUDevice> = devices
            .into_iter()
            .enumerate()
            .map(|(idx, (mut d, _))| {
                d.id = idx;
                d
            })
            .collect();

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
            });
        }

        devices
    }

    /// Try to run nvidia-smi to get total VRAM for one GPU (3-second timeout).
    /// nvidia-smi prints one line per GPU; pick the line matching `device_idx`.
    fn try_detect_nvidia_memory(device_idx: usize) -> Option<u64> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let out = Command::new("nvidia-smi")
                .arg("--query-gpu=memory.total")
                .arg("--format=csv,noheader,nounits")
                .output();
            let _ = tx.send(out);
        });
        let output = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .ok()?
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.lines().nth(device_idx).or_else(|| stdout.lines().next())?;
            if let Ok(mb) = line.trim().parse::<u64>() {
                return Some(mb * 1024 * 1024);
            }
        }
        None
    }

    /// Try to run rocm-smi to get total VRAM for AMD (3-second timeout).
    fn try_detect_amd_memory(device_idx: usize) -> Option<u64> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let out = Command::new("rocm-smi")
                .arg("--showmeminfo")
                .arg("vram")
                .arg("--json")
                .output();
            let _ = tx.send(out);
        });
        let output = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .ok()?
            .ok()?;
            
        if output.status.success() {
            let json_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
                // rocm-smi returns a JSON object where keys are usually "card0", "card1", etc.
                // We try "card{idx}" first, then fallback to any available card if idx fails.
                let card_key = format!("card{}", device_idx);
                
                if let Some(card_data) = v.get(&card_key).or_else(|| v.as_object().and_then(|obj| obj.values().next())) {
                    if let Some(vram_str) = card_data.get("VRAM Total Memory (B)").and_then(|v| v.as_str()) {
                        if let Ok(vram_bytes) = vram_str.parse::<u64>() {
                            return Some(vram_bytes);
                        }
                    }
                }
            }
        }
        None
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

    /// Get available memory for a device (O(1) via atomic running sum).
    pub async fn get_available_memory(&self, device_id: usize) -> u64 {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return 0,
        };
        let used = self.used_bytes
            .get(device_id)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(0);
        device.available_memory.saturating_sub(used)
    }

    /// Reserve memory on a device (tracking only — actual GPU alloc via Burn).
    /// `tag` identifies the owner (model name) so `free_memory` can release it.
    pub async fn allocate_memory(
        &self,
        device_id: usize,
        size: u64,
        tag: &str,
    ) -> Result<(), WorkerError> {
        let lock = self.memory_locks.get(device_id).ok_or_else(|| {
            WorkerError::Gpu(format!(
                "GPU {} not managed by this worker ({} device(s))",
                device_id,
                self.devices.len()
            ))
        })?;
        let _permit = lock
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

        self.allocations.entry(device_id).or_default().push(MemoryAllocation {
            tag: tag.to_string(),
            size,
            _timestamp: std::time::Instant::now(),
        });
        if let Some(counter) = self.used_bytes.get(device_id) {
            counter.fetch_add(size, Ordering::Relaxed);
        }

        debug!("Reserved {} bytes on GPU {} for {}", size, device_id, tag);
        Ok(())
    }

    /// Release every reservation carrying `tag` on `device_id`. Returns bytes freed.
    pub async fn free_memory(&self, device_id: usize, tag: &str) -> u64 {
        let mut freed: u64 = 0;
        if let Some(mut allocs) = self.allocations.get_mut(&device_id) {
            allocs.retain(|a| {
                if a.tag == tag {
                    freed += a.size;
                    false
                } else {
                    true
                }
            });
        }
        if freed > 0 {
            if let Some(counter) = self.used_bytes.get(device_id) {
                counter.fetch_sub(freed, Ordering::Relaxed);
            }
            debug!("Freed {} bytes on GPU {} for {}", freed, device_id, tag);
        }
        freed
    }

    /// Check if all GPUs are healthy
    pub async fn is_healthy(&self) -> bool {
        for device in &self.devices {
            if device.temperature > 100.0 {
                return false;
            }
        }
        true
    }

    /// Get system memory information from /proc/meminfo: (available, total) bytes.
    pub async fn system_memory(&self) -> (u64, u64) {
        let Ok(contents) = tokio::fs::read_to_string("/proc/meminfo").await else {
            return (0, 0);
        };
        let mut total = 0u64;
        let mut available = 0u64;
        for line in contents.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("MemTotal:") => total = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                Some("MemAvailable:") => {
                    available = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0)
                }
                _ => {}
            }
        }
        (available * 1024, total * 1024) // /proc/meminfo reports kB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager_with_one_device() -> GPUManager {
        let device = GPUDevice {
            id: 0,
            name: "test-gpu".to_string(),
            total_memory: 1_000,
            available_memory: 1_000,
            utilization: 0.0,
            temperature: 0.0,
            power_usage: 0,
            capabilities: vec!["fp32".to_string()],
        };
        GPUManager {
            devices: vec![device],
            allocations: Arc::new(DashMap::new()),
            used_bytes: Arc::new(vec![AtomicU64::new(0)]),
            memory_locks: vec![Arc::new(Semaphore::new(1))],
            _p2p_enabled: false,
        }
    }

    #[tokio::test]
    async fn test_allocate_then_free_restores_capacity() {
        let m = make_manager_with_one_device();
        m.allocate_memory(0, 600, "model-a").await.unwrap();
        assert_eq!(m.get_available_memory(0).await, 400);
        // Second allocation of 600 must fail (OOM guard)
        assert!(m.allocate_memory(0, 600, "model-b").await.is_err());
        let freed = m.free_memory(0, "model-a").await;
        assert_eq!(freed, 600);
        assert_eq!(m.get_available_memory(0).await, 1_000);
        // Now it fits
        m.allocate_memory(0, 600, "model-b").await.unwrap();
    }

    #[tokio::test]
    async fn test_allocate_invalid_device_errors_instead_of_panicking() {
        let m = make_manager_with_one_device();
        let err = m.allocate_memory(7, 100, "x").await.unwrap_err();
        assert!(err.to_string().contains("not managed"));
    }
}