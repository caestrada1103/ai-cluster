//! GPU management for AI inference workers
//!
//! Provides a unified interface for GPU detection, memory tracking,
//! and device operations. Uses Burn's wgpu backend by default for
//! automatic GPU detection across NVIDIA, AMD, and Intel.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use std::process::Command;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::cluster::GpuInfo;
use crate::error::WorkerError;

/// Fraction of system RAM held back from a unified-memory GPU budget, for the
/// OS, the coordinator, page cache, and llama.cpp's own compute buffers.
const DEFAULT_UNIFIED_HEADROOM_PERCENT: u64 = 15;

/// Subtract `headroom_pct` percent from `total`, saturating at 0.
///
/// Split out from [`GPUManager::unified_memory_budget`] so the arithmetic is
/// unit-testable without touching process-wide environment state.
fn apply_headroom(total: u64, headroom_pct: u64) -> u64 {
    let keep = 100u64.saturating_sub(headroom_pct.min(100));
    total / 100 * keep
}

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

    /// Current utilization (0-100) at detection time — live values come from
    /// `GpuTelemetry` via `refresh_telemetry`/`get_all_gpu_info` instead.
    #[allow(dead_code)]
    pub utilization: f32,

    /// Temperature in Celsius at detection time — see `utilization` note.
    #[allow(dead_code)]
    pub temperature: f32,

    /// Power usage in watts at detection time — see `utilization` note.
    #[allow(dead_code)]
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

/// Mutable GPU telemetry sampled from vendor tools at scrape time.
#[derive(Debug, Clone, Default)]
struct GpuTelemetry {
    utilization: f32,
    temperature: f32,
    power_usage: u32,
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

    /// Per-managed-device telemetry, refreshed on demand (metrics scrape / health check).
    telemetry: Arc<tokio::sync::RwLock<Vec<GpuTelemetry>>>,
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
            telemetry: Arc::new(tokio::sync::RwLock::new(vec![
                GpuTelemetry::default();
                num_devices
            ])),
        })
    }

    /// Backend preference order for [`select_hardware_adapters`]. Highest
    /// priority first.
    const BACKEND_PRIORITY: [wgpu::Backend; 4] = [
        wgpu::Backend::Vulkan,
        wgpu::Backend::Metal,
        wgpu::Backend::Dx12,
        wgpu::Backend::Gl,
    ];

    /// Pick ONE backend's adapter list and discard the rest, so the same
    /// physical GPU enumerated under multiple backends contributes exactly
    /// once. wgpu's `AdapterInfo` has no per-instance identifier, so this is
    /// the only strategy that never merges distinct hardware. See docs/configuration.md.
    fn select_hardware_adapters(infos: Vec<wgpu::AdapterInfo>) -> Vec<wgpu::AdapterInfo> {
        let mut by_backend: std::collections::HashMap<wgpu::Backend, Vec<wgpu::AdapterInfo>> =
            std::collections::HashMap::new();
        for info in infos {
            by_backend.entry(info.backend).or_default().push(info);
        }

        let has_real_gpu = |b: &wgpu::Backend| {
            by_backend
                .get(b)
                .is_some_and(|v| v.iter().any(|i| i.device_type != wgpu::DeviceType::Cpu))
        };

        let chosen = Self::BACKEND_PRIORITY
            .iter()
            .find(|b| has_real_gpu(b))
            .or_else(|| {
                Self::BACKEND_PRIORITY
                    .iter()
                    .find(|b| by_backend.contains_key(*b))
            })
            .copied()
            .or_else(|| by_backend.keys().next().copied());

        match chosen {
            Some(b) => by_backend.remove(&b).unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Detect available GPU devices via wgpu adapter enumeration. Collapses
    /// the same physical card enumerated under multiple backends down to one
    /// device — see [`select_hardware_adapters`].
    async fn detect_devices() -> Vec<GPUDevice> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default()
                .difference(wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG),
            ..Default::default()
        });

        let adapter_infos: Vec<wgpu::AdapterInfo> = instance
            .enumerate_adapters(wgpu::Backends::all())
            .iter()
            .map(|a| a.get_info())
            .collect();
        let selected = Self::select_hardware_adapters(adapter_infos);

        let mut devices: Vec<(GPUDevice, bool)> = Vec::new();

        for info in selected {
            let is_cpu = info.device_type == wgpu::DeviceType::Cpu;

            let idx = devices.len();
            let name = info.name.clone();
            // Integrated adapters share system RAM with the CPU, so their
            // budget comes from /proc/meminfo rather than a vendor VRAM query.
            let is_unified = info.device_type == wgpu::DeviceType::IntegratedGpu;
            let (total_memory, available_memory) = Self::detect_memory(idx, &info.name, is_unified);

            debug!(
                "Detected {} adapter {}: {} ({:?}) - VRAM: {}MB total, {}MB free",
                info.backend,
                idx,
                name,
                info.device_type,
                total_memory / 1024 / 1024,
                available_memory / 1024 / 1024,
            );

            devices.push((
                GPUDevice {
                    id: idx,
                    name: format!("{} ({})", name, info.backend),
                    total_memory,
                    available_memory,
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

    /// Resolve `(total_bytes, available_bytes)` for one adapter. Precedence:
    /// `GPU_VRAM_GB` override, then a real vendor query (CUDA/`nvidia-smi` or
    /// `rocm-smi`), then [`estimate_total_memory`].
    fn detect_memory(device_idx: usize, adapter_name: &str, is_unified: bool) -> (u64, u64) {
        if let Some(bytes) = Self::vram_override_bytes() {
            return (bytes, bytes);
        }

        let lname = adapter_name.to_lowercase();
        let vendor = if lname.contains("nvidia") {
            Self::detect_nvidia_memory(device_idx)
        } else if lname.contains("amd") || lname.contains("radeon") {
            Self::detect_amd_memory(device_idx)
        } else {
            None
        };

        if let Some((total, free)) = vendor {
            // On unified memory the vendor "free" figure treats reclaimable
            // page cache as used, so it can understate what is actually
            // obtainable by tens of GB. Prefer the headroom-adjusted budget,
            // which is derived from total RAM. See docs/configuration.md.
            if is_unified {
                let budget = Self::unified_memory_budget().unwrap_or(total).min(total);
                return (total, budget.max(free));
            }
            return (total, free);
        }

        let total = Self::estimate_total_memory(is_unified);
        (total, total)
    }

    /// Parse the `GPU_VRAM_GB` operator override, in bytes.
    fn vram_override_bytes() -> Option<u64> {
        std::env::var("GPU_VRAM_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|gb| gb * 1024 * 1024 * 1024)
    }

    /// NVIDIA `(total, free)` bytes: try the CUDA driver API first, then
    /// `nvidia-smi`. Returns `None` (falling through to the generic estimate)
    /// if neither source is usable.
    fn detect_nvidia_memory(device_idx: usize) -> Option<(u64, u64)> {
        if let Some(pair) = Self::try_detect_nvidia_memory_cuda(device_idx) {
            return Some(pair);
        }
        Self::try_detect_nvidia_memory_smi(device_idx)
    }

    /// Best-effort NVIDIA `(total, free)` bytes via the CUDA driver API,
    /// loaded at runtime with `dlopen` (see [`cuda_driver`]) so non-NVIDIA
    /// hosts/builds never link against it. Falls through to `nvidia-smi` on
    /// any failure. Needed because `nvidia-smi`/NVML both fail to report
    /// memory on unified-memory NVIDIA hardware (e.g. DGX Spark) — see
    /// docs/configuration.md for the measured numbers and rationale.
    #[cfg(target_os = "linux")]
    fn try_detect_nvidia_memory_cuda(device_idx: usize) -> Option<(u64, u64)> {
        cuda_driver::device_memory(device_idx)
    }

    /// Non-Linux stub: falls through to `nvidia-smi` (Windows/macOS don't
    /// exhibit the unified-memory `[N/A]` failure mode this works around).
    #[cfg(not(target_os = "linux"))]
    fn try_detect_nvidia_memory_cuda(_device_idx: usize) -> Option<(u64, u64)> {
        None
    }

    /// Try to run nvidia-smi to get total/free VRAM for one GPU (3-second
    /// timeout). nvidia-smi prints one line per GPU; pick the line matching
    /// `device_idx` (single-vendor-host assumption, same as
    /// [`GPUManager::refresh_telemetry`]).
    fn try_detect_nvidia_memory_smi(device_idx: usize) -> Option<(u64, u64)> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let out = Command::new("nvidia-smi")
                .arg("--query-gpu=memory.total,memory.free")
                .arg("--format=csv,noheader,nounits")
                .output();
            let _ = tx.send(out);
        });
        let output = rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .ok()?
            .ok()?;

        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .nth(device_idx)
            .or_else(|| stdout.lines().next())?;
        let mut fields = line.split(',').map(|s| s.trim());
        let total_mb: u64 = fields.next()?.parse().ok()?;
        let total = total_mb * 1024 * 1024;
        // Free is a bonus field: if it's missing or unparsable ("[N/A]"),
        // fall back to assuming the whole card is free rather than losing
        // the (successfully-parsed) total too.
        let free = fields
            .next()
            .and_then(|f| f.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(total);
        Some((total, free))
    }

    /// Try to run rocm-smi to get total/free VRAM for AMD (3-second timeout).
    fn detect_amd_memory(device_idx: usize) -> Option<(u64, u64)> {
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

        if !output.status.success() {
            return None;
        }
        let json_str = String::from_utf8_lossy(&output.stdout);
        let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;
        // rocm-smi returns a JSON object where keys are usually "card0", "card1", etc.
        // We try "card{idx}" first, then fallback to any available card if idx fails.
        let card_key = format!("card{}", device_idx);
        let card_data = v
            .get(&card_key)
            .or_else(|| v.as_object().and_then(|obj| obj.values().next()))?;

        let total_bytes: u64 = card_data
            .get("VRAM Total Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())?;
        // "Used" is a bonus field, same reasoning as the nvidia-smi free
        // fallback above: missing/unparsable used memory means "assume free".
        let free_bytes = card_data
            .get("VRAM Total Used Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|used| total_bytes.saturating_sub(used))
            .unwrap_or(total_bytes);
        Some((total_bytes, free_bytes))
    }

    /// Total system RAM in bytes from `/proc/meminfo`.
    ///
    /// Read synchronously because [`GPUManager::detect_devices`] runs once at
    /// startup, before the manager (and its async `system_memory` helper) exists.
    fn system_memory_total() -> Option<u64> {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024); // /proc/meminfo reports kB
            }
        }
        None
    }

    /// Memory budget for a unified-memory (integrated) adapter — reserves a
    /// headroom slice of system RAM (default 15%, override via
    /// `GPU_MEMORY_HEADROOM_PERCENT`) so admission can't OOM-kill the host.
    fn unified_memory_budget() -> Option<u64> {
        let headroom_pct = std::env::var("GPU_MEMORY_HEADROOM_PERCENT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|p| *p < 100)
            .unwrap_or(DEFAULT_UNIFIED_HEADROOM_PERCENT);
        Some(apply_headroom(Self::system_memory_total()?, headroom_pct))
    }

    /// Estimate total GPU memory for one adapter when no vendor query
    /// succeeded: `GPU_VRAM_GB` override, then a headroom-adjusted slice of
    /// system RAM for unified adapters, else a conservative 8 GiB.
    fn estimate_total_memory(is_unified: bool) -> u64 {
        if let Some(bytes) = Self::vram_override_bytes() {
            return bytes;
        }
        if is_unified {
            if let Some(bytes) = Self::unified_memory_budget() {
                return bytes;
            }
        }
        8 * 1024 * 1024 * 1024
    }

    /// Get number of GPU devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Re-sample utilization/temperature/power via nvidia-smi (one CSV line per GPU).
    /// AMD/Intel adapters keep zeros until a rocm-smi refresh is added.
    /// Managed device i maps to vendor-tool line i (single-vendor hosts; documented limitation).
    pub async fn refresh_telemetry(&self) {
        let output = tokio::task::spawn_blocking(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let out = Command::new("nvidia-smi")
                    .arg("--query-gpu=utilization.gpu,temperature.gpu,power.draw")
                    .arg("--format=csv,noheader,nounits")
                    .output();
                let _ = tx.send(out);
            });
            rx.recv_timeout(std::time::Duration::from_secs(3))
                .ok()
                .and_then(|r| r.ok())
        })
        .await
        .ok()
        .flatten();

        let Some(output) = output else { return };
        if !output.status.success() {
            return;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut telemetry = self.telemetry.write().await;
        for (i, line) in stdout.lines().enumerate() {
            if i >= telemetry.len() {
                break;
            }
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() >= 3 {
                telemetry[i] = GpuTelemetry {
                    utilization: fields[0].parse().unwrap_or(0.0),
                    temperature: fields[1].parse().unwrap_or(0.0),
                    power_usage: fields[2].parse::<f32>().map(|w| w as u32).unwrap_or(0),
                };
            }
        }
    }

    /// Get all GPU info for status reporting (gRPC)
    pub async fn get_all_gpu_info(&self) -> Vec<GpuInfo> {
        let telemetry = self.telemetry.read().await.clone();
        let mut infos = Vec::new();

        for (i, device) in self.devices.iter().enumerate() {
            let available = self.get_available_memory(device.id).await;
            let t = telemetry.get(i).cloned().unwrap_or_default();

            infos.push(GpuInfo {
                id: device.id as i32,
                name: device.name.clone(),
                total_memory: device.total_memory,
                available_memory: available,
                utilization: t.utilization,
                temperature: t.temperature,
                power_usage: t.power_usage,
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
        let used = self
            .used_bytes
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

        self.allocations
            .entry(device_id)
            .or_default()
            .push(MemoryAllocation {
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

    /// Check if all GPUs are healthy (refreshes telemetry first).
    pub async fn is_healthy(&self) -> bool {
        self.refresh_telemetry().await;
        let telemetry = self.telemetry.read().await;
        for t in telemetry.iter() {
            if t.temperature > 100.0 {
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

/// Minimal, `dlopen`-based bindings for the handful of CUDA driver API calls
/// needed to read total/free device memory. Deliberately not a Cargo
/// dependency — see [`GPUManager::try_detect_nvidia_memory_cuda`]. Linux
/// only; every failure mode folds into `None`.
#[cfg(target_os = "linux")]
mod cuda_driver {
    use std::ffi::{c_char, c_int, c_void, CString};

    type CuResult = c_int;
    const CUDA_SUCCESS: CuResult = 0;
    type CuDevice = c_int;
    type CuContext = *mut c_void;

    // Deliberately no `#[link(...)]`: dlopen/dlsym/dlclose ship in every
    // glibc/musl libc already linked by this binary. See docs/configuration.md.
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }
    const RTLD_NOW: c_int = 0x2;
    const RTLD_LOCAL: c_int = 0x0;

    type CuInitFn = unsafe extern "C" fn(c_int) -> CuResult;
    type CuDeviceGetCountFn = unsafe extern "C" fn(*mut c_int) -> CuResult;
    type CuDeviceGetFn = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDevicePrimaryCtxRetainFn = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
    type CuDevicePrimaryCtxReleaseFn = unsafe extern "C" fn(CuDevice) -> CuResult;
    type CuCtxSetCurrentFn = unsafe extern "C" fn(CuContext) -> CuResult;
    type CuMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;

    /// RAII `dlopen` handle: guarantees `dlclose` runs on every exit path
    /// (early `?` returns included), and looks up typed function pointers.
    struct DlHandle(*mut c_void);

    impl DlHandle {
        fn open(candidates: &[&str]) -> Option<Self> {
            for name in candidates {
                let Ok(cname) = CString::new(*name) else {
                    continue;
                };
                // SAFETY: `cname` is a valid, NUL-terminated C string that
                // outlives this call. A null return is `dlopen`'s documented
                // "not found" signal, handled below.
                let handle = unsafe { dlopen(cname.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
                if !handle.is_null() {
                    return Some(DlHandle(handle));
                }
            }
            None
        }

        /// # Safety
        /// Caller must ensure `T` exactly matches the C ABI signature of the
        /// symbol named `name` — `dlsym` cannot verify this.
        unsafe fn sym<T: Copy>(&self, name: &str) -> Option<T> {
            let cname = CString::new(name).ok()?;
            let ptr = dlsym(self.0, cname.as_ptr());
            if ptr.is_null() {
                return None;
            }
            debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
            Some(std::mem::transmute_copy::<*mut c_void, T>(&ptr))
        }
    }

    impl Drop for DlHandle {
        fn drop(&mut self) {
            // SAFETY: `self.0` was returned by a successful `dlopen` above
            // and is only ever closed once (owned by this struct).
            unsafe {
                dlclose(self.0);
            }
        }
    }

    /// Query `(total_bytes, free_bytes)` for the primary CUDA context on
    /// device `device_idx`. `device_idx` is treated as a CUDA device
    /// ordinal — the same "index i maps to vendor-tool line/ordinal i"
    /// single-vendor-host assumption already documented for the
    /// `nvidia-smi` fallback.
    pub(super) fn device_memory(device_idx: usize) -> Option<(u64, u64)> {
        let handle = DlHandle::open(&["libcuda.so.1", "libcuda.so"])?;

        // SAFETY: every function pointer below is looked up by exact CUDA
        // driver API symbol name and immediately called with argument types
        // matching its documented C prototype; pointers passed out-params
        // are valid, live local `&mut` targets for the duration of the call.
        unsafe {
            let cu_init: CuInitFn = handle.sym("cuInit")?;
            let cu_device_get_count: CuDeviceGetCountFn = handle.sym("cuDeviceGetCount")?;
            let cu_device_get: CuDeviceGetFn = handle.sym("cuDeviceGet")?;
            let cu_ctx_retain: CuDevicePrimaryCtxRetainFn =
                handle.sym("cuDevicePrimaryCtxRetain")?;
            let cu_ctx_release: CuDevicePrimaryCtxReleaseFn =
                handle.sym("cuDevicePrimaryCtxRelease_v2")?;
            let cu_ctx_set_current: CuCtxSetCurrentFn = handle.sym("cuCtxSetCurrent")?;
            let cu_mem_get_info: CuMemGetInfoFn = handle.sym("cuMemGetInfo_v2")?;

            if cu_init(0) != CUDA_SUCCESS {
                return None;
            }

            let mut count: c_int = 0;
            let ordinal = c_int::try_from(device_idx).ok()?;
            if cu_device_get_count(&mut count) != CUDA_SUCCESS || ordinal >= count {
                return None;
            }

            let mut device: CuDevice = 0;
            if cu_device_get(&mut device, ordinal) != CUDA_SUCCESS {
                return None;
            }

            let mut ctx: CuContext = std::ptr::null_mut();
            if cu_ctx_retain(&mut ctx, device) != CUDA_SUCCESS || ctx.is_null() {
                return None;
            }

            let mem = (|| -> Option<(u64, u64)> {
                if cu_ctx_set_current(ctx) != CUDA_SUCCESS {
                    return None;
                }
                let mut free: usize = 0;
                let mut total: usize = 0;
                if cu_mem_get_info(&mut free, &mut total) != CUDA_SUCCESS {
                    return None;
                }
                Some((total as u64, free as u64))
            })();

            // Always release our primary-context reference, regardless of
            // whether the memory query above succeeded.
            let _ = cu_ctx_release(device);

            mem
        }
    }
}

#[cfg(test)]
impl GPUManager {
    /// Single-device manager with an exact, caller-chosen capacity (bytes) —
    /// lets other modules' tests exercise real admission/refusal without
    /// depending on detected hardware.
    pub(crate) fn test_with_capacity(bytes: u64) -> Self {
        let device = GPUDevice {
            id: 0,
            name: "test-gpu".to_string(),
            total_memory: bytes,
            available_memory: bytes,
            utilization: 0.0,
            temperature: 0.0,
            power_usage: 0,
            capabilities: vec![],
        };
        GPUManager {
            devices: vec![device],
            allocations: Arc::new(DashMap::new()),
            used_bytes: Arc::new(vec![AtomicU64::new(0)]),
            memory_locks: vec![Arc::new(Semaphore::new(1))],
            telemetry: Arc::new(tokio::sync::RwLock::new(vec![Default::default()])),
            _p2p_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_info(
        name: &str,
        vendor: u32,
        device: u32,
        device_type: wgpu::DeviceType,
        backend: wgpu::Backend,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor,
            device,
            device_type,
            driver: String::new(),
            driver_info: String::new(),
            backend,
        }
    }

    // --- select_hardware_adapters -------------------------------------------

    #[test]
    fn same_card_across_two_backends_collapses_to_one() {
        // Reproduces the exact reported bug: one physical GB10 enumerated as
        // an IntegratedGpu via Vulkan AND as an "Other" device via GL. Name
        // AND device_type both differ, so a name/vendor/device_type dedup
        // key does not collapse them — backend selection must.
        let infos = vec![
            adapter_info(
                "NVIDIA GB10",
                0x10de,
                0x0001,
                wgpu::DeviceType::IntegratedGpu,
                wgpu::Backend::Vulkan,
            ),
            adapter_info(
                "NVIDIA GB10/PCIe",
                0x10de,
                0x0001,
                wgpu::DeviceType::Other,
                wgpu::Backend::Gl,
            ),
            adapter_info(
                "llvmpipe",
                0x10005,
                0,
                wgpu::DeviceType::Cpu,
                wgpu::Backend::Vulkan,
            ),
        ];

        let selected = GPUManager::select_hardware_adapters(infos);

        // Only the Vulkan backend's adapters survive (it's the highest
        // priority backend with a real GPU); the GL duplicate is dropped.
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|i| i.backend == wgpu::Backend::Vulkan));
        assert!(selected.iter().any(|i| i.name == "NVIDIA GB10"));
    }

    #[test]
    fn two_identical_cards_on_one_backend_stay_two_devices() {
        // The dangerous failure mode this design must avoid: a multi-GPU rig
        // with two IDENTICAL cards (same name/vendor/device/device_type) must
        // never collapse to one managed device.
        let infos = vec![
            adapter_info(
                "NVIDIA GeForce RTX 3080",
                0x10de,
                0x2206,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Vulkan,
            ),
            adapter_info(
                "NVIDIA GeForce RTX 3080",
                0x10de,
                0x2206,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Vulkan,
            ),
        ];

        let selected = GPUManager::select_hardware_adapters(infos);

        assert_eq!(
            selected.len(),
            2,
            "two genuinely distinct identical cards must not be merged"
        );
    }

    #[test]
    fn two_identical_cards_still_two_even_with_a_duplicate_gl_backend() {
        // Combines both scenarios: a real two-GPU Vulkan rig, plus each card
        // also visible via GL (as GB10 was). Backend selection must keep
        // both real Vulkan devices and drop the GL ones.
        let infos = vec![
            adapter_info(
                "AMD Radeon RX 9060 XT",
                0x1002,
                0x7550,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Vulkan,
            ),
            adapter_info(
                "AMD Radeon RX 9060 XT",
                0x1002,
                0x7550,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Vulkan,
            ),
            adapter_info(
                "AMD Radeon RX 9060 XT",
                0x1002,
                0x7550,
                wgpu::DeviceType::Other,
                wgpu::Backend::Gl,
            ),
            adapter_info(
                "AMD Radeon RX 9060 XT",
                0x1002,
                0x7550,
                wgpu::DeviceType::Other,
                wgpu::Backend::Gl,
            ),
        ];

        let selected = GPUManager::select_hardware_adapters(infos);

        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|i| i.backend == wgpu::Backend::Vulkan));
    }

    #[test]
    fn backend_priority_prefers_vulkan_over_dx12_and_gl() {
        let infos = vec![
            adapter_info(
                "Intel Arc A770",
                0x8086,
                0x56a0,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Dx12,
            ),
            adapter_info(
                "Intel Arc A770",
                0x8086,
                0x56a0,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Vulkan,
            ),
            adapter_info(
                "Intel Arc A770",
                0x8086,
                0x56a0,
                wgpu::DeviceType::DiscreteGpu,
                wgpu::Backend::Gl,
            ),
        ];

        let selected = GPUManager::select_hardware_adapters(infos);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].backend, wgpu::Backend::Vulkan);
    }

    #[test]
    fn only_cpu_adapters_falls_back_deterministically() {
        // No backend has real hardware — must still pick deterministically
        // (highest-priority backend present) rather than depend on HashMap
        // iteration order.
        let infos = vec![
            adapter_info("llvmpipe", 0, 0, wgpu::DeviceType::Cpu, wgpu::Backend::Gl),
            adapter_info(
                "llvmpipe",
                0,
                0,
                wgpu::DeviceType::Cpu,
                wgpu::Backend::Vulkan,
            ),
        ];

        let selected = GPUManager::select_hardware_adapters(infos);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].backend, wgpu::Backend::Vulkan);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(GPUManager::select_hardware_adapters(Vec::new()).is_empty());
    }

    // --- GPU_VRAM_GB precedence ----------------------------------------------

    #[test]
    fn vram_override_bytes_parses_gib() {
        // SAFETY: single-threaded test process; no other test reads/writes
        // this exact env var concurrently within this test.
        unsafe {
            std::env::set_var("GPU_VRAM_GB", "12");
        }
        assert_eq!(
            GPUManager::vram_override_bytes(),
            Some(12 * 1024 * 1024 * 1024)
        );
        unsafe {
            std::env::remove_var("GPU_VRAM_GB");
        }
        assert_eq!(GPUManager::vram_override_bytes(), None);
    }

    #[test]
    #[ignore = "requires real NVIDIA hardware + driver (run with --ignored on an NVIDIA host)"]
    #[cfg(target_os = "linux")]
    fn cuda_driver_reports_real_memory_on_nvidia_hardware() {
        // Manual verification gate: unified-memory NVIDIA hardware (e.g.
        // DGX Spark) can't report memory via nvidia-smi/NVML, so this is
        // the only way to confirm the driver-API path works end-to-end.
        let (total, free) =
            cuda_driver::device_memory(0).expect("CUDA driver API query must succeed on GPU 0");
        println!(
            "CUDA driver API: {} MiB total, {} MiB free",
            total / 1024 / 1024,
            free / 1024 / 1024
        );
        assert!(total > 0, "total memory must be nonzero");
        assert!(free <= total, "free memory cannot exceed total");
    }

    #[test]
    fn apply_headroom_reserves_the_requested_slice() {
        let total = 128 * 1024 * 1024 * 1024u64;

        // 0% headroom returns the total modulo integer-division rounding
        // (at most 99 bytes lost — irrelevant against a GiB-scale budget).
        assert!(total - apply_headroom(total, 0) < 100);
        // 15% headroom on 128 GiB leaves ~108.8 GiB.
        assert_eq!(apply_headroom(total, 15), total / 100 * 85);
        assert!(apply_headroom(total, 15) < total);
    }

    #[test]
    fn apply_headroom_saturates_instead_of_underflowing() {
        // A nonsensical headroom must yield zero, never wrap around to a huge
        // budget — that would be the one failure mode worth crashing over.
        assert_eq!(apply_headroom(1_000, 100), 0);
        assert_eq!(apply_headroom(1_000, 250), 0);
    }

    #[test]
    fn unified_budget_beats_the_8gb_default_on_this_host() {
        // Guards the DGX Spark regression: nvidia-smi reports "[N/A]" for
        // unified memory, so a unified adapter must not fall back to 8 GiB.
        // Skipped where /proc/meminfo is unavailable or the host genuinely has
        // under ~9 GiB of RAM (the fallback would then be correct anyway).
        let Some(total) = GPUManager::system_memory_total() else {
            return;
        };
        if total <= 9 * 1024 * 1024 * 1024 {
            return;
        }
        let budget = GPUManager::unified_memory_budget().expect("meminfo readable");
        assert!(budget > 8 * 1024 * 1024 * 1024);
        assert!(budget < total, "must hold back headroom for the OS");
    }

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
            telemetry: Arc::new(tokio::sync::RwLock::new(vec![Default::default()])),
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
