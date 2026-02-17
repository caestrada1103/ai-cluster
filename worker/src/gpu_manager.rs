//! GPU management for AMD and NVIDIA devices
//!
//! This module provides a unified interface for GPU detection,
//! memory management, and device operations across different backends.

use std::collections::HashMap;
use std::sync::Arc;

use burn::backend::{Backend, Device};
use burn::tensor::Tensor;
use dashmap::DashMap;
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, debug, error};

#[cfg(feature = "hip")]
use burn::backend::HipBackend;

#[cfg(feature = "cuda")]
use burn::backend::CudaBackend;

#[cfg(feature = "wgpu")]
use burn::backend::WgpuBackend;

use crate::cluster::GpuInfo;
use crate::error::WorkerError;

/// Type alias for the active backend
#[cfg(feature = "hip")]
pub type ActiveBackend = HipBackend;

#[cfg(feature = "cuda")]
pub type ActiveBackend = CudaBackend;

#[cfg(all(not(feature = "hip"), not(feature = "cuda")))]
pub type ActiveBackend = WgpuBackend;

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
    
    /// Burn device handle
    pub device: Device<ActiveBackend>,
    
    /// Whether device supports peer-to-peer access
    pub supports_p2p: bool,
}

/// GPU memory allocation tracking
struct MemoryAllocation {
    /// Size in bytes
    size: u64,
    
    /// Allocation timestamp
    timestamp: std::time::Instant,
    
    /// Owner (model name or request ID)
    owner: String,
}

/// GPU Manager
pub struct GPUManager {
    /// Available GPU devices
    devices: Vec<GPUDevice>,
    
    /// Device index mapping
    device_map: HashMap<usize, usize>, // GPU ID -> index in devices
    
    /// Memory allocations per device
    allocations: Arc<DashMap<usize, Vec<MemoryAllocation>>>,
    
    /// Memory locks per device (for concurrent access)
    memory_locks: Vec<Arc<Semaphore>>,
    
    /// Stream pools for concurrent execution
    stream_pools: Vec<Arc<RwLock<Vec<ActiveBackend::Stream>>>>,
    
    /// Whether peer-to-peer is enabled
    p2p_enabled: bool,
}

impl GPUManager {
    /// Create a new GPU manager for the specified device IDs
    pub async fn new(gpu_ids: &[usize]) -> Result<Self, WorkerError> {
        info!("Initializing GPU manager with devices: {:?}", gpu_ids);
        
        let mut devices = Vec::new();
        let mut device_map = HashMap::new();
        let mut memory_locks = Vec::new();
        let mut stream_pools = Vec::new();
        
        // Detect available devices
        let available_devices = Self::detect_devices()?;
        
        // Filter to requested IDs
        for (idx, &gpu_id) in gpu_ids.iter().enumerate() {
            if gpu_id >= available_devices.len() {
                warn!("GPU {} not available, skipping", gpu_id);
                continue;
            }
            
            let mut device = available_devices[gpu_id].clone();
            device.id = idx; // Renumber sequentially for internal use
            
            // Get detailed info
            if let Ok(info) = Self::get_device_info(&device).await {
                device.total_memory = info.total_memory;
                device.available_memory = info.available_memory;
                device.utilization = info.utilization;
                device.temperature = info.temperature;
                device.power_usage = info.power_usage;
                device.capabilities = info.capabilities;
            }
            
            devices.push(device);
            device_map.insert(gpu_id, idx);
            memory_locks.push(Arc::new(Semaphore::new(1))); // Allow one allocation at a time
            stream_pools.push(Arc::new(RwLock::new(Vec::new())));
            
            info!("Initialized GPU {}: {} ({}MB total, {}MB free)",
                idx,
                devices.last().unwrap().name,
                devices.last().unwrap().total_memory / 1024 / 1024,
                devices.last().unwrap().available_memory / 1024 / 1024
            );
        }
        
        if devices.is_empty() {
            return Err(WorkerError::NoGpusFound);
        }
        
        // Check peer-to-peer support
        let p2p_enabled = Self::check_p2p_support(&devices).await;
        if p2p_enabled {
            info!("Peer-to-peer GPU communication enabled");
        }
        
        Ok(Self {
            devices,
            device_map,
            allocations: Arc::new(DashMap::new()),
            memory_locks,
            stream_pools,
            p2p_enabled,
        })
    }
    
    /// Detect available GPU devices
    fn detect_devices() -> Result<Vec<GPUDevice>, WorkerError> {
        let mut devices = Vec::new();
        
        // Use Burn's device detection
        let burn_devices = ActiveBackend::devices();
        
        for (i, device) in burn_devices.iter().enumerate() {
            // Get device name (backend-specific)
            let name = Self::get_device_name(device);
            
            devices.push(GPUDevice {
                id: i,
                name,
                total_memory: 0, // Will be filled later
                available_memory: 0,
                utilization: 0.0,
                temperature: 0.0,
                power_usage: 0,
                capabilities: Vec::new(),
                device: device.clone(),
                supports_p2p: false,
            });
        }
        
        Ok(devices)
    }
    
    /// Get device name (backend-specific)
    fn get_device_name(device: &Device<ActiveBackend>) -> String {
        #[cfg(feature = "hip")]
        {
            use burn::backend::hip::HipDevice;
            if let Some(hip_device) = device.as_hip() {
                // Try to get actual device name via ROCm API
                if let Ok(name) = Self::get_hip_device_name(hip_device.id()) {
                    return name;
                }
                return format!("AMD GPU {}", hip_device.id());
            }
        }
        
        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::CudaDevice;
            if let Some(cuda_device) = device.as_cuda() {
                // Try to get actual device name via CUDA API
                if let Ok(name) = Self::get_cuda_device_name(cuda_device.id()) {
                    return name;
                }
                return format!("NVIDIA GPU {}", cuda_device.id());
            }
        }
        
        "Unknown GPU".to_string()
    }
    
    /// Get HIP device name (AMD)
    #[cfg(feature = "hip")]
    fn get_hip_device_name(device_id: usize) -> Result<String, WorkerError> {
        use std::ffi::CStr;
        use std::ptr;
        
        unsafe {
            let mut name = [0i8; 256];
            let result = hip_sys::hipDeviceGetName(
                name.as_mut_ptr() as *mut std::ffi::c_char,
                name.len() as i32,
                device_id as i32,
            );
            
            if result == hip_sys::hipError_t::hipSuccess {
                let c_str = CStr::from_ptr(name.as_ptr());
                Ok(c_str.to_string_lossy().into_owned())
            } else {
                Err(WorkerError::Gpu(format!("Failed to get HIP device name: {}", result)))
            }
        }
    }
    
    /// Get CUDA device name (NVIDIA)
    #[cfg(feature = "cuda")]
    fn get_cuda_device_name(device_id: usize) -> Result<String, WorkerError> {
        use std::ffi::CStr;
        
        unsafe {
            let mut name = [0i8; 256];
            let result = cuda_sys::cudaDeviceGetName(
                name.as_mut_ptr(),
                name.len() as i32,
                device_id as i32,
            );
            
            if result == cuda_sys::cudaError_enum::cudaSuccess {
                let c_str = CStr::from_ptr(name.as_ptr());
                Ok(c_str.to_string_lossy().into_owned())
            } else {
                Err(WorkerError::Gpu(format!("Failed to get CUDA device name: {}", result)))
            }
        }
    }
    
    /// Get detailed device information
    async fn get_device_info(device: &GPUDevice) -> Result<GPUDevice, WorkerError> {
        let mut device = device.clone();
        
        #[cfg(feature = "hip")]
        {
            // Use ROCm-smi or hip APIs to get real-time info
            if let Ok(info) = Self::get_hip_device_info(device.id).await {
                device.total_memory = info.total_memory;
                device.available_memory = info.available_memory;
                device.utilization = info.utilization;
                device.temperature = info.temperature;
                device.power_usage = info.power_usage;
                device.capabilities = info.capabilities;
            }
        }
        
        #[cfg(feature = "cuda")]
        {
            // Use nvml or cuda APIs to get real-time info
            if let Ok(info) = Self::get_cuda_device_info(device.id).await {
                device.total_memory = info.total_memory;
                device.available_memory = info.available_memory;
                device.utilization = info.utilization;
                device.temperature = info.temperature;
                device.power_usage = info.power_usage;
                device.capabilities = info.capabilities;
            }
        }
        
        Ok(device)
    }
    
    /// Get HIP device information (AMD)
    #[cfg(feature = "hip")]
    async fn get_hip_device_info(device_id: usize) -> Result<GPUDevice, WorkerError> {
        use std::mem;
        
        unsafe {
            // Get memory info
            let mut free: usize = 0;
            let mut total: usize = 0;
            
            // Set current device
            let result = hip_sys::hipSetDevice(device_id as i32);
            if result != hip_sys::hipError_t::hipSuccess {
                return Err(WorkerError::Gpu("Failed to set HIP device".to_string()));
            }
            
            let result = hip_sys::hipMemGetInfo(&mut free, &mut total);
            if result != hip_sys::hipError_t::hipSuccess {
                return Err(WorkerError::Gpu("Failed to get HIP memory info".to_string()));
            }
            
            // Get utilization (would need ROCm-smi for real-time)
            // For now, return estimates
            Ok(GPUDevice {
                id: device_id,
                name: String::new(), // Will be filled by caller
                total_memory: total as u64,
                available_memory: free as u64,
                utilization: 0.0,
                temperature: 0.0,
                power_usage: 0,
                capabilities: vec!["fp16".to_string(), "int8".to_string()],
                device: Device::from(device_id),
                supports_p2p: false,
            })
        }
    }
    
    /// Get CUDA device information (NVIDIA)
    #[cfg(feature = "cuda")]
    async fn get_cuda_device_info(device_id: usize) -> Result<GPUDevice, WorkerError> {
        unsafe {
            // Initialize NVML for detailed stats
            let mut nvml: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = nvml_sys::nvmlInit_v2();
            if result == nvml_sys::nvmlReturn_enum::NVML_SUCCESS {
                // Get device handle
                let mut device: nvml_sys::nvmlDevice_t = std::ptr::null_mut();
                let result = nvml_sys::nvmlDeviceGetHandleByIndex_v2(device_id as u32, &mut device);
                
                if result == nvml_sys::nvmlReturn_enum::NVML_SUCCESS {
                    // Get memory info
                    let mut memory = mem::zeroed();
                    let result = nvml_sys::nvmlDeviceGetMemoryInfo(device, &mut memory);
                    
                    if result == nvml_sys::nvmlReturn_enum::NVML_SUCCESS {
                        // Get utilization
                        let mut utilization = mem::zeroed();
                        let result = nvml_sys::nvmlDeviceGetUtilizationRates(device, &mut utilization);
                        
                        // Get temperature
                        let mut temp: u32 = 0;
                        let result = nvml_sys::nvmlDeviceGetTemperature(
                            device,
                            nvml_sys::nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU,
                            &mut temp
                        );
                        
                        // Get power usage
                        let mut power: u32 = 0;
                        let result = nvml_sys::nvmlDeviceGetPowerUsage(device, &mut power);
                        
                        nvml_sys::nvmlShutdown();
                        
                        return Ok(GPUDevice {
                            id: device_id,
                            name: String::new(),
                            total_memory: memory.total,
                            available_memory: memory.free,
                            utilization: utilization.gpu as f32,
                            temperature: temp as f32,
                            power_usage: power / 1000, // Convert to watts
                            capabilities: vec![
                                "fp16".to_string(),
                                "int8".to_string(),
                                "tensorcore".to_string(),
                            ],
                            device: Device::from(device_id),
                            supports_p2p: true,
                        });
                    }
                }
                nvml_sys::nvmlShutdown();
            }
            
            // Fallback to CUDA runtime API
            let mut free: usize = 0;
            let mut total: usize = 0;
            let result = cuda_sys::cudaSetDevice(device_id as i32);
            let result = cuda_sys::cudaMemGetInfo(&mut free, &mut total);
            
            if result == cuda_sys::cudaError_enum::cudaSuccess {
                Ok(GPUDevice {
                    id: device_id,
                    name: String::new(),
                    total_memory: total as u64,
                    available_memory: free as u64,
                    utilization: 0.0,
                    temperature: 0.0,
                    power_usage: 0,
                    capabilities: vec!["fp16".to_string()],
                    device: Device::from(device_id),
                    supports_p2p: true,
                })
            } else {
                Err(WorkerError::Gpu("Failed to get CUDA memory info".to_string()))
            }
        }
    }
    
    /// Check peer-to-peer support between devices
    async fn check_p2p_support(devices: &[GPUDevice]) -> bool {
        if devices.len() < 2 {
            return false;
        }
        
        #[cfg(feature = "cuda")]
        {
            // Check CUDA P2P support
            for i in 0..devices.len() {
                for j in i+1..devices.len() {
                    unsafe {
                        let mut can_access = 0;
                        let result = cuda_sys::cudaDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);
                        if result != cuda_sys::cudaError_enum::cudaSuccess || can_access == 0 {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        
        #[cfg(feature = "hip")]
        {
            // Check HIP P2P support (similar to CUDA)
            for i in 0..devices.len() {
                for j in i+1..devices.len() {
                    unsafe {
                        let mut can_access = 0;
                        let result = hip_sys::hipDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);
                        if result != hip_sys::hipError_t::hipSuccess || can_access == 0 {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        
        // Default to false for other backends
        false
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
        self.device_map.get(&original_id).and_then(|&idx| self.devices.get(idx))
    }
    
    /// Get all GPU info for status reporting
    pub async fn get_all_gpu_info(&self) -> Vec<crate::cluster::GpuInfo> {
        let mut infos = Vec::new();
        
        for device in &self.devices {
            // Update available memory
            let available = self.get_available_memory(device.id).await;
            
            infos.push(crate::cluster::GpuInfo {
                id: device.id as i32,
                name: device.name.clone(),
                total_memory: device.total_memory,
                available_memory: available,
                utilization: device.utilization as f32,
                temperature: device.temperature as f32,
                power_usage: device.power_usage,
                capabilities: device.capabilities.clone(),
            });
        }
        
        infos
    }
    
    /// Get available memory for a device
    pub async fn get_available_memory(&self, device_id: usize) -> u64 {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return 0,
        };
        
        // Start with device-reported available
        let mut available = device.available_memory;
        
        // Subtract tracked allocations
        if let Some(allocations) = self.allocations.get(&device_id) {
            for alloc in allocations.value() {
                available = available.saturating_sub(alloc.size);
            }
        }
        
        available
    }
    
    /// Allocate memory on a device
    pub async fn allocate_memory(
        &self,
        device_id: usize,
        size: u64,
        owner: String,
    ) -> Result<(), WorkerError> {
        // Acquire lock for this device
        let _permit = self.memory_locks[device_id].acquire().await.map_err(|e| {
            WorkerError::Resource(format!("Failed to acquire memory lock: {}", e))
        })?;
        
        // Check if enough memory available
        let available = self.get_available_memory(device_id).await;
        if available < size {
            return Err(WorkerError::OutOfMemory {
                requested: size,
                available,
                device: device_id,
            });
        }
        
        // Record allocation
        let mut allocations = self.allocations.entry(device_id).or_insert_with(Vec::new);
        allocations.push(MemoryAllocation {
            size,
            timestamp: std::time::Instant::now(),
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
            
            // Remove allocations owned by this owner
            allocations.retain(|alloc| alloc.owner != owner);
            
            debug!("Freed memory for {} on GPU {}", owner, device_id);
        }
    }
    
    /// Get memory usage statistics for a device
    pub async fn get_memory_stats(&self, device_id: usize) -> (u64, u64, usize) {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return (0, 0, 0),
        };
        
        let allocated = self.allocations
            .get(&device_id)
            .map(|alloc| alloc.value().iter().map(|a| a.size).sum())
            .unwrap_or(0);
        
        let available = device.available_memory.saturating_sub(allocated);
        let allocation_count = self.allocations
            .get(&device_id)
            .map(|alloc| alloc.value().len())
            .unwrap_or(0);
        
        (device.total_memory, available, allocation_count)
    }
    
    /// Get a stream for concurrent execution on a device
    pub async fn get_stream(&self, device_id: usize) -> Result<ActiveBackend::Stream, WorkerError> {
        let pool = self.stream_pools.get(device_id)
            .ok_or_else(|| WorkerError::Gpu(format!("Invalid device ID: {}", device_id)))?;
        
        let mut pool = pool.write().await;
        
        // Reuse existing stream or create new one
        if let Some(stream) = pool.pop() {
            Ok(stream)
        } else {
            // Create new stream (backend-specific)
            #[cfg(feature = "hip")]
            {
                use burn::backend::hip::HipStream;
                Ok(HipStream::new(device_id).into())
            }
            
            #[cfg(feature = "cuda")]
            {
                use burn::backend::cuda::CudaStream;
                Ok(CudaStream::new(device_id).into())
            }
            
            #[cfg(not(any(feature = "hip", feature = "cuda")))]
            {
                Err(WorkerError::Gpu("Streams not supported on this backend".to_string()))
            }
        }
    }
    
    /// Return a stream to the pool
    pub async fn return_stream(&self, device_id: usize, stream: ActiveBackend::Stream) {
        if let Some(pool) = self.stream_pools.get(device_id) {
            let mut pool = pool.write().await;
            pool.push(stream);
        }
    }
    
    /// Execute a function with a stream
    pub async fn with_stream<F, T>(&self, device_id: usize, f: F) -> Result<T, WorkerError>
    where
        F: FnOnce(ActiveBackend::Stream) -> T + Send,
        T: Send,
    {
        let stream = self.get_stream(device_id).await?;
        let result = f(stream.clone());
        self.return_stream(device_id, stream).await;
        Ok(result)
    }
    
    /// Check if all GPUs are healthy
    pub async fn is_healthy(&self) -> bool {
        for device in &self.devices {
            // Check if device is responsive
            let available = self.get_available_memory(device.id).await;
            if available == 0 && device.total_memory > 0 {
                // Device might be hung if no memory available but total > 0
                return false;
            }
            
            // Check temperature (if over 100°C, probably not healthy)
            if device.temperature > 100.0 {
                return false;
            }
        }
        
        true
    }
    
    /// Synchronize all devices
    pub async fn synchronize_all(&self) -> Result<(), WorkerError> {
        for device in &self.devices {
            self.synchronize_device(device.id).await?;
        }
        Ok(())
    }
    
    /// Synchronize a specific device
    pub async fn synchronize_device(&self, device_id: usize) -> Result<(), WorkerError> {
        #[cfg(feature = "hip")]
        {
            unsafe {
                let result = hip_sys::hipSetDevice(device_id as i32);
                if result != hip_sys::hipError_t::hipSuccess {
                    return Err(WorkerError::Gpu("Failed to set HIP device".to_string()));
                }
                
                let result = hip_sys::hipDeviceSynchronize();
                if result != hip_sys::hipError_t::hipSuccess {
                    return Err(WorkerError::Gpu("Failed to synchronize HIP device".to_string()));
                }
            }
        }
        
        #[cfg(feature = "cuda")]
        {
            unsafe {
                let result = cuda_sys::cudaSetDevice(device_id as i32);
                if result != cuda_sys::cudaError_enum::cudaSuccess {
                    return Err(WorkerError::Gpu("Failed to set CUDA device".to_string()));
                }
                
                let result = cuda_sys::cudaDeviceSynchronize();
                if result != cuda_sys::cudaError_enum::cudaSuccess {
                    return Err(WorkerError::Gpu("Failed to synchronize CUDA device".to_string()));
                }
            }
        }
        
        Ok(())
    }
    
    /// Enable peer-to-peer access between two devices
    pub async fn enable_p2p(&self, device1: usize, device2: usize) -> Result<(), WorkerError> {
        if !self.p2p_enabled {
            return Ok(());
        }
        
        #[cfg(feature = "cuda")]
        {
            unsafe {
                let result = cuda_sys::cudaSetDevice(device1 as i32);
                if result != cuda_sys::cudaError_enum::cudaSuccess {
                    return Err(WorkerError::Gpu("Failed to set CUDA device".to_string()));
                }
                
                let result = cuda_sys::cudaDeviceEnablePeerAccess(device2 as i32, 0);
                if result != cuda_sys::cudaError_enum::cudaSuccess && 
                   result != cuda_sys::cudaError_enum::cudaErrorPeerAccessAlreadyEnabled {
                    return Err(WorkerError::Gpu("Failed to enable peer access".to_string()));
                }
            }
        }
        
        #[cfg(feature = "hip")]
        {
            unsafe {
                let result = hip_sys::hipSetDevice(device1 as i32);
                if result != hip_sys::hipError_t::hipSuccess {
                    return Err(WorkerError::Gpu("Failed to set HIP device".to_string()));
                }
                
                let result = hip_sys::hipDeviceEnablePeerAccess(device2 as i32, 0);
                if result != hip_sys::hipError_t::hipSuccess && 
                   result != hip_sys::hipError_t::hipErrorPeerAccessAlreadyEnabled {
                    return Err(WorkerError::Gpu("Failed to enable peer access".to_string()));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get system memory information
    pub async fn system_memory(&self) -> (u64, u64) {
        // Use system APIs to get RAM info
        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = sys_info::mem_info() {
                return (info.free as u64, info.total as u64);
            }
        }
        
        // Fallback
        (0, 0)
    }
}

impl Drop for GPUManager {
    fn drop(&mut self) {
        // Synchronize all devices on drop
        let rt = tokio::runtime::Handle::try_current();
        if let Ok(rt) = rt {
            rt.block_on(async {
                let _ = self.synchronize_all().await;
            });
        }
    }
}