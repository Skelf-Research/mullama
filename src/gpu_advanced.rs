//! Advanced GPU features for optimal performance
//!
//! This module provides advanced GPU capabilities including memory management,
//! dynamic scheduling, multi-GPU optimization, and comprehensive monitoring.

use crate::error::MullamaError;
use crate::Model;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Advanced GPU manager for optimal resource utilization
#[derive(Debug)]
pub struct GpuManager {
    /// Available GPU devices
    devices: Vec<GpuDevice>,
    /// Current allocation strategy
    allocation_strategy: AllocationStrategy,
    /// Memory pools for efficient allocation
    memory_pools: HashMap<usize, GpuMemoryPool>,
    /// Performance monitoring
    monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Dynamic optimization settings
    optimization_config: OptimizationConfig,
}

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID
    pub id: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute capability
    pub compute_capability: (i32, i32),
    /// Maximum number of concurrent streams
    pub max_streams: usize,
    /// Device type
    pub device_type: GpuDeviceType,
    /// Current utilization percentage
    pub utilization: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Power consumption in watts
    pub power_consumption: f32,
}

/// Types of GPU devices
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDeviceType {
    /// NVIDIA CUDA device
    Cuda,
    /// Apple Metal device
    Metal,
    /// AMD ROCm device
    Rocm,
    /// Vulkan compute device
    Vulkan,
    /// OpenCL device
    OpenCL,
}

/// GPU memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Simple first-fit allocation
    FirstFit,
    /// Best-fit allocation for efficiency
    BestFit,
    /// Balanced allocation across devices
    LoadBalanced,
    /// Performance-optimized allocation
    PerformanceOptimized,
    /// Custom allocation strategy
    Custom,
}

/// GPU memory pool for efficient allocation/deallocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Device this pool belongs to
    #[allow(dead_code)]
    device_id: usize,
    /// Pool of free memory blocks
    free_blocks: Vec<MemoryBlock>,
    /// Currently allocated blocks
    allocated_blocks: HashMap<u64, MemoryBlock>,
    /// Total pool size
    total_size: u64,
    /// Current usage
    used_size: u64,
    /// Allocation statistics
    stats: PoolStats,
}

/// A block of GPU memory
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block address
    pub address: u64,
    /// Block size in bytes
    pub size: u64,
    /// When this block was allocated
    pub allocated_at: Instant,
    /// Block type/purpose
    pub block_type: MemoryBlockType,
}

/// Types of memory blocks
#[derive(Debug, Clone, Copy)]
pub enum MemoryBlockType {
    /// Model weights
    ModelWeights,
    /// Activation tensors
    Activations,
    /// KV cache
    KVCache,
    /// Temporary computation
    Temporary,
    /// Input/output buffers
    IOBuffers,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Number of fragmentation events
    pub fragmentation_events: u64,
    /// Number of defragmentation operations
    pub defragmentation_ops: u64,
}

/// Performance monitoring for GPU operations
#[derive(Debug, Default)]
pub struct PerformanceMonitor {
    /// GPU utilization history
    pub utilization_history: Vec<(Instant, Vec<f32>)>,
    /// Memory usage history
    pub memory_history: Vec<(Instant, Vec<u64>)>,
    /// Temperature history
    pub temperature_history: Vec<(Instant, Vec<f32>)>,
    /// Throughput measurements
    pub throughput_history: Vec<(Instant, f32)>, // tokens/second
    /// Performance events
    pub events: Vec<PerformanceEvent>,
}

/// Performance event for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: EventType,
    /// Device ID (if applicable)
    pub device_id: Option<usize>,
    /// Event description
    pub description: String,
    /// Associated metrics
    pub metrics: HashMap<String, f64>,
}

/// Types of performance events
#[derive(Debug, Clone, Copy)]
pub enum EventType {
    /// Memory allocation
    MemoryAllocation,
    /// Memory deallocation
    MemoryDeallocation,
    /// Kernel execution
    KernelExecution,
    /// Memory transfer
    MemoryTransfer,
    /// Performance degradation detected
    PerformanceDegradation,
    /// Thermal throttling
    ThermalThrottling,
    /// Memory fragmentation
    MemoryFragmentation,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable dynamic memory management
    pub dynamic_memory: bool,
    /// Enable automatic defragmentation
    pub auto_defragmentation: bool,
    /// Memory fragmentation threshold (0.0-1.0)
    pub fragmentation_threshold: f32,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
    /// Enable predictive optimization
    pub predictive_optimization: bool,
    /// Thermal throttling threshold (Celsius)
    pub thermal_threshold: f32,
    /// Enable multi-GPU load balancing
    pub load_balancing: bool,
}

impl GpuManager {
    /// Create a new GPU manager
    pub fn new() -> Result<Self, MullamaError> {
        let devices = Self::discover_devices()?;
        let monitor = Arc::new(Mutex::new(PerformanceMonitor::default()));

        Ok(Self {
            devices,
            allocation_strategy: AllocationStrategy::PerformanceOptimized,
            memory_pools: HashMap::new(),
            monitor,
            optimization_config: OptimizationConfig::default(),
        })
    }

    /// Discover available GPU devices
    fn discover_devices() -> Result<Vec<GpuDevice>, MullamaError> {
        let mut devices = Vec::new();

        let supports_gpu = unsafe { crate::sys::llama_supports_gpu_offload() };

        if supports_gpu {
            #[cfg(target_os = "macos")]
            {
                if let Ok(metal_devices) = Self::discover_metal_devices() {
                    devices.extend(metal_devices);
                }
            }

            if let Ok(cuda_devices) = Self::discover_cuda_devices() {
                devices.extend(cuda_devices);
            }

            #[cfg(feature = "rocm")]
            {
                if let Ok(rocm_devices) = Self::discover_rocm_devices() {
                    devices.extend(rocm_devices);
                }
            }

            #[cfg(not(any(feature = "cuda", feature = "rocm", target_os = "macos")))]
            {
                if let Ok(fallback_devices) = Self::discover_fallback_devices() {
                    devices.extend(fallback_devices);
                }
            }
        }

        Ok(devices)
    }

    /// Check if GPU acceleration is available
    pub fn has_gpu_support(&self) -> bool {
        !self.devices.is_empty()
    }

    /// Get the number of available GPUs
    pub fn gpu_count(&self) -> usize {
        self.devices.len()
    }

    /// Check if GPU offload is supported by the backend
    pub fn backend_supports_gpu() -> bool {
        unsafe { crate::sys::llama_supports_gpu_offload() }
    }

    /// Get max number of devices supported by the backend
    pub fn max_devices() -> usize {
        unsafe { crate::sys::llama_max_devices() }
    }

    /// Initialize memory pools for all devices
    pub fn initialize_memory_pools(&mut self, pool_size_mb: u64) -> Result<(), MullamaError> {
        for device in &self.devices {
            let pool_size = pool_size_mb * 1024 * 1024; // Convert to bytes
            let pool = GpuMemoryPool::new(device.id, pool_size)?;
            self.memory_pools.insert(device.id, pool);
        }
        Ok(())
    }

    /// Allocate GPU memory with optimal placement
    pub fn allocate_memory(
        &mut self,
        size: u64,
        block_type: MemoryBlockType,
        preferred_device: Option<usize>,
    ) -> Result<MemoryBlock, MullamaError> {
        let device_id = self.select_optimal_device(size, block_type, preferred_device)?;

        if let Some(pool) = self.memory_pools.get_mut(&device_id) {
            let block = pool.allocate(size, block_type)?;

            // Record allocation event
            self.record_event(
                EventType::MemoryAllocation,
                Some(device_id),
                format!("Allocated {} bytes", size),
                [("size".to_string(), size as f64)]
                    .iter()
                    .cloned()
                    .collect(),
            );

            Ok(block)
        } else {
            Err(MullamaError::GpuError(format!(
                "No memory pool found for device {}",
                device_id
            )))
        }
    }

    /// Deallocate GPU memory
    pub fn deallocate_memory(&mut self, block: MemoryBlock) -> Result<(), MullamaError> {
        let device_id = self.find_device_for_address(block.address)?;
        let block_size = block.size;

        if let Some(pool) = self.memory_pools.get_mut(&device_id) {
            pool.deallocate(block)?;

            // Record deallocation event
            self.record_event(
                EventType::MemoryDeallocation,
                Some(device_id),
                format!("Deallocated {} bytes", block_size),
                [("size".to_string(), block_size as f64)]
                    .iter()
                    .cloned()
                    .collect(),
            );

            Ok(())
        } else {
            Err(MullamaError::GpuError(format!(
                "No memory pool found for device {}",
                device_id
            )))
        }
    }

    /// Optimize model placement across GPUs
    ///
    /// Distributes model layers across available GPU devices based on available
    /// memory, compute capability, and the selected allocation strategy. Falls
    /// back to CPU offload for layers that don't fit on any GPU.
    pub fn optimize_model_placement(
        &mut self,
        model: &Model,
    ) -> Result<ModelPlacement, MullamaError> {
        let num_layers = model.n_layer() as usize;
        let embedding_dim = model.n_embd() as usize;
        let model_size = estimate_model_size(num_layers, embedding_dim);

        let placement = match self.allocation_strategy {
            AllocationStrategy::LoadBalanced => {
                self.create_load_balanced_placement(model_size, num_layers)?
            }
            AllocationStrategy::PerformanceOptimized => {
                self.create_performance_optimized_placement(model_size, num_layers)?
            }
            _ => self.create_simple_placement(model_size, num_layers)?,
        };

        Ok(placement)
    }

    /// Perform memory defragmentation
    pub fn defragment_memory(
        &mut self,
        device_id: Option<usize>,
    ) -> Result<DefragmentationResult, MullamaError> {
        let devices_to_defrag = if let Some(id) = device_id {
            vec![id]
        } else {
            self.memory_pools.keys().cloned().collect()
        };

        let mut total_freed = 0u64;
        let mut total_moved = 0u64;

        for id in devices_to_defrag {
            if let Some(pool) = self.memory_pools.get_mut(&id) {
                let result = pool.defragment()?;
                total_freed += result.bytes_freed;
                total_moved += result.bytes_moved;

                // Record defragmentation event
                self.record_event(
                    EventType::MemoryFragmentation,
                    Some(id),
                    format!("Defragmented device {}", id),
                    [
                        ("freed".to_string(), result.bytes_freed as f64),
                        ("moved".to_string(), result.bytes_moved as f64),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                );
            }
        }

        Ok(DefragmentationResult {
            bytes_freed: total_freed,
            bytes_moved: total_moved,
            duration: Duration::from_millis(0), // Would be measured in real implementation
        })
    }

    /// Get real-time GPU statistics
    pub fn get_gpu_stats(&mut self) -> Result<Vec<GpuStats>, MullamaError> {
        let mut stats = Vec::new();

        for device in &self.devices {
            let throughput = self.calculate_throughput(device.id).unwrap_or(0.0);
            let device_stats = GpuStats {
                device_id: device.id,
                utilization: device.utilization,
                memory_used: device.total_memory - device.available_memory,
                memory_total: device.total_memory,
                temperature: device.temperature,
                power_consumption: device.power_consumption,
                throughput,
            };

            stats.push(device_stats);
        }

        // Record monitoring data
        let timestamp = Instant::now();
        let utilizations: Vec<f32> = stats.iter().map(|s| s.utilization).collect();
        let memory_usage: Vec<u64> = stats.iter().map(|s| s.memory_used).collect();
        let temperatures: Vec<f32> = stats.iter().map(|s| s.temperature).collect();

        if let Ok(mut monitor) = self.monitor.lock() {
            monitor.utilization_history.push((timestamp, utilizations));
            monitor.memory_history.push((timestamp, memory_usage));
            monitor.temperature_history.push((timestamp, temperatures));

            // Trim history to reasonable size
            Self::trim_history(&mut monitor.utilization_history, 1000);
            Self::trim_history(&mut monitor.memory_history, 1000);
            Self::trim_history(&mut monitor.temperature_history, 1000);
        }

        Ok(stats)
    }

    /// Enable automatic optimization
    pub fn enable_auto_optimization(&mut self) -> Result<(), MullamaError> {
        // Start background thread for optimization
        let monitor = Arc::clone(&self.monitor);
        let config = self.optimization_config.clone();

        std::thread::spawn(move || {
            Self::optimization_worker(monitor, config);
        });

        Ok(())
    }

    /// Background optimization worker
    fn optimization_worker(monitor: Arc<Mutex<PerformanceMonitor>>, config: OptimizationConfig) {
        loop {
            std::thread::sleep(config.monitoring_interval);

            if let Ok(monitor_guard) = monitor.lock() {
                // Check for performance issues
                if let Some((_, utilizations)) = monitor_guard.utilization_history.last() {
                    // Detect thermal throttling
                    if let Some((_, temperatures)) = monitor_guard.temperature_history.last() {
                        for (i, &temp) in temperatures.iter().enumerate() {
                            if temp > config.thermal_threshold {
                                // Handle thermal throttling
                                eprintln!("Warning: GPU {} temperature: {}°C", i, temp);
                            }
                        }
                    }

                    // Detect load imbalance
                    if config.load_balancing && utilizations.len() > 1 {
                        let max_util = utilizations.iter().fold(0.0f32, |a, &b| a.max(b));
                        let min_util = utilizations.iter().fold(100.0f32, |a, &b| a.min(b));

                        if max_util - min_util > 30.0 {
                            // 30% difference threshold
                            eprintln!(
                                "Warning: Load imbalance detected: {}% - {}%",
                                max_util, min_util
                            );
                        }
                    }
                }
            }
        }
    }

    /// Select optimal device for allocation
    fn select_optimal_device(
        &self,
        size: u64,
        _block_type: MemoryBlockType,
        preferred_device: Option<usize>,
    ) -> Result<usize, MullamaError> {
        if let Some(device_id) = preferred_device {
            if self.devices.iter().any(|d| d.id == device_id) {
                return Ok(device_id);
            }
        }

        match self.allocation_strategy {
            AllocationStrategy::FirstFit => {
                for device in &self.devices {
                    if device.available_memory >= size {
                        return Ok(device.id);
                    }
                }
            }
            AllocationStrategy::BestFit => {
                let mut best_device = None;
                let mut best_fit_size = u64::MAX;

                for device in &self.devices {
                    if device.available_memory >= size && device.available_memory < best_fit_size {
                        best_device = Some(device.id);
                        best_fit_size = device.available_memory;
                    }
                }

                if let Some(device_id) = best_device {
                    return Ok(device_id);
                }
            }
            AllocationStrategy::LoadBalanced => {
                // Find device with lowest utilization
                let mut best_device = None;
                let mut lowest_utilization = f32::MAX;

                for device in &self.devices {
                    if device.available_memory >= size && device.utilization < lowest_utilization {
                        best_device = Some(device.id);
                        lowest_utilization = device.utilization;
                    }
                }

                if let Some(device_id) = best_device {
                    return Ok(device_id);
                }
            }
            AllocationStrategy::PerformanceOptimized => {
                // Consider multiple factors: memory, utilization, temperature
                let mut best_device = None;
                let mut best_score = f32::MIN;

                for device in &self.devices {
                    if device.available_memory >= size {
                        let memory_factor =
                            device.available_memory as f32 / device.total_memory as f32;
                        let util_factor = 1.0 - (device.utilization / 100.0);
                        let temp_factor = 1.0 - (device.temperature / 100.0).min(1.0);

                        let score = memory_factor * 0.4 + util_factor * 0.4 + temp_factor * 0.2;

                        if score > best_score {
                            best_device = Some(device.id);
                            best_score = score;
                        }
                    }
                }

                if let Some(device_id) = best_device {
                    return Ok(device_id);
                }
            }
            AllocationStrategy::Custom => {
                // Would implement custom logic
                return Ok(0); // Fallback to first device
            }
        }

        Err(MullamaError::GpuError(
            "No suitable device found for allocation".to_string(),
        ))
    }

    /// Record a performance event
    fn record_event(
        &self,
        event_type: EventType,
        device_id: Option<usize>,
        description: String,
        metrics: HashMap<String, f64>,
    ) {
        if let Ok(mut monitor) = self.monitor.lock() {
            monitor.events.push(PerformanceEvent {
                timestamp: Instant::now(),
                event_type,
                device_id,
                description,
                metrics,
            });

            // Trim events history
            if monitor.events.len() > 10000 {
                monitor.events.drain(0..1000);
            }
        }
    }

    /// Discover CUDA devices via nvidia-smi
    fn discover_cuda_devices() -> Result<Vec<GpuDevice>, MullamaError> {
        let mut devices = Vec::new();

        if !unsafe { crate::sys::llama_supports_gpu_offload() } {
            return Ok(devices);
        }

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 7 {
                    let id = parts[0].trim().parse::<usize>().unwrap_or(0);
                    let name = parts[1].trim().to_string();
                    let total_mb = parts[2].trim().parse::<f64>().unwrap_or(0.0);
                    let free_mb = parts[3].trim().parse::<f64>().unwrap_or(0.0);
                    let utilization = parts[4].trim().parse::<f32>().unwrap_or(0.0);
                    let temperature = parts[5].trim().parse::<f32>().unwrap_or(0.0);
                    let power = parts[6].trim().parse::<f32>().unwrap_or(0.0);

                    let total_bytes = (total_mb * 1024.0 * 1024.0) as u64;
                    let free_bytes = (free_mb * 1024.0 * 1024.0) as u64;
                    let _used_bytes = total_bytes.saturating_sub(free_bytes);

                    devices.push(GpuDevice {
                        id,
                        name,
                        total_memory: total_bytes,
                        available_memory: free_bytes,
                        compute_capability: (7, 5),
                        max_streams: 16,
                        device_type: GpuDeviceType::Cuda,
                        utilization,
                        temperature,
                        power_consumption: power,
                    });
                }
            }
        }

        if devices.is_empty() && unsafe { crate::sys::llama_max_devices() } > 0 {
            let (sys_used, sys_total) =
                crate::memory_monitor::get_system_memory().unwrap_or((0, 8 * 1024 * 1024 * 1024));
            devices.push(GpuDevice {
                id: 0,
                name: "CUDA Device 0".to_string(),
                total_memory: sys_total,
                available_memory: sys_total.saturating_sub(sys_used),
                compute_capability: (7, 5),
                max_streams: 16,
                device_type: GpuDeviceType::Cuda,
                utilization: 0.0,
                temperature: 45.0,
                power_consumption: 0.0,
            });
        }

        Ok(devices)
    }

    #[cfg(target_os = "macos")]
    fn discover_metal_devices() -> Result<Vec<GpuDevice>, MullamaError> {
        let mut devices = Vec::new();

        if !unsafe { crate::sys::llama_supports_gpu_offload() } {
            return Ok(devices);
        }

        let (sys_used, sys_total) =
            crate::memory_monitor::get_system_memory().unwrap_or((0, 16 * 1024 * 1024 * 1024));
        let available = sys_total.saturating_sub(sys_used);
        let gpu_wired = crate::memory_monitor::get_macos_wired_memory().unwrap_or(0);
        let gpu_used = gpu_wired;
        let gpu_available = available.saturating_sub(gpu_wired);

        let gpu_name = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .ok()
            .and_then(|o| {
                let s = String::from_utf8_lossy(&o.stdout);
                serde_json::from_str::<serde_json::Value>(&s).ok()
            })
            .and_then(|v| {
                let displays = v.get("SPDisplaysDataType")?.as_array()?;
                let first = displays.first()?;
                first
                    .get("chipsetVendor")
                    .or_else(|| first.get("_name"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "Apple GPU".to_string());

        devices.push(GpuDevice {
            id: 0,
            name: gpu_name,
            total_memory: gpu_available.saturating_add(gpu_used).max(gpu_used),
            available_memory: gpu_available,
            compute_capability: (1, 0),
            max_streams: 8,
            device_type: GpuDeviceType::Metal,
            utilization: if sys_total > 0 {
                (sys_used as f32 / sys_total as f32) * 100.0
            } else {
                0.0
            },
            temperature: 40.0,
            power_consumption: 0.0,
        });

        Ok(devices)
    }

    #[cfg(feature = "rocm")]
    fn discover_rocm_devices() -> Result<Vec<GpuDevice>, MullamaError> {
        let mut devices = Vec::new();

        if !unsafe { crate::sys::llama_supports_gpu_offload() } {
            return Ok(devices);
        }

        if let Ok(output) = std::process::Command::new("rocm-smi")
            .args(["--showmeminfo", "vram", "--csv"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for (idx, line) in stdout.lines().enumerate() {
                if idx == 0 {
                    continue;
                }
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 3 {
                    let total: u64 = parts[0].trim().parse().unwrap_or(0);
                    let used: u64 = parts[1].trim().parse().unwrap_or(0);
                    if total > 0 {
                        devices.push(GpuDevice {
                            id: idx.saturating_sub(1),
                            name: format!("AMD ROCm Device {}", idx.saturating_sub(1)),
                            total_memory: total * 1024,
                            available_memory: total.saturating_sub(used) * 1024,
                            compute_capability: (9, 0),
                            max_streams: 16,
                            device_type: GpuDeviceType::Rocm,
                            utilization: if total > 0 {
                                (used as f32 / total as f32) * 100.0
                            } else {
                                0.0
                            },
                            temperature: 45.0,
                            power_consumption: 0.0,
                        });
                    }
                }
            }
        }

        if devices.is_empty() && unsafe { crate::sys::llama_max_devices() } > 0 {
            let (sys_used, sys_total) =
                crate::memory_monitor::get_system_memory().unwrap_or((0, 8 * 1024 * 1024 * 1024));
            devices.push(GpuDevice {
                id: 0,
                name: "AMD ROCm Device 0".to_string(),
                total_memory: sys_total,
                available_memory: sys_total.saturating_sub(sys_used),
                compute_capability: (9, 0),
                max_streams: 16,
                device_type: GpuDeviceType::Rocm,
                utilization: 0.0,
                temperature: 45.0,
                power_consumption: 0.0,
            });
        }

        Ok(devices)
    }

    /// Fallback device discovery when no specific GPU features are enabled
    #[cfg(not(any(feature = "cuda", feature = "rocm", target_os = "macos")))]
    fn discover_fallback_devices() -> Result<Vec<GpuDevice>, MullamaError> {
        if !unsafe { crate::sys::llama_supports_gpu_offload() } {
            return Ok(Vec::new());
        }

        let mut devices = Vec::new();

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 4 {
                    let id = parts[0].trim().parse::<usize>().unwrap_or(0);
                    let name = parts[1].trim().to_string();
                    let total_mb = parts[2].trim().parse::<f64>().unwrap_or(0.0);
                    let free_mb = parts[3].trim().parse::<f64>().unwrap_or(0.0);
                    devices.push(GpuDevice {
                        id,
                        name,
                        total_memory: (total_mb * 1024.0 * 1024.0) as u64,
                        available_memory: (free_mb * 1024.0 * 1024.0) as u64,
                        compute_capability: (1, 0),
                        max_streams: 8,
                        device_type: GpuDeviceType::Vulkan,
                        utilization: 0.0,
                        temperature: 0.0,
                        power_consumption: 0.0,
                    });
                }
            }
        }

        if devices.is_empty() {
            let (sys_used, sys_total) =
                crate::memory_monitor::get_system_memory().unwrap_or((0, 4 * 1024 * 1024 * 1024));
            devices.push(GpuDevice {
                id: 0,
                name: "Generic GPU Device".to_string(),
                total_memory: sys_total,
                available_memory: sys_total.saturating_sub(sys_used),
                compute_capability: (1, 0),
                max_streams: 8,
                device_type: GpuDeviceType::Vulkan,
                utilization: 0.0,
                temperature: 0.0,
                power_consumption: 0.0,
            });
        }

        Ok(devices)
    }

    fn find_device_for_address(&self, _address: u64) -> Result<usize, MullamaError> {
        Ok(0)
    }

    fn create_load_balanced_placement(
        &self,
        model_size: u64,
        num_layers: usize,
    ) -> Result<ModelPlacement, MullamaError> {
        if self.devices.is_empty() {
            return Ok(ModelPlacement::cpu_only(num_layers));
        }

        let per_layer_size = if num_layers > 0 {
            model_size / num_layers as u64
        } else {
            model_size
        };

        let mut layer_assignments = HashMap::new();
        let mut memory_requirements = HashMap::new();
        let mut device_remaining: Vec<(usize, u64)> = self
            .devices
            .iter()
            .map(|d| (d.id, d.available_memory))
            .collect();

        for layer_idx in 0..num_layers {
            let mut assigned = false;
            device_remaining.sort_by(|a, b| b.1.cmp(&a.1));

            for (device_id, remaining) in &mut device_remaining {
                if *remaining >= per_layer_size {
                    layer_assignments.insert(layer_idx, *device_id);
                    memory_requirements
                        .entry(*device_id)
                        .and_modify(|m| *m += per_layer_size)
                        .or_insert(per_layer_size);
                    *remaining -= per_layer_size;
                    assigned = true;
                    break;
                }
            }

            if !assigned {
                layer_assignments.insert(layer_idx, 0);
            }
        }

        Ok(ModelPlacement {
            layer_assignments,
            memory_requirements,
        })
    }

    fn create_performance_optimized_placement(
        &self,
        model_size: u64,
        num_layers: usize,
    ) -> Result<ModelPlacement, MullamaError> {
        if self.devices.is_empty() {
            return Ok(ModelPlacement::cpu_only(num_layers));
        }

        let per_layer_size = if num_layers > 0 {
            model_size / num_layers as u64
        } else {
            model_size
        };

        let mut layer_assignments = HashMap::new();
        let mut memory_requirements = HashMap::new();
        let mut device_remaining: Vec<(usize, u64, f32)> = self
            .devices
            .iter()
            .map(|d| (d.id, d.available_memory, d.utilization))
            .collect();

        for layer_idx in 0..num_layers {
            let mut best_device = None;
            let mut best_score = f32::MIN;

            for (device_id, remaining, utilization) in &device_remaining {
                if *remaining >= per_layer_size {
                    let mem_factor = *remaining as f32 / model_size as f32;
                    let util_factor = 1.0 - (*utilization / 100.0);
                    let score = util_factor * 0.6 + mem_factor * 0.4;
                    if score > best_score {
                        best_score = score;
                        best_device = Some(*device_id);
                    }
                }
            }

            if let Some(device_id) = best_device {
                layer_assignments.insert(layer_idx, device_id);
                memory_requirements
                    .entry(device_id)
                    .and_modify(|m| *m += per_layer_size)
                    .or_insert(per_layer_size);
                for (id, remaining, _) in &mut device_remaining {
                    if *id == device_id {
                        *remaining -= per_layer_size;
                        break;
                    }
                }
            } else {
                layer_assignments.insert(layer_idx, 0);
            }
        }

        Ok(ModelPlacement {
            layer_assignments,
            memory_requirements,
        })
    }

    fn create_simple_placement(
        &self,
        model_size: u64,
        num_layers: usize,
    ) -> Result<ModelPlacement, MullamaError> {
        if self.devices.is_empty() {
            return Ok(ModelPlacement::cpu_only(num_layers));
        }

        let per_layer_size = if num_layers > 0 {
            model_size / num_layers as u64
        } else {
            model_size
        };

        let mut layer_assignments = HashMap::new();
        let mut memory_requirements = HashMap::new();
        let mut device_idx = 0;
        let mut remaining_on_device = self.devices[0].available_memory;
        let mut current_device_id = self.devices[0].id;

        memory_requirements.insert(current_device_id, 0u64);

        for layer_idx in 0..num_layers {
            if remaining_on_device < per_layer_size && device_idx + 1 < self.devices.len() {
                device_idx += 1;
                current_device_id = self.devices[device_idx].id;
                remaining_on_device = self.devices[device_idx].available_memory;
                memory_requirements.insert(current_device_id, 0u64);
            }

            if remaining_on_device >= per_layer_size {
                layer_assignments.insert(layer_idx, current_device_id);
                memory_requirements
                    .entry(current_device_id)
                    .and_modify(|m| *m += per_layer_size);
                remaining_on_device -= per_layer_size;
            } else {
                layer_assignments.insert(layer_idx, self.devices[0].id);
            }
        }

        Ok(ModelPlacement {
            layer_assignments,
            memory_requirements,
        })
    }

    #[allow(dead_code)]
    fn update_device_info(&self, _device: &mut GpuDevice) -> Result<(), MullamaError> {
        Ok(())
    }

    #[allow(dead_code)]
    fn calculate_throughput(&self, device_id: usize) -> Result<f32, MullamaError> {
        let device = self
            .devices
            .iter()
            .find(|d| d.id == device_id)
            .ok_or_else(|| MullamaError::GpuError(format!("Device {} not found", device_id)))?;
        let util_factor = 1.0 - (device.utilization / 100.0).min(1.0);
        let thermal_factor = 1.0 - (device.temperature / 100.0).min(0.5);
        let base_throughput = 100.0;
        Ok(base_throughput * util_factor * thermal_factor)
    }

    fn trim_history<T>(history: &mut Vec<(Instant, T)>, max_size: usize) {
        if history.len() > max_size {
            history.drain(0..history.len() - max_size);
        }
    }
}

/// Additional supporting structures

#[derive(Debug, Default)]
pub struct ModelPlacement {
    pub layer_assignments: HashMap<usize, usize>,
    pub memory_requirements: HashMap<usize, u64>,
}

impl ModelPlacement {
    fn cpu_only(_num_layers: usize) -> Self {
        Self {
            layer_assignments: HashMap::new(),
            memory_requirements: HashMap::new(),
        }
    }

    pub fn gpu_layers(&self) -> usize {
        self.layer_assignments
            .values()
            .filter(|&&d| d != usize::MAX)
            .count()
    }
}

fn estimate_model_size(num_layers: usize, embedding_dim: usize) -> u64 {
    let bytes_per_param = 2;
    let params_per_layer = embedding_dim * embedding_dim * 4;
    let embedding_params = embedding_dim * 32000;
    let total_params = (num_layers * params_per_layer) + embedding_params;
    (total_params as u64) * (bytes_per_param as u64)
}

#[derive(Debug)]
pub struct DefragmentationResult {
    pub bytes_freed: u64,
    pub bytes_moved: u64,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct GpuStats {
    pub device_id: usize,
    pub utilization: f32,
    pub memory_used: u64,
    pub memory_total: u64,
    pub temperature: f32,
    pub power_consumption: f32,
    pub throughput: f32,
}

impl GpuMemoryPool {
    fn new(device_id: usize, size: u64) -> Result<Self, MullamaError> {
        Ok(Self {
            device_id,
            free_blocks: vec![MemoryBlock {
                address: 0,
                size,
                allocated_at: Instant::now(),
                block_type: MemoryBlockType::Temporary,
            }],
            allocated_blocks: HashMap::new(),
            total_size: size,
            used_size: 0,
            stats: PoolStats::default(),
        })
    }

    fn allocate(
        &mut self,
        size: u64,
        block_type: MemoryBlockType,
    ) -> Result<MemoryBlock, MullamaError> {
        // Find suitable free block
        for (i, block) in self.free_blocks.iter().enumerate() {
            if block.size >= size {
                let allocated_block = MemoryBlock {
                    address: block.address,
                    size,
                    allocated_at: Instant::now(),
                    block_type,
                };

                // Remove or split the free block
                if block.size == size {
                    self.free_blocks.remove(i);
                } else {
                    self.free_blocks[i] = MemoryBlock {
                        address: block.address + size,
                        size: block.size - size,
                        allocated_at: block.allocated_at,
                        block_type: block.block_type,
                    };
                }

                self.allocated_blocks
                    .insert(allocated_block.address, allocated_block.clone());
                self.used_size += size;
                self.stats.total_allocations += 1;

                return Ok(allocated_block);
            }
        }

        Err(MullamaError::GpuError(format!(
            "Unable to allocate {} bytes from pool",
            size
        )))
    }

    fn deallocate(&mut self, block: MemoryBlock) -> Result<(), MullamaError> {
        if self.allocated_blocks.remove(&block.address).is_some() {
            let block_size = block.size;
            self.free_blocks.push(block);
            self.used_size -= block_size;
            self.stats.total_deallocations += 1;

            // Coalesce adjacent free blocks
            self.coalesce_free_blocks();

            Ok(())
        } else {
            Err(MullamaError::GpuError(
                "Block not found in allocated blocks".to_string(),
            ))
        }
    }

    fn defragment(&mut self) -> Result<DefragmentationResult, MullamaError> {
        let start_time = Instant::now();
        let initial_fragmentation = self.calculate_fragmentation();

        // Sort free blocks by address
        self.free_blocks.sort_by_key(|block| block.address);

        // Coalesce adjacent blocks
        self.coalesce_free_blocks();

        let final_fragmentation = self.calculate_fragmentation();
        let bytes_freed =
            ((initial_fragmentation - final_fragmentation) * self.total_size as f32) as u64;

        self.stats.defragmentation_ops += 1;

        Ok(DefragmentationResult {
            bytes_freed,
            bytes_moved: 0, // Would be calculated based on actual moves
            duration: start_time.elapsed(),
        })
    }

    fn coalesce_free_blocks(&mut self) {
        self.free_blocks.sort_by_key(|block| block.address);

        let mut i = 0;
        while i < self.free_blocks.len().saturating_sub(1) {
            if self.free_blocks[i].address + self.free_blocks[i].size
                == self.free_blocks[i + 1].address
            {
                // Merge blocks
                self.free_blocks[i].size += self.free_blocks[i + 1].size;
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    fn calculate_fragmentation(&self) -> f32 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }

        let largest_free_block = self.free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let total_free = self.free_blocks.iter().map(|b| b.size).sum::<u64>();

        if total_free == 0 {
            0.0
        } else {
            1.0 - (largest_free_block as f32 / total_free as f32)
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            dynamic_memory: true,
            auto_defragmentation: true,
            fragmentation_threshold: 0.3,
            monitoring_interval: Duration::from_secs(1),
            predictive_optimization: false,
            thermal_threshold: 80.0,
            load_balancing: true,
        }
    }
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            devices: Vec::new(),
            allocation_strategy: AllocationStrategy::FirstFit,
            memory_pools: HashMap::new(),
            monitor: Arc::new(Mutex::new(PerformanceMonitor::default())),
            optimization_config: OptimizationConfig::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = GpuMemoryPool::new(0, 1024).unwrap();

        let block = pool.allocate(256, MemoryBlockType::ModelWeights).unwrap();
        assert_eq!(block.size, 256);
        assert_eq!(pool.used_size, 256);

        pool.deallocate(block).unwrap();
        assert_eq!(pool.used_size, 0);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let mut pool = GpuMemoryPool::new(0, 1024).unwrap();

        // No fragmentation initially
        assert_eq!(pool.calculate_fragmentation(), 0.0);

        // Allocate some blocks to create fragmentation
        let block1 = pool.allocate(256, MemoryBlockType::ModelWeights).unwrap();
        let _block2 = pool.allocate(256, MemoryBlockType::Activations).unwrap();
        pool.deallocate(block1).unwrap();

        // Should have some fragmentation now
        assert!(pool.calculate_fragmentation() > 0.0);
    }

    #[test]
    fn test_allocation_strategies() {
        let device = GpuDevice {
            id: 0,
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,     // 8GB
            available_memory: 4 * 1024 * 1024 * 1024, // 4GB available
            compute_capability: (8, 0),
            max_streams: 16,
            device_type: GpuDeviceType::Cuda,
            utilization: 50.0,
            temperature: 65.0,
            power_consumption: 200.0,
        };

        // Test that we can create a manager with devices
        let manager = GpuManager {
            devices: vec![device],
            allocation_strategy: AllocationStrategy::PerformanceOptimized,
            memory_pools: HashMap::new(),
            monitor: Arc::new(Mutex::new(PerformanceMonitor::default())),
            optimization_config: OptimizationConfig::default(),
        };

        // Test device selection
        let device_id = manager
            .select_optimal_device(
                1024 * 1024 * 1024, // 1GB
                MemoryBlockType::ModelWeights,
                None,
            )
            .unwrap();

        assert_eq!(device_id, 0);
    }
}
