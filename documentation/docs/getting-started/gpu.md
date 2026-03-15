---
title: GPU Acceleration
description: Configure GPU acceleration for Mullama with NVIDIA CUDA, Apple Metal, AMD ROCm, or OpenCL. Includes driver setup, layer offloading, VRAM guidelines, multi-GPU, and verification steps.
---

# GPU Acceleration

GPU acceleration dramatically improves inference speed, especially for larger models. Mullama supports multiple GPU backends through llama.cpp's build system.

---

## Supported Backends

| Backend | GPU Hardware | Platform | Environment Variable |
|---------|-------------|----------|---------------------|
| **CUDA** | NVIDIA GeForce, RTX, Tesla, A100, H100 | Linux, Windows | `LLAMA_CUDA=1` |
| **Metal** | Apple M1/M2/M3/M4 (integrated GPU) | macOS | `LLAMA_METAL=1` |
| **ROCm (HIP)** | AMD Radeon RX 6000+, Instinct MI | Linux | `LLAMA_HIPBLAS=1` |
| **OpenCL (CLBlast)** | Any OpenCL-capable GPU | Linux, macOS, Windows | `LLAMA_CLBLAST=1` |

!!! tip "Which Backend Should I Use?"

    - **NVIDIA GPU:** Use CUDA (best performance)
    - **Apple Silicon Mac:** Use Metal (automatic, no setup needed)
    - **AMD GPU on Linux:** Use ROCm
    - **Other GPUs or older hardware:** Use OpenCL (broadest compatibility, lower performance)

---

## NVIDIA CUDA

CUDA provides the best performance for NVIDIA GPUs. Requires CUDA Toolkit 11.0 or later (12.x recommended).

### Driver Requirements

| GPU Generation | Minimum Driver | Recommended Driver |
|---------------|---------------|-------------------|
| RTX 40-series (Ada) | 525.60+ | 545.xx+ |
| RTX 30-series (Ampere) | 470.42+ | 545.xx+ |
| RTX 20-series (Turing) | 440.33+ | 545.xx+ |
| GTX 10-series (Pascal) | 410.48+ | 535.xx+ |
| Data Center (A100, H100) | 450.80+ | 545.xx+ |

### Step 1: Install NVIDIA Drivers

=== "Ubuntu / Debian"

    ```bash
    sudo apt update
    sudo apt install -y nvidia-driver-545

    # Reboot to load the driver
    sudo reboot
    ```

=== "Fedora"

    ```bash
    sudo dnf install -y akmod-nvidia
    sudo reboot
    ```

=== "Windows"

    Download the latest driver from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) and run the installer.

### Step 2: Install CUDA Toolkit

=== "Ubuntu / Debian"

    ```bash
    # Add NVIDIA CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update

    # Install CUDA Toolkit
    sudo apt install -y cuda-toolkit-12-4

    # Configure environment
    echo 'export CUDA_PATH="/usr/local/cuda"' >> ~/.bashrc
    echo 'export PATH="$CUDA_PATH/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "Windows"

    Download and install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

    ```powershell
    # Set environment (usually done by installer)
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    [Environment]::SetEnvironmentVariable("CUDA_PATH", $env:CUDA_PATH, "Machine")
    ```

### Step 3: Verify Installation

```bash
# Check driver is loaded
nvidia-smi

# Check CUDA toolkit version
nvcc --version
```

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.4, V12.4.131
```

### Step 4: Build Mullama with CUDA

```bash
export LLAMA_CUDA=1
cargo build --release --features full
```

!!! info "Persist the Environment Variable"

    Add to your shell profile so it is set on every build:

    ```bash
    echo 'export LLAMA_CUDA=1' >> ~/.bashrc
    source ~/.bashrc
    ```

---

## Apple Metal

Metal provides GPU acceleration on Apple Silicon Macs (M1, M2, M3, M4 series). It leverages the unified memory architecture for efficient CPU-GPU data sharing.

### Setup

Metal support is **automatically available** on macOS with Apple Silicon. No additional drivers or toolkits are needed.

```bash
export LLAMA_METAL=1
cargo build --release --features full
```

!!! success "No Extra Setup Needed"

    On Apple Silicon, Metal acceleration works out of the box. The environment variable simply enables the Metal compute backend in llama.cpp during compilation.

### Verify Metal Support

```bash
system_profiler SPDisplaysDataType | grep Metal
```

Expected output:

```
Metal Family: Supported, Metal GPUFamily Apple 9
```

### Metal-Specific Configuration

```bash
# Enable Metal (required for build)
export LLAMA_METAL=1

# Disable Metal debug output for production (faster)
export GGML_METAL_NDEBUG=1
```

!!! tip "Unified Memory Advantage"

    On Apple Silicon, the GPU shares memory with the CPU. This means you can set `n_gpu_layers = -1` to offload ALL layers without worrying about separate VRAM limits. The only constraint is your total system RAM.

### Intel Macs

Metal is also available on Intel Macs with dedicated AMD GPUs, but performance gains are more modest. Consider using OpenCL (`LLAMA_CLBLAST=1`) as an alternative on Intel Macs.

---

## AMD ROCm

ROCm (Radeon Open Compute) provides GPU acceleration for AMD GPUs on Linux.

### Supported GPUs

- AMD Instinct MI-series (MI100, MI210, MI250X, MI300X)
- AMD Radeon RX 7000 series (RDNA 3)
- AMD Radeon RX 6000 series (RDNA 2)
- AMD Radeon Pro W6000/W7000 series

### Step 1: Install ROCm

=== "Ubuntu 22.04"

    ```bash
    # Add ROCm repository
    wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
    sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb

    # Install ROCm
    sudo amdgpu-install --usecase=rocm

    # Add user to required groups
    sudo usermod -a -G render,video $USER
    ```

=== "Ubuntu 24.04"

    ```bash
    wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/noble/amdgpu-install_6.0.60000-1_all.deb
    sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
    sudo amdgpu-install --usecase=rocm
    sudo usermod -a -G render,video $USER
    ```

=== "Fedora"

    ```bash
    sudo dnf install -y rocm-hip-devel rocblas-devel hipblas-devel
    sudo usermod -a -G render,video $USER
    ```

### Step 2: Configure Environment

```bash
echo 'export ROCM_PATH="/opt/rocm"' >> ~/.bashrc
echo 'export PATH="$ROCM_PATH/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Reboot and Verify

```bash
sudo reboot

# After reboot
rocm-smi
rocminfo | grep "Name:"
```

!!! warning "Reboot Required"

    A reboot is required after installing ROCm for the kernel modules to load properly.

### Step 4: Build with ROCm

```bash
export LLAMA_HIPBLAS=1
cargo build --release --features full
```

---

## OpenCL (CLBlast)

OpenCL provides broad GPU compatibility across vendors. Performance is generally lower than native backends (CUDA, Metal, ROCm) but works on a wider range of hardware, including older GPUs and integrated graphics.

### Setup

=== "Linux"

    ```bash
    # Install OpenCL development packages
    sudo apt install -y \
        opencl-headers \
        ocl-icd-opencl-dev \
        clinfo

    # Install CLBlast
    sudo apt install -y libclblast-dev

    # Verify OpenCL devices
    clinfo | head -20
    ```

=== "macOS"

    ```bash
    brew install clblast
    ```

=== "Windows"

    OpenCL runtime is typically included with GPU drivers. Install CLBlast from [GitHub releases](https://github.com/CNugteren/CLBlast/releases).

### Build with OpenCL

```bash
export LLAMA_CLBLAST=1
cargo build --release --features full
```

!!! info "When to Use OpenCL"

    Use OpenCL when:

    - You have an older GPU not supported by CUDA/ROCm
    - You have integrated graphics (Intel HD/Iris, AMD APU)
    - You need cross-vendor GPU support
    - Native backends are unavailable on your platform

---

## GPU Layer Offloading

The `n_gpu_layers` parameter controls how many transformer layers are offloaded from CPU to GPU. This is the single most important parameter for GPU performance.

### How It Works

A language model consists of many transformer layers stacked sequentially. Each layer that runs on the GPU avoids a CPU computation and memory transfer. More layers on the GPU means faster inference.

| Value | Behavior |
|-------|----------|
| `0` | CPU only -- no GPU acceleration |
| `1` to `N` | Offload the first N layers to GPU |
| `-1` | Offload ALL layers to GPU (recommended if VRAM allows) |

### Usage Examples

=== "Node.js"

    ```javascript
    const { Model } = require('mullama');

    // Offload all layers to GPU
    const model = new Model('model.gguf', {
        nGpuLayers: -1
    });

    // Offload 35 layers (partial, for limited VRAM)
    const model2 = new Model('model.gguf', {
        nGpuLayers: 35
    });
    ```

=== "Python"

    ```python
    from mullama import Model

    # Offload all layers to GPU
    model = Model("model.gguf", n_gpu_layers=-1)

    # Offload 35 layers (partial, for limited VRAM)
    model2 = Model("model.gguf", n_gpu_layers=35)
    ```

=== "Rust"

    ```rust
    use mullama::ModelParams;

    // Offload all layers
    let params = ModelParams::default()
        .with_n_gpu_layers(-1);

    // Partial offload
    let params = ModelParams::default()
        .with_n_gpu_layers(35);
    ```

=== "CLI"

    ```bash
    # All layers on GPU
    mullama run llama3.2:1b --n-gpu-layers -1 "Hello!"

    # Partial offload
    mullama run llama3.2:1b --n-gpu-layers 20 "Hello!"
    ```

!!! tip "Finding the Right Number"

    Start with `-1` (all layers). If you get out-of-memory errors, reduce the value by 5 until it fits. Each layer uses approximately `model_size / total_layers` of VRAM.

---

## Memory Considerations

### VRAM Requirements by Model Size

Approximate VRAM needed for full GPU offload (`n_gpu_layers = -1`):

| Model Size | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 |
|-----------|--------|--------|------|------|-----|
| **1B** | ~0.9 GB | ~1.0 GB | ~1.2 GB | ~1.5 GB | ~2.5 GB |
| **3B** | ~2.2 GB | ~2.5 GB | ~2.9 GB | ~3.6 GB | ~6.5 GB |
| **7B** | ~4.5 GB | ~5.1 GB | ~5.9 GB | ~7.5 GB | ~14 GB |
| **13B** | ~8.0 GB | ~9.2 GB | ~10.5 GB | ~13.5 GB | ~26 GB |
| **30B** | ~19 GB | ~22 GB | ~25 GB | ~32 GB | ~60 GB |
| **70B** | ~40 GB | ~46 GB | ~53 GB | ~68 GB | ~140 GB |

### Recommended GPUs by Model Size

| Model Size | Minimum GPU | Recommended GPU |
|-----------|-------------|-----------------|
| 1B-3B | Any 4 GB GPU | GTX 1650, M1 |
| 7B | 6 GB VRAM | RTX 3060 12GB, M1 Pro |
| 13B | 10 GB VRAM | RTX 3080, RTX 4070, M2 Pro |
| 30B | 24 GB VRAM | RTX 3090, RTX 4090, M2 Max |
| 70B | 48+ GB VRAM | 2x RTX 4090, A100, M2 Ultra |

!!! info "Context Size Affects VRAM"

    The VRAM figures above are for model weights only. Additional VRAM is needed for the KV cache, which scales with context size:

    - 2048 context: +0.5-1 GB
    - 4096 context: +1-2 GB
    - 8192 context: +2-4 GB
    - 32768 context: +8-16 GB

---

## Multi-GPU Configuration

For systems with multiple GPUs, Mullama supports distributing model layers across devices.

### Tensor Split

Control how model weight tensors are distributed across GPUs:

=== "Node.js"

    ```javascript
    const model = new Model('large-model.gguf', {
        nGpuLayers: -1,
        tensorSplit: [0.6, 0.4]  // 60% GPU 0, 40% GPU 1
    });
    ```

=== "Python"

    ```python
    model = Model("large-model.gguf",
        n_gpu_layers=-1,
        tensor_split=[0.6, 0.4]  # 60% GPU 0, 40% GPU 1
    )
    ```

=== "Rust"

    ```rust
    use mullama::{ModelParams, SplitMode};

    // Layer split: distribute layers across GPUs
    let params = ModelParams::default()
        .with_n_gpu_layers(-1)
        .with_split_mode(SplitMode::Layer)
        .with_tensor_split(&[0.6, 0.4]);  // 60% GPU 0, 40% GPU 1

    // Row split: split individual tensors (better for very large layers)
    let params = ModelParams::default()
        .with_n_gpu_layers(-1)
        .with_split_mode(SplitMode::Row)
        .with_tensor_split(&[0.5, 0.5]);  // Equal split
    ```

### Split Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Layer** | Distributes entire layers across GPUs | Unequal GPU sizes |
| **Row** | Splits individual weight matrices across GPUs | Equal GPUs, very large models |

### Selecting a Primary GPU

```rust
let params = ModelParams::default()
    .with_n_gpu_layers(-1)
    .with_main_gpu(1);  // Use GPU 1 as primary (0-indexed)
```

### Multi-GPU Examples

```bash
# Two GPUs: 70/30 split (GPU 0 has more VRAM)
# GPU 0: RTX 4090 (24 GB), GPU 1: RTX 3090 (24 GB)
mullama run llama3.3:70b --n-gpu-layers -1 --tensor-split "0.5,0.5"

# Three GPUs: equal split
mullama run llama3.3:70b --n-gpu-layers -1 --tensor-split "0.33,0.33,0.34"
```

---

## Verification

### Check GPU Detection

```bash
# NVIDIA
nvidia-smi
nvcc --version

# AMD ROCm
rocm-smi
rocminfo

# Apple Metal
system_profiler SPDisplaysDataType | grep Metal

# OpenCL (any vendor)
clinfo
```

### Verify GPU Is Being Used

=== "Node.js"

    ```javascript
    const { Model, gpu } = require('mullama');

    console.log('CUDA available:', gpu.cudaAvailable());
    console.log('Metal available:', gpu.metalAvailable());

    const model = new Model('model.gguf', { nGpuLayers: -1 });
    console.log('GPU layers loaded:', model.gpuLayers);
    ```

=== "Python"

    ```python
    from mullama import Model, gpu

    print(f"CUDA available: {gpu.cuda_available()}")
    print(f"Metal available: {gpu.metal_available()}")

    model = Model("model.gguf", n_gpu_layers=-1)
    print(f"GPU layers loaded: {model.gpu_layers}")
    ```

=== "Rust"

    ```rust
    use mullama::gpu;

    fn check_gpu_status() {
        if gpu::cuda_available() {
            println!("CUDA available");
            println!("  Devices: {}", gpu::cuda_device_count());
            for i in 0..gpu::cuda_device_count() {
                let (used, total) = gpu::cuda_memory_info(i);
                println!("  GPU {}: {:.1} GB / {:.1} GB",
                    i, used as f64 / 1e9, total as f64 / 1e9);
            }
        }

        if gpu::metal_available() {
            println!("Metal available (Apple Silicon)");
        }

        if gpu::rocm_available() {
            println!("ROCm available");
            println!("  Devices: {}", gpu::rocm_device_count());
        }
    }
    ```

### Monitor GPU During Inference

```bash
# NVIDIA: watch GPU utilization in real-time
watch -n 1 nvidia-smi

# AMD: monitor ROCm devices
watch -n 1 rocm-smi

# All platforms: check if GPU layers were loaded (in log output)
MULLAMA_LOG=debug mullama run llama3.2:1b "Hello"
```

!!! tip "What to Look For"

    When GPU is active, you should see:

    - **nvidia-smi:** GPU utilization > 0%, memory usage increases when model loads
    - **Log output:** Messages like `offloaded 32/32 layers to GPU`
    - **Performance:** Significantly higher tokens/second compared to CPU-only

---

## Performance Tuning

### Batch Size

Larger batch sizes improve GPU utilization but require more VRAM:

=== "Node.js"

    ```javascript
    const model = new Model('model.gguf', {
        nGpuLayers: -1,
        contextSize: 4096,
        batchSize: 1024  // Higher = better GPU utilization
    });
    ```

=== "Python"

    ```python
    model = Model("model.gguf",
        n_gpu_layers=-1,
        context_size=4096,
        batch_size=1024  # Higher = better GPU utilization
    )
    ```

=== "Rust"

    ```rust
    let ctx_params = ContextParams::default()
        .with_n_ctx(4096)
        .with_n_batch(1024)    // Higher = better GPU utilization
        .with_n_threads(1);    // Fewer CPU threads when GPU is primary
    ```

### CPU Thread Count

When using GPU acceleration, reduce CPU threads since most compute is on the GPU:

```rust
let ctx_params = ContextParams::default()
    .with_n_threads(2)          // Minimal CPU threads for GPU workloads
    .with_n_threads_batch(4);   // Slightly more for batch processing
```

---

## Troubleshooting

??? question "CUDA out of memory"

    Reduce GPU layers or context size:

    ```python
    # Reduce layers
    model = Model("model.gguf", n_gpu_layers=20)

    # Or use a smaller context
    model = Model("model.gguf", n_gpu_layers=-1, context_size=1024)
    ```

    Or use a more aggressively quantized model (Q4_0 instead of Q5_K_M).

??? question "GPU not detected at runtime"

    1. Verify the environment variable was set **before** building:

        ```bash
        echo $LLAMA_CUDA  # Should print "1"
        ```

    2. Rebuild from clean:

        ```bash
        cargo clean
        export LLAMA_CUDA=1
        cargo build --release --features full
        ```

    3. Check that the GPU driver is loaded:

        ```bash
        nvidia-smi  # Should show GPU info, not an error
        ```

??? question "Metal performance is slow"

    Disable debug output:

    ```bash
    export GGML_METAL_NDEBUG=1
    ```

    Use `-1` for `n_gpu_layers` to offload all layers. Apple Silicon unified memory can handle this.

??? question "ROCm build fails"

    Ensure HIP development packages are installed:

    ```bash
    sudo apt install -y hip-dev rocblas-dev hipblas-dev
    ```

    Verify your GPU is supported:

    ```bash
    rocminfo | grep "Marketing Name"
    ```

??? question "OpenCL performance is poor"

    OpenCL/CLBlast is the slowest GPU backend. Consider:

    - Switching to CUDA (NVIDIA) or ROCm (AMD) if available
    - Using Metal on macOS
    - Ensuring CLBlast (not just OpenCL headers) is properly installed
    - Trying different `n_gpu_layers` values -- sometimes partial offload is better with OpenCL

??? question "Multi-GPU not splitting correctly"

    Verify all GPUs are detected:

    ```bash
    nvidia-smi -L  # Should list all GPUs
    ```

    Ensure tensor_split values sum to 1.0 and match the number of GPUs.

---

!!! success "Next Steps"

    - [Your First Project](first-project.md) -- Build a chatbot with GPU-accelerated inference
    - [Installation](installation.md) -- Feature flags and build options
    - [Streaming Guide](../guide/streaming.md) -- Real-time token generation
