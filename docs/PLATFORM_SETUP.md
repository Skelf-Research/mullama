# 🖥️ Platform-Specific Setup Guide

Comprehensive setup instructions for building Mullama on Windows, Linux, and macOS with all dependencies and platform-specific optimizations.

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Windows Setup](#-windows-setup)
- [Linux Setup](#-linux-setup)
- [macOS Setup](#-macos-setup)
- [GPU Acceleration](#-gpu-acceleration)
- [Platform-Specific Optimizations](#-platform-specific-optimizations)
- [Troubleshooting](#-troubleshooting)

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: x86_64 processor with SSE4.1 support
- **RAM**: 8 GB (16 GB recommended for full features)
- **Storage**: 10 GB free space (models require additional space)
- **Rust**: 1.70.0 or later
- **CMake**: 3.12 or later
- **Git**: 2.20 or later

### Recommended Requirements
- **CPU**: Modern multi-core processor (8+ cores recommended)
- **RAM**: 32 GB (for large models and parallel processing)
- **GPU**: NVIDIA RTX series, AMD RX series, or Apple Silicon M-series
- **Storage**: SSD with 50+ GB free space
- **Network**: Stable internet for model downloads

---

## 🪟 Windows Setup

### Prerequisites Installation

#### 1. Install Visual Studio Build Tools

**Option A: Visual Studio Community (Recommended)**
```powershell
# Download and install Visual Studio Community 2022
# https://visualstudio.microsoft.com/vs/community/

# Required components:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - Windows 11 SDK (latest version)
# - CMake tools for Visual Studio
# - Git for Windows
```

**Option B: Build Tools Only**
```powershell
# Download Visual Studio Build Tools 2022
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Install with workload: C++ build tools
```

#### 2. Install Package Manager (Chocolatey)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### 3. Install Core Dependencies
```powershell
# Install CMake, Git, and Rust
choco install cmake git rustup.install -y

# Refresh environment variables
refreshenv

# Install Rust stable toolchain
rustup default stable
rustup update

# Install additional tools
choco install llvm ninja -y
```

#### 4. Install Audio Dependencies
```powershell
# Windows audio dependencies are included in Windows SDK
# Additional audio processing libraries
choco install vcredist-all -y

# For advanced audio processing (optional)
# Download and install ASIO SDK from Steinberg (for professional audio)
```

#### 5. Install Development Tools
```powershell
# Visual Studio Code with Rust extension
choco install vscode -y

# Additional useful tools
choco install windows-sdk-10-version-2004-all -y
choco install dotnet-sdk -y
```

### Environment Configuration

#### Set Environment Variables
```powershell
# Add to System Environment Variables (permanent)
[Environment]::SetEnvironmentVariable("LLAMA_CUDA", "1", "Machine")
[Environment]::SetEnvironmentVariable("CMAKE_GENERATOR", "Ninja", "Machine")

# For current session
$env:LLAMA_CUDA = "1"
$env:CMAKE_GENERATOR = "Ninja"
```

#### Configure Rust
```powershell
# Set MSVC as default linker
rustup toolchain install stable-x86_64-pc-windows-msvc
rustup default stable-x86_64-pc-windows-msvc

# Install additional targets
rustup target add x86_64-pc-windows-msvc
```

### Building Mullama

#### Basic Build
```powershell
# Clone repository
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama

# Basic build
cargo build --release

# Build with all features
cargo build --release --features full
```

#### Windows-Specific Build Flags
```powershell
# High-performance build with Windows optimizations
$env:RUSTFLAGS = "-C target-cpu=native -C opt-level=3 -C lto=fat"
$env:CMAKE_BUILD_TYPE = "Release"

# Build with optimizations
cargo build --release --features full

# For debugging issues
$env:RUST_LOG = "debug"
$env:RUST_BACKTRACE = "1"
cargo build --features full
```

#### GPU Acceleration (NVIDIA)
```powershell
# Install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Set CUDA environment
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$env:LLAMA_CUDA = "1"
$env:CUDA_DOCKER_ARCH = "compute_75,compute_80,compute_86"

# Build with CUDA support
cargo build --release --features "full,cuda"
```

### Windows-Specific Configuration

#### Audio System Configuration
```toml
# Add to Cargo.toml for Windows-specific audio
[target.'cfg(windows)'.dependencies]
windows = { version = "0.48", features = [
    "Win32_Media_Audio",
    "Win32_Media_Audio_DirectSound",
    "Win32_System_Com",
    "Win32_Foundation"
]}
```

#### Firewall Configuration
```powershell
# Allow Mullama through Windows Firewall (for WebSocket features)
New-NetFirewallRule -DisplayName "Mullama WebSocket" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
New-NetFirewallRule -DisplayName "Mullama API" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow
```

---

## 🐧 Linux Setup

### Ubuntu/Debian Dependencies

#### System Dependencies
```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    unzip

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default stable
```

#### Audio Dependencies
```bash
# ALSA development libraries
sudo apt install -y \
    libasound2-dev \
    alsa-utils \
    alsa-oss \
    alsa-tools

# PulseAudio support
sudo apt install -y \
    libpulse-dev \
    pulseaudio-utils

# JACK Audio Connection Kit (for professional audio)
sudo apt install -y \
    libjack-jackd2-dev \
    jackd2

# Additional audio processing libraries
sudo apt install -y \
    libsamplerate0-dev \
    libsndfile1-dev \
    libflac-dev \
    libvorbis-dev \
    libopus-dev \
    libmp3lame-dev
```

#### Graphics and Image Processing
```bash
# Image processing libraries
sudo apt install -y \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libwebp-dev \
    libgif-dev

# FFmpeg for advanced media processing
sudo apt install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev

# Additional graphics libraries
sudo apt install -y \
    libsdl2-dev \
    libgtk-3-dev
```

#### System Libraries
```bash
# Threading and synchronization
sudo apt install -y \
    libnuma-dev \
    libssl-dev \
    libffi-dev

# Database and networking (for web features)
sudo apt install -y \
    libsqlite3-dev \
    libpq-dev \
    libmysqlclient-dev

# Development tools
sudo apt install -y \
    valgrind \
    gdb \
    strace \
    htop \
    perf-tools-unstable
```

### CentOS/RHEL/Fedora Dependencies

#### Package Installation
```bash
# For CentOS/RHEL (with EPEL)
sudo yum groupinstall -y "Development Tools"
sudo yum install -y epel-release
sudo yum install -y \
    cmake3 \
    pkg-config \
    git \
    curl \
    wget \
    alsa-lib-devel \
    pulseaudio-libs-devel \
    jack-audio-connection-kit-devel

# For Fedora
sudo dnf groupinstall -y "Development Tools" "Development Libraries"
sudo dnf install -y \
    cmake \
    pkg-config \
    git \
    curl \
    alsa-lib-devel \
    pulseaudio-libs-devel \
    jack-audio-connection-kit-devel \
    libsamplerate-devel \
    libsndfile-devel \
    flac-devel \
    libvorbis-devel \
    opus-devel \
    lame-devel \
    ffmpeg-devel
```

### Arch Linux Dependencies
```bash
# Install base development tools
sudo pacman -S base-devel cmake pkg-config git curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Audio dependencies
sudo pacman -S \
    alsa-lib \
    pulseaudio \
    jack2 \
    libsamplerate \
    libsndfile \
    flac \
    libvorbis \
    opus \
    lame \
    ffmpeg

# Image processing
sudo pacman -S \
    libpng \
    libjpeg-turbo \
    libtiff \
    libwebp \
    giflib

# Additional libraries
sudo pacman -S \
    openssl \
    sqlite \
    postgresql-libs
```

### Linux Build Configuration

#### Environment Setup
```bash
# Add to ~/.bashrc or ~/.zshrc
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CMAKE_BUILD_TYPE="Release"
export PKG_CONFIG_PATH="/usr/lib/pkgconfig:/usr/local/lib/pkgconfig"

# For development
export RUST_LOG="info"
export RUST_BACKTRACE="1"

# Reload environment
source ~/.bashrc
```

#### Audio System Configuration
```bash
# Configure ALSA
echo "pcm.!default pulse" | sudo tee -a /etc/asound.conf
echo "ctl.!default pulse" | sudo tee -a /etc/asound.conf

# Start audio services
sudo systemctl enable --now pulseaudio
sudo systemctl enable --now alsa-state

# Test audio
aplay /usr/share/sounds/alsa/Front_Left.wav
```

#### Build Mullama
```bash
# Clone and build
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama

# Install additional Rust components
rustup component add clippy rustfmt

# Build with all features
cargo build --release --features full

# Run tests
cargo test --release --features full

# Install for system-wide use
cargo install --path . --features full
```

---

## 🍎 macOS Setup

### Prerequisites Installation

#### Install Xcode Command Line Tools
```bash
# Install Xcode command line tools
xcode-select --install

# Accept license
sudo xcodebuild -license accept

# Install Xcode (optional, for full IDE)
# Download from Mac App Store
```

#### Install Homebrew
```bash
# Install Homebrew package manager
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### Install Core Dependencies
```bash
# Install build essentials
brew install \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    llvm \
    ninja

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default stable
```

#### Audio Dependencies
```bash
# Core Audio framework is included in macOS
# Install additional audio processing libraries
brew install \
    portaudio \
    libsamplerate \
    libsndfile \
    flac \
    libvorbis \
    opus \
    lame \
    mad \
    jack

# FFmpeg for media processing
brew install ffmpeg

# Audio development tools
brew install \
    aubio \
    fftw \
    vamp-plugin-sdk
```

#### Image Processing Dependencies
```bash
# Image processing libraries
brew install \
    libpng \
    jpeg \
    libtiff \
    webp \
    giflib \
    imagemagick

# Graphics libraries
brew install \
    sdl2 \
    cairo \
    pango
```

#### System Libraries
```bash
# Development libraries
brew install \
    openssl@3 \
    libffi \
    sqlite \
    postgresql@14 \
    mysql

# Development tools
brew install \
    gdb \
    lldb \
    valgrind \
    htop
```

### macOS-Specific Configuration

#### Environment Setup
```bash
# Add to ~/.zshrc or ~/.bash_profile
export PATH="/opt/homebrew/bin:$PATH"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"

# For OpenSSL
export OPENSSL_DIR="/opt/homebrew/opt/openssl@3"
export OPENSSL_LIB_DIR="/opt/homebrew/opt/openssl@3/lib"
export OPENSSL_INCLUDE_DIR="/opt/homebrew/opt/openssl@3/include"

# Rust optimization flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Reload environment
source ~/.zshrc
```

#### macOS Metal Configuration (Apple Silicon)
```bash
# For Apple Silicon Macs with Metal acceleration
export LLAMA_METAL=1
export GGML_METAL=1

# Metal performance shaders
export GGML_METAL_NDEBUG=1

# For Intel Macs with OpenCL
export LLAMA_CLBLAST=1
```

#### Audio System Configuration
```bash
# Configure Core Audio
# Core Audio is automatically configured on macOS

# For Jack Audio (professional audio)
# Start JackPilot or use command line
# brew services start jack

# Test audio
say "Audio test successful"
```

### Building on macOS

#### Apple Silicon (M1/M2/M3) Build
```bash
# Clone repository
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama

# Apple Silicon optimized build
export RUSTFLAGS="-C target-cpu=apple-m1 -C opt-level=3"
export LLAMA_METAL=1

# Build with Metal acceleration
cargo build --release --features full

# For debugging
export RUST_LOG="debug"
cargo build --features full
```

#### Intel Mac Build
```bash
# Intel Mac optimized build
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# With OpenCL acceleration (if available)
export LLAMA_CLBLAST=1
brew install clblast

# Build
cargo build --release --features full
```

#### macOS-Specific Cargo Configuration
```toml
# Add to .cargo/config.toml
[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

# For Apple Silicon with Metal
[target.aarch64-apple-darwin.env]
LLAMA_METAL = "1"
GGML_METAL = "1"
```

---

## 🚀 GPU Acceleration

### NVIDIA CUDA (Windows/Linux)

#### Windows CUDA Setup
```powershell
# Download and install CUDA Toolkit 12.0+
# https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# https://developer.nvidia.com/cudnn

# Set environment variables
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$env:CUDNN_PATH = "C:\tools\cuda"
$env:LLAMA_CUDA = "1"

# Add to PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDNN_PATH\bin;$env:PATH"

# Build with CUDA
cargo build --release --features "full,cuda"
```

#### Linux CUDA Setup
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-525

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-0

# Install cuDNN
sudo apt install libcudnn8 libcudnn8-dev

# Set environment
export CUDA_PATH="/usr/local/cuda"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LLAMA_CUDA=1

# Build with CUDA
cargo build --release --features "full,cuda"
```

### AMD ROCm (Linux)

#### ROCm Installation
```bash
# Add ROCm repository (Ubuntu)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dkms rocm-dev rocm-libs

# Add user to render group
sudo usermod -a -G render,video $USER

# Set environment
export ROCM_PATH="/opt/rocm"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LLAMA_HIPBLAS=1

# Reboot and build
cargo build --release --features "full,rocm"
```

### Apple Metal (macOS)

#### Metal Configuration
```bash
# Metal is included with macOS
# Enable Metal acceleration
export LLAMA_METAL=1
export GGML_METAL=1

# For maximum performance
export GGML_METAL_NDEBUG=1

# Build with Metal
cargo build --release --features "full,metal"
```

### OpenCL (Cross-platform)

#### OpenCL Setup
```bash
# Linux - Install OpenCL
sudo apt install opencl-headers ocl-icd-opencl-dev clinfo

# macOS - Install OpenCL support
brew install clblast

# Windows - OpenCL usually included with GPU drivers

# Set environment
export LLAMA_CLBLAST=1

# Build with OpenCL
cargo build --release --features "full,opencl"
```

---

## ⚙️ Platform-Specific Optimizations

### Windows Optimizations

#### Performance Tuning
```powershell
# Windows-specific compiler optimizations
$env:RUSTFLAGS = "-C target-cpu=native -C opt-level=3 -C lto=fat -C embed-bitcode=yes"

# Enable Windows-specific features
$env:CARGO_FEATURES = "full,windows-optimization"

# Memory management
$env:MALLOC_CONF = "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
```

#### Windows Service Configuration
```toml
# Add to Cargo.toml for Windows service support
[target.'cfg(windows)'.dependencies]
windows-service = "0.6"
winsw = "3.0"

[target.'cfg(windows)'.features]
windows-service = ["windows-service"]
```

### Linux Optimizations

#### Performance Tuning
```bash
# CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Huge pages
echo 'vm.nr_hugepages = 1024' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Process limits
echo '*               soft    nofile          65536' | sudo tee -a /etc/security/limits.conf
echo '*               hard    nofile          65536' | sudo tee -a /etc/security/limits.conf

# Compiler optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat"
export CFLAGS="-O3 -march=native -mtune=native"
export CXXFLAGS="-O3 -march=native -mtune=native"
```

#### Systemd Service
```ini
# /etc/systemd/system/mullama.service
[Unit]
Description=Mullama AI Service
After=network.target

[Service]
Type=simple
User=mullama
WorkingDirectory=/opt/mullama
ExecStart=/opt/mullama/bin/mullama-server
Restart=always
RestartSec=5
Environment=RUST_LOG=info
Environment=LLAMA_CUDA=1

[Install]
WantedBy=multi-user.target
```

### macOS Optimizations

#### Apple Silicon Optimization
```bash
# Apple Silicon specific optimizations
export RUSTFLAGS="-C target-cpu=apple-m1 -C opt-level=3 -C lto=fat"

# Memory management for Apple Silicon
export MALLOC_CONF="background_thread:true,metadata_thp:auto"

# Metal performance shaders optimization
export GGML_METAL_NDEBUG=1
export MTL_HUD_ENABLED=0
```

#### macOS Service (LaunchAgent)
```xml
<!-- ~/Library/LaunchAgents/com.mullama.service.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mullama.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/mullama-server</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>LLAMA_METAL</key>
        <string>1</string>
        <key>RUST_LOG</key>
        <string>info</string>
    </dict>
</dict>
</plist>
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Windows Issues

**Issue: MSVC compiler not found**
```powershell
# Solution: Install Visual Studio Build Tools
# Ensure C++ build tools workload is installed
# Set environment variable
$env:CC = "cl.exe"
$env:CXX = "cl.exe"
```

**Issue: Link errors with CUDA**
```powershell
# Solution: Check CUDA installation
nvcc --version
# Ensure CUDA_PATH is set correctly
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

**Issue: Audio features not working**
```powershell
# Solution: Install Visual C++ Redistributables
choco install vcredist-all -y
# Check Windows audio service
Get-Service -Name "AudioEndpointBuilder"
```

#### Linux Issues

**Issue: Missing ALSA development headers**
```bash
# Solution: Install ALSA development packages
sudo apt install libasound2-dev alsa-utils

# Test ALSA
aplay -l
```

**Issue: Permission denied for audio devices**
```bash
# Solution: Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again

# Check audio permissions
ls -la /dev/snd/
```

**Issue: CMake version too old**
```bash
# Solution: Install newer CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt update
sudo apt install cmake
```

#### macOS Issues

**Issue: Command line tools not found**
```bash
# Solution: Reinstall Xcode command line tools
sudo xcode-select --reset
xcode-select --install
```

**Issue: Homebrew packages not found**
```bash
# Solution: Update Homebrew and fix paths
brew update
brew doctor

# Fix PATH in shell profile
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Issue: Metal compilation errors**
```bash
# Solution: Check Metal support
system_profiler SPDisplaysDataType | grep Metal

# Update to latest macOS
softwareupdate -l
```

### Performance Issues

#### Memory Usage Too High
```bash
# Solution: Reduce model size or context length
export LLAMA_N_CTX=2048  # Reduce context size
export LLAMA_N_BATCH=256 # Reduce batch size

# Use quantized models
# Download Q4_0 or Q4_1 quantized versions
```

#### Slow Compilation
```bash
# Solution: Use faster linker
export RUSTFLAGS="-C link-arg=-fuse-ld=lld"

# Increase parallel jobs
export CARGO_BUILD_JOBS=8

# Use incremental compilation for development
export CARGO_INCREMENTAL=1
```

#### Poor GPU Performance
```bash
# Check GPU usage
nvidia-smi  # NVIDIA
rocm-smi    # AMD
system_profiler SPDisplaysDataType  # macOS

# Ensure GPU acceleration is enabled
export LLAMA_CUDA=1      # NVIDIA
export LLAMA_HIPBLAS=1   # AMD
export LLAMA_METAL=1     # Apple
```

### Build Validation

#### Test Build Success
```bash
# Basic functionality test
cargo run --example simple --features async

# Audio functionality test
cargo run --example streaming_audio_demo --features streaming-audio

# GPU acceleration test
cargo run --example gpu_benchmark --features full

# Web features test
cargo run --example web_service --features web
```

#### Performance Benchmarks
```bash
# Run benchmarks
cargo bench --features full

# Memory usage test
valgrind --tool=massif cargo run --example memory_test --features full

# CPU profiling
perf record cargo run --example cpu_benchmark --features full
perf report
```

---

## 📦 Distribution Packages

### Windows Installer
```powershell
# Create Windows installer with dependencies
# Use NSIS or WiX Toolset
# Include Visual C++ Redistributables
# Include CUDA runtime (optional)
```

### Linux Packages
```bash
# Debian/Ubuntu package
cargo deb --features full

# RPM package (CentOS/RHEL/Fedora)
cargo rpm --features full

# Arch Linux package
makepkg -si
```

### macOS Bundle
```bash
# Create macOS application bundle
cargo bundle --features full

# Sign and notarize for distribution
codesign --sign "Developer ID Application" target/release/bundle/osx/Mullama.app
xcrun notarytool submit Mullama.app.zip --keychain-profile "notarytool-password"
```

This comprehensive platform setup guide ensures developers can successfully build and run Mullama with all features on any supported platform! 🚀