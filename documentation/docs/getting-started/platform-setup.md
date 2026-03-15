---
title: Platform Setup
description: Install platform-specific dependencies for Mullama on Linux, macOS, and Windows. Covers audio, image, and FFmpeg libraries required for multimodal and format conversion features.
---

# Platform Setup

Mullama builds native code via llama.cpp and supports optional audio, image, and video processing. This page covers the platform-specific dependencies you need to install.

!!! info "Install Only What You Need"

    Not all dependencies are required. The sections below are organized by feature -- only install the packages for the features you plan to use. If you only need text generation, the essential build tools are sufficient.

---

## Linux

### Ubuntu / Debian

#### Essential Build Tools

Required for all builds:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget
```

#### Audio Libraries

!!! note "Required for features: `multimodal`, `streaming-audio`"

```bash
sudo apt install -y \
    libasound2-dev \
    libpulse-dev \
    libflac-dev \
    libvorbis-dev \
    libopus-dev
```

#### Image Libraries

!!! note "Required for feature: `multimodal`"

```bash
sudo apt install -y \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libwebp-dev
```

#### FFmpeg Libraries

!!! note "Required for feature: `format-conversion`"

```bash
sudo apt install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev
```

#### All Dependencies (One Command)

Install everything at once for the `full` feature set:

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config git curl wget \
    libasound2-dev libpulse-dev \
    libflac-dev libvorbis-dev libopus-dev \
    libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    libssl-dev
```

---

### Fedora / RHEL

#### Essential Build Tools

```bash
sudo dnf groupinstall -y "Development Tools" "Development Libraries"
sudo dnf install -y \
    cmake \
    pkg-config \
    git \
    curl \
    wget
```

#### Audio Libraries

```bash
sudo dnf install -y \
    alsa-lib-devel \
    pulseaudio-libs-devel \
    flac-devel \
    libvorbis-devel \
    opus-devel
```

#### Image Libraries

```bash
sudo dnf install -y \
    libpng-devel \
    libjpeg-turbo-devel \
    libtiff-devel \
    libwebp-devel
```

#### FFmpeg Libraries

```bash
# Enable RPM Fusion for FFmpeg packages
sudo dnf install -y \
    https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y \
    ffmpeg-devel
```

#### All Dependencies (One Command)

```bash
sudo dnf install -y \
    cmake pkg-config git curl wget \
    alsa-lib-devel pulseaudio-libs-devel \
    flac-devel libvorbis-devel opus-devel \
    libpng-devel libjpeg-turbo-devel libtiff-devel libwebp-devel \
    ffmpeg-devel openssl-devel
```

---

### Arch Linux

#### Essential Build Tools

```bash
sudo pacman -S --needed \
    base-devel \
    cmake \
    pkgconf \
    git \
    curl \
    wget
```

#### Audio Libraries

```bash
sudo pacman -S --needed \
    alsa-lib \
    libpulse \
    flac \
    libvorbis \
    opus
```

#### Image Libraries

```bash
sudo pacman -S --needed \
    libpng \
    libjpeg-turbo \
    libtiff \
    libwebp
```

#### FFmpeg Libraries

```bash
sudo pacman -S --needed ffmpeg
```

#### All Dependencies (One Command)

```bash
sudo pacman -S --needed \
    base-devel cmake pkgconf git curl wget \
    alsa-lib libpulse flac libvorbis opus \
    libpng libjpeg-turbo libtiff libwebp \
    ffmpeg openssl
```

---

## macOS

### Xcode Command Line Tools

Required for all builds on macOS:

```bash
xcode-select --install
```

!!! info "Accept the License"

    If prompted, accept the Xcode license agreement:

    ```bash
    sudo xcodebuild -license accept
    ```

### Homebrew

Install the [Homebrew](https://brew.sh) package manager if you do not already have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add Homebrew to your PATH (Apple Silicon):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Core Build Dependencies

```bash
brew install cmake pkg-config
```

### Audio Libraries

!!! note "Required for features: `multimodal`, `streaming-audio`"

macOS includes Core Audio natively. Install additional codec libraries:

```bash
brew install \
    portaudio \
    libsndfile \
    flac \
    libvorbis \
    opus
```

### Image Libraries

!!! note "Required for feature: `multimodal`"

```bash
brew install \
    libpng \
    jpeg \
    libtiff \
    webp
```

### FFmpeg

!!! note "Required for feature: `format-conversion`"

```bash
brew install ffmpeg
```

### All Dependencies (One Command)

```bash
brew install \
    cmake pkg-config \
    portaudio libsndfile flac libvorbis opus \
    libpng jpeg libtiff webp \
    ffmpeg
```

### Environment Configuration

Add to your `~/.zshrc` (or `~/.bash_profile`):

```bash
export PATH="/opt/homebrew/bin:$PATH"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
```

Reload your shell:

```bash
source ~/.zshrc
```

!!! tip "Metal GPU Acceleration"

    On Apple Silicon (M1/M2/M3/M4), Metal GPU acceleration is available automatically. No additional drivers or packages are needed. See [GPU Acceleration](gpu.md) for details.

---

## Windows

### Visual Studio Build Tools

!!! warning "Required"

    A C++17 compiler is required to build the native llama.cpp code.

=== "Visual Studio Community (Recommended)"

    Download [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select:

    - **Desktop development with C++** workload
        - MSVC v143 -- VS 2022 C++ x64/x86 build tools
        - Windows 11 SDK (latest version)
        - CMake tools for Visual Studio

=== "Build Tools Only (Smaller)"

    Download [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) and install the **C++ build tools** workload.

### CMake

CMake is included with Visual Studio. If you need a standalone installation:

```powershell
# Using Chocolatey
choco install cmake -y

# Or download from https://cmake.org/download/
```

### Core Dependencies

```powershell
choco install git cmake ninja -y
refreshenv
```

### Audio Dependencies

Windows audio APIs (WASAPI, DirectSound) are included in the Windows SDK. No additional audio packages are needed.

!!! info "Windows SDK"

    The Windows SDK is automatically installed with the Visual Studio C++ workload. It provides all audio interfaces needed by Mullama.

### Image Dependencies

Image processing libraries are bundled by the Rust `image` crate on Windows. No additional system packages are needed.

### FFmpeg (Optional)

!!! note "Required for feature: `format-conversion`"

```powershell
choco install ffmpeg -y
```

Or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system PATH.

### CUDA Toolkit (Optional)

!!! note "Required for NVIDIA GPU acceleration"

    Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) if you have an NVIDIA GPU. See [GPU Acceleration](gpu.md) for full instructions.

### Environment Variables

Set permanently via PowerShell (administrator):

```powershell
# Ensure Ninja is used for CMake (faster builds)
[Environment]::SetEnvironmentVariable("CMAKE_GENERATOR", "Ninja", "Machine")

# For CUDA GPU acceleration (if applicable)
[Environment]::SetEnvironmentVariable("LLAMA_CUDA", "1", "Machine")
```

---

## Required Submodules

Mullama includes llama.cpp as a git submodule. This is required for all source builds.

```bash
# If you cloned with --recurse-submodules, you are already set.
# Otherwise, initialize submodules:
git submodule update --init --recursive
```

!!! warning "Build Will Fail Without Submodules"

    If you see errors about missing `llama.cpp` source files, header files not found, or `ggml.h` not found, it means submodules are not initialized. Run the command above to fix this.

To update submodules to the latest upstream version:

```bash
git submodule update --remote --merge
```

---

## Dependency Summary Table

| Feature | Linux (apt) | macOS (brew) | Windows |
|---------|-------------|--------------|---------|
| Core (no features) | `build-essential cmake pkg-config` | `cmake pkg-config` | VS Build Tools, CMake |
| `multimodal` (audio) | `libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev` | `portaudio libsndfile flac libvorbis opus` | Windows SDK (included) |
| `multimodal` (image) | `libpng-dev libjpeg-dev libtiff-dev libwebp-dev` | `libpng jpeg libtiff webp` | None (bundled) |
| `streaming-audio` | Same as multimodal audio | Same as multimodal audio | Same as multimodal audio |
| `format-conversion` | `ffmpeg libavcodec-dev libavformat-dev libavutil-dev` | `ffmpeg` | `ffmpeg` via Chocolatey |
| `web`, `websockets` | `libssl-dev` | Included via macOS | Included via Schannel |
| GPU (CUDA) | CUDA Toolkit + nvidia-driver | N/A | CUDA Toolkit |
| GPU (Metal) | N/A | Automatic (Apple Silicon) | N/A |
| GPU (ROCm) | ROCm packages | N/A | N/A |

---

## Install Rust

If you do not have Rust installed, use rustup:

=== "Linux / macOS"

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env
    rustup default stable
    ```

=== "Windows"

    ```powershell
    # Using Chocolatey
    choco install rustup.install -y
    refreshenv
    rustup default stable-x86_64-pc-windows-msvc
    ```

    Or download from [rustup.rs](https://rustup.rs).

Verify the installation:

```bash
rustc --version   # Should be 1.75.0 or later
cargo --version
cmake --version   # Should be 3.12 or later
```

---

## Common Issues and Solutions

??? question "pkg-config cannot find a library"

    Ensure the development packages are installed (the `-dev` or `-devel` suffix packages, not just the runtime libraries). On macOS, verify that `PKG_CONFIG_PATH` includes Homebrew:

    ```bash
    export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
    pkg-config --list-all | grep pulse  # Should show libpulse
    ```

??? question "CMake version too old (Linux)"

    Install a newer CMake from Kitware's repository:

    ```bash
    # Ubuntu/Debian
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
        gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'
    sudo apt update && sudo apt install -y cmake
    cmake --version
    ```

??? question "Permission denied for audio devices (Linux)"

    Add your user to the `audio` group:

    ```bash
    sudo usermod -a -G audio $USER
    ```

    Log out and back in for the change to take effect. Verify with:

    ```bash
    groups | grep audio
    ```

??? question "Homebrew packages not found (macOS)"

    Run `brew doctor` and ensure your shell profile exports the correct paths:

    ```bash
    eval "$(/opt/homebrew/bin/brew shellenv)"
    brew doctor
    ```

??? question "MSVC compiler not found (Windows)"

    Ensure the C++ build tools workload is installed in Visual Studio. Open a **Developer Command Prompt** or set the compiler explicitly:

    ```powershell
    $env:CC = "cl.exe"
    $env:CXX = "cl.exe"
    ```

??? question "'alsa/asoundlib.h' not found"

    Install the ALSA development package:

    ```bash
    # Ubuntu/Debian
    sudo apt install -y libasound2-dev

    # Fedora
    sudo dnf install -y alsa-lib-devel

    # Arch
    sudo pacman -S alsa-lib
    ```

??? question "FFmpeg headers not found"

    Install the FFmpeg development packages (not just the `ffmpeg` binary):

    ```bash
    # Ubuntu/Debian
    sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev

    # Fedora (requires RPM Fusion)
    sudo dnf install -y ffmpeg-devel

    # macOS
    brew install ffmpeg
    ```

??? question "OpenSSL not found (Linux)"

    Install the OpenSSL development package:

    ```bash
    # Ubuntu/Debian
    sudo apt install -y libssl-dev

    # Fedora/RHEL
    sudo dnf install -y openssl-devel
    ```

---

!!! success "Next Steps"

    - [GPU Acceleration](gpu.md) -- Configure CUDA, Metal, ROCm, or OpenCL
    - [Installation](installation.md) -- Feature flags and build options
    - [Your First Project](first-project.md) -- Build a chatbot from scratch
