---
title: Installation
description: Install Mullama as a library for Node.js, Python, Rust, Go, PHP, or C/C++, or as a CLI daemon. Includes feature flags, building from source, Docker setup, and verification steps.
---

# Installation

Install Mullama for your language or as a standalone CLI/daemon. Pre-built binaries are available for all major platforms.

---

## Install the Package

=== "Node.js"

    Install via npm (prebuilt native binaries included):

    ```bash
    npm install mullama
    ```

    Or with yarn:

    ```bash
    yarn add mullama
    ```

    Or with pnpm:

    ```bash
    pnpm add mullama
    ```

    !!! note "Native Addon"
        The Node.js package includes prebuilt binaries for Linux (x86_64, aarch64), macOS (Apple Silicon, Intel), and Windows (x86_64). If a prebuilt binary is not available for your platform, it will compile from source during installation -- ensure you have a C++ compiler and CMake installed.

=== "Python"

    Install via pip:

    ```bash
    pip install mullama
    ```

    Or with a virtual environment (recommended):

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    pip install mullama
    ```

    Or with conda:

    ```bash
    conda install -c conda-forge mullama
    ```

    !!! tip "GPU Support"
        For GPU-accelerated inference, install the GPU variant:

        ```bash
        pip install mullama[cuda]    # NVIDIA GPU
        pip install mullama[metal]   # Apple Silicon
        ```

=== "Rust"

    Add to your `Cargo.toml`:

    ```bash
    cargo add mullama
    ```

    Or manually specify features:

    ```toml title="Cargo.toml"
    [dependencies]
    mullama = { version = "0.1.1", features = ["async", "streaming"] }
    ```

=== "Go"

    ```bash
    go get github.com/skelf-research/mullama-go
    ```

    !!! note "CGo Required"
        The Go binding uses cgo to interface with the native library. Ensure you have a C compiler available and `CGO_ENABLED=1` (the default).

=== "PHP"

    ```bash
    composer require skelf-research/mullama
    ```

    !!! note "FFI Extension"
        Requires PHP 7.4+ with the FFI extension enabled. Add `extension=ffi` to your `php.ini` if not already enabled.

=== "C/C++ (FFI)"

    Build the shared library from source:

    ```bash
    git clone --recurse-submodules https://github.com/skelf-research/mullama.git
    cd mullama/bindings/ffi
    cargo build --release
    ```

    The compiled library will be located at:

    | Platform | Path |
    |----------|------|
    | Linux | `target/release/libmullama_ffi.so` |
    | macOS | `target/release/libmullama_ffi.dylib` |
    | Windows | `target/release/mullama_ffi.dll` |

    Copy the header file and library to your project:

    ```bash
    cp bindings/ffi/include/mullama.h /your/project/include/
    cp target/release/libmullama_ffi.so /your/project/lib/
    ```

=== "CLI / Daemon"

    Install the Mullama daemon and CLI:

    ```bash
    cargo install mullama --features daemon
    ```

    For the daemon with embedded Web UI:

    ```bash
    # Build the UI first
    cd ui && npm install && npm run build && cd ..

    # Install with embedded UI
    cargo install mullama --features "daemon,embedded-ui"
    ```

---

## Feature Flags (Rust)

Mullama uses Cargo feature flags to control which capabilities are compiled. This keeps binary size minimal and avoids unnecessary dependencies.

### Common Presets

```toml title="Cargo.toml"
[dependencies]
# Web API service with streaming responses
mullama = { version = "0.1.1", features = ["web", "websockets", "streaming"] }

# Multimodal AI (text + image + audio)
mullama = { version = "0.1.1", features = ["multimodal", "streaming-audio", "format-conversion"] }

# High-throughput batch processing
mullama = { version = "0.1.1", features = ["parallel", "async", "tokio-runtime"] }

# Semantic search / RAG pipeline
mullama = { version = "0.1.1", features = ["late-interaction", "parallel", "async"] }

# Everything enabled
mullama = { version = "0.1.1", features = ["full"] }
```

### Feature Reference

| Feature | Description | Key Dependencies |
|---------|-------------|-----------------|
| `async` | Tokio-based non-blocking operations | `tokio`, `futures` |
| `streaming` | Real-time token-by-token generation | `async`, `async-stream` |
| `web` | Axum REST API framework integration | `async`, `axum`, `tower` |
| `websockets` | Bidirectional real-time communication | `async`, `tokio-tungstenite` |
| `multimodal` | Text, image, and audio processing | `image`, `hound`, `symphonia` |
| `streaming-audio` | Real-time audio capture with VAD | `multimodal`, `cpal`, `ringbuf` |
| `format-conversion` | Audio/image format conversion | `multimodal`, `ffmpeg-next` |
| `parallel` | Rayon work-stealing parallelism | `rayon` |
| `late-interaction` | ColBERT-style semantic search | Core only |
| `tokio-runtime` | Advanced Tokio runtime management | `tokio`, `tokio-util` |
| `daemon` | Full CLI, daemon, TUI, and REST API | Multiple (see below) |
| `embedded-ui` | Embed Web UI in daemon binary | `include_dir`, `mime_guess` |
| `full` | All features enabled | All of the above |

### Feature Dependency Chain

```
full
 |-- async .............. tokio, futures, tokio-util
 |-- streaming .......... async + async-stream
 |-- web ................ async + axum, tower, tower-http
 |-- websockets ......... async + axum, tokio-tungstenite
 |-- multimodal ......... image, hound, symphonia, rubato, dasp
 |-- streaming-audio .... multimodal + cpal, ringbuf
 |-- format-conversion .. multimodal + ffmpeg-next
 |-- parallel ........... rayon
 |-- late-interaction ... (core only)
 |-- tokio-runtime ...... tokio, tokio-util
 |-- daemon ............. async, tokio-runtime, web + clap, ratatui, nng, ...
```

!!! tip "Minimal Builds"

    For the smallest possible binary, use no features:

    ```toml
    mullama = { version = "0.1.1", default-features = false }
    ```

    This gives you synchronous model loading, inference, and sampling with zero additional dependencies.

---

## Building from Source

For the latest features or custom builds, compile Mullama from source.

### Step 1: Clone the Repository

```bash
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama
```

!!! warning "Submodules are Required"

    Mullama includes llama.cpp as a git submodule. If you cloned without `--recurse-submodules`:

    ```bash
    git submodule update --init --recursive
    ```

### Step 2: Install Platform Dependencies

See [Platform Setup](platform-setup.md) for OS-specific packages. At minimum you need:

```bash
# Ubuntu/Debian
sudo apt install -y build-essential cmake pkg-config git

# macOS
xcode-select --install && brew install cmake pkg-config

# Windows: Install Visual Studio Build Tools + CMake
```

### Step 3: Build

```bash
# Debug build (faster compilation, slower runtime)
cargo build --features full

# Release build (slower compilation, optimized runtime)
cargo build --release --features full

# Build only the daemon
cargo build --release --features daemon
```

### Step 4: Run Tests

```bash
# Run all tests
cargo test --features full

# Run specific test modules
cargo test --features async test_async_generation
```

### From Git (Latest Unreleased)

Use the git dependency for bleeding-edge features:

```toml title="Cargo.toml"
[dependencies]
mullama = { git = "https://github.com/skelf-research/mullama.git", features = ["full"] }
```

---

## Verifying Installation

Confirm that Mullama is correctly installed for your language.

=== "Node.js"

    ```bash
    node -e "const m = require('mullama'); console.log('Mullama version:', m.version)"
    ```

    Expected output:

    ```
    Mullama version: 0.1.1
    ```

=== "Python"

    ```bash
    python -c "import mullama; print(f'Mullama version: {mullama.__version__}')"
    ```

    Expected output:

    ```
    Mullama version: 0.1.1
    ```

=== "Rust"

    ```bash
    # Verify the crate compiles
    cargo check --features "async,streaming"
    ```

    Or in code:

    ```rust
    fn main() {
        println!("Mullama version: {}", mullama::VERSION);
    }
    ```

=== "Go"

    ```bash
    go run -v . 2>&1 | head -5
    ```

    ```go
    package main

    import (
        "fmt"
        "github.com/skelf-research/mullama-go"
    )

    func main() {
        fmt.Println("Mullama version:", mullama.Version())
    }
    ```

=== "PHP"

    ```bash
    php -r "require 'vendor/autoload.php'; echo 'Mullama version: ' . Mullama\version() . PHP_EOL;"
    ```

=== "CLI / Daemon"

    ```bash
    mullama --version
    ```

    Expected output:

    ```
    mullama 0.1.1
    ```

    Test with a quick generation:

    ```bash
    mullama pull llama3.2:1b
    mullama run llama3.2:1b "Say hello in one word."
    ```

---

## Docker

Run Mullama in a container for reproducible deployments.

### CPU-Only

```dockerfile title="Dockerfile"
FROM rust:1.77-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config git \
    libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev \
    libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --recurse-submodules https://github.com/skelf-research/mullama.git .
RUN cargo build --release --features daemon

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libasound2 libpulse0 libflac12 libvorbis0a libopus0 \
    libpng16-16 libjpeg62-turbo libtiff6 libwebp7 \
    ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/mullama /usr/local/bin/mullama

EXPOSE 8080
VOLUME ["/models"]

ENTRYPOINT ["mullama"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--model-dir", "/models"]
```

### With NVIDIA GPU

```dockerfile title="Dockerfile.cuda"
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config git curl \
    libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev \
    libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
RUN git clone --recurse-submodules https://github.com/skelf-research/mullama.git .

ENV LLAMA_CUDA=1
RUN cargo build --release --features daemon

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    libasound2 libpulse0 libflac12 libvorbis0a libopus0 \
    libpng16-16 libjpeg62-turbo libtiff6 libwebp7 \
    ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/mullama /usr/local/bin/mullama

EXPOSE 8080
VOLUME ["/models"]

ENTRYPOINT ["mullama"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--model-dir", "/models"]
```

### Running the Container

```bash
# CPU-only
docker build -t mullama .
docker run -p 8080:8080 -v ./models:/models mullama

# With NVIDIA GPU
docker build -f Dockerfile.cuda -t mullama-cuda .
docker run --gpus all -p 8080:8080 -v ./models:/models mullama-cuda
```

### Docker Compose

```yaml title="docker-compose.yml"
services:
  mullama:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - MULLAMA_MODEL=llama3.2:1b
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Pre-built Binaries

Download pre-built daemon binaries from the [GitHub Releases](https://github.com/skelf-research/mullama/releases) page.

=== "Linux (x86_64)"

    ```bash
    curl -LO https://github.com/skelf-research/mullama/releases/latest/download/mullama-linux-x86_64.tar.gz
    tar xzf mullama-linux-x86_64.tar.gz
    sudo mv mullama /usr/local/bin/
    mullama --version
    ```

=== "Linux (ARM64)"

    ```bash
    curl -LO https://github.com/skelf-research/mullama/releases/latest/download/mullama-linux-aarch64.tar.gz
    tar xzf mullama-linux-aarch64.tar.gz
    sudo mv mullama /usr/local/bin/
    mullama --version
    ```

=== "macOS (Apple Silicon)"

    ```bash
    curl -LO https://github.com/skelf-research/mullama/releases/latest/download/mullama-macos-aarch64.tar.gz
    tar xzf mullama-macos-aarch64.tar.gz
    sudo mv mullama /usr/local/bin/
    mullama --version
    ```

=== "macOS (Intel)"

    ```bash
    curl -LO https://github.com/skelf-research/mullama/releases/latest/download/mullama-macos-x86_64.tar.gz
    tar xzf mullama-macos-x86_64.tar.gz
    sudo mv mullama /usr/local/bin/
    mullama --version
    ```

=== "Windows"

    ```powershell
    Invoke-WebRequest -Uri "https://github.com/skelf-research/mullama/releases/latest/download/mullama-windows-x86_64.zip" -OutFile mullama.zip
    Expand-Archive mullama.zip -DestinationPath .
    Move-Item mullama.exe C:\Windows\System32\
    mullama --version
    ```

---

## Troubleshooting

??? question "npm install fails with compilation errors"

    Ensure you have build tools installed:

    ```bash
    # Ubuntu/Debian
    sudo apt install -y build-essential cmake

    # macOS
    xcode-select --install

    # Windows: Install Visual Studio Build Tools
    ```

??? question "pip install fails on Linux"

    Install the Python development headers:

    ```bash
    sudo apt install -y python3-dev python3-pip
    ```

??? question "cargo build fails with 'llama.cpp not found'"

    Initialize git submodules:

    ```bash
    git submodule update --init --recursive
    ```

??? question "Linking errors on macOS"

    Ensure Homebrew paths are configured:

    ```bash
    export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LDFLAGS="-L/opt/homebrew/lib"
    export CPPFLAGS="-I/opt/homebrew/include"
    ```

!!! success "Next Steps"

    - [Platform Setup](platform-setup.md) -- Install OS-specific audio, image, and video dependencies
    - [GPU Acceleration](gpu.md) -- Configure GPU for faster inference
    - [Your First Project](first-project.md) -- Build a chatbot from scratch
