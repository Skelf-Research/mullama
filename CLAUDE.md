# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Building and Testing
```bash
# Basic build check (no features)
cargo check --no-default-features

# Build with specific features
cargo check --features "async,streaming"
cargo check --features "multimodal,streaming-audio"
cargo check --features "web,websockets"

# Full build with all features
cargo build --release --features full

# Run examples
cargo run --example simple --features async
cargo run --example streaming_audio_demo --features "streaming-audio,multimodal"
cargo run --example web_service --features "web,websockets"

# Run tests
cargo test
cargo test --features full
```

### Platform-Specific Setup Requirements
**Linux (Ubuntu/Debian):**
```bash
sudo apt install -y libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev
sudo apt install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

**Required Submodules:**
```bash
git submodule update --init --recursive
```

### GPU Acceleration
Set environment variables before building:
```bash
export LLAMA_CUDA=1      # NVIDIA CUDA
export LLAMA_METAL=1     # Apple Silicon
export LLAMA_HIPBLAS=1   # AMD ROCm
export LLAMA_CLBLAST=1   # OpenCL
```

## Architecture Overview

### Core Library Structure
Mullama is a comprehensive Rust wrapper for llama.cpp with 14,000+ lines of integration code. The architecture is built around several key layers:

**Foundation Layer (sys.rs):**
- Low-level FFI bindings to llama.cpp C++ library
- Platform-specific build configuration in `build.rs`
- Memory-safe wrappers around unsafe C operations

**Core API Layer:**
- `Model`: GGUF model loading and management with builder patterns
- `Context`: Inference context with configurable parameters
- `Sampling`: Advanced sampling strategies (top-k, top-p, temperature, penalties)
- `Batch`: Efficient multi-sequence token processing
- `Embedding`: Text embedding generation and manipulation

**Integration Layer (Feature-Gated):**
- `async_support`: Tokio integration with non-blocking operations
- `streaming`: Real-time token generation with backpressure handling
- `multimodal`: Text, image, and audio processing pipeline
- `streaming_audio`: Real-time audio capture with voice activity detection
- `format_conversion`: Audio/image format conversion (WAV, MP3, FLAC, JPEG, PNG, WebP)
- `web`: Axum framework integration for REST APIs
- `websockets`: Real-time bidirectional communication
- `parallel`: Rayon work-stealing parallelism for batch operations

### Key Design Patterns

**Builder Pattern Extensively Used:**
- `ModelBuilder` for complex model configuration
- `AudioStreamConfig` for audio processing setup
- `StreamConfig` for streaming parameters
- All builders follow the same `.build()` completion pattern

**Feature Flag Architecture:**
- Core functionality available without features
- Optional integrations gated behind Cargo features
- `full` feature enables all capabilities
- Platform-specific features auto-detected in build.rs

**Error Handling Strategy:**
- Central `MullamaError` enum covering all error types
- Result-based APIs throughout with meaningful error messages
- No panics in public API surface

**Memory Management:**
- RAII patterns with automatic resource cleanup
- Arc/Rc for shared ownership where needed
- Zero unsafe operations exposed in public API

### Critical Integration Points

**Async/Sync Duality:**
- Core API is synchronous for simplicity
- Async wrappers in `async_support` module when `async` feature enabled
- Tokio runtime management in `tokio_integration` module

**Multimodal Pipeline:**
- `MultimodalProcessor` coordinates text, audio, and image processing
- Format conversion happens automatically between components
- Streaming capabilities preserve real-time performance

**Audio Processing Chain:**
- `StreamingAudioProcessor` handles real-time capture
- Voice activity detection and noise reduction built-in
- Ring buffer architecture for low-latency processing

### Configuration System
- Serde-based configuration with JSON/YAML support
- Environment variable overrides for build-time options
- Platform-specific defaults auto-detected
- Extensive validation with helpful error messages

## Integration Feature Dependencies

When working with features, understand these dependency chains:
- `streaming-audio` requires `multimodal`
- `format-conversion` requires `multimodal`
- `web` and `websockets` require `async`
- `full` enables all features and their dependencies

Platform-specific features are automatically enabled:
- `windows-optimization` on Windows
- `macos-optimization` on macOS (with Metal support on Apple Silicon)

## Important Implementation Notes

**Build System (build.rs):**
- Comprehensive platform detection and configuration
- GPU acceleration detection and linking
- Audio library detection using pkg-config
- Provides detailed error messages for missing dependencies

**Memory Safety:**
- All llama.cpp operations wrapped in safe Rust APIs
- Automatic cleanup of C++ resources using Drop implementations
- No manual memory management required by library users

**Performance Considerations:**
- Batch processing preferred for high throughput
- Streaming interfaces for real-time applications
- Parallel processing with work-stealing for CPU-bound tasks
- GPU acceleration when available

The codebase emphasizes comprehensive integration capabilities rather than just basic LLM bindings, with production-ready features for building sophisticated AI applications.