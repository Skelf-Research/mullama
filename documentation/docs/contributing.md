---
title: Contributing
description: Development setup, code standards, testing, and contribution workflow
---

# Contributing

Thank you for your interest in contributing to Mullama. This guide covers the development environment setup, code standards, testing practices, and the contribution workflow.

---

## Getting Started

### 1. Fork and Clone

Fork the repository on GitHub, then clone with submodules (required for llama.cpp):

```bash
git clone --recursive https://github.com/neul-labs/mullama.git
cd mullama
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Create a Feature Branch

Always work on a feature branch:

```bash
git checkout -b feature/my-improvement
```

---

## Development Environment Setup

### System Dependencies

=== "Linux (Ubuntu/Debian)"
    ```bash
    # Audio libraries
    sudo apt install -y libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev

    # Image libraries
    sudo apt install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev

    # Video/FFmpeg libraries
    sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

    # Build tools
    sudo apt install -y build-essential cmake pkg-config
    ```

=== "macOS"
    ```bash
    # Most dependencies are available via system frameworks
    brew install cmake pkg-config ffmpeg
    ```

=== "Windows"
    ```powershell
    # Install Visual Studio Build Tools with C++ workload
    # Install vcpkg for native dependencies
    vcpkg install ffmpeg libpng libjpeg-turbo
    ```

### Rust Toolchain

Ensure you have the latest stable Rust:

```bash
rustup update stable
rustup component add rustfmt clippy
```

### GPU Development (Optional)

For GPU-accelerated development, set the appropriate environment variable before building:

```bash
# NVIDIA CUDA
export LLAMA_CUDA=1

# Apple Metal (macOS)
export LLAMA_METAL=1

# AMD ROCm
export LLAMA_HIPBLAS=1

# OpenCL
export LLAMA_CLBLAST=1
```

---

## Building from Source

### Quick Build Check

```bash
# Verify core compiles (no features)
cargo check --no-default-features

# Verify with specific features
cargo check --features "async,streaming"
cargo check --features "multimodal,streaming-audio"
cargo check --features "web,websockets"
```

### Full Build

```bash
# Debug build with all features
cargo build --features full

# Release build
cargo build --release --features full
```

### Building the Daemon

```bash
# CLI daemon
cargo build --release --features daemon

# Daemon with embedded Web UI
cd ui && npm install && npm run build && cd ..
cargo build --release --features daemon,embedded-ui
```

---

## Running Tests

### Unit Tests

```bash
# Basic tests (no model required)
cargo test

# Full test suite with all features
cargo test --features full

# Tests for a specific module
cargo test --lib sampling
cargo test --lib embedding

# With output visible
cargo test -- --nocapture
```

### Integration Tests

Integration tests may require a model file. Set the `MULLAMA_TEST_MODEL` environment variable:

```bash
export MULLAMA_TEST_MODEL="path/to/test-model.gguf"
cargo test --features full -- --test-threads=1
```

### Verifying Examples Compile

```bash
# Ensure all examples compile
cargo build --examples --features full

# Run a specific example
cargo run --example simple --features async
cargo run --example embedding
```

---

## Code Style

### Formatting

Always run `cargo fmt` before committing:

```bash
cargo fmt
```

The project uses the default Rust formatting style.

### Linting

Run clippy with all features enabled and treat warnings as errors:

```bash
cargo clippy --features full -- -D warnings
```

Address all warnings before submitting a PR. Common clippy suggestions include:

- Use `if let` instead of single-arm `match`
- Prefer `&str` over `&String` in function parameters
- Use `Default::default()` where appropriate
- Avoid redundant closures

### Safety Rules

- **No `unsafe` in public API surface** -- All unsafe operations must be encapsulated within safe wrappers in internal modules
- **Document unsafe blocks** -- Every `unsafe` block must have a `// SAFETY:` comment explaining why it is sound
- **Minimize unsafe scope** -- Keep unsafe blocks as small as possible

```rust
// Good: Minimal unsafe scope with safety documentation
pub fn vocab_size(&self) -> i32 {
    // SAFETY: self.model is a valid pointer initialized in Model::load()
    // and remains valid for the lifetime of the Model struct
    unsafe { llama_n_vocab(self.model) }
}
```

### Error Handling

- Use `MullamaError` for all public functions returning `Result`
- Never panic (`unwrap()`, `expect()`, `panic!()`) in library code
- Include context in error messages about what went wrong

```rust
// Good: Descriptive error with context
pub fn load(path: &str) -> Result<Self, MullamaError> {
    if !std::path::Path::new(path).exists() {
        return Err(MullamaError::ModelLoadError(
            format!("Model file not found: {}", path)
        ));
    }
    // ...
}
```

### Documentation Standards

- All public types, functions, and modules must have doc comments (`///`)
- Include usage examples in doc comments where helpful
- Use `//!` for module-level documentation

```rust
/// Load a GGUF model from the filesystem.
///
/// # Arguments
///
/// * `path` - Path to the GGUF model file
///
/// # Errors
///
/// Returns `MullamaError::ModelLoadError` if the file cannot be opened
/// or is not a valid GGUF format.
///
/// # Example
///
/// ```no_run
/// use mullama::Model;
/// let model = Model::load("path/to/model.gguf")?;
/// ```
pub fn load(path: &str) -> Result<Self, MullamaError> {
    // ...
}
```

---

## Feature-Gated Code

When adding code that depends on optional features:

```rust
// Gate the module
#[cfg(feature = "streaming")]
pub mod streaming;

// Gate imports
#[cfg(feature = "async")]
use tokio::runtime::Runtime;

// Gate implementations
#[cfg(feature = "web")]
impl AppState {
    pub fn create_router(self) -> axum::Router {
        // ...
    }
}
```

Always provide helpful error messages when features are missing:

```rust
#[cfg(not(feature = "multimodal"))]
pub fn process_image(_path: &str) -> Result<(), MullamaError> {
    Err(MullamaError::FeatureDisabled(
        "The 'multimodal' feature is required for image processing. \
         Build with: cargo build --features multimodal".to_string()
    ))
}
```

---

## Pull Request Process

### Before Submitting

1. **Run the full check suite:**

```bash
cargo fmt
cargo clippy --features full -- -D warnings
cargo test --features full
cargo build --features full
```

2. **Update documentation** if your change affects the public API.

3. **Add tests** for new functionality.

4. **Keep commits focused** -- one logical change per commit.

### PR Description Template

Include in your PR description:

- **What** -- Brief summary of the changes
- **Why** -- Motivation and context
- **How** -- Approach taken (for non-trivial changes)
- **Testing** -- How you verified the changes work
- **Breaking changes** -- Any API changes that affect existing users

### Review Criteria

PRs are reviewed for:

- [ ] Code compiles with `--features full`
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Code is formatted with `cargo fmt`
- [ ] Public API changes are documented
- [ ] No unsafe code in public API
- [ ] Error handling uses `MullamaError` (no panics)
- [ ] New features are properly gated behind feature flags
- [ ] Commit messages are clear and descriptive

---

## Documentation Contributions

The documentation uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

### Setup

```bash
cd documentation

# Install dependencies
pip install mkdocs-material

# Local preview with live reload
mkdocs serve

# Build static site
mkdocs build
```

### Writing Guidelines

- Use clear, concise language
- Include code examples for all API descriptions
- Use admonitions (`!!! note`, `!!! warning`, `!!! tip`) for callouts
- Parameter tables use format: Name | Type | Default | Description
- Test all code examples mentally for correctness
- Cross-reference related pages with relative links

### File Organization

```
documentation/docs/
  index.md              # Landing page
  getting-started/      # Installation, quickstart
  guide/                # Conceptual guides
  api/                  # API reference (type signatures, parameters)
  examples/             # Complete runnable examples
  advanced/             # Advanced topics
  bindings/             # Language binding docs
  daemon/               # Daemon documentation
```

---

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Rust version** (`rustc --version`)
2. **Platform** (OS, architecture, GPU if relevant)
3. **Mullama version** (commit hash or release tag)
4. **Feature flags** used
5. **Minimal reproduction** -- smallest code that triggers the issue
6. **Expected vs. actual behavior**
7. **Error messages** -- full output including any backtraces

```bash
# Gather debug info
rustc --version
cargo --version
uname -a  # or systeminfo on Windows
git log -1 --format="%H"
```

### Feature Requests

For feature requests, describe:

1. **Use case** -- What you are trying to accomplish
2. **Current workaround** -- How you handle it today (if applicable)
3. **Proposed solution** -- Your suggested approach
4. **Alternatives considered** -- Other approaches you thought about

---

## Binding Development

Mullama supports language bindings via FFI, N-API, PyO3, and CGo.

### Adding a New Binding

1. Create a directory under `bindings/` (e.g., `bindings/ruby/`)
2. Implement the FFI wrapper using the C API from `bindings/ffi/`
3. Add type mappings for the target language
4. Write tests that exercise the core API through the binding
5. Add documentation in `documentation/docs/bindings/`

### Updating Existing Bindings

When the Rust API changes:

1. Update the C FFI layer in `bindings/ffi/`
2. Regenerate or update each language binding
3. Update binding tests
4. Update binding documentation

### Binding Architecture

```
bindings/
  ffi/       # C header and implementation (base layer)
  node/      # Node.js via N-API (napi-rs)
  python/    # Python via PyO3
  go/        # Go via CGo
  php/       # PHP via FFI
```

---

## Release Process Overview

Releases follow semantic versioning:

1. **Version bump** -- Update `Cargo.toml` version
2. **Changelog** -- Document changes since last release
3. **CI checks** -- All tests pass on Linux, macOS, Windows
4. **Tag** -- Create annotated git tag: `git tag -a v0.x.y -m "Release v0.x.y"`
5. **Publish** -- `cargo publish` (maintainers only)
6. **Bindings** -- Update and publish language binding packages

---

## Code of Conduct

All contributors are expected to follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). We are committed to providing a welcoming and inclusive experience for everyone.

Key points:

- Be respectful and constructive in all interactions
- Focus on technical merit in code reviews
- Welcome newcomers and help them get started
- Report unacceptable behavior to project maintainers

---

## Getting Help

- Open a [GitHub Issue](https://github.com/neul-labs/mullama/issues) for bugs or feature requests
- Check existing issues before creating new ones
- For usage questions, refer to the [Examples](examples/index.md) section
- Review the [API Reference](api/index.md) for type details

Thank you for contributing to Mullama.
