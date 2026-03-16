# Contributing to Mullama

Thank you for your interest in contributing to Mullama! We welcome contributions from the community and are excited to work with you to make Mullama the best Rust bindings for llama.cpp.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone --recurse-submodules https://github.com/skelf-research/mullama.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes
5. **Test** your changes: `cargo test`
6. **Commit** your changes: `git commit -m 'Add amazing feature'`
7. **Push** to your branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

## 📋 Development Setup

### Prerequisites

- **Rust 1.70+** with `cargo`
- **CMake 3.12+**
- **C++17 compiler** (GCC 8+, Clang 7+, or MSVC 2019+)
- **Git** with submodule support

### Development Dependencies

```bash
# Install development tools
cargo install cargo-watch
cargo install cargo-tarpaulin
cargo install cargo-clippy
cargo install rustfmt

# For documentation
cargo install mdbook
```

### Building

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/skelf-research/mullama.git
cd mullama

# Build
cargo build

# Run tests
cargo test

# Run tests with coverage
cargo tarpaulin --out html
```

### Development Workflow

```bash
# Watch for changes and run tests
cargo watch -x test

# Run specific test suites
cargo test --test unit_tests
cargo test --test integration_tests
cargo test --test sampling_tests

# Check formatting
cargo fmt --check

# Run linter
cargo clippy -- -D warnings

# Build documentation
cargo doc --open
```

## 🎯 Areas for Contribution

### High Priority

- **🧪 Test Coverage**: Add more test cases, especially edge cases
- **📚 Documentation**: Improve guides, examples, and API docs
- **⚡ Performance**: Optimize critical paths and memory usage
- **🔧 GPU Support**: Enhance CUDA, Metal, and ROCm integration
- **🐛 Bug Fixes**: Address issues and improve stability

### Medium Priority

- **✨ New Features**: Implement additional llama.cpp functionality
- **📦 Examples**: Create more real-world usage examples
- **🛠️ Developer Tools**: Improve build system and debugging tools
- **📊 Benchmarks**: Add comprehensive performance tests

### Lower Priority

- **🎨 API Ergonomics**: Improve the developer experience
- **📱 Platform Support**: Enhance compatibility across platforms
- **🔍 Profiling**: Add detailed performance profiling tools

## 📝 Contribution Guidelines

### Code Style

We follow standard Rust conventions:

```bash
# Format code
cargo fmt

# Check with clippy
cargo clippy -- -D warnings
```

**Key principles:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and traits
- Prefer explicit types in public APIs
- Add comprehensive documentation to public items
- Follow the existing error handling patterns

### Documentation

All public APIs must be documented:

```rust
/// Brief description of the function.
///
/// Longer description explaining the behavior, parameters,
/// and return values in detail.
///
/// # Arguments
///
/// * `param1` - Description of parameter 1
/// * `param2` - Description of parameter 2
///
/// # Returns
///
/// Description of what the function returns.
///
/// # Errors
///
/// Description of when and what errors can occur.
///
/// # Examples
///
/// ```rust
/// use mullama::Model;
///
/// let model = Model::from_file("path/to/model.gguf")?;
/// ```
pub fn example_function(param1: &str, param2: i32) -> Result<String, MullamaError> {
    // Implementation
}
```

### Testing

All contributions should include appropriate tests:

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test implementation
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_error_conditions() {
        let result = function_that_should_fail();
        assert!(result.is_err());
    }
}
```

#### Integration Tests
Place integration tests in the `tests/` directory:

```rust
// tests/my_feature_tests.rs
use mullama::*;

#[test]
fn test_feature_integration() {
    // Test with actual models/contexts
}
```

### Error Handling

Use the existing error types consistently:

```rust
use mullama::MullamaError;

pub fn my_function() -> Result<T, MullamaError> {
    match some_operation() {
        Ok(value) => Ok(value),
        Err(_) => Err(MullamaError::OperationFailed("Descriptive message".to_string())),
    }
}
```

### Performance

- Profile performance-critical code
- Use `cargo bench` for benchmarks
- Avoid unnecessary allocations
- Consider memory usage impact
- Test with large models when relevant

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test files
cargo test --test unit_tests
cargo test --test integration_tests
cargo test --test sampling_tests
cargo test --test error_tests
cargo test --test benchmark_tests

# Run tests with output
cargo test -- --nocapture

# Run tests matching a pattern
cargo test tokenize

# Run with coverage
cargo tarpaulin --out html
```

### Test Categories

1. **Unit Tests** (`src/lib.rs`, `src/*/mod.rs`)
   - Test individual functions and modules
   - Mock external dependencies
   - Fast execution

2. **Integration Tests** (`tests/integration_tests.rs`)
   - Test component interactions
   - Use real llama.cpp functionality
   - May require model files

3. **Performance Tests** (`tests/benchmark_tests.rs`)
   - Benchmark critical operations
   - Regression testing
   - Memory usage validation

4. **Error Tests** (`tests/error_tests.rs`)
   - Test error conditions
   - Edge cases and boundary conditions
   - Resource exhaustion scenarios

### Test Requirements

- **Fast**: Unit tests should run quickly
- **Reliable**: Tests should be deterministic
- **Isolated**: Tests shouldn't depend on external resources
- **Comprehensive**: Cover both success and error paths

## 📚 Documentation Standards

### Code Documentation

- **Public APIs**: Must have comprehensive rustdoc
- **Examples**: Include practical usage examples
- **Error Cases**: Document when and why errors occur
- **Performance**: Note performance characteristics

### Guide Documentation

- **Clear Structure**: Use consistent headings and formatting
- **Practical Examples**: Show real-world usage
- **Progressive Complexity**: Start simple, build up
- **Cross-references**: Link to related concepts

### Example Code

All examples should:
- Be runnable (when possible)
- Include error handling
- Show best practices
- Be well-commented

## 🔄 Pull Request Process

### Before Submitting

1. **Run Tests**: `cargo test`
2. **Check Formatting**: `cargo fmt --check`
3. **Run Linter**: `cargo clippy -- -D warnings`
4. **Update Documentation**: Add/update relevant docs
5. **Add Tests**: Include appropriate test coverage

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Changes
- List of specific changes made
- Breaking changes (if any)
- New features added

## Testing
- How the changes were tested
- New tests added
- Performance impact (if any)

## Documentation
- Documentation updates made
- Examples added/updated

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Comprehensive test coverage
4. **Documentation**: Appropriate documentation updates

## 🏗️ Architecture Guidelines

### Module Organization

```
src/
├── lib.rs          # Public API exports
├── model.rs        # Model loading and management
├── context.rs      # Context and evaluation
├── sampling.rs     # Sampling strategies
├── batch.rs        # Batch processing
├── sys.rs          # FFI bindings
├── error.rs        # Error types
└── utils.rs        # Utility functions
```

### FFI Guidelines

- Keep unsafe code isolated in `sys.rs`
- Validate all inputs from C
- Handle null pointers gracefully
- Use proper error propagation
- Document safety requirements

### Memory Management

- Use RAII patterns consistently
- Implement proper Drop traits
- Avoid memory leaks
- Handle resource cleanup on errors
- Document lifetime requirements

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Rust version, hardware
2. **Steps to Reproduce**: Minimal example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Logs/Errors**: Full error messages
6. **Model Information**: Model type, size, format

### Bug Report Template

```markdown
**Environment**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Rust version: [e.g., 1.70.0]
- Mullama version: [e.g., 0.1.0]
- Hardware: [e.g., CPU, GPU model]

**Description**
A clear description of the bug.

**Steps to Reproduce**
1. Load model X
2. Call function Y with parameters Z
3. Observe error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Error Messages**
```
Full error output here
```

**Additional Context**
Any other relevant information.
```

## 🚀 Feature Requests

For new features, please:

1. **Check Existing Issues**: Avoid duplicates
2. **Describe Use Case**: Why is this needed?
3. **Propose API**: What should the interface look like?
4. **Consider Impact**: Performance, compatibility, complexity
5. **Implementation Ideas**: How might it work?

## 📞 Getting Help

- **💬 Discussions**: [GitHub Discussions](https://github.com/skelf-research/mullama/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/skelf-research/mullama/issues)
- **📚 Documentation**: [docs.rs/mullama](https://docs.rs/mullama)
- **📧 Email**: support@skelfresearch.com

## 🎉 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the maintainer team for sustained contributions

## 📄 License

By contributing to Mullama, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Mullama! Together, we're building the best Rust bindings for llama.cpp. 🦙❤️