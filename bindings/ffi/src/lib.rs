//! # Mullama FFI
//!
//! C ABI bindings for Mullama LLM library.
//!
//! This crate provides a stable C API for integrating Mullama into
//! Node.js, Python, PHP, Go, and other languages.
//!
//! ## Features
//!
//! - Model loading and management
//! - Tokenization and detokenization
//! - Text generation with customizable sampling
//! - Streaming generation with callbacks
//! - Embedding generation
//! - Thread-local error handling
//!
//! ## Example C Usage
//!
//! ```c
//! #include <mullama.h>
//!
//! int main() {
//!     // Initialize backend
//!     mullama_backend_init();
//!
//!     // Load model
//!     MullamaModel* model = mullama_model_load("model.gguf", NULL);
//!     if (!model) {
//!         printf("Error: %s\n", mullama_get_last_error());
//!         return 1;
//!     }
//!
//!     // Create context
//!     MullamaContext* ctx = mullama_context_new(model, NULL);
//!
//!     // Tokenize
//!     int32_t tokens[1024];
//!     int n_tokens = mullama_tokenize(model, "Hello, AI!", tokens, 1024, true, false);
//!
//!     // Generate
//!     char output[4096];
//!     int result = mullama_generate(ctx, tokens, n_tokens, 100, NULL, output, 4096);
//!
//!     printf("Generated: %s\n", output);
//!
//!     // Cleanup
//!     mullama_context_free(ctx);
//!     mullama_model_free(model);
//!     mullama_backend_free();
//!
//!     return 0;
//! }
//! ```

#![allow(clippy::missing_safety_doc)]

pub mod context;
pub mod embedding;
pub mod error;
pub mod handle;
pub mod model;
pub mod sampler;
pub mod streaming;

// Re-export everything for convenience
pub use context::*;
pub use embedding::*;
pub use error::*;
pub use model::*;
pub use sampler::*;
pub use streaming::*;

// ============================================================================
// Backend Initialization
// ============================================================================

/// Initialize the Mullama/llama.cpp backend
///
/// This should be called once before using any other functions.
/// It is safe to call multiple times.
#[no_mangle]
pub extern "C" fn mullama_backend_init() {
    mullama::backend_init();
}

/// Free the Mullama/llama.cpp backend resources
///
/// Call this when completely done with the library.
#[no_mangle]
pub extern "C" fn mullama_backend_free() {
    mullama::backend_free();
}

// ============================================================================
// System Information
// ============================================================================

/// Check if GPU offloading is supported
#[no_mangle]
pub extern "C" fn mullama_supports_gpu_offload() -> bool {
    mullama::supports_gpu_offload()
}

/// Check if memory mapping is supported
#[no_mangle]
pub extern "C" fn mullama_supports_mmap() -> bool {
    mullama::supports_mmap()
}

/// Check if memory locking is supported
#[no_mangle]
pub extern "C" fn mullama_supports_mlock() -> bool {
    mullama::supports_mlock()
}

/// Get maximum number of devices supported
#[no_mangle]
pub extern "C" fn mullama_max_devices() -> usize {
    mullama::max_devices()
}

/// Get system information string
///
/// # Arguments
/// * `output` - Output buffer
/// * `max_output` - Size of output buffer
///
/// # Returns
/// Number of bytes written, or negative required size
#[no_mangle]
pub extern "C" fn mullama_system_info(
    output: *mut libc::c_char,
    max_output: libc::size_t,
) -> libc::c_int {
    let info = mullama::print_system_info();
    let bytes = info.as_bytes();
    let len = bytes.len();

    if output.is_null() || max_output < len + 1 {
        return -(len as libc::c_int + 1);
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, len);
        *output.add(len) = 0;
    }

    len as libc::c_int
}

// ============================================================================
// Version Information
// ============================================================================

/// Library version major number
pub const MULLAMA_VERSION_MAJOR: u32 = 0;
/// Library version minor number
pub const MULLAMA_VERSION_MINOR: u32 = 2;
/// Library version patch number
pub const MULLAMA_VERSION_PATCH: u32 = 0;

/// Get library version as a string (derived from Cargo.toml)
#[no_mangle]
pub extern "C" fn mullama_version() -> *const libc::c_char {
    // Null-terminated version string derived from Cargo.toml at compile time
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const libc::c_char
}

/// Get library version major number
#[no_mangle]
pub extern "C" fn mullama_version_major() -> u32 {
    MULLAMA_VERSION_MAJOR
}

/// Get library version minor number
#[no_mangle]
pub extern "C" fn mullama_version_minor() -> u32 {
    MULLAMA_VERSION_MINOR
}

/// Get library version patch number
#[no_mangle]
pub extern "C" fn mullama_version_patch() -> u32 {
    MULLAMA_VERSION_PATCH
}

// ============================================================================
// Time Utilities
// ============================================================================

/// Get current timestamp in microseconds
#[no_mangle]
pub extern "C" fn mullama_time_us() -> i64 {
    mullama::time_us()
}

// ============================================================================
// Hardware Presets
// ============================================================================

/// Get the number of available hardware presets
#[no_mangle]
pub extern "C" fn mullama_preset_count() -> libc::c_int {
    mullama::presets::HardwarePreset::all().len() as libc::c_int
}

/// Get the name of a preset by index.
/// Returns null if index is out of range.
/// The returned pointer points to a static string valid for the lifetime of the program.
#[no_mangle]
pub extern "C" fn mullama_preset_name(index: libc::c_int) -> *const libc::c_char {
    match mullama::presets::HardwarePreset::from_index(index as usize) {
        Some(preset) => {
            let name = preset.name();
            // Return static string pointer (valid for lifetime of program)
            name.as_ptr() as *const libc::c_char
        }
        None => std::ptr::null(),
    }
}

/// Get the description of a preset by index.
/// Returns null if index is out of range.
#[no_mangle]
pub extern "C" fn mullama_preset_description(index: libc::c_int) -> *const libc::c_char {
    match mullama::presets::HardwarePreset::from_index(index as usize) {
        Some(preset) => {
            let desc = preset.description();
            desc.as_ptr() as *const libc::c_char
        }
        None => std::ptr::null(),
    }
}

/// Get the recommended quantization format for a preset by index.
/// Returns null if index is out of range.
#[no_mangle]
pub extern "C" fn mullama_preset_recommended_quant(index: libc::c_int) -> *const libc::c_char {
    match mullama::presets::HardwarePreset::from_index(index as usize) {
        Some(preset) => {
            let quant = preset.recommended_quant();
            quant.as_ptr() as *const libc::c_char
        }
        None => std::ptr::null(),
    }
}

/// Get the recommended GPU layers for a preset by index.
/// Returns 0 if index is out of range.
#[no_mangle]
pub extern "C" fn mullama_preset_gpu_layers(index: libc::c_int) -> libc::c_int {
    match mullama::presets::HardwarePreset::from_index(index as usize) {
        Some(preset) => preset.model_params().n_gpu_layers,
        None => 0,
    }
}

/// Get the recommended context size for a preset by index.
/// Returns 0 if index is out of range.
#[no_mangle]
pub extern "C" fn mullama_preset_context_size(index: libc::c_int) -> u32 {
    match mullama::presets::HardwarePreset::from_index(index as usize) {
        Some(preset) => preset.context_params().n_ctx,
        None => 0,
    }
}

/// Detect the best preset for the current hardware.
/// Returns the preset index.
#[no_mangle]
pub extern "C" fn mullama_preset_detect() -> libc::c_int {
    mullama::presets::HardwarePreset::detect().index() as libc::c_int
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_init() {
        mullama_backend_init();
        mullama_backend_free();
    }

    #[test]
    fn test_version() {
        assert_eq!(mullama_version_major(), MULLAMA_VERSION_MAJOR);
        assert_eq!(mullama_version_minor(), MULLAMA_VERSION_MINOR);
        assert_eq!(mullama_version_patch(), MULLAMA_VERSION_PATCH);

        let version = mullama_version();
        assert!(!version.is_null());
    }

    #[test]
    fn test_system_capabilities() {
        // These should not panic
        let _ = mullama_supports_gpu_offload();
        let _ = mullama_supports_mmap();
        let _ = mullama_supports_mlock();
        let _ = mullama_max_devices();
    }

    #[test]
    fn test_system_info() {
        let mut buffer = vec![0u8; 1024];
        let result = mullama_system_info(buffer.as_mut_ptr() as *mut libc::c_char, 1024);
        assert!(result >= 0);
    }

    #[test]
    fn test_time() {
        let t1 = mullama_time_us();
        let t2 = mullama_time_us();
        assert!(t2 >= t1);
    }

    #[test]
    fn test_error_handling() {
        // Test that error functions work
        error::set_last_error("test error");
        let ptr = error::mullama_get_last_error();
        assert!(!ptr.is_null());

        error::mullama_clear_error();
        let ptr2 = error::mullama_get_last_error();
        assert!(ptr2.is_null());
    }
}
