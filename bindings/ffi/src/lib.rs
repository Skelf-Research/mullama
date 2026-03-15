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
pub const MULLAMA_VERSION_MINOR: u32 = 1;
/// Library version patch number
pub const MULLAMA_VERSION_PATCH: u32 = 0;

/// Get library version as a string
#[no_mangle]
pub extern "C" fn mullama_version() -> *const libc::c_char {
    static VERSION: &[u8] = b"0.1.0\0";
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
