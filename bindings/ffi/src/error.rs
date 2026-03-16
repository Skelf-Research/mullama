//! Error handling for FFI layer
//!
//! This module provides error codes and thread-local error message storage
//! for communicating errors across the FFI boundary.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

/// Error codes returned by FFI functions.
///
/// All functions return these codes to indicate success or failure.
/// Use `mullama_get_last_error` to get a detailed error message.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MullamaErrorCode {
    /// Operation completed successfully
    Ok = 0,
    /// A null pointer was passed where a valid pointer was expected
    NullPointer = -1,
    /// Failed to load the model file
    ModelLoad = -2,
    /// Failed to create or operate on context
    Context = -3,
    /// Tokenization failed
    Tokenization = -4,
    /// Text generation failed
    Generation = -5,
    /// Sampler operation failed
    Sampler = -6,
    /// Embedding generation failed
    Embedding = -7,
    /// Invalid input parameters
    InvalidInput = -8,
    /// Buffer too small for output
    BufferTooSmall = -9,
    /// Backend initialization failed
    BackendInit = -10,
    /// Operation was cancelled
    Cancelled = -11,
    /// Internal error
    Internal = -12,
    /// Feature not available
    NotAvailable = -13,
    /// UTF-8 encoding error
    Utf8Error = -14,
    /// Lock acquisition failed
    LockError = -15,
}

impl MullamaErrorCode {
    /// Convert to i32 for FFI return values
    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

impl From<MullamaErrorCode> for i32 {
    fn from(code: MullamaErrorCode) -> Self {
        code as i32
    }
}

// Thread-local storage for the last error message
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Set the last error message for the current thread.
///
/// This should be called before returning an error code from an FFI function.
pub fn set_last_error(message: impl AsRef<str>) {
    LAST_ERROR.with(|cell| {
        let msg = CString::new(message.as_ref())
            .unwrap_or_else(|_| CString::new("Error message contained null bytes").unwrap());
        *cell.borrow_mut() = Some(msg);
    });
}

/// Set the last error from a Rust error type.
pub fn set_last_error_from<E: std::fmt::Display>(error: E) {
    set_last_error(error.to_string());
}

/// Clear the last error message for the current thread.
pub fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Get a pointer to the last error message for the current thread.
///
/// Returns null if no error has been set. The returned pointer is valid
/// until the next FFI call on the same thread.
///
/// # Safety
/// This function is safe to call, but the returned pointer should not
/// be stored long-term as it may be invalidated by subsequent FFI calls.
#[no_mangle]
pub extern "C" fn mullama_get_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match cell.borrow().as_ref() {
        Some(msg) => msg.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Clear the last error message.
#[no_mangle]
pub extern "C" fn mullama_clear_error() {
    clear_last_error();
}

/// Get a human-readable description of an error code.
#[no_mangle]
pub extern "C" fn mullama_error_code_description(code: i32) -> *const c_char {
    static OK: &[u8] = b"Success\0";
    static NULL_POINTER: &[u8] = b"Null pointer error\0";
    static MODEL_LOAD: &[u8] = b"Model loading failed\0";
    static CONTEXT: &[u8] = b"Context operation failed\0";
    static TOKENIZATION: &[u8] = b"Tokenization failed\0";
    static GENERATION: &[u8] = b"Text generation failed\0";
    static SAMPLER: &[u8] = b"Sampler operation failed\0";
    static EMBEDDING: &[u8] = b"Embedding generation failed\0";
    static INVALID_INPUT: &[u8] = b"Invalid input parameters\0";
    static BUFFER_TOO_SMALL: &[u8] = b"Buffer too small for output\0";
    static BACKEND_INIT: &[u8] = b"Backend initialization failed\0";
    static CANCELLED: &[u8] = b"Operation was cancelled\0";
    static INTERNAL: &[u8] = b"Internal error\0";
    static NOT_AVAILABLE: &[u8] = b"Feature not available\0";
    static UTF8_ERROR: &[u8] = b"UTF-8 encoding error\0";
    static LOCK_ERROR: &[u8] = b"Lock acquisition failed\0";
    static UNKNOWN: &[u8] = b"Unknown error code\0";

    let desc = match code {
        0 => OK,
        -1 => NULL_POINTER,
        -2 => MODEL_LOAD,
        -3 => CONTEXT,
        -4 => TOKENIZATION,
        -5 => GENERATION,
        -6 => SAMPLER,
        -7 => EMBEDDING,
        -8 => INVALID_INPUT,
        -9 => BUFFER_TOO_SMALL,
        -10 => BACKEND_INIT,
        -11 => CANCELLED,
        -12 => INTERNAL,
        -13 => NOT_AVAILABLE,
        -14 => UTF8_ERROR,
        -15 => LOCK_ERROR,
        _ => UNKNOWN,
    };

    desc.as_ptr() as *const c_char
}

/// Helper macro for setting error and returning error code
#[macro_export]
macro_rules! ffi_try {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                $crate::error::set_last_error_from(&e);
                return $crate::error::MullamaErrorCode::Internal.to_i32();
            }
        }
    };
    ($expr:expr, $code:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                $crate::error::set_last_error_from(&e);
                return $code.to_i32();
            }
        }
    };
}

/// Helper macro for null pointer checks
#[macro_export]
macro_rules! ffi_null_check {
    ($ptr:expr) => {
        if $ptr.is_null() {
            $crate::error::set_last_error("Null pointer passed to function");
            return $crate::error::MullamaErrorCode::NullPointer.to_i32();
        }
    };
    ($ptr:expr, $ret:expr) => {
        if $ptr.is_null() {
            $crate::error::set_last_error("Null pointer passed to function");
            return $ret;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_values() {
        assert_eq!(MullamaErrorCode::Ok.to_i32(), 0);
        assert_eq!(MullamaErrorCode::NullPointer.to_i32(), -1);
        assert_eq!(MullamaErrorCode::ModelLoad.to_i32(), -2);
    }

    #[test]
    fn test_set_and_get_error() {
        set_last_error("Test error message");

        let ptr = mullama_get_last_error();
        assert!(!ptr.is_null());

        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "Test error message");
    }

    #[test]
    fn test_clear_error() {
        set_last_error("Some error");
        assert!(!mullama_get_last_error().is_null());

        clear_last_error();
        assert!(mullama_get_last_error().is_null());
    }

    #[test]
    fn test_error_description() {
        let desc = mullama_error_code_description(0);
        let msg = unsafe { std::ffi::CStr::from_ptr(desc) };
        assert_eq!(msg.to_str().unwrap(), "Success");

        let desc = mullama_error_code_description(-1);
        let msg = unsafe { std::ffi::CStr::from_ptr(desc) };
        assert_eq!(msg.to_str().unwrap(), "Null pointer error");
    }
}
