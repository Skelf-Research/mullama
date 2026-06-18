//! Handle management for FFI layer
//!
//! This module provides Arc-based opaque handles for safely crossing
//! the FFI boundary. Each handle wraps a Rust type in an Arc, allowing
//! multiple references while ensuring proper cleanup.

use std::sync::Arc;

/// A wrapper for creating opaque handles that can cross FFI boundaries.
///
/// Handles are created by boxing an Arc, which ensures:
/// 1. The data lives as long as the handle exists
/// 2. Multiple handles can safely reference the same data
/// 3. Proper cleanup when the last handle is freed
pub struct Handle<T>(pub Arc<T>);

impl<T> Handle<T> {
    /// Create a new handle wrapping the given value.
    pub fn new(value: T) -> Self {
        Handle(Arc::new(value))
    }

    /// Create a handle from an existing Arc.
    pub fn from_arc(arc: Arc<T>) -> Self {
        Handle(arc)
    }

    /// Convert this handle into a raw pointer suitable for FFI.
    ///
    /// The caller is responsible for eventually calling `from_raw`
    /// to reclaim ownership and drop the handle.
    pub fn into_raw(self) -> *mut Handle<T> {
        Box::into_raw(Box::new(self))
    }

    /// Reconstruct a handle from a raw pointer.
    ///
    /// # Safety
    /// The pointer must have been created by `into_raw` and must not
    /// have been freed already.
    pub unsafe fn from_raw(ptr: *mut Handle<T>) -> Self {
        *Box::from_raw(ptr)
    }

    /// Get a reference to the inner value without consuming the handle.
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    pub unsafe fn as_ref<'a>(ptr: *const Handle<T>) -> Option<&'a T> {
        if ptr.is_null() {
            None
        } else {
            Some(&(*ptr).0)
        }
    }

    /// Clone the Arc inside the handle (increases reference count).
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    pub unsafe fn clone_arc(ptr: *const Handle<T>) -> Option<Arc<T>> {
        if ptr.is_null() {
            None
        } else {
            Some(Arc::clone(&(*ptr).0))
        }
    }

    /// Get the inner Arc.
    pub fn inner(&self) -> &Arc<T> {
        &self.0
    }

    /// Consume the handle and return the inner Arc.
    pub fn into_inner(self) -> Arc<T> {
        self.0
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle(Arc::clone(&self.0))
    }
}

/// A mutable handle for types that need interior mutability.
///
/// Uses a Mutex to provide safe mutable access across FFI.
pub struct MutableHandle<T>(pub Arc<std::sync::Mutex<T>>);

impl<T> MutableHandle<T> {
    /// Create a new mutable handle wrapping the given value.
    pub fn new(value: T) -> Self {
        MutableHandle(Arc::new(std::sync::Mutex::new(value)))
    }

    /// Convert this handle into a raw pointer suitable for FFI.
    pub fn into_raw(self) -> *mut MutableHandle<T> {
        Box::into_raw(Box::new(self))
    }

    /// Reconstruct a handle from a raw pointer.
    ///
    /// # Safety
    /// The pointer must have been created by `into_raw` and must not
    /// have been freed already.
    pub unsafe fn from_raw(ptr: *mut MutableHandle<T>) -> Self {
        *Box::from_raw(ptr)
    }

    /// Get a reference to execute a closure with the inner value.
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    pub unsafe fn with_ref<'a, R, F>(ptr: *const MutableHandle<T>, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        if ptr.is_null() {
            None
        } else {
            let guard = (*ptr).0.lock().ok()?;
            Some(f(&*guard))
        }
    }

    /// Get a mutable reference to execute a closure with the inner value.
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    pub unsafe fn with_mut<'a, R, F>(ptr: *mut MutableHandle<T>, f: F) -> Option<R>
    where
        F: FnOnce(&mut T) -> R,
    {
        if ptr.is_null() {
            None
        } else {
            let mut guard = (*ptr).0.lock().ok()?;
            Some(f(&mut *guard))
        }
    }
}

impl<T> Clone for MutableHandle<T> {
    fn clone(&self) -> Self {
        MutableHandle(Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_lifecycle() {
        let handle = Handle::new(42i32);
        let ptr = handle.into_raw();

        unsafe {
            let value = Handle::as_ref(ptr).unwrap();
            assert_eq!(*value, 42);

            let handle = Handle::from_raw(ptr);
            assert_eq!(**handle.inner(), 42);
        }
    }

    #[test]
    fn test_handle_null_safety() {
        unsafe {
            let result: Option<&i32> = Handle::as_ref(std::ptr::null());
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_mutable_handle() {
        let handle = MutableHandle::new(42i32);
        let ptr = handle.into_raw();

        unsafe {
            MutableHandle::with_mut(ptr, |val| {
                *val = 100;
            });

            let result = MutableHandle::with_ref(ptr, |val| *val);
            assert_eq!(result, Some(100));

            let _ = MutableHandle::from_raw(ptr);
        }
    }
}
