use crate::{sys, error::MullamaError, context::Context};

/// Represents the complete state of a context that can be saved and restored
#[derive(Debug)]
pub struct Session {
    data: Vec<u8>,
}

impl Session {
    /// Create a session from a context's current state
    pub fn from_context(context: &Context) -> Result<Self, MullamaError> {
        // Get the required buffer size
        let size = unsafe { 
            sys::llama_state_get_size(context.as_ptr()) 
        };
        
        // Allocate buffer
        let mut data = vec![0u8; size];
        
        // Get the state data
        let written = unsafe {
            sys::llama_state_get_data(
                context.as_ptr(),
                data.as_mut_ptr(),
                size,
            )
        };
        
        if written != size {
            return Err(MullamaError::SessionError(
                "Failed to get session data".to_string()
            ));
        }
        
        // Truncate to actual size written
        data.truncate(written);
        
        Ok(Session { data })
    }
    
    /// Restore a session to a context
    pub fn restore_to_context(&self, context: &mut Context) -> Result<(), MullamaError> {
        let read = unsafe {
            sys::llama_state_set_data(
                context.as_ptr(),
                self.data.as_ptr(),
                self.data.len(),
            )
        };
        
        if read != self.data.len() {
            return Err(MullamaError::SessionError(
                "Failed to restore session data".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Save session to a file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), MullamaError> {
        std::fs::write(path, &self.data).map_err(MullamaError::IoError)
    }
    
    /// Load session from a file
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, MullamaError> {
        let data = std::fs::read(path).map_err(MullamaError::IoError)?;
        Ok(Session { data })
    }
}