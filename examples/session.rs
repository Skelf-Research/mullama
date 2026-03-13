//! Session management example showing how to save and restore model states
//!
//! This example demonstrates:
//! 1. Loading a model
//! 2. Creating a context
//! 3. Processing some tokens
//! 4. Saving the session state
//! 5. Loading the session state in a new context

use mullama::{Model, ContextParams, Session};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mullama session management example");
    
    // In a real implementation, you would load an actual GGUF model file:
    // let model = Model::load("path/to/model.gguf")?;
    
    // For this example, we'll just create a placeholder model
    println!("Creating model placeholder...");
    let model = create_placeholder_model();
    
    println!("Creating context...");
    let mut ctx = model.create_context(ContextParams::default())?;
    
    println!("Processing some tokens...");
    let tokens = model.tokenize("The quick brown fox jumps over the lazy dog", true, false)?;
    ctx.decode(&tokens)?;
    
    println!("Saving session...");
    let session = Session::from_context(&ctx)?;
    session.save_to_file("session.bin")?;
    
    println!("Creating new context...");
    let mut new_ctx = model.create_context(ContextParams::default())?;
    
    println!("Restoring session...");
    let loaded_session = Session::load_from_file("session.bin")?;
    new_ctx.restore_to_context(&loaded_session)?;
    
    println!("Session restored successfully!");
    
    Ok(())
}

/// Create a placeholder model for demonstration purposes
/// In a real implementation, you would load an actual GGUF model file
fn create_placeholder_model() -> Model {
    // This creates a model with a null pointer
    // In a real implementation, this would load an actual model
    Model {
        model_ptr: std::ptr::null_mut(),
    }
}