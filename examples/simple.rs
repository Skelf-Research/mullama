//! Simple example showing how to use Mullama
//!
//! This example demonstrates the basic usage pattern:
//! 1. Load a model (placeholder)
//! 2. Create a context
//! 3. Show API usage patterns

use mullama::{ContextParams, Model, MullamaError};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    println!("Mullama example - demonstrating API usage");

    // In a real implementation, you would load an actual GGUF model file:
    // let model = Model::load("path/to/model.gguf")?;

    // For this example, we'll show how the API would be used
    // without requiring an actual model file
    println!("Creating model example...");

    // Example of how to load a model (this would work with a real model file)
    println!("To load a real model, you would call:");
    println!("  let model = Model::load(\"path/to/model.gguf\")?;");

    // Example of creating context parameters
    println!("Creating context parameters...");
    let ctx_params = ContextParams::default();
    println!("  Default context size: {}", ctx_params.n_ctx);
    println!("  Default batch size: {}", ctx_params.n_batch);
    println!("  Default threads: {}", ctx_params.n_threads);

    // Example of what model API calls would look like
    println!("Model API examples:");
    println!("  model.vocab_size() - Get vocabulary size");
    println!("  model.n_ctx_train() - Get training context size");
    println!("  model.tokenize(text, add_bos, special) - Tokenize text");
    println!("  model.token_to_str(token, lstrip, special) - Convert token to string");

    // Example of what context API calls would look like
    println!("Context API examples:");
    println!("  Context::new(model, params) - Create new context");
    println!("  context.generate(tokens, max_len) - Generate text");

    println!("API demonstration complete!");

    Ok(())
}
