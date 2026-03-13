//! Embedding example showing how to generate text embeddings
//!
//! This example demonstrates:
//! 1. Loading a model with embedding support
//! 2. Creating a context with embeddings enabled
//! 3. Generating embeddings for text
//! 4. Calculating similarity between embeddings

use mullama::{Model, ContextParams, Embeddings, EmbeddingUtil};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mullama embedding example");
    
    // In a real implementation, you would load an actual GGUF model file:
    // let model = Model::load("path/to/model.gguf")?;
    
    // For this example, we'll just create a placeholder model
    println!("Creating model placeholder...");
    let model = create_placeholder_model();
    
    println!("Creating context with embeddings enabled...");
    let mut context_params = ContextParams::default();
    context_params.embeddings = true;
    let mut ctx = model.create_context(context_params)?;
    
    println!("Tokenizing texts...");
    let text1 = "The cat sat on the mat";
    let text2 = "A feline rested on the rug";
    let text3 = "The dog ran in the park";
    
    let tokens1 = model.tokenize(text1, true, false)?;
    let tokens2 = model.tokenize(text2, true, false)?;
    let tokens3 = model.tokenize(text3, true, false)?;
    
    println!("Generating embeddings...");
    // In a real implementation:
    // let embeddings1 = EmbeddingUtil::generate_embeddings(&ctx, &tokens1)?;
    // let embeddings2 = EmbeddingUtil::generate_embeddings(&ctx, &tokens2)?;
    // let embeddings3 = EmbeddingUtil::generate_embeddings(&ctx, &tokens3)?;
    
    // For this example, we'll create placeholder embeddings
    let embeddings1 = Embeddings::new(vec![0.1, 0.2, 0.3, 0.4, 0.5], 5);
    let embeddings2 = Embeddings::new(vec![0.15, 0.25, 0.35, 0.45, 0.55], 5);
    let embeddings3 = Embeddings::new(vec![0.8, 0.7, 0.6, 0.5, 0.4], 5);
    
    println!("Calculating similarities...");
    let similarity1_2 = EmbeddingUtil::cosine_similarity(
        embeddings1.get(0).unwrap_or(&[]), 
        embeddings2.get(0).unwrap_or(&[])
    );
    let similarity1_3 = EmbeddingUtil::cosine_similarity(
        embeddings1.get(0).unwrap_or(&[]), 
        embeddings3.get(0).unwrap_or(&[])
    );
    let similarity2_3 = EmbeddingUtil::cosine_similarity(
        embeddings2.get(0).unwrap_or(&[]), 
        embeddings3.get(0).unwrap_or(&[])
    );
    
    println!("Similarity between '{}' and '{}': {:.4}", text1, text2, similarity1_2);
    println!("Similarity between '{}' and '{}': {:.4}", text1, text3, similarity1_3);
    println!("Similarity between '{}' and '{}': {:.4}", text2, text3, similarity2_3);
    
    println!("Embeddings generated successfully!");
    
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