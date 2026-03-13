//! Embedding example showing how to generate text embeddings
//!
//! This example demonstrates:
//! 1. Creating embeddings
//! 2. Working with embedding utilities
//! 3. Embedding comparison and similarity

use mullama::{ContextParams, EmbeddingUtil, Embeddings, Model, MullamaError};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    println!("Mullama embedding example");

    // In a real implementation, you would load an actual GGUF model file:
    // let model = Model::load("path/to/model.gguf")?;

    // For this example, we'll demonstrate the embedding API
    println!("Demonstrating embedding API...");

    println!("Creating context parameters for embeddings...");
    let mut context_params = ContextParams::default();
    context_params.embeddings = true;
    context_params.n_ctx = 2048;
    println!(" Context parameters configured for embeddings");
    println!("   Embeddings enabled: {}", context_params.embeddings);
    println!("   Context size: {}", context_params.n_ctx);

    println!("Creating sample embeddings...");

    // Create some example embeddings
    let embedding1_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let embedding1 = Embeddings::new(embedding1_data.clone(), 5);

    let embedding2_data = vec![0.2, 0.3, 0.4, 0.5, 0.6];
    let embedding2 = Embeddings::new(embedding2_data.clone(), 5);

    let embedding3_data = vec![0.9, 0.8, 0.7, 0.6, 0.5];
    let embedding3 = Embeddings::new(embedding3_data.clone(), 5);

    println!(" Created {} embeddings", 3);
    println!(
        "   Embedding 1 - Dimension: {}, Length: {}",
        embedding1.dimension,
        embedding1.len()
    );
    println!(
        "   Embedding 2 - Dimension: {}, Length: {}",
        embedding2.dimension,
        embedding2.len()
    );
    println!(
        "   Embedding 3 - Dimension: {}, Length: {}",
        embedding3.dimension,
        embedding3.len()
    );

    println!("Computing similarities...");
    let similarity_1_2 = EmbeddingUtil::cosine_similarity(&embedding1_data, &embedding2_data);
    let similarity_1_3 = EmbeddingUtil::cosine_similarity(&embedding1_data, &embedding3_data);
    let similarity_2_3 = EmbeddingUtil::cosine_similarity(&embedding2_data, &embedding3_data);

    println!(" Similarity calculations completed:");
    println!("   Embedding 1 vs 2: {:.4}", similarity_1_2);
    println!("   Embedding 1 vs 3: {:.4}", similarity_1_3);
    println!("   Embedding 2 vs 3: {:.4}", similarity_2_3);

    println!("Working with embedding normalization...");
    let normalized = EmbeddingUtil::normalize(&embedding1_data);
    println!(" Normalized embedding created");
    println!("   Original: {:?}", &embedding1_data[..3]);
    println!("   Normalized: {:?}", &normalized[..3]);

    // Example of what real embedding generation would look like
    println!("Embedding generation concepts:");
    println!("  1. Enable embeddings in ContextParams");
    println!("  2. Use embedding-capable models");
    println!("  3. Generate embeddings with EmbeddingUtil::generate_embeddings()");
    println!("  4. Compare embeddings using cosine similarity");
    println!("  5. Normalize embeddings for better comparison");
    println!("  6. Use embeddings for semantic search and similarity");

    println!("Embedding example completed successfully!");

    Ok(())
}
