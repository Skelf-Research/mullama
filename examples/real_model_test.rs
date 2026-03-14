//! Test with real GGUF model file
//!
//! Run with: cargo run --example real_model_test

use mullama::{Context, ContextParams, Model, MullamaError};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), MullamaError> {
    let model_path = "models/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== Mullama Real Model Test ===\n");

    // Test 1: Load model
    println!("1. Loading model: {}", model_path);
    let start = Instant::now();
    let model = Arc::new(Model::load(model_path)?);
    println!("   Model loaded in {:?}", start.elapsed());

    // Test 2: Model info
    println!("\n2. Model Information:");
    println!("   Vocab size: {}", model.vocab_size());
    println!("   Context size (train): {}", model.n_ctx_train());
    println!("   Embedding size: {}", model.n_embd());

    // Test 3: Create context
    println!("\n3. Creating context...");
    let ctx_params = ContextParams {
        n_ctx: 512,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let start = Instant::now();
    let mut context = Context::new(model.clone(), ctx_params)?;
    println!("   Context created in {:?}", start.elapsed());

    // Test 4: Tokenization
    println!("\n4. Testing tokenization...");
    let test_text = "Hello, how are you?";
    let tokens = model.tokenize(test_text, true, false)?;
    println!("   Text: \"{}\"", test_text);
    println!("   Tokens: {:?}", tokens);
    println!("   Token count: {}", tokens.len());

    // Test 5: Token to piece (single token)
    println!("\n5. Testing token_to_str...");
    for &token in &tokens[1..] {
        // Skip BOS token
        let piece = model.token_to_str(token, 0, false)?;
        print!("{}", piece);
    }
    println!();

    // Test 6: Simple generation
    println!("\n6. Testing text generation...");
    let prompt = "The quick brown fox";
    println!("   Prompt: \"{}\"", prompt);

    let prompt_tokens = model.tokenize(prompt, true, false)?;

    let start = Instant::now();
    let generated_text = context.generate(&prompt_tokens, 32)?;
    let gen_time = start.elapsed();

    println!("   Generated: \"{}\"", generated_text);
    println!("   Generation time: {:?}", gen_time);

    println!("\n=== All tests passed! ===");

    Ok(())
}
