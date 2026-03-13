//! # Simple Text Generation Example
//!
//! This is the most basic example of using Mullama for text generation.
//! Perfect for getting started and understanding the core concepts.

use mullama::{
    Context, ContextParams, Model, MullamaError, SamplerChain, SamplerChainParams, SamplerParams,
};
use std::io::{self, Write};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path> [prompt]", args[0]);
        eprintln!(
            "Example: {} path/to/model.gguf \"The future of AI is\"",
            args[0]
        );
        return Ok(());
    }

    let model_path = &args[1];
    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("The future of artificial intelligence is");

    println!("Mullama Simple Generation Example");
    println!("Model: {}", model_path);
    println!("Prompt: \"{}\"", prompt);
    println!("{}", "=".repeat(50));

    // Load the model
    println!("Loading model...");
    let model = Arc::new(Model::load(model_path)?);

    println!("Model loaded successfully!");
    println!("   Vocabulary size: {}", model.vocab_size());
    println!("   Context size: {}", model.n_ctx_train());

    // Create context
    let mut ctx_params = ContextParams::default();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    let mut context = Context::new(model.clone(), ctx_params)?;

    // Set up sampling
    let mut sampler_params = SamplerParams::default();
    sampler_params.temperature = 0.7;
    sampler_params.top_k = 40;
    sampler_params.top_p = 0.9;

    let mut sampler = sampler_params.build_chain(model.clone())?;

    // Tokenize prompt
    println!("Tokenizing prompt...");
    let tokens = model.tokenize(prompt, true, false)?;
    println!("Prompt tokenized into {} tokens", tokens.len());

    // Evaluate prompt (placeholder - method not implemented)
    println!("Processing prompt...");
    for _token in tokens {
        // context.eval_token(token)?; // Method not implemented yet
    }

    // Generate text
    println!("Generating text...\n");
    print!("Output: {}", prompt);
    io::stdout().flush()?;

    let max_tokens = 100;
    let mut generated_tokens = 0;

    while generated_tokens < max_tokens {
        // Sample next token
        let next_token = sampler.sample(&mut context, 0);

        // Check for end of generation (placeholder)
        if next_token == 0 {
            // Placeholder for EOS token check
            println!("\n\nGeneration completed (end token reached)");
            break;
        }

        // Convert token to text
        let text = model.token_to_str(next_token, 0, false)?;
        print!("{}", text);
        io::stdout().flush()?;

        // Evaluate token for next iteration (placeholder - method not implemented)
        // context.eval_token(next_token)?; // Method not implemented yet
        generated_tokens += 1;
    }

    if generated_tokens >= max_tokens {
        println!("\n\nGeneration completed (max tokens reached)");
    }

    println!("Generated {} tokens", generated_tokens);

    Ok(())
}
