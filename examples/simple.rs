//! Simple text generation example
//!
//! Shows the minimal code to load a model and generate text.
//!
//! Usage:
//!   cargo run --example simple -- path/to/model.gguf "Your prompt here"

use mullama::{Context, ContextParams, Model, SamplerParams};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <prompt>", args[0]);
        eprintln!("Example: {} ./models/llama-3.2-1b-q4_k_m.gguf \"The meaning of life is\"", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];
    let prompt = &args[2];

    // Initialize backend
    mullama::backend_init();

    // Load model
    println!("Loading model: {}", model_path);
    let model = Arc::new(Model::load(model_path)?);
    println!("  Parameters: {}M", model.n_params() / 1_000_000);
    println!("  Vocab size: {}", model.n_vocab());

    // Create context
    let ctx_params = ContextParams {
        n_ctx: 2048,
        n_batch: 512,
        n_threads: (num_cpus::get() / 2).max(1) as i32,
        ..ContextParams::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    // Configure sampling
    let sampler_params = SamplerParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ..SamplerParams::default()
    };
    let mut sampler = sampler_params.build_chain(model.clone())?;

    // Tokenize and decode prompt
    let tokens = model.tokenize(prompt, true, false)?;
    println!("\nPrompt: {} ({} tokens)", prompt, tokens.len());
    context.decode(&tokens)?;

    // Generate
    print!("\nGeneration: ");
    let mut generated_tokens = 0;
    let start = std::time::Instant::now();

    for _ in 0..256 {
        let token = sampler.sample(&mut context, -1);

        if model.vocab_is_eog(token) {
            break;
        }

        if let Ok(text) = model.token_to_str(token, 0, false) {
            print!("{}", text);
        }

        sampler.accept(token);
        context.decode_single(token)?;
        generated_tokens += 1;
    }

    let elapsed = start.elapsed();
    let tok_per_sec = generated_tokens as f64 / elapsed.as_secs_f64();
    println!("\n\n--- {} tokens in {:.1}s ({:.1} tok/s) ---", generated_tokens, elapsed.as_secs_f64(), tok_per_sec);

    mullama::backend_free();
    Ok(())
}
