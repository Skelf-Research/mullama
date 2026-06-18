//! Multi-model example
//!
//! Demonstrates loading multiple models and switching between them.
//!
//! Usage:
//!   cargo run --example multi_model -- model1.gguf model2.gguf

use mullama::{Context, ContextParams, Model, SamplerParams};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model1.gguf> <model2.gguf>", args[0]);
        std::process::exit(1);
    }

    mullama::backend_init();

    // Load two models
    println!("Loading model 1: {}", args[1]);
    let model1 = Arc::new(Model::load(&args[1])?);
    println!("  {}M parameters", model1.n_params() / 1_000_000);

    println!("Loading model 2: {}", args[2]);
    let model2 = Arc::new(Model::load(&args[2])?);
    println!("  {}M parameters", model2.n_params() / 1_000_000);

    let ctx_params = ContextParams {
        n_ctx: 2048,
        n_batch: 512,
        n_threads: (num_cpus::get() / 2).max(1) as i32,
        ..ContextParams::default()
    };

    let prompt = "Explain quantum computing in one sentence:";

    // Generate with model 1
    println!("\n=== Model 1 ===");
    generate(&model1, &ctx_params, prompt)?;

    // Generate with model 2
    println!("\n=== Model 2 ===");
    generate(&model2, &ctx_params, prompt)?;

    // System info
    println!("\n=== System Info ===");
    let info = mullama::system_info();
    println!("GPU offload: {}", info.supports_gpu_offload);
    println!("mmap: {}", info.supports_mmap);
    println!("Details: {}", info.details);

    mullama::backend_free();
    Ok(())
}

fn generate(
    model: &Arc<Model>,
    ctx_params: &ContextParams,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::new(model.clone(), ctx_params.clone())?;

    let sampler_params = SamplerParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ..SamplerParams::default()
    };
    let mut sampler = sampler_params.build_chain(model.clone())?;

    let tokens = model.tokenize(prompt, true, false)?;
    print!("{}", prompt);
    context.decode(&tokens)?;

    let start = std::time::Instant::now();
    let mut count = 0u32;

    for _ in 0..128 {
        let token = sampler.sample(&mut context, -1);
        if model.vocab_is_eog(token) {
            break;
        }
        if let Ok(text) = model.token_to_str(token, 0, false) {
            print!("{}", text);
        }
        sampler.accept(token);
        context.decode_single(token)?;
        count += 1;
    }

    let elapsed = start.elapsed();
    println!(
        "\n  ({} tokens, {:.1} tok/s)",
        count,
        count as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}
