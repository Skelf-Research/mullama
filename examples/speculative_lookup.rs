//! Prompt-lookup speculative decoding: parity + measurement.
//!
//! Verifies that prompt-lookup speculative decoding produces *token-identical*
//! output to plain greedy decoding, and reports the acceptance rate and
//! realized speedup (tokens emitted per target forward pass).
//!
//! Usage:
//!   cargo run --release --example speculative_lookup -- <model.gguf> [max_new]
//!
//! For exact parity with Ollama numerics, set the matching backend, e.g.:
//!   GGML_BACKEND_PATH=/usr/local/lib/ollama/libggml-cpu-alderlake.so \
//!     cargo run --release --example speculative_lookup -- <model.gguf>

use std::sync::Arc;
use std::time::Instant;

use mullama::prompt_lookup::{
    generate_greedy, generate_greedy_baseline, PromptLookupConfig,
};
use mullama::{Context, ContextParams, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [max_new]", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(160);

    mullama::backend_init();
    let model = Arc::new(Model::load(model_path)?);

    // A prompt that induces verbatim repetition (where prompt-lookup shines):
    // a long, self-similar block the continuation is likely to copy. Overridable
    // via $SPEC_PROMPT for ad-hoc experiments.
    let default_prompt = "Repeat the following JSON array exactly, then continue it \
        with three more identical-shaped entries. \
        [{\"file\":\"src/main.rs\",\"role\":\"entry\"},\
        {\"file\":\"src/lib.rs\",\"role\":\"library\"},\
        {\"file\":\"src/context.rs\",\"role\":\"context\"},\
        {\"file\":\"src/model.rs\",\"role\":\"model\"},\
        {\"file\":\"src/batch.rs\",\"role\":\"batch\"}]";
    let prompt = std::env::var("SPEC_PROMPT").unwrap_or_else(|_| default_prompt.to_string());
    let prompt_tokens = model.tokenize(&prompt, true, false)?;
    println!("prompt tokens: {}", prompt_tokens.len());

    let ctx_params = ContextParams {
        n_ctx: 4096,
        n_batch: 512,
        ..Default::default()
    };

    // --- Baseline: plain greedy, one token per forward pass ---
    let mut ctx_base = Context::new(model.clone(), ctx_params.clone())?;
    ctx_base.kv_cache_clear();
    ctx_base.decode(&prompt_tokens)?;
    let t0 = Instant::now();
    let baseline = generate_greedy_baseline(&mut ctx_base, &model, &prompt_tokens, max_new)?;
    let base_ms = t0.elapsed().as_secs_f64() * 1e3;

    // --- Speculative: prompt-lookup n-gram drafting + batched verification ---
    let cfg = PromptLookupConfig { ngram: 2, max_draft: 10 };
    let mut ctx_spec = Context::new(model.clone(), ctx_params)?;
    ctx_spec.kv_cache_clear();
    ctx_spec.decode(&prompt_tokens)?;
    let t1 = Instant::now();
    let (spec, stats) = generate_greedy(&mut ctx_spec, &model, &prompt_tokens, max_new, &cfg)?;
    let spec_ms = t1.elapsed().as_secs_f64() * 1e3;

    // --- Parity check ---
    let parity = baseline == spec;
    println!("\n=== PARITY ===");
    println!("baseline tokens: {}", baseline.len());
    println!("spec     tokens: {}", spec.len());
    println!("token-identical: {}", if parity { "YES ✓" } else { "NO ✗" });
    if !parity {
        // Show first divergence for debugging.
        let n = baseline.len().min(spec.len());
        for i in 0..n {
            if baseline[i] != spec[i] {
                println!("  first divergence at {}: base={} spec={}", i, baseline[i], spec[i]);
                break;
            }
        }
    }

    println!("\n=== MEASUREMENT ===");
    println!("rounds (target passes): {}", stats.rounds);
    println!("drafted tokens:         {}", stats.drafted);
    println!("accepted drafts:        {}", stats.accepted);
    println!("draft misses:           {}", stats.draft_misses);
    println!("acceptance rate:        {:.1}%", stats.acceptance_rate() * 100.0);
    println!("tokens/forward-pass:    {:.2}", stats.tokens_per_pass());
    println!("baseline:  {:.1} ms ({:.1} tok/s)", base_ms, baseline.len() as f64 / (base_ms / 1e3));
    println!("spec:      {:.1} ms ({:.1} tok/s)", spec_ms, spec.len() as f64 / (spec_ms / 1e3));
    println!("wall-clock speedup:     {:.2}x", base_ms / spec_ms);

    let text = model
        .detokenize(&spec, false, false)
        .unwrap_or_else(|_| "<detok failed>".to_string());
    println!("\n=== OUTPUT (spec) ===\n{}", text);

    if !parity {
        std::process::exit(2);
    }
    Ok(())
}
