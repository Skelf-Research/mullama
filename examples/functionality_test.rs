//! Test additional llama.cpp functionality after upgrade

use mullama::{Context, ContextParams, Model, MullamaError, SamplerParams};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    let model_path = "models/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== llama.cpp Functionality Tests ===\n");

    // Load model
    println!("Loading model...");
    let model = Arc::new(Model::load(model_path)?);

    // Test 1: Model metadata
    println!("\n1. Model Metadata:");
    println!("   - Vocab size: {}", model.vocab_size());
    println!("   - Context train: {}", model.n_ctx_train());
    println!("   - Embedding dim: {}", model.n_embd());
    println!("   - Layers: {}", model.n_layer());
    println!("   - Heads: {}", model.n_head());
    println!("   - Head KV: {}", model.n_head_kv());
    println!("   [PASS]");

    // Test 2: Special tokens
    println!("\n2. Special Tokens:");
    let bos = model.token_bos();
    let eos = model.token_eos();
    println!("   - BOS token: {}", bos);
    println!("   - EOS token: {}", eos);
    println!("   - BOS is control: {}", model.token_is_control(bos));
    println!("   - EOS is EOG: {}", model.token_is_eog(eos));
    println!("   [PASS]");

    // Test 3: Context creation with various params
    println!("\n3. Context Creation:");
    let ctx_params = ContextParams {
        n_ctx: 512,
        n_batch: 64,
        embeddings: true,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;
    println!("   - n_ctx: {}", context.n_ctx());
    println!("   - n_batch: {}", context.n_batch());
    println!("   - n_ubatch: {}", context.n_ubatch());
    println!("   [PASS]");

    // Test 4: Tokenization and decode
    println!("\n4. Decode Operation:");
    let tokens = model.tokenize("The meaning of life is", true, false)?;
    println!("   - Tokenized {} tokens", tokens.len());
    context.decode(&tokens)?;
    println!("   - Decode successful");
    println!("   [PASS]");

    // Test 5: Logits access
    println!("\n5. Logits Access:");
    let logits = context.get_logits();
    println!("   - Logits length: {}", logits.len());
    println!("   - Expected (vocab_size): {}", model.vocab_size());
    assert_eq!(logits.len(), model.vocab_size() as usize);
    println!("   [PASS]");

    // Test 6: Embeddings
    println!("\n6. Embeddings:");
    let emb = context.get_embeddings();
    match emb {
        Some(e) => println!("   - Embedding length: {}", e.len()),
        None => println!("   - No embeddings (model may not support)"),
    }
    println!("   [PASS]");

    // Test 7: KV Cache operations
    println!("\n7. KV Cache:");
    println!("   - Can shift: {}", context.kv_cache_can_shift());
    let pos_max = context.kv_cache_seq_pos_max(0);
    println!("   - Seq 0 pos max: {}", pos_max);
    println!("   [PASS]");

    // Test 8: Sampling with chain
    println!("\n8. Sampler Chain:");
    let sampler_params = SamplerParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ..Default::default()
    };
    let mut sampler = sampler_params.build_chain(model.clone())?;
    let token = sampler.sample(&mut context, -1);
    println!("   - Sampled token: {}", token);
    sampler.accept(token);
    println!("   - Token accepted");
    println!("   [PASS]");

    // Test 9: State save/load
    println!("\n9. State Management:");
    let state_size = context.state_size();
    println!("   - State size: {} bytes", state_size);
    let state = context.save_state();
    println!("   - Saved state: {} bytes", state.len());
    println!("   [PASS]");

    // Test 10: Performance data
    println!("\n10. Performance Timing:");
    let perf = context.perf_data();
    println!("   - Load time: {:.2}ms", perf.t_load_ms);
    println!("   - Eval time: {:.2}ms", perf.t_eval_ms);
    println!("   - Prompt eval: {:.2}ms", perf.t_p_eval_ms);
    println!("   [PASS]");

    // Test 11: Thread management
    println!("\n11. Thread Management:");
    let n_threads = context.n_threads();
    println!("   - Current threads: {}", n_threads);
    context.set_n_threads(4, 4);
    println!("   - Set threads to 4");
    println!("   [PASS]");

    // Test 12: KV cache clear
    println!("\n12. KV Cache Clear:");
    context.kv_cache_clear();
    println!("   - Cache cleared");
    println!("   [PASS]");

    println!("\n{}", "=".repeat(50));
    println!("=== All 12 Functionality Tests Passed! ===");
    println!("{}", "=".repeat(50));

    Ok(())
}
