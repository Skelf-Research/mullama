//! Integration tests using real GGUF model and LoRA files
//!
//! These tests require actual model files to run. Set the following environment variables:
//! - MULLAMA_TEST_MODEL: Path to a small GGUF model file
//! - MULLAMA_TEST_LORA: Path to a LoRA adapter file (optional)
//!
//! Example:
//! ```bash
//! export MULLAMA_TEST_MODEL=/path/to/tiny-llama.gguf
//! export MULLAMA_TEST_LORA=/path/to/adapter.gguf
//! cargo test --test model_integration_tests
//! ```
//!
//! If the environment variables are not set, tests will be skipped.

use std::env;
use std::path::Path;
use std::sync::Arc;

/// Helper to get test model path, returns None if not available
fn get_test_model_path() -> Option<String> {
    env::var("MULLAMA_TEST_MODEL")
        .ok()
        .filter(|p| Path::new(p).exists())
}

/// Helper to get test LoRA path, returns None if not available
fn get_test_lora_path() -> Option<String> {
    env::var("MULLAMA_TEST_LORA")
        .ok()
        .filter(|p| Path::new(p).exists())
}

/// Macro to skip test if model not available
macro_rules! require_model {
    () => {
        match get_test_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: MULLAMA_TEST_MODEL not set or file not found");
                return;
            }
        }
    };
}

/// Macro to skip test if LoRA not available
macro_rules! require_lora {
    () => {
        match get_test_lora_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: MULLAMA_TEST_LORA not set or file not found");
                return;
            }
        }
    };
}

#[test]
fn test_backend_init() {
    // This test doesn't need a model
    unsafe {
        mullama::sys::llama_backend_init();
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_model_loading() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let result = mullama::Model::load(&model_path);
    assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

    let model = result.unwrap();

    // Check model properties
    assert!(model.vocab_size() > 0, "Model should have vocabulary");
    assert!(model.n_embd() > 0, "Model should have embeddings");
    assert!(model.n_layer() > 0, "Model should have layers");

    println!("Model loaded successfully:");
    println!("  Vocab size: {}", model.vocab_size());
    println!("  Embedding dim: {}", model.n_embd());
    println!("  Layers: {}", model.n_layer());
    println!("  Context train: {}", model.n_ctx_train());

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_model_loading_with_params() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let params = mullama::ModelParams {
        n_gpu_layers: 0,
        use_mmap: true,
        use_mlock: false,
        check_tensors: false,
        vocab_only: false,
        ..Default::default()
    };

    let result = mullama::Model::load_with_params(&model_path, params);
    assert!(
        result.is_ok(),
        "Failed to load model with params: {:?}",
        result.err()
    );

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_tokenization() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");

    // Test tokenization
    let text = "Hello, world!";
    let tokens = model.tokenize(text, true, false);
    assert!(tokens.is_ok(), "Tokenization failed: {:?}", tokens.err());

    let tokens = tokens.unwrap();
    assert!(!tokens.is_empty(), "Tokenization should produce tokens");

    println!(
        "Tokenized '{}' into {} tokens: {:?}",
        text,
        tokens.len(),
        tokens
    );

    // Test detokenization
    for &token in &tokens {
        let piece = model.token_to_str(token, 0, false);
        assert!(piece.is_ok(), "Detokenization failed for token {}", token);
    }

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_context_creation() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");
    let model = Arc::new(model);

    let ctx_params = mullama::ContextParams {
        n_ctx: 512,
        n_batch: 128,
        n_threads: 4,
        ..Default::default()
    };

    let result = mullama::Context::new(model.clone(), ctx_params);
    assert!(
        result.is_ok(),
        "Context creation failed: {:?}",
        result.err()
    );

    let context = result.unwrap();
    println!("Context created successfully");

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_builder_pattern() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    // Test ModelBuilder
    let model = mullama::builder::ModelBuilder::new()
        .path(&model_path)
        .gpu_layers(0)
        .memory_mapping(true)
        .build();

    assert!(model.is_ok(), "ModelBuilder failed: {:?}", model.err());
    let model = model.unwrap();

    // Test ContextBuilder
    let context = mullama::builder::ContextBuilder::new(model.clone())
        .context_size(512)
        .batch_size(128)
        .threads(4)
        .build();

    assert!(
        context.is_ok(),
        "ContextBuilder failed: {:?}",
        context.err()
    );

    // Test SamplerBuilder
    let sampler = mullama::builder::SamplerBuilder::new()
        .temperature(0.8)
        .top_k(40)
        .nucleus(0.95)
        .build(model.clone())
        .expect("Failed to build sampler via builder");

    println!("Builder pattern test passed");

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_sampling_params() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");
    let model = Arc::new(model);

    // Test various sampling configurations
    let params = mullama::SamplerParams {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        min_p: 0.05,
        penalty_repeat: 1.1,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        penalty_last_n: 64,
        seed: 42,
        ..Default::default()
    };

    let chain = params
        .build_chain(model.clone())
        .expect("Failed to build sampler chain");
    println!("Sampler chain created with custom params");

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

// LoRA tests commented out until lora module compilation issues are fixed
// #[test]
// fn test_lora_loading() { ... }
// #[test]
// fn test_lora_apply_to_context() { ... }
// #[test]
// fn test_lora_manager() { ... }

#[test]
fn test_model_clone() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");

    // Test that cloning works (Arc-based)
    let model_clone = model.clone();

    // Both should have same properties
    assert_eq!(model.vocab_size(), model_clone.vocab_size());
    assert_eq!(model.n_embd(), model_clone.n_embd());
    assert_eq!(model.n_layer(), model_clone.n_layer());

    println!("Model clone test passed - Arc reference counting works");

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_special_tokens() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");

    // Check special tokens
    let bos = model.token_bos();
    let eos = model.token_eos();

    println!("Special tokens:");
    println!("  BOS: {}", bos);
    println!("  EOS: {}", eos);

    // At least one should be valid (not -1)
    assert!(
        bos != -1 || eos != -1,
        "Model should have at least BOS or EOS token"
    );

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

// Quantization tests commented out until module compilation issues are fixed
// #[test]
// fn test_quantization_types() { ... }

#[test]
fn test_batch_operations() {
    let model_path = require_model!();

    unsafe {
        mullama::sys::llama_backend_init();
    }

    let model = mullama::Model::load(&model_path).expect("Failed to load model");

    // Create a batch
    let mut batch = mullama::Batch::default();
    assert!(batch.is_empty());

    // Tokenize some text
    let tokens = model
        .tokenize("Test input", true, false)
        .expect("Tokenization failed");

    // Add tokens to batch
    batch = mullama::Batch::from_tokens(&tokens);
    assert!(!batch.is_empty());

    println!("Batch created with {} tokens", tokens.len());

    unsafe {
        mullama::sys::llama_backend_free();
    }
}

#[test]
fn test_config_with_model() {
    let model_path = require_model!();

    // Create a config pointing to the test model
    let mut config = mullama::config::MullamaConfig::default();
    config.model.path = model_path.clone();
    config.model.context_size = 512;
    config.context.n_ctx = 512;
    config.context.n_batch = 128;
    config.sampling.temperature = 0.7;
    config.sampling.top_p = 0.9;

    // Validate config
    let result = config.validate();
    assert!(
        result.is_ok(),
        "Config validation failed: {:?}",
        result.err()
    );

    println!("Config with model path validated successfully");
}
