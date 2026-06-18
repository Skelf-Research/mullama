//! # Async Text Generation Example
//!
//! This example demonstrates async/await support in Mullama for non-blocking
//! text generation operations.
//!
//! Run with: cargo run --example async_generation --features async

use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🚀 Async Text Generation Example");
    println!("=================================");

    #[cfg(feature = "async")]
    {
        // Load model asynchronously without blocking
        println!("📂 Loading model asynchronously...");

        // Note: Replace with actual model path
        let _model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "path/to/model.gguf".to_string());

        // This would load the model in a real scenario:
        // let model = AsyncModel::load(&model_path).await?;
        // println!("✅ Model loaded successfully");

        // Get model information asynchronously
        // let info = model.info_async().await;
        // println!("📊 Model Info:");
        // println!("   Vocabulary size: {}", info.vocab_size);
        // println!("   Context size: {}", info.n_ctx_train);
        // println!("   Embedding size: {}", info.n_embd);
        // println!("   Layers: {}", info.n_layer);

        // Create context with custom parameters
        let _context_params = ContextParams {
            n_ctx: 2048,
            n_batch: 512,
            n_threads: 8,
            embeddings: false,
            flash_attn_type: mullama::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED,
            ..Default::default()
        };

        // let context = model.create_context_async(context_params).await?;
        // println!("🧠 Context created with {} tokens capacity", 2048);

        // Generate text asynchronously
        let prompts = vec![
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important aspect of machine learning is",
        ];

        // Process multiple prompts concurrently
        let mut handles = Vec::new();

        for (i, prompt) in prompts.iter().enumerate() {
            // In a real scenario:
            // let model_clone = model.clone();
            let prompt = prompt.to_string();

            let handle = tokio::spawn(async move {
                println!("🤖 Task {}: Starting generation for: \"{}\"", i + 1, prompt);

                // Simulate async generation
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // In real scenario:
                // let result = model_clone.generate_async(&prompt_clone, 100).await?;
                let result = format!("Generated text for prompt: {}", prompt);

                println!("✅ Task {}: Completed", i + 1);
                Ok::<String, MullamaError>(result)
            });

            handles.push(handle);
        }

        // Wait for all generations to complete
        println!("⏳ Waiting for all generations to complete...");
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(result)) => {
                    println!("📝 Result {}: {}", i + 1, result);
                }
                Ok(Err(e)) => {
                    eprintln!("❌ Task {} failed: {}", i + 1, e);
                }
                Err(e) => {
                    eprintln!("❌ Task {} panicked: {}", i + 1, e);
                }
            }
        }

        // Demonstrate advanced async patterns
        demonstrate_async_patterns().await?;
        demonstrate_error_handling().await?;
    }

    #[cfg(not(feature = "async"))]
    {
        println!("❌ This example requires the 'async' feature to be enabled");
        println!("Run with: cargo run --example async_generation --features async");
    }

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_async_patterns() -> Result<(), MullamaError> {
    println!("\n🔄 Advanced Async Patterns");
    println!("==========================");

    // Pattern 1: Async model loading with custom parameters
    println!("1️⃣ Custom model loading...");
    let _model_params = ModelParams {
        n_gpu_layers: 32,
        use_mmap: true,
        use_mlock: false,
        check_tensors: true,
        vocab_only: false,
        ..Default::default()
    };

    // In real scenario:
    // let model = AsyncModel::load_with_params("model.gguf", model_params).await?;
    println!("   ✅ Model would be loaded with custom parameters");

    // Pattern 2: Concurrent context creation
    println!("2️⃣ Concurrent context creation...");
    let _context_configs = vec![
        ContextParams {
            n_ctx: 1024,
            n_batch: 256,
            ..Default::default()
        },
        ContextParams {
            n_ctx: 2048,
            n_batch: 512,
            ..Default::default()
        },
        ContextParams {
            n_ctx: 4096,
            n_batch: 1024,
            ..Default::default()
        },
    ];

    // In real scenario, create multiple contexts concurrently:
    // let contexts = futures::future::try_join_all(
    //     context_configs.into_iter().map(|params| {
    //         model.create_context_async(params)
    //     })
    // ).await?;

    println!("   ✅ Multiple contexts would be created concurrently");

    // Pattern 3: Async generation with timeout
    println!("3️⃣ Generation with timeout...");
    let generation_task = async {
        // Simulate generation
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok::<String, MullamaError>("Generated text".to_string())
    };

    match tokio::time::timeout(tokio::time::Duration::from_millis(100), generation_task).await {
        Ok(Ok(result)) => println!("   ✅ Generation completed: {}", result),
        Ok(Err(e)) => println!("   ❌ Generation failed: {}", e),
        Err(_) => println!("   ⏰ Generation timed out"),
    }

    // Pattern 4: Batch processing with async
    println!("4️⃣ Batch processing...");
    let batch_prompts = vec!["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"];

    // Process in batches of 2
    for chunk in batch_prompts.chunks(2) {
        let batch_futures = chunk.iter().map(|prompt| async move {
            // Simulate processing
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            format!("Processed: {}", prompt)
        });

        let results = futures::future::join_all(batch_futures).await;
        println!("   📦 Batch processed: {:?}", results);
    }

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_error_handling() -> Result<(), MullamaError> {
    println!("\n🛡️ Async Error Handling");
    println!("=======================");

    // Graceful error handling in async context
    let result = async {
        // Simulate potential failure
        if rand::random::<bool>() {
            Err(MullamaError::ModelLoadError("Simulated error".to_string()))
        } else {
            Ok("Success".to_string())
        }
    }
    .await;

    match result {
        Ok(value) => println!("✅ Operation succeeded: {}", value),
        Err(e) => println!("❌ Operation failed gracefully: {}", e),
    }

    Ok(())
}
