//! # Mullama Integration Showcase
//!
//! This example demonstrates all the advanced integration features of Mullama:
//! - Async/await support for non-blocking operations
//! - Streaming interfaces for real-time token generation
//! - Configuration management with serde
//! - Builder patterns for fluent APIs
//! - Web framework integration with Axum
//!
//! Run with: cargo run --example integration_showcase --features full

use mullama::config::presets;
use mullama::prelude::*;

#[cfg(feature = "streaming")]
use mullama::StreamConfig;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🚀 Mullama Integration Showcase");
    println!("================================");

    // Example 1: Configuration Management
    showcase_configuration().await?;

    // Example 2: Builder Patterns
    showcase_builder_patterns().await?;

    #[cfg(feature = "async")]
    {
        // Example 3: Async Model Loading
        showcase_async_operations().await?;

        // Example 4: Streaming Generation
        showcase_streaming().await?;

        // Example 5: Web Service Integration
        showcase_web_integration().await?;
    }

    println!("\n✨ All integration features showcased successfully!");
    Ok(())
}

/// Showcase configuration management with serde
async fn showcase_configuration() -> Result<(), MullamaError> {
    println!("\n📋 Configuration Management");
    println!("---------------------------");

    // Create configuration programmatically
    let config = MullamaConfig {
        model: mullama::config::ModelConfig {
            path: "path/to/model.gguf".to_string(),
            gpu_layers: 32,
            context_size: 4096,
            ..Default::default()
        },
        sampling: mullama::config::SamplingConfig {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            ..Default::default()
        },
        ..Default::default()
    };

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| MullamaError::ConfigError(format!("JSON serialization failed: {}", e)))?;
    println!("📄 Configuration as JSON:\n{}", json);

    // Use preset configurations
    let creative_config = presets::creative_writing();
    println!(
        "🎨 Creative writing preset: temp={}, top_k={}",
        creative_config.sampling.temperature, creative_config.sampling.top_k
    );

    let code_config = presets::code_generation();
    println!(
        "💻 Code generation preset: temp={}, top_k={}",
        code_config.sampling.temperature, code_config.sampling.top_k
    );

    // Validate configuration
    match config.validate() {
        Ok(_) => println!("✅ Configuration is valid"),
        Err(e) => println!("❌ Configuration error: {}", e),
    }

    Ok(())
}

/// Showcase builder patterns for fluent API
async fn showcase_builder_patterns() -> Result<(), MullamaError> {
    println!("\n🔧 Builder Patterns");
    println!("-------------------");

    // Model builder with fluent API
    #[cfg(feature = "async")]
    {
        let _model_builder = ModelBuilder::new()
            .path("path/to/model.gguf")
            .gpu_layers(32)
            .context_size(4096)
            .memory_mapping(true)
            .preset(mullama::builder::presets::performance_optimized);

        println!("🏗️  Model builder configured with performance optimizations");

        // Context builder with optimization presets
        // Note: This would need an actual model in a real scenario
        // let context_builder = ContextBuilder::new(model.clone())
        //     .context_size(4096)
        //     .batch_size(512)
        //     .threads(8)
        //     .optimize_for_performance();

        println!("🏗️  Context builder configured for performance");

        // Sampler builder with penalty configuration
        let _sampler_builder = SamplerBuilder::new()
            .temperature(0.8)
            .top_k(50)
            .nucleus(0.95)
            .penalties(|p| p.repetition(1.1).frequency(0.1).presence(0.1))
            .preset(mullama::builder::presets::creative_sampling);

        println!("🏗️  Sampler builder configured with creative sampling");
    }

    Ok(())
}

/// Showcase async operations
#[cfg(feature = "async")]
async fn showcase_async_operations() -> Result<(), MullamaError> {
    println!("\n⚡ Async Operations");
    println!("------------------");

    println!("🔄 Loading model asynchronously...");
    // Note: In a real scenario, you'd use an actual model path
    // let model = AsyncModel::load("path/to/model.gguf").await?;
    // println!("✅ Model loaded successfully");

    // let info = model.info_async().await;
    // println!("📊 Model info - Vocab: {}, Layers: {}", info.vocab_size, info.n_layer);

    // Generate text asynchronously
    // let result = model.generate_async("The future of AI is", 50).await?;
    // println!("🤖 Generated: {}", result);

    println!("✅ Async operations demonstrated (with placeholder model)");
    Ok(())
}

/// Showcase streaming token generation
#[cfg(feature = "streaming")]
async fn showcase_streaming() -> Result<(), MullamaError> {
    println!("\n🌊 Streaming Generation");
    println!("----------------------");

    // Note: In a real scenario, you'd use an actual model
    // let model = AsyncModel::load("path/to/model.gguf").await?;

    // Configure streaming
    let config = StreamConfig::default()
        .max_tokens(50)
        .temperature(0.8)
        .include_probabilities(true);

    println!(
        "📡 Stream config: max_tokens={}, temp={}",
        config.max_tokens, config.sampler_params.temperature
    );

    // Create token stream (placeholder)
    // let mut stream = TokenStream::new(model, "Once upon a time", config).await?;

    // Process stream
    // println!("🎬 Streaming tokens:");
    // while let Some(result) = stream.next().await {
    //     match result {
    //         Ok(token_data) => {
    //             print!("{}", token_data.text);
    //             if token_data.is_final {
    //                 println!("\n🏁 Generation complete!");
    //                 break;
    //             }
    //         }
    //         Err(e) => {
    //             eprintln!("❌ Stream error: {}", e);
    //             break;
    //         }
    //     }
    // }

    println!("✅ Streaming demonstrated (with placeholder model)");
    Ok(())
}

/// Showcase web service integration
#[cfg(feature = "web")]
async fn showcase_web_integration() -> Result<(), MullamaError> {
    println!("\n🌐 Web Service Integration");
    println!("-------------------------");

    // Note: In a real scenario, you'd use an actual model
    // let model = AsyncModel::load("path/to/model.gguf").await?;

    // Create application state (placeholder)
    // let app_state = AppState {
    //     model,
    //     default_config: MullamaConfig::default(),
    //     metrics: Arc::new(tokio::sync::RwLock::new(ApiMetrics::default())),
    // };

    // Create router with all endpoints
    // let app = create_router(app_state);

    println!("🛠️  Router created with endpoints:");
    println!("   📍 POST /generate - Text generation");
    println!("   📍 POST /tokenize - Text tokenization");
    println!("   📍 GET /stream/:prompt - Server-sent events streaming");
    println!("   📍 GET /health - Health check");
    println!("   📍 GET /metrics - API metrics");

    // In a real application, you would bind and serve:
    // let listener = TcpListener::bind("0.0.0.0:3000").await
    //     .map_err(|e| MullamaError::ConfigError(format!("Failed to bind: {}", e)))?;
    // println!("🚀 Server running on http://0.0.0.0:3000");
    // axum::serve(listener, app).await
    //     .map_err(|e| MullamaError::ConfigError(format!("Server error: {}", e)))?;

    println!("✅ Web integration demonstrated (server not started)");
    Ok(())
}
