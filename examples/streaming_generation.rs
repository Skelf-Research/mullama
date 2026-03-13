//! # Streaming Text Generation Example
//!
//! This example demonstrates real-time token streaming capabilities in Mullama,
//! allowing tokens to be processed as they are generated.
//!
//! Run with: cargo run --example streaming_generation --features streaming,async

use mullama::prelude::*;

#[cfg(all(feature = "streaming", feature = "async"))]
use futures::StreamExt;
#[cfg(all(feature = "streaming", feature = "async"))]
use mullama::{AsyncModel, StreamConfig, TokenData, TokenStream};
#[cfg(all(feature = "streaming", feature = "async"))]
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🌊 Streaming Text Generation Example");
    println!("====================================");

    #[cfg(all(feature = "streaming", feature = "async"))]
    {
        // Load model for streaming
        println!("📂 Loading model for streaming...");

        // Note: Replace with actual model path
        let model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "path/to/model.gguf".to_string());

        // In real scenario:
        // let model = AsyncModel::load(&model_path).await?;
        // println!("✅ Model loaded successfully");

        // Demonstrate different streaming configurations
        demonstrate_basic_streaming().await?;
        demonstrate_configured_streaming().await?;
        demonstrate_word_streaming().await?;
        demonstrate_text_only_streaming().await?;
        demonstrate_streaming_utilities().await?;
    }

    #[cfg(not(all(feature = "streaming", feature = "async")))]
    {
        println!("❌ This example requires both 'streaming' and 'async' features");
        println!("Run with: cargo run --example streaming_generation --features streaming,async");
    }

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_basic_streaming() -> Result<(), MullamaError> {
    println!("\n🎬 Basic Streaming");
    println!("==================");

    // Note: In real scenario, use actual model
    // let model = AsyncModel::load("model.gguf").await?;

    let config = StreamConfig::default()
        .max_tokens(50)
        .temperature(0.8)
        .include_probabilities(true);

    let prompt = "The future of artificial intelligence";
    println!("📝 Prompt: \"{}\"", prompt);
    println!(
        "⚙️  Config: {} tokens, temp={}",
        config.max_tokens, config.sampler_params.temperature
    );

    // Create token stream (placeholder implementation)
    // let mut stream = TokenStream::new(model, prompt, config).await?;

    println!("🎭 Streaming tokens (simulated):");
    print!("   ");

    // Simulate streaming
    let tokens = vec![
        "is",
        " to",
        " enhance",
        " human",
        " capabilities",
        " and",
        " solve",
        " complex",
        " problems",
    ];
    for (i, token_text) in tokens.iter().enumerate() {
        // Simulate token arrival
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let token_data = TokenData {
            token_id: i as u32 + 1000,
            text: token_text.to_string(),
            position: i,
            is_final: i == tokens.len() - 1,
            probability: Some(0.8 + (i as f32 * 0.02)),
        };

        print!("{}", token_data.text);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        if token_data.is_final {
            println!("\n🏁 Generation complete!");
            break;
        }
    }

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_configured_streaming() -> Result<(), MullamaError> {
    println!("\n⚙️ Configured Streaming");
    println!("=======================");

    // Configure streaming with specific parameters
    let configs = vec![
        (
            "Creative",
            StreamConfig::default()
                .temperature(0.9)
                .top_k(60)
                .max_tokens(30),
        ),
        (
            "Precise",
            StreamConfig::default()
                .temperature(0.2)
                .top_k(10)
                .max_tokens(30),
        ),
        (
            "Balanced",
            StreamConfig::default()
                .temperature(0.7)
                .top_k(40)
                .max_tokens(30),
        ),
    ];

    let prompt = "Once upon a time in a land far away";

    for (name, config) in configs {
        println!("\n🎯 {} configuration:", name);
        println!("   Temperature: {}", config.sampler_params.temperature);
        println!("   Top-k: {}", config.sampler_params.top_k);

        // In real scenario:
        // let model = AsyncModel::load("model.gguf").await?;
        // let mut stream = TokenStream::new(model, prompt, config).await?;

        print!("   Output: ");

        // Simulate different outputs based on temperature
        let output = match name {
            "Creative" => "there lived a magnificent dragon who collected shimmering",
            "Precise" => "there was a small village with a market square",
            "Balanced" => "there lived a young princess who loved to read",
            _ => "sample output",
        };

        for char in output.chars() {
            print!("{}", char);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        println!();
    }

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_word_streaming() -> Result<(), MullamaError> {
    println!("\n📝 Word-Based Streaming");
    println!("=======================");

    // Word streaming buffers tokens until word boundaries
    let prompt = "Explain the concept of machine learning";
    println!("📝 Prompt: \"{}\"", prompt);

    // In real scenario:
    // let model = AsyncModel::load("model.gguf").await?;
    // let config = StreamConfig::default().max_tokens(100);
    // let mut word_stream = TokenStream::word_stream(model, prompt, config).await?;

    println!("🔤 Word-by-word output:");

    // Simulate word streaming
    let words = vec![
        "Machine",
        "learning",
        "is",
        "a",
        "subset",
        "of",
        "artificial",
        "intelligence",
        "that",
        "enables",
        "computers",
        "to",
        "learn",
        "and",
        "improve",
        "automatically",
    ];

    for word in words {
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        print!("{} ", word);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    println!("\n✅ Word streaming complete");

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_text_only_streaming() -> Result<(), MullamaError> {
    println!("\n📄 Text-Only Streaming");
    println!("======================");

    // Text-only streaming for simple use cases
    let prompt = "The benefits of renewable energy include";
    println!("📝 Prompt: \"{}\"", prompt);

    // In real scenario:
    // let model = AsyncModel::load("model.gguf").await?;
    // let config = StreamConfig::default().max_tokens(80);
    // let mut text_stream = TokenStream::text_only(model, prompt, config).await?;

    println!("📜 Text-only output:");
    print!("   ");

    // Simulate text streaming
    let text_parts = vec![
        "reduced",
        " carbon",
        " emissions,",
        " energy",
        " independence,",
        " job",
        " creation,",
        " and",
        " long-term",
        " cost",
        " savings",
        " for",
        " consumers.",
    ];

    for part in text_parts {
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        print!("{}", part);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    println!("\n✅ Text streaming complete");

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_streaming_utilities() -> Result<(), MullamaError> {
    println!("\n🛠️ Streaming Utilities");
    println!("======================");

    // Demonstrate utility functions for streaming
    use mullama::streaming::utils;

    // In real scenario, you would have an actual stream:
    // let model = AsyncModel::load("model.gguf").await?;
    // let config = StreamConfig::default().max_tokens(50);
    // let stream = TokenStream::new(model, "Hello world", config).await?;

    // 1. Collect stream to string
    println!("1️⃣ Collect to string:");
    // let complete_text = utils::collect_to_string(stream).await?;
    let complete_text = "This would be the complete generated text";
    println!("   Result: \"{}\"", complete_text);

    // 2. Collect with metadata
    println!("\n2️⃣ Collect with metadata:");
    // let generation_result = utils::collect_with_metadata(stream).await?;
    let generation_result = utils::GenerationResult {
        text: "Generated text with metadata".to_string(),
        token_count: 25,
        tokens: vec![1, 2, 3, 4, 5],
    };

    println!("   Text: \"{}\"", generation_result.text);
    println!("   Token count: {}", generation_result.token_count);
    println!(
        "   First 5 tokens: {:?}",
        &generation_result.tokens[..5.min(generation_result.tokens.len())]
    );

    // 3. Performance monitoring
    println!("\n3️⃣ Performance monitoring:");
    let start_time = Instant::now();

    // Simulate generation with timing
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let duration = start_time.elapsed();
    let tokens_per_second = 50.0 / duration.as_secs_f64();

    println!("   Generation time: {:.2}s", duration.as_secs_f64());
    println!("   Tokens per second: {:.2}", tokens_per_second);

    Ok(())
}

#[cfg(all(feature = "streaming", feature = "async"))]
async fn demonstrate_error_handling() -> Result<(), MullamaError> {
    println!("\n🛡️ Stream Error Handling");
    println!("========================");

    // Demonstrate robust error handling in streams
    // In real scenario:
    // let model = AsyncModel::load("model.gguf").await?;
    // let config = StreamConfig::default();
    // let mut stream = TokenStream::new(model, "Test prompt", config).await?;

    let mut error_count = 0;
    let mut successful_tokens = 0;

    // Simulate stream with occasional errors
    for i in 0..10 {
        // Simulate random errors
        if i == 3 || i == 7 {
            error_count += 1;
            println!("❌ Stream error at position {}: Simulated network issue", i);
            continue;
        }

        successful_tokens += 1;
        println!("✅ Token {}: Generated successfully", i);
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!("\n📊 Error handling summary:");
    println!("   Successful tokens: {}", successful_tokens);
    println!("   Errors encountered: {}", error_count);
    println!(
        "   Success rate: {:.1}%",
        (successful_tokens as f64 / 10.0) * 100.0
    );

    Ok(())
}
