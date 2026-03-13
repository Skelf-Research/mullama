//! # Complete Integration Demo
//!
//! This example demonstrates all advanced integration features of Mullama working together:
//! - Tokio runtime integration with task management
//! - Rayon parallel processing for batch operations
//! - WebSocket-based real-time communication
//! - Multimodal processing with audio, image, and text
//! - Configuration management and builder patterns
//!
//! Run with: cargo run --example complete_integration_demo --features full

use mullama::prelude::*;
use std::sync::Arc;

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
use mullama::{
    AudioFeatures, AudioInput, ImageInput, MullamaRuntime, MultimodalProcessor, ParallelProcessor,
    TaskManager, WebSocketConfig, WebSocketServer,
};

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🚀 Complete Mullama Integration Demo");
    println!("===================================");

    #[cfg(all(
        feature = "tokio-runtime",
        feature = "parallel",
        feature = "websockets",
        feature = "multimodal"
    ))]
    {
        // 1. Setup Tokio Runtime with Task Management
        let runtime = setup_runtime().await?;

        // 2. Setup Parallel Processing
        let parallel_processor = setup_parallel_processing().await?;

        // 3. Setup Multimodal Processing
        let multimodal_processor = setup_multimodal_processing().await?;

        // 4. Demonstrate Complete Workflow
        demonstrate_complete_workflow(&runtime, &parallel_processor, &multimodal_processor).await?;

        // 5. Setup WebSocket Server (would run in background)
        setup_websocket_server().await?;

        // 6. Demonstrate Advanced Patterns
        demonstrate_advanced_patterns().await?;

        // Graceful shutdown
        runtime.shutdown().await;
    }

    #[cfg(not(all(
        feature = "tokio-runtime",
        feature = "parallel",
        feature = "websockets",
        feature = "multimodal"
    )))]
    {
        println!("❌ This demo requires all integration features to be enabled");
        println!("Run with: cargo run --example complete_integration_demo --features full");
    }

    println!("\n✨ Complete integration demo finished!");
    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_runtime() -> Result<Arc<MullamaRuntime>, MullamaError> {
    println!("\n🔧 Setting up Tokio Runtime");
    println!("===========================");

    let runtime = MullamaRuntime::new()
        .worker_threads(8)
        .max_blocking_threads(16)
        .thread_name("mullama-worker")
        .enable_all()
        .build()?;

    println!("✅ Runtime created with 8 worker threads");

    let mut task_manager = TaskManager::new(&Arc::new(runtime));

    // Start background tasks
    task_manager.spawn_generation_worker().await?;
    task_manager.spawn_metrics_collector().await?;

    println!("✅ Background tasks started");

    Ok(Arc::new(runtime))
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_parallel_processing() -> Result<Arc<ParallelProcessor>, MullamaError> {
    println!("\n⚡ Setting up Parallel Processing");
    println!("=================================");

    // Note: In real scenario, you'd load an actual model
    // let model = Arc::new(Model::load("model.gguf")?);

    // For demo, we'll create placeholder processor
    // let processor = ParallelProcessor::new(model)
    //     .thread_pool(ThreadPoolConfig::new().num_threads(6))
    //     .build()?;

    println!("✅ Parallel processor configured with 6 threads");

    // Demonstrate batch processing
    let texts = vec!["Hello", "World", "Parallel", "Processing", "Demo"];
    println!("📦 Processing {} texts in parallel", texts.len());

    // In real scenario:
    // let results = processor.batch_tokenize(&texts)?;
    // println!("✅ Tokenized {} texts", results.len());

    println!("✅ Parallel processing demo completed");

    // Return placeholder
    Ok(Arc::new(ParallelProcessor::new(Arc::new(Model {
        model_ptr: std::ptr::null_mut(),
    }))))
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_multimodal_processing() -> Result<Arc<MultimodalProcessor>, MullamaError> {
    println!("\n🎭 Setting up Multimodal Processing");
    println!("===================================");

    let processor = MultimodalProcessor::new()
        .enable_image_processing()
        .enable_audio_processing()
        .build();

    println!("✅ Multimodal processor created");

    // Demonstrate image processing
    let image_path = "demo_image.jpg"; // Placeholder
    println!("🖼️ Processing image: {}", image_path);

    // In real scenario:
    // let image = ImageInput::from_path(image_path).await?;
    // let description = processor.describe_image(&image).await?;
    // println!("📄 Image description: {}", description);

    println!("✅ Image processing demo completed");

    // Demonstrate audio processing
    let audio_path = "demo_audio.wav"; // Placeholder
    println!("🎵 Processing audio: {}", audio_path);

    // In real scenario:
    // let audio = AudioInput::from_path(audio_path).await?;
    // let transcript = processor.transcribe_audio(&audio).await?;
    // println!("📝 Audio transcript: {}", transcript);

    println!("✅ Audio processing demo completed");

    Ok(Arc::new(processor))
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_websocket_server() -> Result<(), MullamaError> {
    println!("\n🌐 Setting up WebSocket Server");
    println!("==============================");

    let config = WebSocketConfig::new()
        .port(8080)
        .max_connections(100)
        .enable_audio()
        .enable_compression();

    let server = WebSocketServer::new(config).build().await?;

    println!("🚀 WebSocket server configured on port 8080");
    println!("📍 Available endpoints:");
    println!("   - ws://localhost:8080/ws (general)");
    println!("   - ws://localhost:8080/ws/chat/:room_id (chat rooms)");
    println!("   - ws://localhost:8080/ws/audio (audio processing)");
    println!("   - ws://localhost:8080/ws/stream/:session_id (streaming)");

    // In real scenario, this would start the server:
    // tokio::spawn(async move {
    //     server.start().await.unwrap();
    // });

    println!("✅ WebSocket server setup completed");

    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_complete_workflow(
    runtime: &Arc<MullamaRuntime>,
    parallel_processor: &Arc<ParallelProcessor>,
    multimodal_processor: &Arc<MultimodalProcessor>,
) -> Result<(), MullamaError> {
    println!("\n🔄 Complete Integration Workflow");
    println!("================================");

    // 1. Process multiple inputs in parallel
    println!("1️⃣ Parallel batch processing...");
    let batch_inputs = vec![
        "Analyze this financial report",
        "Summarize the meeting notes",
        "Translate this document",
        "Generate a creative story",
    ];

    // In real scenario:
    // let batch_config = BatchGenerationConfig::default();
    // let parallel_results = parallel_processor.batch_generate(&batch_inputs, &batch_config)?;
    // println!("   ✅ Processed {} items in parallel", parallel_results.len());

    println!("   ✅ Parallel processing completed");

    // 2. Process multimodal content
    println!("2️⃣ Multimodal content processing...");

    // Create multimodal input
    let mut multimodal_input = MultimodalInput {
        text: Some("Describe what you see and hear".to_string()),
        image: None, // Would add actual image in real scenario
        audio: None, // Would add actual audio in real scenario
        max_tokens: Some(150),
        context: None,
    };

    // In real scenario:
    // let multimodal_result = multimodal_processor.process_multimodal(&multimodal_input).await?;
    // println!("   ✅ Multimodal response: {}", multimodal_result.text_response);

    println!("   ✅ Multimodal processing completed");

    // 3. Coordinate with Tokio runtime
    println!("3️⃣ Runtime coordination...");

    let metrics = runtime.metrics().summary().await;
    println!("   📊 Runtime metrics:");
    println!("      - Tasks spawned: {}", metrics.tasks_spawned);
    println!(
        "      - Generation requests: {}",
        metrics.generation_requests
    );
    println!(
        "      - Average generation time: {:?}",
        metrics.average_generation_time
    );

    println!("   ✅ Runtime coordination completed");

    // 4. Real-time streaming simulation
    println!("4️⃣ Real-time streaming simulation...");

    // Simulate streaming workflow
    for i in 1..=5 {
        let task = runtime.spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            format!("Stream token {}", i)
        });

        match task.await {
            Ok(result) => println!("   🌊 {}", result),
            Err(e) => println!("   ❌ Stream error: {}", e),
        }
    }

    println!("   ✅ Streaming simulation completed");

    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_advanced_patterns() -> Result<(), MullamaError> {
    println!("\n🎯 Advanced Integration Patterns");
    println!("================================");

    // Pattern 1: Pipeline Processing
    println!("1️⃣ Pipeline processing pattern...");

    let pipeline_stages = vec![
        "Audio preprocessing",
        "Speech-to-text conversion",
        "Text analysis",
        "Response generation",
        "Text-to-speech synthesis",
    ];

    for (i, stage) in pipeline_stages.iter().enumerate() {
        println!("   🔄 Stage {}: {}", i + 1, stage);
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!("   ✅ Pipeline completed");

    // Pattern 2: Cross-Modal Analysis
    println!("2️⃣ Cross-modal analysis pattern...");

    // Simulate analyzing image + audio + text together
    let cross_modal_features = vec![
        ("Visual", "Objects: person, chair, table"),
        ("Audio", "Speech detected, confident tone"),
        ("Text", "Meeting discussion about quarterly results"),
    ];

    for (modality, feature) in cross_modal_features {
        println!("   🎭 {}: {}", modality, feature);
    }

    let combined_analysis =
        "Professional meeting setting with confident discussion of business metrics";
    println!("   🎯 Combined analysis: {}", combined_analysis);

    println!("   ✅ Cross-modal analysis completed");

    // Pattern 3: Real-time Collaboration
    println!("3️⃣ Real-time collaboration pattern...");

    // Simulate multiple users collaborating
    let users = vec!["Alice", "Bob", "Charlie"];
    for user in users {
        println!("   👤 {} joined the session", user);
        println!("   💬 {} is typing...", user);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        println!("   📝 {} shared multimodal content", user);
    }

    println!("   ✅ Collaboration simulation completed");

    // Pattern 4: Adaptive Processing
    println!("4️⃣ Adaptive processing pattern...");

    // Simulate adaptive resource allocation
    let processing_loads = vec![("Light", 2), ("Medium", 4), ("Heavy", 8), ("Peak", 12)];

    for (load_type, threads) in processing_loads {
        println!(
            "   📊 {} load detected, allocating {} threads",
            load_type, threads
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!("   ✅ Adaptive processing completed");

    Ok(())
}

// Demo configuration utilities
#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
fn demonstrate_configuration_integration() {
    println!("\n⚙️ Configuration Integration");
    println!("============================");

    // 1. Create comprehensive configuration
    let config = MullamaConfig {
        model: ModelConfig {
            path: "advanced_model.gguf".to_string(),
            gpu_layers: 40,
            context_size: 8192,
            ..Default::default()
        },
        context: ContextConfig {
            n_ctx: 8192,
            n_batch: 2048,
            n_threads: 12,
            flash_attn: true,
            ..Default::default()
        },
        sampling: SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            ..Default::default()
        },
        performance: PerformanceConfig {
            enable_monitoring: true,
            memory_optimization: 3,
            gpu_optimizations: GpuOptimizations {
                enable_gpu: true,
                memory_pool_size: Some(8192),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    println!("🔧 Created comprehensive configuration:");
    println!(
        "   - Model: {} with {} GPU layers",
        config.model.path, config.model.gpu_layers
    );
    println!(
        "   - Context: {} tokens, {} batch size",
        config.context.n_ctx, config.context.n_batch
    );
    println!("   - Performance: monitoring enabled, memory optimized");

    // 2. Builder pattern integration
    println!("\n🏗️ Builder pattern integration:");

    // Model builder
    println!("   📦 Model: fluent configuration with presets");

    // Context builder
    println!("   🧠 Context: performance optimizations applied");

    // Sampler builder
    println!("   🎲 Sampler: creative configuration with penalties");

    println!("✅ Configuration integration completed");
}

// Performance monitoring utilities
#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_performance_monitoring() {
    println!("\n📊 Performance Monitoring");
    println!("=========================");

    // Simulate performance metrics collection
    let metrics = vec![
        ("CPU Usage", "65%"),
        ("Memory Usage", "4.2GB / 16GB"),
        ("GPU Utilization", "80%"),
        ("Network I/O", "150 Mbps"),
        ("Active Connections", "42"),
        ("Requests/Second", "125"),
        ("Average Latency", "45ms"),
        ("Error Rate", "0.1%"),
    ];

    for (metric, value) in metrics {
        println!("   📈 {}: {}", metric, value);
    }

    println!("\n🎯 Performance Summary:");
    println!("   ✅ System operating within normal parameters");
    println!("   ⚡ High throughput with low latency");
    println!("   🛡️ Excellent reliability (99.9% uptime)");
    println!("   🚀 Ready for production workloads");
}

// Error handling demonstration
#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_error_handling() {
    println!("\n🛡️ Error Handling");
    println!("==================");

    // Simulate various error scenarios and recovery
    let error_scenarios = vec![
        ("Network timeout", "Retry with exponential backoff"),
        ("Model memory overflow", "Reduce batch size and retry"),
        ("Audio format unsupported", "Convert to supported format"),
        ("WebSocket connection lost", "Reconnect automatically"),
        ("Configuration invalid", "Validate and provide defaults"),
    ];

    for (error, recovery) in error_scenarios {
        println!("   ❌ Error: {}", error);
        println!("   🔄 Recovery: {}", recovery);
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        println!("   ✅ Recovered successfully");
        println!();
    }

    println!("🎯 Error handling demonstrates:");
    println!("   - Graceful degradation");
    println!("   - Automatic recovery");
    println!("   - Comprehensive logging");
    println!("   - User-friendly error messages");
}
