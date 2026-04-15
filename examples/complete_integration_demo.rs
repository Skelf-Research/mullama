//! Complete integration demo for the current high-level feature surface.
//!
//! This example intentionally avoids constructing placeholder models. It demonstrates
//! runtime coordination, parallel-work orchestration, multimodal input shaping, and
//! WebSocket setup without requiring model files.
//!
//! Run with: cargo run --example complete_integration_demo --features full

use mullama::prelude::*;
use std::{collections::HashMap, sync::Arc};

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
use mullama::{
    MullamaRuntime, MultimodalInput, TaskManager, WebSocketConfig, WebSocketServer,
};

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("Complete Mullama Integration Demo");
    println!("=================================");

    #[cfg(all(
        feature = "tokio-runtime",
        feature = "parallel",
        feature = "websockets",
        feature = "multimodal"
    ))]
    {
        let runtime = setup_runtime().await?;

        demonstrate_parallel_planning().await?;
        demonstrate_multimodal_input_shape()?;
        demonstrate_complete_workflow(&runtime).await?;
        setup_websocket_server().await?;
        demonstrate_advanced_patterns().await?;
    }

    #[cfg(not(all(
        feature = "tokio-runtime",
        feature = "parallel",
        feature = "websockets",
        feature = "multimodal"
    )))]
    {
        println!("This demo requires the full feature set.");
        println!("Run with: cargo run --example complete_integration_demo --features full");
    }

    println!("\nComplete integration demo finished.");
    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_runtime() -> Result<Arc<MullamaRuntime>, MullamaError> {
    println!("\nSetting up Tokio runtime");

    let runtime = Arc::new(
        MullamaRuntime::new()
            .worker_threads(8)
            .max_blocking_threads(16)
            .thread_name("mullama-worker")
            .enable_all()
            .build()?,
    );

    let mut task_manager = TaskManager::new(&runtime);
    task_manager.spawn_generation_worker().await?;
    task_manager.spawn_metrics_collector().await?;

    println!("Runtime and background tasks are ready.");
    Ok(runtime)
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_parallel_planning() -> Result<(), MullamaError> {
    println!("\nPlanning parallel work");

    let inputs = ["summarize", "translate", "classify", "extract"];
    for (idx, input) in inputs.iter().enumerate() {
        println!("  shard {} -> {}", idx + 1, input);
    }

    println!("Parallel planning is ready; attach a loaded model to execute batches.");
    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
fn demonstrate_multimodal_input_shape() -> Result<(), MullamaError> {
    println!("\nBuilding multimodal input");

    let input = MultimodalInput {
        text: Some("Describe the provided media.".to_string()),
        images: Vec::new(),
        videos: Vec::new(),
        audio: Vec::new(),
        metadata: HashMap::from([("source".to_string(), "integration-demo".to_string())]),
    };

    println!(
        "Input prepared: text={}, images={}, videos={}, audio={}",
        input.text.is_some(),
        input.images.len(),
        input.videos.len(),
        input.audio.len()
    );

    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn setup_websocket_server() -> Result<(), MullamaError> {
    println!("\nConfiguring WebSocket server");

    let config = WebSocketConfig::new()
        .port(8080)
        .max_connections(100)
        .enable_audio()
        .enable_compression();

    let _server = WebSocketServer::new(config).build().await?;

    println!("WebSocket endpoints are configured.");
    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_complete_workflow(runtime: &Arc<MullamaRuntime>) -> Result<(), MullamaError> {
    println!("\nRunning coordinated async workflow");

    for i in 1..=5 {
        let task = runtime.spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
            format!("stream token {}", i)
        });

        match task.await {
            Ok(result) => println!("  {}", result),
            Err(err) => println!("  task error: {}", err),
        }
    }

    let metrics = runtime.metrics().summary().await;
    println!(
        "Runtime metrics: tasks={}, generations={}",
        metrics.tasks_spawned, metrics.generation_requests
    );

    Ok(())
}

#[cfg(all(
    feature = "tokio-runtime",
    feature = "parallel",
    feature = "websockets",
    feature = "multimodal"
))]
async fn demonstrate_advanced_patterns() -> Result<(), MullamaError> {
    println!("\nAdvanced integration patterns");

    let pipeline_stages = [
        "audio preprocessing",
        "vision preprocessing",
        "prompt assembly",
        "generation",
        "stream fanout",
    ];

    for (idx, stage) in pipeline_stages.iter().enumerate() {
        println!("  stage {}: {}", idx + 1, stage);
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    println!("Pipeline pattern completed.");
    Ok(())
}
