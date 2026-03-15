# Mullama Integration Features

This document provides an overview of the comprehensive integration features implemented in Mullama, making it a standout Rust library for LLM applications.

## 🚀 Core Integration Features

### 1. **Async/Await Support** (`async` feature)
- **Full Tokio Integration**: Native async/await support for all operations
- **Non-blocking Operations**: Parallel inference and processing
- **Streaming Interfaces**: Real-time token generation with backpressure
- **Connection Pooling**: Efficient resource management for concurrent requests

**Example:**
```rust
use mullama::{AsyncModel, AsyncContext};

let model = AsyncModel::load("model.gguf").await?;
let mut context = AsyncContext::new(model.clone()).await?;
let tokens = model.tokenize_async("Hello world", true).await?;
```

### 2. **Web Framework Integration** (`web` feature)
- **Axum Integration**: Direct REST API support with auto-generated routes
- **JSON Serialization**: Serde-based request/response handling
- **Error Handling**: Production-ready error responses
- **Metrics Collection**: Built-in performance monitoring
- **CORS Support**: Cross-origin resource sharing

**Example:**
```rust
use mullama::{create_router, AppState};
use axum::Server;

let app_state = AppState::new(model);
let app = create_router(app_state);
Server::bind(&"0.0.0.0:3000".parse()?)
    .serve(app.into_make_service())
    .await?;
```

### 3. **Tokio Runtime Integration** (`tokio-runtime` feature)
- **Runtime Management**: Advanced Tokio runtime configuration
- **Task Coordination**: Sophisticated task scheduling and management
- **Worker Pools**: Dedicated worker threads for different operations
- **Graceful Shutdown**: Clean resource cleanup on termination
- **Performance Metrics**: Runtime performance monitoring

**Example:**
```rust
use mullama::{MullamaRuntime, TaskManager};

let runtime = MullamaRuntime::new()
    .worker_threads(8)
    .max_blocking_threads(16)
    .build()?;

let mut task_manager = TaskManager::new(&runtime);
task_manager.spawn_generation_worker().await?;
```

### 4. **Parallel Processing** (`parallel` feature)
- **Rayon Integration**: Work-stealing parallelism for batch operations
- **Concurrent Inference**: Process multiple requests simultaneously
- **Thread Pool Management**: Configurable thread allocation
- **Batch Processing**: Efficient processing of multiple inputs
- **Load Balancing**: Automatic work distribution

**Example:**
```rust
use mullama::{ParallelProcessor, BatchGenerationConfig};

let processor = ParallelProcessor::new(model)
    .thread_pool(ThreadPoolConfig::new().num_threads(6))
    .build()?;

let batch_results = processor.batch_generate(&prompts, &config)?;
```

### 5. **WebSocket Support** (`websockets` feature)
- **Real-time Communication**: Bidirectional streaming with clients
- **Audio Processing**: Real-time audio streaming and processing
- **Connection Management**: Advanced connection handling
- **Message Types**: Structured message protocol for different data types
- **Compression**: Built-in message compression support

**Example:**
```rust
use mullama::{WebSocketServer, WebSocketConfig, WSMessage};

let config = WebSocketConfig::new()
    .port(8080)
    .max_connections(100)
    .enable_audio()
    .enable_compression();

let server = WebSocketServer::new(config).build().await?;
```

### 6. **Multimodal Processing** (`multimodal` feature)
- **Vision-Language Models**: Process text and images together
- **Audio Processing**: Speech-to-text and text-to-speech capabilities
- **Cross-Modal Understanding**: Combine multiple data types for richer context
- **Format Support**: Wide range of image and audio formats
- **Feature Extraction**: Advanced multimodal feature extraction

**Example:**
```rust
use mullama::{MultimodalProcessor, MultimodalInput};

let processor = MultimodalProcessor::new()
    .enable_image_processing()
    .enable_audio_processing()
    .build();

let input = MultimodalInput {
    text: Some("Describe this image".to_string()),
    image: Some(image_data),
    audio: Some(audio_data),
    max_tokens: Some(150),
    context: None,
};

let result = processor.process_multimodal(&input).await?;
```

### 7. **Streaming Audio Support** (`streaming-audio` feature)
- **Real-time Audio Capture**: Live microphone input processing
- **Low-latency Processing**: Ring buffer-based audio pipelines
- **Voice Activity Detection**: Automatic speech detection
- **Noise Reduction**: Advanced audio preprocessing
- **Multi-channel Support**: Stereo and multi-channel audio
- **Adaptive Quality**: Dynamic quality adjustment based on performance

**Example:**
```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig};

let config = AudioStreamConfig::new()
    .sample_rate(16000)
    .channels(1)
    .enable_voice_detection(true)
    .enable_noise_reduction(true);

let mut processor = StreamingAudioProcessor::new(config)?;
let audio_stream = processor.start_capture().await?;

while let Some(chunk) = audio_stream.next().await {
    let processed = processor.process_chunk(&chunk).await?;
    // Process with AI model...
}
```

### 8. **Format Conversion** (`format-conversion` feature)
- **Audio Format Conversion**: WAV, MP3, FLAC, AAC, OGG support
- **Image Format Conversion**: JPEG, PNG, WebP, TIFF, BMP support
- **Real-time Conversion**: Streaming format conversion
- **Quality Control**: Configurable quality and compression settings
- **Metadata Preservation**: Maintain metadata during conversion
- **Batch Processing**: Efficient multi-file conversion

**Example:**
```rust
use mullama::{AudioConverter, ImageConverter, ConversionConfig};

// Audio conversion
let audio_converter = AudioConverter::new();
let wav_data = audio_converter.mp3_to_wav("input.mp3", config).await?;

// Image conversion
let image_converter = ImageConverter::new();
let png_data = image_converter.jpeg_to_png("input.jpg", config).await?;

// Real-time streaming conversion
let streaming_converter = StreamingConverter::new(1024);
let converted_stream = streaming_converter.convert_audio_stream(
    input_stream, AudioFormatType::MP3, AudioFormatType::WAV, config
).await?;
```

### 9. **Late Interaction / ColBERT** (`late-interaction` feature)
- **Multi-Vector Embeddings**: Per-token embeddings instead of single pooled vector
- **MaxSim Scoring**: Fine-grained token-level similarity matching
- **Top-K Retrieval**: Efficient document ranking for semantic search
- **Model Agnostic**: Works with any embedding model (ColBERT-trained models optimal)
- **Parallel Scoring**: Rayon-powered parallel document scoring
- **Analysis Tools**: Similarity matrices and token-level match inspection

**Example:**
```rust
use mullama::late_interaction::{
    MultiVectorGenerator, MultiVectorConfig, LateInteractionScorer
};
use std::sync::Arc;

// Create generator with any embedding model
let model = Arc::new(Model::load("model.gguf")?);
let config = MultiVectorConfig::default()
    .normalize(true)
    .skip_special_tokens(true);
let mut generator = MultiVectorGenerator::new(model, config)?;

// Generate multi-vector embeddings (per-token)
let query = generator.embed_text("What is machine learning?")?;
let documents: Vec<_> = texts.iter()
    .map(|t| generator.embed_text(t))
    .collect::<Result<Vec<_>, _>>()?;

// Score with MaxSim: sum of max similarities per query token
let score = LateInteractionScorer::max_sim(&query, &documents[0]);

// Top-k retrieval
let top_k = LateInteractionScorer::find_top_k(&query, &documents, 10);

// Analyze token-level matches
let matrix = LateInteractionScorer::similarity_matrix(&query, &documents[0]);
let matches = LateInteractionScorer::best_matches(&query, &documents[0]);
```

**With parallel processing** (combine with `parallel` feature):
```rust
// Parallel scoring across large document collections
let top_k = LateInteractionScorer::find_top_k_parallel(&query, &documents, 10);
let scores = LateInteractionScorer::batch_score_parallel(&queries, &documents);
```

**Recommended models:**
- `LiquidAI/LFM2-ColBERT-350M-GGUF` - Purpose-trained ColBERT model
- Any GGUF embedding model (functional but suboptimal for retrieval)

### 10. **Daemon Mode** (`daemon` feature)
- **Multi-Model Server**: Run multiple models simultaneously
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Anthropic-Compatible API**: Claude Messages API support
- **Embedded Web UI**: Vue.js management interface
- **Auto-Spawn**: Daemon starts automatically when needed
- **Model Aliases**: Simple names that resolve to HuggingFace repos
- **Modelfile/Mullamafile**: Ollama-compatible model configuration
- **Prometheus Metrics**: Production monitoring endpoint
- **TUI Client**: Interactive terminal chat interface

**Example:**
```bash
# Auto-spawning - daemon starts automatically
mullama run llama3.2:1b "Hello!"

# Or start explicitly with multiple models
mullama serve --model llama3.2:1b --model qwen2.5:7b-instruct

# Use OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hi"}]}'

# Use Anthropic-compatible API
curl http://localhost:8080/v1/messages \
  -d '{"model": "llama3.2:1b", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}'

# Access Web UI
open http://localhost:8080/ui/
```

**Modelfile Support:**
```dockerfile
FROM llama3.2:1b
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
SYSTEM """You are a helpful assistant."""
GPU_LAYERS 32  # Mullama extension
```

```bash
mullama create my-assistant -f ./Modelfile
mullama run my-assistant "Hello!"
```

### 11. **Reproducibility & Audit Features**
- **Content-Addressed Verification**: SHA256 digest verification for model files
- **Revision Pinning**: Pin HuggingFace models to specific commit hashes
- **Execution Records**: Structured audit logging for inference operations
- **Config Hashing**: Deterministic configuration fingerprinting
- **Thinking Content Separation**: Parse and separate reasoning from output

**Content Verification:**
```dockerfile
FROM hf:Qwen/Qwen2.5-7B-Instruct-GGUF@a1b2c3d
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

**Execution Records (JSON-lines audit log):**
```rust
use mullama::modelfile::ExecutionRecord;

let record = ExecutionRecord {
    id: "exec-12345".to_string(),
    timestamp: 1705123456,
    model_digest: "sha256:abc123...".to_string(),
    model_ref: "hf:Qwen/Qwen2.5-7B-Instruct-GGUF".to_string(),
    revision: Some("a1b2c3d".to_string()),
    config_hash: "sha256:config...".to_string(),
    backend_version: "mullama-0.1.1".to_string(),
    gpu_info: Some("NVIDIA RTX 4090".to_string()),
    context_size: 8192,
    gpu_layers: 35,
    temperature: 0.7,
    prompt_tokens: 128,
    completion_tokens: 256,
    duration_ms: 1500,
    success: true,
    error: None,
};

// Append to audit log
let json = serde_json::to_string(&record)?;
writeln!(audit_file, "{}", json)?;
```

**Thinking Content Separation:**
```dockerfile
FROM deepseek-r1:7b

THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true
```

Streaming responses include separate `thinking` field:
```json
{"choices":[{"delta":{"thinking":"Let me reason through this..."}}]}
{"choices":[{"delta":{"content":"The answer is 42."}}]}
```

## 🎯 Advanced Integration Patterns

### 1. **Complete Workflow Integration**
All features work seamlessly together for complex applications:

```rust
use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    // 1. Setup runtime
    let runtime = MullamaRuntime::new()
        .worker_threads(8)
        .build()?;

    // 2. Setup parallel processing
    let parallel_processor = ParallelProcessor::new(model)
        .thread_pool(ThreadPoolConfig::new().num_threads(6))
        .build()?;

    // 3. Setup multimodal processing
    let multimodal_processor = MultimodalProcessor::new()
        .enable_image_processing()
        .enable_audio_processing()
        .build();

    // 4. Setup streaming audio
    let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;
    let audio_stream = audio_processor.start_capture().await?;

    // 5. Setup WebSocket server
    let ws_server = WebSocketServer::new(ws_config).build().await?;

    // 6. Setup web API
    let app_state = AppState::new(model);
    let app = create_router(app_state);

    // All features working together!
    Ok(())
}
```

### 2. **Cross-Modal Pipeline**
Process audio, convert it, analyze with AI, and stream results:

```rust
// Audio capture → Format conversion → AI processing → WebSocket streaming
let audio_stream = audio_processor.start_capture().await?;
let converter = AudioConverter::new();

while let Some(chunk) = audio_stream.next().await {
    // Convert audio format if needed
    let converted = converter.convert_chunk(&chunk).await?;

    // Process with multimodal AI
    let audio_input = converted.to_audio_input();
    let result = multimodal_processor.process_audio(&audio_input).await?;

    // Stream result via WebSocket
    let ws_message = WSMessage::AudioResponse {
        transcript: result.transcript,
        confidence: result.confidence
    };
    ws_server.broadcast(ws_message).await?;
}
```

### 3. **High-Performance Batch Processing**
Combine parallel processing with format conversion:

```rust
// Parallel batch processing with format conversion
let files = vec!["audio1.mp3", "audio2.flac", "audio3.wav"];
let converter = AudioConverter::new();

let converted_inputs: Vec<_> = parallel_processor
    .batch_process(files, |file| async move {
        let audio_data = converter.load_and_convert(file, AudioFormatType::WAV).await?;
        audio_data.to_audio_input()
    })
    .await?;

let results = parallel_processor.batch_generate_audio(&converted_inputs, &config)?;
```

## 🔧 Configuration Management

### Comprehensive Configuration System
- **Serde Integration**: JSON/TOML configuration files
- **Builder Patterns**: Fluent configuration APIs
- **Validation**: Automatic configuration validation
- **Environment Variables**: Environment-based configuration override
- **Hot Reload**: Runtime configuration updates

**Example:**
```rust
use mullama::{MullamaConfig, ModelConfig, ContextConfig};

let config = MullamaConfig {
    model: ModelConfig {
        path: "model.gguf".to_string(),
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
    // ... other configuration sections
};

// Or use builder pattern
let config = MullamaConfig::new()
    .model(ModelConfig::new().path("model.gguf").gpu_layers(40))
    .performance(PerformanceConfig::new().enable_monitoring(true))
    .build()?;
```

## 📊 Performance Features

### 1. **Monitoring and Metrics**
- **Runtime Metrics**: Performance monitoring for all components
- **Resource Tracking**: Memory and CPU usage monitoring
- **Latency Measurement**: Request/response time tracking
- **Throughput Analysis**: Operations per second tracking
- **Error Rate Monitoring**: Error frequency and types

### 2. **Optimization Features**
- **Memory Management**: Automatic resource cleanup and optimization
- **Connection Pooling**: Efficient resource reuse
- **Caching**: Intelligent caching for repeated operations
- **Load Balancing**: Automatic work distribution
- **Adaptive Quality**: Dynamic quality adjustment

## 🛡️ Production Features

### 1. **Error Handling**
- **Comprehensive Error Types**: Detailed error classification
- **Graceful Degradation**: Fallback strategies for failures
- **Automatic Recovery**: Self-healing capabilities
- **Logging Integration**: Comprehensive logging support

### 2. **Security Features**
- **Safe Memory Management**: Zero unsafe operations in public API
- **Resource Limits**: Configurable resource constraints
- **Input Validation**: Comprehensive input sanitization
- **Secure Defaults**: Production-ready default configurations

## 🎉 Why This Stands Out

### Innovation Points:
1. **Complete Integration**: All features work together seamlessly
2. **Production Ready**: Comprehensive error handling and monitoring
3. **Performance Optimized**: Advanced parallelism and resource management
4. **Real-time Capabilities**: Streaming audio and WebSocket integration
5. **Cross-Platform**: Support for all major platforms
6. **Memory Safe**: Zero unsafe operations in public API
7. **Flexible Configuration**: Multiple configuration methods
8. **Extensible Architecture**: Easy to add new features and integrations

### Use Cases:
- **Real-time AI Assistants**: Voice-enabled AI with streaming responses
- **Multimodal Applications**: Combined text, image, and audio processing
- **Semantic Search & RAG**: ColBERT-style retrieval with MaxSim scoring
- **High-Performance Services**: Batch processing with parallel execution
- **Web Applications**: RESTful APIs with WebSocket streaming
- **Edge Computing**: Optimized for resource-constrained environments
- **Research Platforms**: Flexible experimentation and prototyping

This comprehensive integration makes Mullama a unique and powerful library in the Rust ecosystem, providing everything needed to build sophisticated LLM-powered applications with production-grade performance and reliability.