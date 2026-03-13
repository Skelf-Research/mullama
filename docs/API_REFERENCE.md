# 📚 Mullama API Reference

Complete API documentation for all Mullama integration features.

## 📋 Table of Contents

- [Core APIs](#-core-apis)
- [Async Integration](#-async-integration)
- [Streaming APIs](#-streaming-apis)
- [Web Framework Integration](#-web-framework-integration)
- [WebSocket APIs](#-websocket-apis)
- [Multimodal Processing](#-multimodal-processing)
- [Audio Streaming](#-audio-streaming)
- [Format Conversion](#-format-conversion)
- [Parallel Processing](#-parallel-processing)
- [Runtime Management](#-runtime-management)
- [Configuration System](#-configuration-system)
- [Error Handling](#-error-handling)

## 🔧 Core APIs

### Model

The foundational model interface for LLM operations.

```rust
use mullama::{Model, ModelParams};

impl Model {
    /// Load a model from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError>;

    /// Load with custom parameters
    pub fn load_with_params(path: impl AsRef<Path>, params: ModelParams) -> Result<Self, MullamaError>;

    /// Get model vocabulary size
    pub fn vocab_size(&self) -> usize;

    /// Get training context size
    pub fn n_ctx_train(&self) -> usize;

    /// Get embedding dimension
    pub fn n_embd(&self) -> usize;

    /// Get number of layers
    pub fn n_layer(&self) -> usize;

    /// Tokenize text to tokens
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<TokenId>, MullamaError>;

    /// Convert token to text
    pub fn token_to_str(&self, token: TokenId, lstrip: u32, special: bool) -> Result<String, MullamaError>;
}
```

### Context

Manages inference context and state.

```rust
use mullama::{Context, ContextParams};

impl Context {
    /// Create new context
    pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self, MullamaError>;

    /// Generate text from tokens
    pub fn generate(&mut self, tokens: &[TokenId], max_len: usize) -> Result<Vec<TokenId>, MullamaError>;

    /// Get context size
    pub fn n_ctx(&self) -> usize;

    /// Clear context state
    pub fn clear(&mut self);
}
```

### Builder Pattern APIs

Fluent configuration APIs for easy setup.

```rust
use mullama::{ModelBuilder, ContextBuilder, SamplerBuilder};

// Model builder
let model = ModelBuilder::new()
    .path("model.gguf")
    .gpu_layers(40)
    .context_size(8192)
    .threads(8)
    .build().await?;

// Context builder
let context = ContextBuilder::new(model.clone())
    .context_size(4096)
    .batch_size(1024)
    .flash_attention(true)
    .build()?;

// Sampler builder
let sampler = SamplerBuilder::new()
    .temperature(0.7)
    .top_k(50)
    .top_p(0.9)
    .build(model.clone())?;
```

---

## 🚀 Async Integration

### AsyncModel

Async-first model interface with full Tokio integration.

```rust
use mullama::{AsyncModel, AsyncContext, ModelInfo, AsyncConfig};

impl AsyncModel {
    /// Load model asynchronously
    pub async fn load(path: impl AsRef<Path>) -> Result<Self, MullamaError>;

    /// Load with configuration
    pub async fn load_with_config(config: AsyncConfig) -> Result<Self, MullamaError>;

    /// Generate text asynchronously
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, MullamaError>;

    /// Generate with streaming
    pub async fn generate_stream(&self, prompt: &str, config: StreamConfig) -> Result<TokenStream, MullamaError>;

    /// Tokenize asynchronously
    pub async fn tokenize_async(&self, text: &str, add_bos: bool) -> Result<Vec<TokenId>, MullamaError>;

    /// Get model information
    pub async fn model_info(&self) -> Result<ModelInfo, MullamaError>;

    /// Check if model is ready
    pub async fn is_ready(&self) -> bool;

    /// Clone for concurrent use
    pub fn clone(&self) -> Self;
}
```

### AsyncContext

Async context management with advanced features.

```rust
use mullama::AsyncContext;

impl AsyncContext {
    /// Create async context
    pub async fn new(model: Arc<AsyncModel>) -> Result<Self, MullamaError>;

    /// Generate with async streaming
    pub async fn generate_stream(&mut self, tokens: &[TokenId]) -> Result<TokenStream, MullamaError>;

    /// Process batch asynchronously
    pub async fn process_batch(&mut self, batches: Vec<Batch>) -> Result<Vec<Vec<TokenId>>, MullamaError>;

    /// Save state asynchronously
    pub async fn save_state(&self, path: impl AsRef<Path>) -> Result<(), MullamaError>;

    /// Load state asynchronously
    pub async fn load_state(&mut self, path: impl AsRef<Path>) -> Result<(), MullamaError>;
}
```

---

## 🌊 Streaming APIs

### TokenStream

Real-time token streaming with backpressure handling.

```rust
use mullama::{TokenStream, StreamConfig, TokenData};

impl TokenStream {
    /// Get next token
    pub async fn next(&mut self) -> Option<Result<TokenData, MullamaError>>;

    /// Try to get next token without waiting
    pub fn try_next(&mut self) -> Option<Result<TokenData, MullamaError>>;

    /// Get stream configuration
    pub fn config(&self) -> &StreamConfig;

    /// Check if stream is finished
    pub fn is_finished(&self) -> bool;

    /// Get total tokens generated
    pub fn tokens_generated(&self) -> usize;
}

// Implement Stream trait for use with StreamExt
impl Stream for TokenStream {
    type Item = Result<TokenData, MullamaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
}
```

### StreamConfig

Configuration for streaming operations.

```rust
use mullama::StreamConfig;

impl StreamConfig {
    pub fn new() -> Self;
    pub fn max_tokens(mut self, max: usize) -> Self;
    pub fn temperature(mut self, temp: f32) -> Self;
    pub fn top_k(mut self, k: u32) -> Self;
    pub fn top_p(mut self, p: f32) -> Self;
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self;
    pub fn stream_delay_ms(mut self, delay: u64) -> Self;
}
```

### Usage Example

```rust
use mullama::{AsyncModel, StreamConfig};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    let model = AsyncModel::load("model.gguf").await?;

    let config = StreamConfig::new()
        .max_tokens(200)
        .temperature(0.7)
        .stop_sequences(vec!["\\n\\n".to_string()]);

    let mut stream = model.generate_stream("Tell me about AI", config).await?;

    print!("AI: ");
    while let Some(token_result) = stream.next().await {
        let token_data = token_result?;
        print!("{}", token_data.text);
        tokio::io::stdout().flush().await?;
    }
    println!();

    Ok(())
}
```

---

## 🌍 Web Framework Integration

### AppState

Application state management for web services.

```rust
use mullama::{AppState, AsyncModel, ApiMetrics};

impl AppState {
    /// Create new app state
    pub fn new(model: Arc<AsyncModel>) -> AppStateBuilder;

    /// Get model reference
    pub fn model(&self) -> &Arc<AsyncModel>;

    /// Get metrics
    pub fn metrics(&self) -> &ApiMetrics;

    /// Check if streaming is enabled
    pub fn streaming_enabled(&self) -> bool;

    /// Get concurrent request limit
    pub fn max_concurrent_requests(&self) -> usize;
}

pub struct AppStateBuilder {
    model: Arc<AsyncModel>,
    enable_streaming: bool,
    enable_metrics: bool,
    max_concurrent_requests: usize,
    rate_limit: Option<RateLimit>,
}

impl AppStateBuilder {
    pub fn enable_streaming(mut self) -> Self;
    pub fn enable_metrics(mut self) -> Self;
    pub fn max_concurrent_requests(mut self, max: usize) -> Self;
    pub fn rate_limit(mut self, requests: u32, window: Duration) -> Self;
    pub fn build(self) -> AppState;
}
```

### Router Creation

Automatic endpoint generation with Axum integration.

```rust
use mullama::{create_router, AppState, GenerateRequest, GenerateResponse};
use axum::{Router, routing::post};

/// Create router with default endpoints
pub fn create_router(state: AppState) -> Router;

/// Available default endpoints:
/// - POST /generate      - Text generation
/// - POST /tokenize      - Text tokenization
/// - POST /embeddings    - Generate embeddings
/// - GET  /metrics       - Performance metrics
/// - GET  /health        - Health check
/// - WS   /ws            - WebSocket streaming

// Usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AsyncModel::load("model.gguf").await?;
    let app_state = AppState::new(model).enable_streaming().build();

    let app = create_router(app_state)
        .route("/custom", post(custom_endpoint));

    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

### Request/Response Types

Structured API types for web endpoints.

```rust
use mullama::{GenerateRequest, GenerateResponse, TokenizeRequest, TokenizeResponse};

#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub processing_time_ms: u64,
    pub model_info: Option<ModelInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    pub add_bos: Option<bool>,
    pub special_tokens: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<TokenId>,
    pub token_count: usize,
    pub text_length: usize,
}
```

---

## 🌐 WebSocket APIs

### WebSocketServer

Real-time bidirectional communication server.

```rust
use mullama::{WebSocketServer, WebSocketConfig, WSMessage, ConnectionManager};

impl WebSocketServer {
    /// Create new WebSocket server
    pub fn new(config: WebSocketConfig) -> WebSocketServerBuilder;

    /// Start the server
    pub async fn start(&self) -> Result<(), MullamaError>;

    /// Stop the server
    pub async fn stop(&self) -> Result<(), MullamaError>;

    /// Broadcast message to all connections
    pub async fn broadcast(&self, message: WSMessage) -> Result<(), MullamaError>;

    /// Send message to specific connection
    pub async fn send_to(&self, connection_id: &str, message: WSMessage) -> Result<(), MullamaError>;

    /// Get connection count
    pub fn connection_count(&self) -> usize;

    /// Get server statistics
    pub fn stats(&self) -> ServerStats;
}

pub struct WebSocketServerBuilder {
    config: WebSocketConfig,
    on_connect: Option<Box<dyn Fn(WSConnection) -> BoxFuture<'_, Result<(), MullamaError>>>>,
    on_message: Option<Box<dyn Fn(WSMessage, &mut WSConnection) -> BoxFuture<'_, Result<(), MullamaError>>>>,
    on_disconnect: Option<Box<dyn Fn(&WSConnection) -> BoxFuture<'_, Result<(), MullamaError>>>>,
}

impl WebSocketServerBuilder {
    pub fn on_connect<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(WSConnection) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), MullamaError>> + Send + 'static;

    pub fn on_message<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(WSMessage, &mut WSConnection) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), MullamaError>> + Send + 'static;

    pub fn on_disconnect<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(&WSConnection) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), MullamaError>> + Send + 'static;

    pub async fn build(self) -> Result<WebSocketServer, MullamaError>;
}
```

### WSMessage Types

Structured message protocol for WebSocket communication.

```rust
use mullama::{WSMessage, AudioFormat, AudioFeatures};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WSMessage {
    // Text messages
    Text {
        content: String,
    },

    // AI generation requests
    Generate {
        prompt: String,
        config: Option<GenerationConfig>,
    },

    // Audio messages
    Audio {
        data: Vec<u8>,
        format: AudioFormat,
    },

    // Audio processing results
    AudioResponse {
        transcript: String,
        confidence: f32,
    },

    // Streaming tokens
    Token {
        text: String,
        is_final: bool,
    },

    // Control messages
    StartStream {
        config: StreamConfig,
    },

    StopStream,

    // Error messages
    Error {
        message: String,
        code: u32,
    },

    // Connection management
    Ping,
    Pong,
}
```

### WebSocketConfig

Configuration for WebSocket server behavior.

```rust
use mullama::WebSocketConfig;

impl WebSocketConfig {
    pub fn new() -> Self;
    pub fn port(mut self, port: u16) -> Self;
    pub fn max_connections(mut self, max: usize) -> Self;
    pub fn enable_audio(mut self) -> Self;
    pub fn enable_compression(mut self) -> Self;
    pub fn ping_interval(mut self, interval: Duration) -> Self;
    pub fn message_size_limit(mut self, limit: usize) -> Self;
    pub fn connection_timeout(mut self, timeout: Duration) -> Self;
}
```

---

## 🎭 Multimodal Processing

### MultimodalProcessor

Cross-modal AI processing for text, images, and audio.

```rust
use mullama::{MultimodalProcessor, MultimodalInput, MultimodalOutput};

impl MultimodalProcessor {
    /// Create new multimodal processor
    pub fn new() -> MultimodalProcessorBuilder;

    /// Process multimodal input
    pub async fn process_multimodal(&self, input: &MultimodalInput) -> Result<MultimodalOutput, MullamaError>;

    /// Process text only
    pub async fn process_text(&self, text: &str) -> Result<String, MullamaError>;

    /// Process image only
    pub async fn process_image(&self, image: &ImageInput) -> Result<ImageProcessingResult, MullamaError>;

    /// Process audio only
    pub async fn process_audio(&self, audio: &AudioInput) -> Result<AudioProcessingResult, MullamaError>;

    /// Check supported modalities
    pub fn supported_modalities(&self) -> Vec<Modality>;
}

pub struct MultimodalProcessorBuilder {
    enable_image: bool,
    enable_audio: bool,
    enable_video: bool,
    image_config: Option<ImageProcessingConfig>,
    audio_config: Option<AudioProcessingConfig>,
}

impl MultimodalProcessorBuilder {
    pub fn enable_image_processing(mut self) -> Self;
    pub fn enable_audio_processing(mut self) -> Self;
    pub fn enable_video_processing(mut self) -> Self;
    pub fn image_config(mut self, config: ImageProcessingConfig) -> Self;
    pub fn audio_config(mut self, config: AudioProcessingConfig) -> Self;
    pub fn build(self) -> MultimodalProcessor;
}
```

### Input/Output Types

Structured types for multimodal data.

```rust
use mullama::{MultimodalInput, MultimodalOutput, ImageInput, AudioInput, VideoInput};

#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub image: Option<ImageInput>,
    pub audio: Option<AudioInput>,
    pub video: Option<VideoInput>,
    pub max_tokens: Option<usize>,
    pub context: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MultimodalOutput {
    pub text_response: String,
    pub image_description: Option<String>,
    pub audio_transcript: Option<String>,
    pub video_description: Option<String>,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ImageInput {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub dimensions: (u32, u32),
    pub metadata: HashMap<String, String>,
}

impl ImageInput {
    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self, MullamaError>;
    pub async fn from_url(url: &str) -> Result<Self, MullamaError>;
    pub fn from_bytes(data: Vec<u8>, format: ImageFormat) -> Result<Self, MullamaError>;
}

#[derive(Debug, Clone)]
pub struct AudioInput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: f32,
    pub format: AudioFormat,
    pub transcript: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl AudioInput {
    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self, MullamaError>;
    pub async fn from_bytes(data: &[u8], format: AudioFormat) -> Result<Self, MullamaError>;
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self;
}
```

---

## 🎵 Audio Streaming

### StreamingAudioProcessor

Real-time audio capture and processing.

```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig, AudioChunk, AudioStream};

impl StreamingAudioProcessor {
    /// Create new audio processor
    pub fn new(config: AudioStreamConfig) -> Result<Self, MullamaError>;

    /// Initialize audio devices
    pub async fn initialize(&mut self) -> Result<(), MullamaError>;

    /// Start audio capture
    pub async fn start_capture(&mut self) -> Result<AudioStream, MullamaError>;

    /// Stop audio capture
    pub async fn stop_capture(&mut self) -> Result<(), MullamaError>;

    /// Process single audio chunk
    pub async fn process_chunk(&self, chunk: &AudioChunk) -> Result<AudioChunk, MullamaError>;

    /// Get streaming metrics
    pub async fn metrics(&self) -> StreamingMetrics;

    /// List available input devices
    pub fn list_input_devices(&self) -> Result<Vec<String>, MullamaError>;
}
```

### AudioStreamConfig

Configuration for real-time audio streaming.

```rust
use mullama::{AudioStreamConfig, DevicePreference};

impl AudioStreamConfig {
    pub fn new() -> Self;
    pub fn sample_rate(mut self, rate: u32) -> Self;
    pub fn channels(mut self, channels: u16) -> Self;
    pub fn buffer_size(mut self, size: usize) -> Self;
    pub fn enable_noise_reduction(mut self, enable: bool) -> Self;
    pub fn enable_voice_detection(mut self, enable: bool) -> Self;
    pub fn vad_threshold(mut self, threshold: f32) -> Self;
    pub fn max_latency_ms(mut self, latency: u32) -> Self;
    pub fn device_preference(mut self, preference: DevicePreference) -> Self;
}

#[derive(Debug, Clone)]
pub enum DevicePreference {
    Default,
    ByName(String),
    LowestLatency,
    HighestQuality,
}
```

### AudioChunk & AudioStream

Real-time audio data structures.

```rust
use mullama::{AudioChunk, AudioStream, AudioFeatures};

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
    pub timestamp: Instant,
    pub duration: Duration,
    pub voice_detected: bool,
    pub signal_level: f32,
    pub features: Option<AudioFeatures>,
}

impl AudioChunk {
    pub fn new(samples: Vec<f32>, channels: u16, sample_rate: u32) -> Self;
    pub fn to_audio_input(&self) -> AudioInput;
    pub fn apply_noise_gate(&mut self, threshold_db: f32);
    pub fn detect_voice_activity(&mut self, threshold: f32);
}

pub struct AudioStream {
    // Internal receiver for audio chunks
}

impl AudioStream {
    pub async fn next(&mut self) -> Option<AudioChunk>;
    pub async fn metrics(&self) -> StreamingMetrics;
}

impl Stream for AudioStream {
    type Item = AudioChunk;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
}
```

---

## 🔄 Format Conversion

### AudioConverter

Audio format conversion with streaming support.

```rust
use mullama::{AudioConverter, AudioConverterConfig, ConversionConfig, AudioConversionResult};

impl AudioConverter {
    /// Create new audio converter
    pub fn new() -> Self;

    /// Create with custom configuration
    pub fn with_config(config: AudioConverterConfig) -> Self;

    /// Convert MP3 to WAV
    pub async fn mp3_to_wav(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<AudioConversionResult, MullamaError>;

    /// Convert WAV to MP3
    pub async fn wav_to_mp3(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<AudioConversionResult, MullamaError>;

    /// Convert FLAC to WAV
    pub async fn flac_to_wav(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<AudioConversionResult, MullamaError>;

    /// Generic format conversion
    pub async fn convert_audio(&self, input_path: impl AsRef<Path>, input_format: AudioFormatType, output_format: AudioFormatType, config: ConversionConfig) -> Result<AudioConversionResult, MullamaError>;

    /// Resample audio
    pub async fn resample_audio(&self, input_data: &[f32], input_rate: u32, output_rate: u32, channels: u16) -> Result<Vec<f32>, MullamaError>;

    /// Batch convert multiple files
    pub async fn batch_convert(&self, conversions: Vec<(PathBuf, AudioFormatType, AudioFormatType, ConversionConfig)>) -> Result<Vec<AudioConversionResult>, MullamaError>;
}
```

### ImageConverter

Image format conversion and processing.

```rust
use mullama::{ImageConverter, ImageConverterConfig, ImageConversionResult};

impl ImageConverter {
    /// Create new image converter
    pub fn new() -> Self;

    /// Create with custom configuration
    pub fn with_config(config: ImageConverterConfig) -> Self;

    /// Convert JPEG to PNG
    pub async fn jpeg_to_png(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<ImageConversionResult, MullamaError>;

    /// Convert PNG to JPEG
    pub async fn png_to_jpeg(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<ImageConversionResult, MullamaError>;

    /// Convert WebP to PNG
    pub async fn webp_to_png(&self, input_path: impl AsRef<Path>, config: ConversionConfig) -> Result<ImageConversionResult, MullamaError>;

    /// Generic image conversion
    pub async fn convert_image(&self, input_path: impl AsRef<Path>, input_format: ImageFormatType, output_format: ImageFormatType, config: ConversionConfig) -> Result<ImageConversionResult, MullamaError>;

    /// Convert from bytes
    pub async fn convert_image_bytes(&self, input_data: &[u8], input_format: ImageFormatType, output_format: ImageFormatType, config: ConversionConfig) -> Result<ImageConversionResult, MullamaError>;

    /// Resize image
    pub async fn resize_image(&self, input_path: impl AsRef<Path>, dimensions: (u32, u32), filter: image::imageops::FilterType) -> Result<ImageConversionResult, MullamaError>;

    /// Batch convert images
    pub async fn batch_convert_images(&self, conversions: Vec<(PathBuf, ImageFormatType, ImageFormatType, ConversionConfig)>) -> Result<Vec<ImageConversionResult>, MullamaError>;
}
```

### ConversionConfig

Unified configuration for format conversion.

```rust
use mullama::ConversionConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    pub quality: Option<u8>,           // 1-100 for lossy formats
    pub sample_rate: Option<u32>,      // Audio sample rate
    pub bit_rate: Option<u32>,         // Audio bit rate
    pub dimensions: Option<(u32, u32)>, // Image dimensions
    pub preserve_metadata: bool,
    pub options: HashMap<String, String>,
}

impl ConversionConfig {
    pub fn new() -> Self;
    pub fn quality(mut self, quality: u8) -> Self;
    pub fn sample_rate(mut self, rate: u32) -> Self;
    pub fn dimensions(mut self, width: u32, height: u32) -> Self;
    pub fn preserve_metadata(mut self, preserve: bool) -> Self;
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self;
}
```

---

## ⚡ Parallel Processing

### ParallelProcessor

High-performance parallel processing with work-stealing.

```rust
use mullama::{ParallelProcessor, BatchGenerationConfig, GenerationResult, ThreadPoolConfig};

impl ParallelProcessor {
    /// Create new parallel processor
    pub fn new(model: Arc<Model>) -> ParallelProcessorBuilder;

    /// Batch tokenize texts
    pub fn batch_tokenize(&self, texts: &[&str]) -> Result<Vec<Vec<TokenId>>, MullamaError>;

    /// Batch generate responses
    pub fn batch_generate(&self, prompts: &[&str], config: &BatchGenerationConfig) -> Result<Vec<GenerationResult>, MullamaError>;

    /// Process tasks in parallel
    pub async fn parallel_process<T, F, R>(&self, items: Vec<T>, processor: F) -> Result<Vec<R>, MullamaError>
    where
        T: Send + 'static,
        F: Fn(T) -> Result<R, MullamaError> + Send + Sync + Clone + 'static,
        R: Send + 'static;

    /// Get processor statistics
    pub fn stats(&self) -> ProcessorStats;
}

pub struct ParallelProcessorBuilder {
    model: Arc<Model>,
    thread_pool: Option<ThreadPoolConfig>,
    max_batch_size: usize,
    enable_metrics: bool,
}

impl ParallelProcessorBuilder {
    pub fn thread_pool(mut self, config: ThreadPoolConfig) -> Self;
    pub fn max_batch_size(mut self, size: usize) -> Self;
    pub fn enable_metrics(mut self) -> Self;
    pub fn build(self) -> Result<ParallelProcessor, MullamaError>;
}
```

### BatchGenerationConfig

Configuration for batch processing operations.

```rust
use mullama::BatchGenerationConfig;

#[derive(Debug, Clone)]
pub struct BatchGenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub timeout_ms: Option<u64>,
    pub retry_attempts: u32,
}

impl BatchGenerationConfig {
    pub fn new() -> Self;
    pub fn max_tokens(mut self, tokens: usize) -> Self;
    pub fn temperature(mut self, temp: f32) -> Self;
    pub fn top_k(mut self, k: u32) -> Self;
    pub fn top_p(mut self, p: f32) -> Self;
    pub fn timeout_ms(mut self, timeout: u64) -> Self;
    pub fn retry_attempts(mut self, attempts: u32) -> Self;
}
```

---

## 🚀 Runtime Management

### MullamaRuntime

Advanced Tokio runtime management and task coordination.

```rust
use mullama::{MullamaRuntime, MullamaRuntimeBuilder, TaskManager, RuntimeMetrics};

impl MullamaRuntime {
    /// Create new runtime builder
    pub fn new() -> MullamaRuntimeBuilder;

    /// Spawn task on runtime
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawn blocking task
    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Get runtime metrics
    pub async fn metrics(&self) -> RuntimeMetrics;

    /// Graceful shutdown
    pub async fn shutdown(self);

    /// Check if runtime is running
    pub fn is_running(&self) -> bool;
}

pub struct MullamaRuntimeBuilder {
    worker_threads: Option<usize>,
    max_blocking_threads: Option<usize>,
    thread_stack_size: Option<usize>,
    thread_name: Option<String>,
    enable_io: bool,
    enable_time: bool,
    enable_metrics: bool,
}

impl MullamaRuntimeBuilder {
    pub fn worker_threads(mut self, threads: usize) -> Self;
    pub fn max_blocking_threads(mut self, threads: usize) -> Self;
    pub fn thread_stack_size(mut self, size: usize) -> Self;
    pub fn thread_name(mut self, name: impl Into<String>) -> Self;
    pub fn enable_io(mut self) -> Self;
    pub fn enable_time(mut self) -> Self;
    pub fn enable_all(mut self) -> Self;
    pub fn build(self) -> Result<MullamaRuntime, MullamaError>;
}
```

### TaskManager

Advanced task coordination and lifecycle management.

```rust
use mullama::TaskManager;

impl TaskManager {
    /// Create new task manager
    pub fn new(runtime: &Arc<MullamaRuntime>) -> Self;

    /// Spawn generation worker
    pub async fn spawn_generation_worker(&mut self) -> Result<(), MullamaError>;

    /// Spawn metrics collector
    pub async fn spawn_metrics_collector(&mut self) -> Result<(), MullamaError>;

    /// Spawn custom worker
    pub async fn spawn_worker<F>(&mut self, name: &str, worker: F) -> Result<(), MullamaError>
    where
        F: Future<Output = Result<(), MullamaError>> + Send + 'static;

    /// Stop all workers
    pub async fn stop_all(&mut self) -> Result<(), MullamaError>;

    /// Get worker status
    pub fn worker_status(&self) -> HashMap<String, WorkerStatus>;
}
```

---

## ⚙️ Configuration System

### MullamaConfig

Comprehensive configuration system with validation.

```rust
use mullama::{MullamaConfig, ModelConfig, ContextConfig, SamplingConfig, PerformanceConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MullamaConfig {
    pub model: ModelConfig,
    pub context: ContextConfig,
    pub sampling: SamplingConfig,
    pub performance: PerformanceConfig,
    pub logging: LoggingConfig,
}

impl MullamaConfig {
    pub fn new() -> MullamaConfigBuilder;
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, MullamaError>;
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), MullamaError>;
    pub fn validate(&self) -> Result<(), MullamaError>;
    pub async fn load_model(&self) -> Result<AsyncModel, MullamaError>;
}

pub struct MullamaConfigBuilder {
    model: Option<ModelConfig>,
    context: Option<ContextConfig>,
    sampling: Option<SamplingConfig>,
    performance: Option<PerformanceConfig>,
    logging: Option<LoggingConfig>,
}

impl MullamaConfigBuilder {
    pub fn model(mut self, config: ModelConfig) -> Self;
    pub fn context(mut self, config: ContextConfig) -> Self;
    pub fn sampling(mut self, config: SamplingConfig) -> Self;
    pub fn performance(mut self, config: PerformanceConfig) -> Self;
    pub fn logging(mut self, config: LoggingConfig) -> Self;
    pub fn build(self) -> Result<MullamaConfig, MullamaError>;
}
```

### Configuration Subsections

Detailed configuration for different components.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub gpu_layers: u32,
    pub context_size: usize,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub kv_overrides: Vec<ModelKvOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub n_threads: usize,
    pub n_threads_batch: Option<usize>,
    pub rope_scaling_type: i32,
    pub flash_attn: bool,
    pub no_perf: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_monitoring: bool,
    pub memory_optimization: u8,
    pub cpu_optimizations: CpuOptimizations,
    pub gpu_optimizations: GpuOptimizations,
}
```

---

## ❌ Error Handling

### MullamaError

Comprehensive error types for all operations.

```rust
use mullama::MullamaError;

#[derive(Error, Debug)]
pub enum MullamaError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Failed to create context: {0}")]
    ContextError(String),

    #[error("Tokenization failed: {0}")]
    TokenizationError(String),

    #[error("Generation failed: {0}")]
    GenerationError(String),

    #[error("Audio error: {0}")]
    AudioError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("Format conversion error: {0}")]
    FormatConversionError(String),

    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    #[error("Multimodal error: {0}")]
    MultimodalError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
```

### Error Handling Patterns

Best practices for error handling in applications.

```rust
use mullama::{MullamaError, AsyncModel};

// Pattern 1: Specific error handling
async fn robust_generation(model: &AsyncModel, prompt: &str) -> Result<String, MullamaError> {
    match model.generate(prompt, 100).await {
        Ok(response) => Ok(response),
        Err(MullamaError::ModelLoadError(msg)) => {
            eprintln!("Model issue: {}", msg);
            // Try fallback or reload
            Err(MullamaError::ModelLoadError(msg))
        }
        Err(MullamaError::GenerationError(msg)) => {
            eprintln!("Generation issue: {}", msg);
            // Retry with different parameters
            model.generate(&format!("Simple: {}", prompt), 50).await
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            Err(e)
        }
    }
}

// Pattern 2: Error recovery
async fn generation_with_fallback(model: &AsyncModel, prompt: &str) -> String {
    match model.generate(prompt, 100).await {
        Ok(response) => response,
        Err(_) => {
            // Fallback to simple response
            "I apologize, but I encountered an error processing your request.".to_string()
        }
    }
}

// Pattern 3: Error context
use anyhow::{Context, Result};

async fn generation_with_context(model: &AsyncModel, prompt: &str) -> Result<String> {
    model.generate(prompt, 100).await
        .context("Failed to generate response for user prompt")
        .map_err(anyhow::Error::from)
}
```

---

## 📋 Quick Reference

### Common Imports

```rust
// Basic functionality
use mullama::prelude::*;
use mullama::{Model, Context, AsyncModel, MullamaError};

// Streaming
use mullama::{TokenStream, StreamConfig};
use tokio_stream::StreamExt;

// Web integration
use mullama::{create_router, AppState, GenerateRequest, GenerateResponse};
use axum::{Server, response::Json};

// WebSocket
use mullama::{WebSocketServer, WebSocketConfig, WSMessage};

// Multimodal
use mullama::{MultimodalProcessor, MultimodalInput, ImageInput, AudioInput};

// Audio streaming
use mullama::{StreamingAudioProcessor, AudioStreamConfig, AudioChunk};

// Format conversion
use mullama::{AudioConverter, ImageConverter, ConversionConfig};

// Parallel processing
use mullama::{ParallelProcessor, BatchGenerationConfig, ThreadPoolConfig};

// Runtime management
use mullama::{MullamaRuntime, TaskManager, MullamaConfig};
```

### Feature Flags Quick Reference

```toml
[dependencies.mullama]
version = "0.1.0"
features = [
    "async",              # AsyncModel, AsyncContext
    "streaming",          # TokenStream, StreamConfig
    "web",                # create_router, AppState
    "websockets",         # WebSocketServer, WSMessage
    "multimodal",         # MultimodalProcessor, ImageInput, AudioInput
    "streaming-audio",    # StreamingAudioProcessor, AudioChunk
    "format-conversion",  # AudioConverter, ImageConverter
    "parallel",           # ParallelProcessor, BatchGenerationConfig
    "tokio-runtime",      # MullamaRuntime, TaskManager
    "full"                # All features enabled
]
```

---

This completes the comprehensive API reference for Mullama. Each section provides detailed information about the available APIs, their usage patterns, and examples. For more specific implementation details, refer to the individual feature documentation and examples in the repository.
