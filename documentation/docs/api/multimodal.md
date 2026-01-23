---
title: Multimodal API
description: Text, image, and audio processing pipeline for vision-language models
---

# Multimodal API

The multimodal module provides a unified processing pipeline for text, images, and audio, enabling interaction with vision-language and audio-language models.

!!! info "Feature Gate"
    This module requires the `multimodal` feature flag:
    ```toml
    mullama = { version = "0.1", features = ["multimodal"] }
    ```

## MultimodalProcessor

The central coordinator for cross-modal AI processing. Manages vision encoders, audio processors, and format conversion.

```rust
pub struct MultimodalProcessor {
    // Internal fields managing encoders and processors
}

impl MultimodalProcessor {
    /// Create a new multimodal processor builder
    pub fn new() -> MultimodalProcessorBuilder;

    /// Process multimodal input (text + image + audio)
    pub async fn process_multimodal(
        &self,
        input: &MultimodalInput,
    ) -> Result<MultimodalOutput, MullamaError>;

    /// Process text only
    pub async fn process_text(&self, text: &str) -> Result<String, MullamaError>;

    /// Process image with optional text prompt
    pub async fn process_image(
        &self,
        image: &ImageInput,
    ) -> Result<ImageProcessingResult, MullamaError>;

    /// Process audio input
    pub async fn process_audio(
        &self,
        audio: &AudioInput,
    ) -> Result<AudioProcessingResult, MullamaError>;

    /// Query supported modalities
    pub fn supported_modalities(&self) -> Vec<Modality>;
}
```

### MultimodalProcessorBuilder

```rust
let processor = MultimodalProcessor::new()
    .enable_image_processing()
    .enable_audio_processing()
    .image_config(ImageProcessingConfig { max_resolution: (1024, 1024) })
    .audio_config(AudioProcessingConfig { sample_rate: 16000 })
    .build();
```

| Method | Description |
|--------|-------------|
| `enable_image_processing()` | Enable image modality support |
| `enable_audio_processing()` | Enable audio modality support |
| `enable_video_processing()` | Enable video modality support |
| `image_config(config)` | Set image processing configuration |
| `audio_config(config)` | Set audio processing configuration |
| `build()` | Build the processor |

## VisionEncoder

Handles image encoding for vision-language models. Converts images to the embedding space the LLM can understand.

```rust
pub struct VisionEncoder {
    encoder_type: VisionEncoderType,
    // Internal state
}

impl VisionEncoder {
    /// Create a new vision encoder
    pub fn new(encoder_type: VisionEncoderType) -> Result<Self, MullamaError>;

    /// Encode an image to embeddings
    pub fn encode(&self, image: &Bitmap) -> Result<Vec<f32>, MullamaError>;

    /// Get the output embedding dimension
    pub fn embedding_dim(&self) -> usize;

    /// Get the expected input resolution
    pub fn input_resolution(&self) -> (u32, u32);
}
```

## VisionEncoderType

Supported vision encoder architectures.

```rust
#[derive(Debug, Clone)]
pub enum VisionEncoderType {
    /// CLIP (Contrastive Language-Image Pre-training)
    Clip {
        model_path: String,
    },
    /// DINO (Self-Distillation with No Labels)
    Dino {
        model_path: String,
    },
    /// Custom vision encoder with user-provided weights
    Custom {
        model_path: String,
        config: CustomEncoderConfig,
    },
}
```

| Variant | Use Case | Description |
|---------|----------|-------------|
| `Clip` | General vision-language | CLIP-based encoding, used by LLaVA and similar models |
| `Dino` | Dense visual features | Self-supervised features for detailed image understanding |
| `Custom` | Specialized models | User-provided encoder with custom configuration |

## Modality

Enum representing supported input/output modalities.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}
```

## InputChunk

Represents a piece of mixed-media input for models that support interleaved content.

```rust
#[derive(Debug, Clone)]
pub enum InputChunk {
    /// Text segment
    Text(String),
    /// Image data
    Image(ImageInput),
    /// Audio data
    Audio(AudioInput),
    /// Video reference
    Video { path: PathBuf },
}
```

### InputChunks

Ordered sequence of chunks for interleaved multimodal input.

```rust
#[derive(Debug, Clone)]
pub struct InputChunks {
    pub chunks: Vec<InputChunk>,
}

impl InputChunks {
    pub fn new() -> Self;
    pub fn add_text(&mut self, text: &str) -> &mut Self;
    pub fn add_image(&mut self, image: ImageInput) -> &mut Self;
    pub fn add_audio(&mut self, audio: AudioInput) -> &mut Self;
}
```

**Example:**

```rust
use mullama::multimodal::{InputChunks, ImageInput};

let mut chunks = InputChunks::new();
chunks.add_text("Describe this image:");
chunks.add_image(ImageInput::from_path("photo.jpg").await?);
chunks.add_text("Focus on the colors and composition.");
```

## AudioFeatures

Extracted audio feature representation for model consumption.

```rust
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub features: Vec<f32>,
    pub n_frames: usize,
    pub n_features: usize,
    pub sample_rate: u32,
    pub duration_ms: u64,
}
```

### Fields

| Name | Type | Description |
|------|------|-------------|
| `features` | `Vec<f32>` | Flat array of audio features (n_frames * n_features) |
| `n_frames` | `usize` | Number of time frames |
| `n_features` | `usize` | Features per frame (e.g., mel bands) |
| `sample_rate` | `u32` | Original sample rate |
| `duration_ms` | `u64` | Audio duration in milliseconds |

## AudioFormat

Supported audio formats for input and conversion.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioFormat {
    /// 16-bit signed integer PCM
    PCM16,
    /// 32-bit signed integer PCM
    PCM32,
    /// 32-bit floating point samples
    Float32,
    /// MP3 compressed audio
    MP3,
    /// WAV container (various internal formats)
    WAV,
    /// FLAC lossless compressed audio
    FLAC,
}
```

| Format | Quality | Size | Use Case |
|--------|---------|------|----------|
| `PCM16` | Lossless | Large | Raw audio capture, maximum compatibility |
| `PCM32` | Lossless | Very large | High-precision audio processing |
| `Float32` | Lossless | Very large | Internal processing format |
| `MP3` | Lossy | Small | Compressed storage, web delivery |
| `WAV` | Lossless | Large | Standard audio file format |
| `FLAC` | Lossless | Medium | Compressed lossless storage |

## Bitmap

Raw image data for direct pixel manipulation and model input.

```rust
#[derive(Debug, Clone)]
pub struct Bitmap {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,  // 1=grayscale, 3=RGB, 4=RGBA
}

impl Bitmap {
    /// Create empty bitmap with given dimensions
    pub fn new(width: u32, height: u32, channels: u8) -> Self;

    /// Create from ImageInput (decode compressed format)
    pub fn from_image_input(input: &ImageInput) -> Result<Self, MullamaError>;

    /// Resize to new dimensions (bilinear interpolation)
    pub fn resize(&self, new_width: u32, new_height: u32) -> Self;

    /// Convert to RGB format
    pub fn to_rgb(&self) -> Self;

    /// Get pixel value at coordinates
    pub fn pixel_at(&self, x: u32, y: u32) -> &[u8];
}
```

## ImageInput

Represents image data for multimodal processing.

```rust
#[derive(Debug, Clone)]
pub struct ImageInput {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub dimensions: (u32, u32),
    pub metadata: HashMap<String, String>,
}
```

### Fields

| Name | Type | Description |
|------|------|-------------|
| `data` | `Vec<u8>` | Raw image bytes (compressed or raw) |
| `format` | `ImageFormat` | Image format identifier |
| `dimensions` | `(u32, u32)` | Width and height in pixels |
| `metadata` | `HashMap<String, String>` | Optional metadata (EXIF, etc.) |

### Factory Methods

```rust
impl ImageInput {
    /// Load from file path (auto-detects format)
    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self, MullamaError>;

    /// Load from URL
    pub async fn from_url(url: &str) -> Result<Self, MullamaError>;

    /// Create from raw bytes with known format
    pub fn from_bytes(data: Vec<u8>, format: ImageFormat) -> Result<Self, MullamaError>;
}
```

**Example:**

```rust
use mullama::multimodal::ImageInput;

// From file
let image = ImageInput::from_path("photo.jpg").await?;

// From bytes
let bytes = std::fs::read("photo.png")?;
let image = ImageInput::from_bytes(bytes, ImageFormat::PNG)?;
```

## Image Processing Pipeline

The image processing pipeline follows these steps:

1. **Load** -- Read image from file/bytes/URL
2. **Decode** -- Convert compressed format to raw bitmap
3. **Resize** -- Scale to encoder's expected resolution
4. **Normalize** -- Convert pixel values to float range
5. **Encode** -- Pass through vision encoder to get embeddings
6. **Interleave** -- Combine image embeddings with text tokens
7. **Inference** -- Run the LLM with combined input

```rust
use mullama::{Model, Context, ContextParams};
use mullama::multimodal::{MtmdContext, MtmdParams, ImageInput};
use std::sync::Arc;

let model = Arc::new(Model::load("llava-model.gguf")?);
let mut ctx = Context::new(model.clone(), ContextParams::default())?;
let mut mtmd = MtmdContext::new("mmproj.gguf", &model, MtmdParams::default())?;

// Load and process image
let image = mtmd.bitmap_from_file("photo.jpg")?;
let prompt = "What do you see? <__media__>";
let chunks = mtmd.tokenize(prompt, &[&image])?;

// Evaluate multimodal input
let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;
// Generate response from context...
```

## Audio Processing Pipeline

The audio processing pipeline:

1. **Load** -- Read audio from file/bytes/stream
2. **Decode** -- Convert compressed format to PCM samples
3. **Resample** -- Convert to model's expected sample rate
4. **Feature extraction** -- Compute mel spectrograms or other features
5. **Encode** -- Pass through audio encoder
6. **Interleave** -- Combine audio features with text tokens
7. **Inference** -- Run the LLM with combined input

```rust
use mullama::multimodal::{AudioInput, AudioFormat};

// Load audio
let audio = AudioInput::from_path("speech.wav").await?;
println!("Duration: {:.1}s, Rate: {}Hz", audio.duration, audio.sample_rate);

// Process with multimodal processor
let processor = MultimodalProcessor::new()
    .enable_audio_processing()
    .build();

let result = processor.process_audio(&audio).await?;
println!("Transcript: {:?}", result.transcript);
```

## MultimodalInput

Combined input supporting multiple modalities simultaneously.

```rust
#[derive(Debug, Clone)]
pub enum MultimodalInput {
    Text(String),
    Image { image: ImageInput, prompt: Option<String> },
    Audio { audio: AudioInput, context: Option<String> },
    Video { path: PathBuf, prompt: Option<String> },
    Mixed {
        text: Option<String>,
        image: Option<ImageInput>,
        audio: Option<AudioInput>,
        max_tokens: Option<usize>,
    },
}
```

## MultimodalOutput

Result from multimodal processing.

```rust
#[derive(Debug, Clone)]
pub struct MultimodalOutput {
    pub text_response: String,
    pub image_description: Option<String>,
    pub audio_transcript: Option<String>,
    pub video_description: Option<String>,
    pub confidence: f32,
    pub processing_time_ms: u64,
}
```

## Complete Example

```rust
use mullama::multimodal::{MultimodalProcessor, ImageInput, MultimodalInput};

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    let processor = MultimodalProcessor::new()
        .enable_image_processing()
        .enable_audio_processing()
        .build();

    // Image description
    let image = ImageInput::from_path("landscape.jpg").await?;
    let input = MultimodalInput::Image {
        image,
        prompt: Some("Describe what you see in detail.".to_string()),
    };

    let output = processor.process_multimodal(&input).await?;
    println!("Description: {}", output.text_response);
    println!("Confidence: {:.2}", output.confidence);
    println!("Processing time: {}ms", output.processing_time_ms);

    Ok(())
}
```
