//! Multimodal types, enums, and configuration structs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::error::MullamaError;

/// Types of vision encoders
#[derive(Debug, Clone, Copy)]
pub enum VisionEncoderType {
    /// CLIP-style encoder
    Clip,
    /// DINOv2 encoder
    Dino,
    /// Custom vision encoder
    Custom,
}

/// Supported modalities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modality {
    /// Text input/output
    Text,
    /// Image input
    Image,
    /// Video input (experimental)
    Video,
    /// Audio input (experimental)
    Audio,
}

/// Multimodal configuration
#[derive(Debug, Clone)]
pub struct MultimodalConfig {
    /// Maximum image resolution
    pub max_image_resolution: (u32, u32),
    /// Image patch size for vision transformer
    pub patch_size: u32,
    /// Number of vision tokens per image
    pub vision_tokens_per_image: usize,
    /// Enable image-to-text generation
    pub enable_image_to_text: bool,
    /// Enable text-to-image generation (experimental)
    pub enable_text_to_image: bool,
    /// Cross-attention configuration
    pub cross_attention_config: CrossAttentionConfig,
    /// Temperature for multimodal generation
    pub temperature: f32,
}

/// Cross-attention configuration for multimodal fusion
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Number of cross-attention layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout_rate: f32,
}

/// Image preprocessing configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    /// Target image size
    pub target_size: (u32, u32),
    /// Normalization mean values (RGB)
    pub mean: [f32; 3],
    /// Normalization standard deviation values (RGB)
    pub std: [f32; 3],
    /// Whether to resize and center crop
    pub resize_and_crop: bool,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Image interpolation methods
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

/// Multimodal input combining text and visual data
#[derive(Debug)]
pub struct MultimodalInput {
    /// Text prompt
    pub text: Option<String>,
    /// Image data
    pub images: Vec<ImageInput>,
    /// Video data (experimental)
    pub videos: Vec<VideoInput>,
    /// Audio data (experimental)
    pub audio: Vec<AudioInput>,
    /// Input metadata
    pub metadata: HashMap<String, String>,
}

/// Image input data
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Image data (RGB bytes)
    pub data: Vec<u8>,
    /// Image dimensions (width, height)
    pub dimensions: (u32, u32),
    /// Image format
    pub format: ImageFormat,
    /// Optional caption or description
    pub caption: Option<String>,
}

/// Supported image formats
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    /// RGB format
    Rgb,
    /// RGBA format
    Rgba,
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// WebP format
    WebP,
}

/// Video input data (experimental)
#[derive(Debug, Clone)]
pub struct VideoInput {
    /// Frame data
    pub frames: Vec<ImageInput>,
    /// Frame rate
    pub fps: f32,
    /// Duration in seconds
    pub duration: f32,
    /// Optional description
    pub description: Option<String>,
}

/// Enhanced audio input data with comprehensive format support
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// Audio samples (normalized to -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Audio format information
    pub format: AudioFormat,
    /// Optional transcript for speech audio
    pub transcript: Option<String>,
    /// Audio metadata (artist, title, etc.)
    pub metadata: HashMap<String, String>,
}

/// Enhanced audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Container format (wav, mp3, flac, etc.)
    pub container: String,
    /// Codec used (pcm, mp3, flac, aac, etc.)
    pub codec: String,
    /// Bit depth (8, 16, 24, 32)
    pub bit_depth: u16,
    /// Bitrate for compressed formats
    pub bitrate: Option<u32>,
}

/// Audio processor for advanced audio processing
#[cfg(feature = "multimodal")]
pub struct AudioProcessor {
    pub(crate) config: AudioProcessingConfig,
    pub(crate) supported_formats: Vec<String>,
}

/// Configuration for audio processing
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct AudioProcessingConfig {
    /// Default sample rate for processing
    pub default_sample_rate: u32,
    /// Default number of channels
    pub default_channels: u16,
    /// Maximum audio duration in seconds
    pub max_duration: Duration,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// Speech-to-text configuration
    pub stt_config: Option<SpeechToTextConfig>,
    /// Text-to-speech configuration
    pub tts_config: Option<TextToSpeechConfig>,
}

/// Speech-to-text configuration
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct SpeechToTextConfig {
    /// Language model for transcription
    pub language: String,
    /// Enable speaker identification
    pub enable_speaker_id: bool,
    /// Enable confidence scores
    pub enable_confidence: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

/// Text-to-speech configuration
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct TextToSpeechConfig {
    /// Voice to use for synthesis
    pub voice: String,
    /// Speaking rate (0.5 = half speed, 2.0 = double speed)
    pub rate: f32,
    /// Pitch adjustment (-1.0 to 1.0)
    pub pitch: f32,
    /// Volume level (0.0 to 1.0)
    pub volume: f32,
    /// Output audio format
    pub output_format: AudioFormat,
}

/// Multimodal generation output
#[derive(Debug)]
pub struct MultimodalOutput {
    /// Generated text
    pub text: Option<String>,
    /// Generated image features (for text-to-image)
    pub image_features: Option<Vec<f32>>,
    /// Attention weights for interpretability
    pub attention_weights: Option<AttentionWeights>,
    /// Generation metadata
    pub metadata: HashMap<String, f64>,
}

/// Attention weights for multimodal interpretability
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Text-to-image attention weights
    pub text_to_image: Vec<Vec<f32>>,
    /// Image-to-text attention weights
    pub image_to_text: Vec<Vec<f32>>,
    /// Self-attention weights
    pub self_attention: Vec<Vec<f32>>,
}

/// Multimodal generation parameters
#[derive(Debug, Clone)]
pub struct MultimodalGenerationParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Whether to include attention weights in output
    pub include_attention: bool,
    /// Custom stopping criteria
    pub stop_sequences: Vec<String>,
}

/// Audio feature extraction results
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub duration: f32,
    pub energy: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
    pub mfcc: Vec<f32>,
    pub pitch: f32,
    pub tempo: f32,
    pub has_speech: bool,
}

// ---- Default impls ----

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            max_image_resolution: (512, 512),
            patch_size: 16,
            vision_tokens_per_image: 256,
            enable_image_to_text: true,
            enable_text_to_image: false,
            cross_attention_config: CrossAttentionConfig::default(),
            temperature: 0.7,
        }
    }
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            num_heads: 8,
            hidden_dim: 768,
            dropout_rate: 0.1,
        }
    }
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (224, 224),
            mean: [0.485, 0.456, 0.406], // ImageNet normalization
            std: [0.229, 0.224, 0.225],  // ImageNet normalization
            resize_and_crop: true,
            interpolation: InterpolationMethod::Bilinear,
        }
    }
}

impl Default for MultimodalGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            include_attention: false,
            stop_sequences: vec!["<|end|>".to_string(), "</s>".to_string()],
        }
    }
}

impl MultimodalInput {
    /// Create a new multimodal input
    pub fn new() -> Self {
        Self {
            text: None,
            images: Vec::new(),
            videos: Vec::new(),
            audio: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set text prompt
    pub fn set_text<S: Into<String>>(&mut self, text: S) {
        self.text = Some(text.into());
    }

    /// Add an image from raw data
    pub fn add_image(&mut self, data: Vec<u8>, dimensions: (u32, u32), format: ImageFormat) {
        self.images.push(ImageInput {
            data,
            dimensions,
            format,
            caption: None,
        });
    }

    /// Add an image from file path
    pub fn add_image_from_path<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), MullamaError> {
        // This would load and decode the image file
        // For now, return a placeholder implementation
        let placeholder_data = vec![128u8; 224 * 224 * 3]; // 224x224 RGB
        self.add_image(placeholder_data, (224, 224), ImageFormat::Rgb);
        Ok(())
    }

    /// Add metadata
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }
}
