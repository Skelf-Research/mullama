//! Comprehensive multimodal processing for text, image, and audio integration
//!
//! This module provides advanced multimodal processing capabilities, enabling seamless
//! integration of text, image, and audio data for sophisticated AI applications.
//!
//! ## Features
//!
//! - **Vision-Language Models**: Process text and visual inputs together
//! - **Audio Processing**: Handle audio input/output with speech-to-text/text-to-speech
//! - **Cross-Modal Understanding**: Combine multiple modalities for richer context
//! - **Batch Processing**: Efficient processing of multimodal datasets
//! - **Format Support**: Wide range of image and audio formats
//! - **Pipeline Integration**: Seamless integration with generation pipelines

mod mtmd;
mod types;
mod vision;

// Re-export all public types
pub use mtmd::{Bitmap, ChunkType, InputChunk, InputChunks, MtmdContext, MtmdParams};
pub use types::*;
pub use vision::{MultimodalProcessor, VisionEncoder};

use crate::error::MullamaError;
use std::collections::HashMap;
use std::path::Path;

#[cfg(all(feature = "multimodal", feature = "async"))]
use crate::AsyncModel;

#[cfg(feature = "multimodal")]
use tokio::{fs, io::AsyncReadExt};

/// Utility functions for multimodal processing
pub mod utils {
    use super::*;

    /// Create a basic image-to-text configuration
    pub fn image_to_text_config() -> MultimodalConfig {
        MultimodalConfig {
            enable_image_to_text: true,
            enable_text_to_image: false,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for visual question answering
    pub fn vqa_config() -> MultimodalConfig {
        MultimodalConfig {
            max_image_resolution: (384, 384),
            vision_tokens_per_image: 196, // 14x14 patches
            cross_attention_config: CrossAttentionConfig {
                num_layers: 12,
                num_heads: 12,
                hidden_dim: 768,
                dropout_rate: 0.05,
            },
            temperature: 0.1, // Lower temperature for factual answers
            ..Default::default()
        }
    }

    /// Create a configuration for image captioning
    pub fn captioning_config() -> MultimodalConfig {
        MultimodalConfig {
            max_image_resolution: (224, 224),
            vision_tokens_per_image: 196,
            temperature: 0.8, // Higher temperature for creative captions
            ..Default::default()
        }
    }

    /// Validate image format compatibility
    pub fn validate_image_format(format: ImageFormat) -> bool {
        matches!(
            format,
            ImageFormat::Rgb | ImageFormat::Rgba | ImageFormat::Jpeg | ImageFormat::Png
        )
    }

    /// Calculate optimal batch size for multimodal processing
    pub fn calculate_optimal_batch_size(
        model_size: u64,
        available_memory: u64,
        image_resolution: (u32, u32),
    ) -> usize {
        let base_model_memory = model_size;
        let image_memory = (image_resolution.0 * image_resolution.1 * 3 * 4) as u64; // RGB, f32
        let safety_factor = 0.8; // Use 80% of available memory

        let usable_memory = (available_memory as f64 * safety_factor) as u64;
        let memory_per_sample = base_model_memory / 10 + image_memory; // Rough estimate

        std::cmp::max(1, (usable_memory / memory_per_sample) as usize)
    }

    /// Create an audio input from file path
    pub async fn load_audio_from_path(path: impl AsRef<Path>) -> Result<AudioInput, MullamaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(MullamaError::ConfigError(format!(
                "Audio file not found: {}",
                path.display()
            )));
        }

        // Placeholder for actual audio loading
        // In real implementation, this would use libraries like rodio, symphonia, etc.
        let samples = vec![0.0; 44100]; // 1 second of silence
        let format = AudioFormat {
            container: "wav".to_string(),
            codec: "pcm".to_string(),
            bit_depth: 16,
            bitrate: None,
        };

        Ok(AudioInput {
            samples,
            sample_rate: 44100,
            channels: 1,
            duration: 1.0,
            format,
            transcript: None,
            metadata: HashMap::new(),
        })
    }

    /// Process audio with noise reduction and normalization
    pub fn process_audio(
        audio: &mut AudioInput,
        config: &AudioProcessingConfig,
    ) -> Result<(), MullamaError> {
        if config.enable_noise_reduction {
            apply_noise_reduction(&mut audio.samples);
        }

        if config.enable_agc {
            apply_automatic_gain_control(&mut audio.samples);
        }

        // Resample if needed
        if audio.sample_rate != config.default_sample_rate {
            audio.samples = resample_audio(
                &audio.samples,
                audio.sample_rate,
                config.default_sample_rate,
            )?;
            audio.sample_rate = config.default_sample_rate;
        }

        Ok(())
    }

    /// Convert between audio formats
    pub fn convert_audio_format(
        input: &AudioInput,
        target_format: &AudioFormat,
    ) -> Result<AudioInput, MullamaError> {
        // Placeholder for audio format conversion
        let mut output = input.clone();
        output.format = target_format.clone();
        Ok(output)
    }

    /// Extract audio features for analysis
    pub fn extract_audio_features(audio: &AudioInput) -> AudioFeatures {
        // Placeholder for actual feature extraction
        AudioFeatures {
            duration: audio.duration,
            energy: calculate_energy(&audio.samples),
            zero_crossing_rate: calculate_zero_crossing_rate(&audio.samples),
            spectral_centroid: 1000.0,           // Placeholder
            mfcc: vec![0.1, 0.2, 0.3, 0.4, 0.5], // 5 MFCC coefficients
            pitch: detect_pitch(&audio.samples, audio.sample_rate),
            tempo: detect_tempo(&audio.samples, audio.sample_rate),
            has_speech: detect_speech(&audio.samples),
        }
    }
}

// Audio processing helper functions
fn apply_noise_reduction(samples: &mut [f32]) {
    // Placeholder for noise reduction algorithm
    // In real implementation, this would use spectral subtraction or Wiener filtering
    for sample in samples.iter_mut() {
        if sample.abs() < 0.01 {
            *sample = 0.0; // Simple noise gate
        }
    }
}

fn apply_automatic_gain_control(samples: &mut [f32]) {
    // Simple AGC implementation
    let max_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    if max_amplitude > 0.0 {
        let gain = 0.8 / max_amplitude; // Normalize to 80% of full scale
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, MullamaError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Simple linear interpolation resampling (placeholder)
    let ratio = to_rate as f32 / from_rate as f32;
    let new_length = (samples.len() as f32 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let original_index = i as f32 / ratio;
        let index = original_index as usize;

        if index < samples.len() - 1 {
            let frac = original_index - index as f32;
            let sample = samples[index] * (1.0 - frac) + samples[index + 1] * frac;
            resampled.push(sample);
        } else if index < samples.len() {
            resampled.push(samples[index]);
        }
    }

    Ok(resampled)
}

fn calculate_energy(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32
}

fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
    let mut crossings = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / samples.len() as f32
}

fn detect_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    // Placeholder for pitch detection (would use autocorrelation or YIN algorithm)
    440.0 // A4 note
}

fn detect_tempo(samples: &[f32], sample_rate: u32) -> f32 {
    // Placeholder for tempo detection
    120.0 // 120 BPM
}

fn detect_speech(samples: &[f32]) -> bool {
    // Simple speech detection based on energy and zero-crossing rate
    let energy = calculate_energy(samples);
    let zcr = calculate_zero_crossing_rate(samples);

    // Heuristic thresholds for speech detection
    energy > 0.01 && zcr > 0.1 && zcr < 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_input_creation() {
        let mut input = MultimodalInput::new();
        input.set_text("Describe this image");
        input.add_image(vec![128u8; 224 * 224 * 3], (224, 224), ImageFormat::Rgb);

        assert!(input.text.is_some());
        assert_eq!(input.images.len(), 1);
        assert_eq!(input.images[0].dimensions, (224, 224));
    }

    #[test]
    fn test_config_defaults() {
        let config = MultimodalConfig::default();
        assert_eq!(config.max_image_resolution, (512, 512));
        assert_eq!(config.patch_size, 16);
        assert!(config.enable_image_to_text);
        assert!(!config.enable_text_to_image);
    }

    #[test]
    fn test_image_preprocessing_config() {
        let config = ImagePreprocessConfig::default();
        assert_eq!(config.target_size, (224, 224));
        assert_eq!(config.mean, [0.485, 0.456, 0.406]);
        assert_eq!(config.std, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_modality_support() {
        let _config = MultimodalConfig::default();
        // Placeholder for test model - skip for now as we don't have a model in tests
    }

    #[test]
    fn test_generation_params() {
        let params = MultimodalGenerationParams::default();
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 40);
        assert!(!params.include_attention);
    }

    #[test]
    fn test_utility_configs() {
        let vqa_config = utils::vqa_config();
        assert_eq!(vqa_config.temperature, 0.1);
        assert_eq!(vqa_config.cross_attention_config.num_layers, 12);

        let caption_config = utils::captioning_config();
        assert_eq!(caption_config.temperature, 0.8);
    }

    #[test]
    fn test_batch_size_calculation() {
        let batch_size = utils::calculate_optimal_batch_size(
            1_000_000_000, // 1GB model
            8_000_000_000, // 8GB available memory
            (224, 224),    // Image resolution
        );

        assert!(batch_size > 0);
        assert!(batch_size < 100); // Should be reasonable
    }
}
