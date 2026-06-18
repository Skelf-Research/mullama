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

mod audio;
mod high_level;
mod mtmd;
mod types;
pub mod utils;

// Re-export all public types
pub use high_level::{MultimodalProcessor, VisionEncoder};
pub use mtmd::{Bitmap, ChunkType, InputChunk, InputChunks, MtmdContext, MtmdParams};
pub use types::*;

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
