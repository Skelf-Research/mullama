use super::{CrossAttentionConfig, ImageFormat, MultimodalConfig};

pub use super::audio::{
    convert_audio_format, extract_audio_features, load_audio_from_path, process_audio,
};

/// Create a basic image-to-text configuration.
pub fn image_to_text_config() -> MultimodalConfig {
    MultimodalConfig {
        enable_image_to_text: true,
        enable_text_to_image: false,
        ..Default::default()
    }
}

/// Create a configuration optimized for visual question answering.
pub fn vqa_config() -> MultimodalConfig {
    MultimodalConfig {
        max_image_resolution: (384, 384),
        vision_tokens_per_image: 196,
        cross_attention_config: CrossAttentionConfig {
            num_layers: 12,
            num_heads: 12,
            hidden_dim: 768,
            dropout_rate: 0.05,
        },
        temperature: 0.1,
        ..Default::default()
    }
}

/// Create a configuration for image captioning.
pub fn captioning_config() -> MultimodalConfig {
    MultimodalConfig {
        max_image_resolution: (224, 224),
        vision_tokens_per_image: 196,
        temperature: 0.8,
        ..Default::default()
    }
}

/// Validate image format compatibility.
pub fn validate_image_format(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Rgb | ImageFormat::Rgba | ImageFormat::Jpeg | ImageFormat::Png
    )
}

/// Calculate optimal batch size for multimodal processing.
pub fn calculate_optimal_batch_size(
    model_size: u64,
    available_memory: u64,
    image_resolution: (u32, u32),
) -> usize {
    let base_model_memory = model_size;
    let image_memory = (image_resolution.0 * image_resolution.1 * 3 * 4) as u64;
    let safety_factor = 0.8;

    let usable_memory = (available_memory as f64 * safety_factor) as u64;
    let memory_per_sample = base_model_memory / 10 + image_memory;

    std::cmp::max(1, (usable_memory / memory_per_sample) as usize)
}
