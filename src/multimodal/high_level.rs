//! High-level multimodal processor APIs backed by `mtmd`.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::MullamaError;
use crate::{Context, ContextParams, Model, SamplerParams};

use super::mtmd::{Bitmap, ChunkType, InputChunks, MtmdContext, MtmdParams};
use super::types::*;

/// Multimodal processor for handling text, image, and audio inputs.
pub struct MultimodalProcessor {
    model: Arc<Model>,
    generation_context: Context,
    mtmd_context: Option<MtmdContext>,
    config: MultimodalConfig,
    supported_modalities: Vec<Modality>,
    media_marker: String,
}

impl fmt::Debug for MultimodalProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultimodalProcessor")
            .field("config", &self.config)
            .field("supported_modalities", &self.supported_modalities)
            .field("has_mtmd_context", &self.mtmd_context.is_some())
            .field("media_marker", &self.media_marker)
            .finish()
    }
}

/// Configuration wrapper for the multimodal projector used by `mtmd`.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    projector_path: PathBuf,
    preprocess_config: ImagePreprocessConfig,
    encoder_type: VisionEncoderType,
    mtmd_params: MtmdParams,
}

impl MultimodalProcessor {
    /// Create a new multimodal processor.
    ///
    /// When `vision_encoder` is provided, the processor initializes a real
    /// `MtmdContext` using the projector path stored there.
    pub fn new(
        text_model: Model,
        vision_encoder: Option<VisionEncoder>,
        config: MultimodalConfig,
    ) -> Result<Self, MullamaError> {
        let model = Arc::new(text_model);
        let generation_context = Context::new(model.clone(), ContextParams::default())?;
        let mut supported_modalities = vec![Modality::Text];

        let (mtmd_context, media_marker) = if let Some(encoder) = vision_encoder {
            let media_marker = encoder.media_marker();
            let projector_path = encoder.projector_path_str()?.to_owned();
            let mtmd_params = encoder.mtmd_params;
            let mtmd = MtmdContext::new(&projector_path, &model, mtmd_params)?;

            if mtmd.supports_vision() {
                supported_modalities.push(Modality::Image);
                supported_modalities.push(Modality::Video);
            }
            if mtmd.supports_audio() {
                supported_modalities.push(Modality::Audio);
            }

            (Some(mtmd), media_marker)
        } else {
            (None, MtmdContext::default_marker())
        };

        Ok(Self {
            model,
            generation_context,
            mtmd_context,
            config,
            supported_modalities,
            media_marker,
        })
    }

    /// Load a multimodal model from files.
    pub fn from_files<P: AsRef<Path>>(
        text_model_path: P,
        vision_model_path: Option<P>,
        config: MultimodalConfig,
    ) -> Result<Self, MullamaError> {
        let text_model = Model::load(text_model_path)?;
        let vision_encoder = if let Some(vision_path) = vision_model_path {
            Some(VisionEncoder::from_file(vision_path)?)
        } else {
            None
        };

        Self::new(text_model, vision_encoder, config)
    }

    /// Process multimodal input and generate a response.
    pub fn generate(
        &mut self,
        input: &MultimodalInput,
        params: &MultimodalGenerationParams,
    ) -> Result<MultimodalOutput, MullamaError> {
        self.validate_input(input)?;
        self.generation_context.kv_cache_clear();

        let (text, image_features, prompt_tokens, completion_tokens) =
            if self.media_items(input) > 0 {
                self.generate_with_media(input, params)?
            } else {
                let prompt = input.text.as_deref().ok_or_else(|| {
                    MullamaError::InvalidInput(
                        "Multimodal input must include text or media".to_string(),
                    )
                })?;
                let prompt_tokens = self.model.tokenize(prompt, true, false)?;
                let text = self.generate_from_active_context(&prompt_tokens, params)?;
                (Some(text.0), None, prompt_tokens.len() as u32, text.1)
            };

        let mut metadata = HashMap::new();
        metadata.insert("prompt_tokens".to_string(), prompt_tokens as f64);
        metadata.insert("completion_tokens".to_string(), completion_tokens as f64);
        metadata.insert("images".to_string(), input.images.len() as f64);
        metadata.insert("audio".to_string(), input.audio.len() as f64);
        metadata.insert("videos".to_string(), input.videos.len() as f64);
        metadata.insert(
            "video_frames".to_string(),
            self.video_frame_count(input) as f64,
        );

        Ok(MultimodalOutput {
            text,
            image_features,
            attention_weights: if params.include_attention {
                Some(self.attention_weights()?)
            } else {
                None
            },
            metadata,
        })
    }

    /// Process a batch of multimodal inputs.
    pub fn generate_batch(
        &mut self,
        inputs: &[MultimodalInput],
        params: &MultimodalGenerationParams,
    ) -> Result<Vec<MultimodalOutput>, MullamaError> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.generate(input, params)?);
        }
        Ok(outputs)
    }

    /// Get supported modalities.
    pub fn supported_modalities(&self) -> &[Modality] {
        &self.supported_modalities
    }

    /// Check if a specific modality is supported.
    pub fn supports_modality(&self, modality: Modality) -> bool {
        self.supported_modalities.contains(&modality)
    }

    /// Update processor configuration.
    pub fn update_config(&mut self, config: MultimodalConfig) {
        self.config = config;
    }

    /// Get current configuration.
    pub fn config(&self) -> &MultimodalConfig {
        &self.config
    }

    fn generate_with_media(
        &mut self,
        input: &MultimodalInput,
        params: &MultimodalGenerationParams,
    ) -> Result<(Option<String>, Option<Vec<f32>>, u32, u32), MullamaError> {
        let prompt = compose_prompt(input, &self.media_marker)?;
        let bitmaps = self.build_bitmaps(input)?;
        let bitmap_refs: Vec<&Bitmap> = bitmaps.iter().collect();
        let generation_context = &mut self.generation_context;

        let mtmd = self.mtmd_context.as_mut().ok_or_else(|| {
            MullamaError::NotSupported(
                "Multimodal projector not configured; provide a VisionEncoder/mmproj file"
                    .to_string(),
            )
        })?;

        let chunks = mtmd.tokenize(&prompt, &bitmap_refs)?;
        let image_features = extract_image_features(mtmd, &chunks)?;
        let prompt_tokens = mtmd.eval_chunks(generation_context, &chunks, 0, 0, 512, true)? as u32;
        let (generated, completion_tokens) = self.generate_from_active_context(&[], params)?;

        Ok((
            Some(generated),
            image_features,
            prompt_tokens,
            completion_tokens,
        ))
    }

    fn generate_from_active_context(
        &mut self,
        prompt_tokens: &[i32],
        params: &MultimodalGenerationParams,
    ) -> Result<(String, u32), MullamaError> {
        if !prompt_tokens.is_empty() {
            self.generation_context.decode(prompt_tokens)?;
        }

        let sampler_params = self.sampler_params_from_generation(params);
        let mut sampler = sampler_params.build_chain(self.model.clone())?;
        let stop_sequences: Vec<String> = params
            .stop_sequences
            .iter()
            .filter(|stop| !stop.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences
            .iter()
            .map(|stop| stop.len())
            .max()
            .unwrap_or(0);

        let mut generated = String::new();
        let mut completion_tokens = 0u32;

        for _ in 0..params.max_tokens {
            let token = sampler.sample(&mut self.generation_context, -1);
            sampler.accept(token);

            if self.model.token_is_eog(token) {
                break;
            }

            let piece = self.model.token_to_str(token, 0, false)?;
            let previous_len = generated.len();
            generated.push_str(&piece);

            if let Some(pos) =
                find_stop_in_recent_window(&generated, previous_len, &stop_sequences, max_stop_len)
            {
                generated.truncate(pos);
                break;
            }

            self.generation_context.decode_single(token)?;
            completion_tokens += 1;
        }

        Ok((generated, completion_tokens))
    }

    fn build_bitmaps(&self, input: &MultimodalInput) -> Result<Vec<Bitmap>, MullamaError> {
        let mtmd = self.mtmd_context.as_ref().ok_or_else(|| {
            MullamaError::NotSupported(
                "Multimodal projector not configured; provide a VisionEncoder/mmproj file"
                    .to_string(),
            )
        })?;

        let mut bitmaps = Vec::with_capacity(self.media_items(input));

        for image in &input.images {
            bitmaps.push(self.bitmap_from_image(mtmd, image)?);
        }

        for video in &input.videos {
            for frame in &video.frames {
                bitmaps.push(self.bitmap_from_image(mtmd, frame)?);
            }
        }

        for audio in &input.audio {
            bitmaps.push(Bitmap::from_audio(&audio.samples)?);
        }

        Ok(bitmaps)
    }

    fn bitmap_from_image(
        &self,
        mtmd: &MtmdContext,
        image: &ImageInput,
    ) -> Result<Bitmap, MullamaError> {
        let (width, height) = image.dimensions;
        match image.format {
            ImageFormat::Rgb => Bitmap::from_image(width, height, &image.data),
            ImageFormat::Rgba => {
                let rgb = rgba_to_rgb(&image.data, width, height)?;
                Bitmap::from_image(width, height, &rgb)
            }
            ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::WebP => {
                mtmd.bitmap_from_buffer(&image.data)
            }
        }
    }

    fn media_items(&self, input: &MultimodalInput) -> usize {
        input.images.len() + self.video_frame_count(input) + input.audio.len()
    }

    fn video_frame_count(&self, input: &MultimodalInput) -> usize {
        input.videos.iter().map(|video| video.frames.len()).sum()
    }

    fn validate_input(&self, input: &MultimodalInput) -> Result<(), MullamaError> {
        if !input.images.is_empty() && !self.supports_modality(Modality::Image) {
            return Err(MullamaError::NotSupported(
                "Image processing not supported".to_string(),
            ));
        }

        if !input.videos.is_empty() && !self.supports_modality(Modality::Video) {
            return Err(MullamaError::NotSupported(
                "Video processing not supported".to_string(),
            ));
        }

        if !input.audio.is_empty() && !self.supports_modality(Modality::Audio) {
            return Err(MullamaError::NotSupported(
                "Audio processing not supported".to_string(),
            ));
        }

        if input.videos.iter().any(|video| video.frames.is_empty()) {
            return Err(MullamaError::InvalidInput(
                "Video input must contain at least one decoded frame".to_string(),
            ));
        }

        if (!input.images.is_empty() || !input.videos.is_empty())
            && !self.config.enable_image_to_text
        {
            return Err(MullamaError::NotSupported(
                "Image-to-text generation is disabled in the processor config".to_string(),
            ));
        }

        if self.media_items(input) == 0 && input.text.is_none() {
            return Err(MullamaError::InvalidInput(
                "Multimodal input must include text or media".to_string(),
            ));
        }

        Ok(())
    }

    fn attention_weights(&self) -> Result<AttentionWeights, MullamaError> {
        Err(MullamaError::NotSupported(
            "Attention weight extraction is not available in the current llama.cpp/mtmd FFI surface exposed by this crate".to_string(),
        ))
    }

    fn sampler_params_from_generation(&self, params: &MultimodalGenerationParams) -> SamplerParams {
        let mut sampler = SamplerParams::default();
        sampler.temperature = params.temperature;
        sampler.top_p = params.top_p;
        sampler.top_k = params.top_k as i32;
        sampler
    }
}

impl VisionEncoder {
    /// Create a multimodal projector wrapper from a projector file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(MullamaError::ModelLoadError(format!(
                "Multimodal projector file not found: {}",
                path.display()
            )));
        }

        Ok(Self {
            projector_path: path.to_path_buf(),
            preprocess_config: ImagePreprocessConfig::default(),
            encoder_type: VisionEncoderType::Custom,
            mtmd_params: MtmdParams::default(),
        })
    }

    /// Override the underlying `mtmd` initialization parameters.
    pub fn with_mtmd_params(mut self, params: MtmdParams) -> Self {
        self.mtmd_params = params;
        self
    }

    /// Expose the configured image preprocessing options.
    pub fn preprocess_config(&self) -> &ImagePreprocessConfig {
        &self.preprocess_config
    }

    /// Expose the logical encoder type metadata.
    pub fn encoder_type(&self) -> VisionEncoderType {
        self.encoder_type
    }

    /// Legacy helper retained for API compatibility.
    pub fn encode_images(&self, _images: &[ImageInput]) -> Result<Vec<f32>, MullamaError> {
        Err(MullamaError::NotSupported(
            "Standalone VisionEncoder image encoding is not supported; use MultimodalProcessor::generate with a projector-backed processor".to_string(),
        ))
    }

    fn projector_path_str(&self) -> Result<&str, MullamaError> {
        self.projector_path.to_str().ok_or_else(|| {
            MullamaError::InvalidInput("Projector path is not valid UTF-8".to_string())
        })
    }

    fn media_marker(&self) -> String {
        self.mtmd_params
            .media_marker
            .clone()
            .unwrap_or_else(MtmdContext::default_marker)
    }
}

fn rgba_to_rgb(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, MullamaError> {
    let expected_len = (width * height * 4) as usize;
    if data.len() != expected_len {
        return Err(MullamaError::InvalidInput(format!(
            "RGBA image data length {} doesn't match expected {} ({}x{}x4)",
            data.len(),
            expected_len,
            width,
            height
        )));
    }

    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for pixel in data.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
    }
    Ok(rgb)
}

fn compose_prompt(input: &MultimodalInput, media_marker: &str) -> Result<String, MullamaError> {
    let media_items = input.images.len()
        + input.audio.len()
        + input.videos.iter().map(|v| v.frames.len()).sum::<usize>();
    let base = input.text.as_deref().unwrap_or("").trim();

    if media_items == 0 {
        if base.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Multimodal input must include text or media".to_string(),
            ));
        }
        return Ok(base.to_string());
    }

    let marker_count = base.matches(media_marker).count();
    if marker_count > 0 && marker_count != media_items {
        return Err(MullamaError::InvalidInput(format!(
            "Prompt contains {} media markers but input provides {} media items",
            marker_count, media_items
        )));
    }

    if marker_count == media_items {
        return Ok(base.to_string());
    }

    let appended_markers = vec![media_marker; media_items].join(" ");
    if base.is_empty() {
        Ok(appended_markers)
    } else {
        Ok(format!("{} {}", base, appended_markers))
    }
}

fn extract_image_features(
    mtmd: &mut MtmdContext,
    chunks: &InputChunks,
) -> Result<Option<Vec<f32>>, MullamaError> {
    let mut features = Vec::new();
    let mut saw_image = false;

    for chunk in chunks.iter() {
        if chunk.chunk_type() != ChunkType::Image {
            continue;
        }
        saw_image = true;
        mtmd.encode_chunk(&chunk)?;
        if let Some(embeddings) = mtmd.get_output_embeddings(&chunk) {
            features.extend_from_slice(embeddings);
        }
    }

    if saw_image {
        Ok(Some(features))
    } else {
        Ok(None)
    }
}

fn find_stop_in_recent_window(
    generated: &str,
    previous_len: usize,
    stop_sequences: &[String],
    max_stop_len: usize,
) -> Option<usize> {
    if stop_sequences.is_empty() || max_stop_len == 0 {
        return None;
    }

    let mut start = previous_len.saturating_sub(max_stop_len.saturating_sub(1));
    while start > 0 && !generated.is_char_boundary(start) {
        start -= 1;
    }

    let window = &generated[start..];
    for stop in stop_sequences {
        if let Some(relative_pos) = window.find(stop) {
            return Some(start + relative_pos);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_prompt_appends_missing_markers() {
        let mut input = MultimodalInput::new();
        input.set_text("Describe this");
        input.images.push(test_image(ImageFormat::Rgb));
        input.audio.push(test_audio());

        let prompt = compose_prompt(&input, "<media>").unwrap();
        assert_eq!(prompt, "Describe this <media> <media>");
    }

    #[test]
    fn compose_prompt_rejects_marker_mismatch() {
        let mut input = MultimodalInput::new();
        input.set_text("Describe <media>");
        input.images.push(test_image(ImageFormat::Rgb));
        input.audio.push(test_audio());

        let err = compose_prompt(&input, "<media>").unwrap_err();
        assert!(matches!(err, MullamaError::InvalidInput(_)));
    }

    #[test]
    fn compose_prompt_counts_video_frames_as_media_items() {
        let mut input = MultimodalInput::new();
        input.set_text("Summarize");
        input.videos.push(VideoInput {
            frames: vec![test_image(ImageFormat::Rgb), test_image(ImageFormat::Rgb)],
            fps: 2.0,
            duration: 1.0,
            description: None,
        });

        let prompt = compose_prompt(&input, "<media>").unwrap();
        assert_eq!(prompt, "Summarize <media> <media>");
    }

    #[test]
    fn rgba_conversion_drops_alpha_channel() {
        let rgba = vec![1, 2, 3, 255, 4, 5, 6, 0];
        let rgb = rgba_to_rgb(&rgba, 2, 1).unwrap();
        assert_eq!(rgb, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn stop_search_finds_sequence_inside_last_piece() {
        let stops = vec!["</s>".to_string()];
        let generated = "hello</s> trailing";
        let pos = find_stop_in_recent_window(generated, 5, &stops, 4).unwrap();
        assert_eq!(pos, 5);
    }

    fn test_image(format: ImageFormat) -> ImageInput {
        ImageInput {
            data: vec![0, 0, 0],
            dimensions: (1, 1),
            format,
            caption: None,
        }
    }

    fn test_audio() -> AudioInput {
        AudioInput {
            samples: vec![0.0, 0.1],
            sample_rate: 16_000,
            channels: 1,
            duration: 0.1,
            format: AudioFormat {
                container: "wav".to_string(),
                codec: "pcm".to_string(),
                bit_depth: 16,
                bitrate: None,
            },
            transcript: None,
            metadata: HashMap::new(),
        }
    }
}
