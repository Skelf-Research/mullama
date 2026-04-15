//! Multimodal types, enums, and configuration structs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::error::MullamaError;

#[cfg(feature = "multimodal")]
use image::AnimationDecoder;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[allow(dead_code)]
    pub(crate) config: AudioProcessingConfig,
    #[allow(dead_code)]
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

    /// Add an image from file path.
    #[cfg(feature = "multimodal")]
    pub fn add_image_from_path<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), MullamaError> {
        let image = ImageInput::from_path(path)?;
        self.images.push(image);
        Ok(())
    }

    /// Add an image from file path.
    #[cfg(not(feature = "multimodal"))]
    pub fn add_image_from_path<P: AsRef<std::path::Path>>(
        &mut self,
        _path: P,
    ) -> Result<(), MullamaError> {
        Err(MullamaError::FeatureNotAvailable(
            "Image loading from path requires the `multimodal` feature".to_string(),
        ))
    }

    /// Add a video from file path.
    #[cfg(feature = "multimodal")]
    pub fn add_video_from_path<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), MullamaError> {
        let video = VideoInput::from_path(path)?;
        self.videos.push(video);
        Ok(())
    }

    /// Add a video from file path.
    #[cfg(not(feature = "multimodal"))]
    pub fn add_video_from_path<P: AsRef<std::path::Path>>(
        &mut self,
        _path: P,
    ) -> Result<(), MullamaError> {
        Err(MullamaError::FeatureNotAvailable(
            "Video loading from path requires the `multimodal` feature".to_string(),
        ))
    }

    /// Add metadata
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }
}

impl ImageInput {
    /// Load an image from disk into RGB bytes.
    #[cfg(feature = "multimodal")]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self, MullamaError> {
        let path = path.as_ref();
        let reader = image::io::Reader::open(path).map_err(|e| {
            MullamaError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to open image {}: {}", path.display(), e),
            ))
        })?;

        let reader = reader.with_guessed_format().map_err(|e| {
            MullamaError::MultimodalError(format!(
                "Failed to detect image format for {}: {}",
                path.display(),
                e
            ))
        })?;
        let guessed = reader.format();
        let image = reader.decode().map_err(|e| {
            MullamaError::MultimodalError(format!(
                "Failed to decode image {}: {}",
                path.display(),
                e
            ))
        })?;

        let rgb = image.to_rgb8();
        Ok(Self {
            data: rgb.into_raw(),
            dimensions: (image.width(), image.height()),
            format: guessed
                .map(ImageFormat::from_image_crate)
                .unwrap_or(ImageFormat::Rgb),
            caption: None,
        })
    }
}

impl VideoInput {
    /// Load a video-like input from disk.
    ///
    /// Real support covers animated GIF decoding via the `image` crate and,
    /// when `format-conversion` is enabled, general video decoding via FFmpeg.
    #[cfg(feature = "multimodal")]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self, MullamaError> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());

        match ext.as_deref() {
            Some("gif") => Self::from_gif_path(path),
            #[cfg(feature = "format-conversion")]
            _ => Self::from_video_path_ffmpeg(path),
            #[cfg(not(feature = "format-conversion"))]
            _ => Err(MullamaError::NotSupported(format!(
                "Video loading currently supports animated GIF files by default. Enable `format-conversion` for FFmpeg-backed formats such as MP4/WebM. Unsupported path: {}",
                path.display()
            ))),
        }
    }

    #[cfg(feature = "multimodal")]
    fn from_gif_path(path: &std::path::Path) -> Result<Self, MullamaError> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let decoder = image::codecs::gif::GifDecoder::new(BufReader::new(file)).map_err(|e| {
            MullamaError::MultimodalError(format!("Failed to open GIF {}: {}", path.display(), e))
        })?;

        let frames = decoder.into_frames().collect_frames().map_err(|e| {
            MullamaError::MultimodalError(format!(
                "Failed to decode GIF frames from {}: {}",
                path.display(),
                e
            ))
        })?;

        if frames.is_empty() {
            return Err(MullamaError::InvalidInput(format!(
                "Animated GIF {} contained no frames",
                path.display()
            )));
        }

        let mut image_frames = Vec::with_capacity(frames.len());
        let mut total_delay_ms = 0u64;

        for frame in frames {
            let delay = frame.delay();
            let (numer_ms, denom_ms) = delay.numer_denom_ms();
            if denom_ms > 0 {
                total_delay_ms += (numer_ms as u64) / (denom_ms as u64);
            }

            let buffer = frame.into_buffer();
            let (width, height) = buffer.dimensions();
            let rgb = image::DynamicImage::ImageRgba8(buffer).to_rgb8();
            image_frames.push(ImageInput {
                data: rgb.into_raw(),
                dimensions: (width, height),
                format: ImageFormat::Rgb,
                caption: None,
            });
        }

        let duration = if total_delay_ms > 0 {
            total_delay_ms as f32 / 1000.0
        } else {
            image_frames.len() as f32 / 10.0
        };
        let fps = if duration > 0.0 {
            image_frames.len() as f32 / duration
        } else {
            0.0
        };

        Ok(Self {
            frames: image_frames,
            fps,
            duration,
            description: None,
        })
    }

    #[cfg(all(feature = "multimodal", feature = "format-conversion"))]
    fn from_video_path_ffmpeg(path: &std::path::Path) -> Result<Self, MullamaError> {
        use ffmpeg_sys_next as ffmpeg;
        use std::ffi::CString;
        use std::ptr;

        let path_string = path.to_string_lossy().into_owned();
        let c_path = CString::new(path_string.as_bytes()).map_err(|_| {
            MullamaError::InvalidInput(format!(
                "Video path contains an interior NUL byte: {}",
                path.display()
            ))
        })?;

        unsafe {
            ffmpeg_check(
                ffmpeg::avformat_network_init(),
                "initialize FFmpeg network components",
            )?;

            let mut format_context = ptr::null_mut();
            ffmpeg_check(
                ffmpeg::avformat_open_input(
                    &mut format_context,
                    c_path.as_ptr(),
                    ptr::null(),
                    ptr::null_mut(),
                ),
                &format!("open video {}", path.display()),
            )?;
            let input = AvFormatInput(format_context);

            ffmpeg_check(
                ffmpeg::avformat_find_stream_info(input.0, ptr::null_mut()),
                &format!("read stream information from {}", path.display()),
            )?;

            let mut decoder = ptr::null();
            let video_stream_index = ffmpeg::av_find_best_stream(
                input.0,
                ffmpeg::AVMediaType::AVMEDIA_TYPE_VIDEO,
                -1,
                -1,
                &mut decoder,
                0,
            );
            if video_stream_index < 0 {
                return Err(MullamaError::InvalidInput(format!(
                    "No video stream found in {}",
                    path.display()
                )));
            }

            let stream_ptr = *(*input.0).streams.add(video_stream_index as usize);
            if stream_ptr.is_null() || (*stream_ptr).codecpar.is_null() {
                return Err(MullamaError::MultimodalError(format!(
                    "FFmpeg returned an invalid video stream for {}",
                    path.display()
                )));
            }

            if decoder.is_null() {
                decoder = ffmpeg::avcodec_find_decoder((*(*stream_ptr).codecpar).codec_id);
            }
            if decoder.is_null() {
                return Err(MullamaError::MultimodalError(format!(
                    "No FFmpeg decoder available for {}",
                    path.display()
                )));
            }

            let codec_context = AvCodecContextHandle::new(decoder, path)?;
            ffmpeg_check(
                ffmpeg::avcodec_parameters_to_context(codec_context.0, (*stream_ptr).codecpar),
                &format!("copy codec parameters for {}", path.display()),
            )?;
            ffmpeg_check(
                ffmpeg::avcodec_open2(codec_context.0, decoder, ptr::null_mut()),
                &format!("open video decoder for {}", path.display()),
            )?;

            let packet = AvPacketHandle::new(path)?;
            let decoded_frame = AvFrameHandle::new(path, "decoded video frame")?;
            let mut scaler = None;
            let mut frames = Vec::new();

            loop {
                let read_result = ffmpeg::av_read_frame(input.0, packet.0);
                if read_result == FFMPEG_AVERROR_EOF {
                    break;
                }
                if read_result < 0 {
                    return Err(ffmpeg_error(
                        read_result,
                        &format!("read frames from {}", path.display()),
                    ));
                }

                if (*packet.0).stream_index == video_stream_index {
                    ffmpeg_check(
                        ffmpeg::avcodec_send_packet(codec_context.0, packet.0),
                        &format!("send packet to decoder for {}", path.display()),
                    )?;
                    drain_video_frames(
                        codec_context.0,
                        decoded_frame.0,
                        &mut scaler,
                        &mut frames,
                        path,
                    )?;
                }

                ffmpeg::av_packet_unref(packet.0);
            }

            ffmpeg_check(
                ffmpeg::avcodec_send_packet(codec_context.0, ptr::null()),
                &format!("flush decoder for {}", path.display()),
            )?;
            drain_video_frames(
                codec_context.0,
                decoded_frame.0,
                &mut scaler,
                &mut frames,
                path,
            )?;

            if frames.is_empty() {
                return Err(MullamaError::InvalidInput(format!(
                    "No decodable video frames found in {}",
                    path.display()
                )));
            }

            let stream = &*stream_ptr;
            let fps = rational_to_f32(stream.avg_frame_rate)
                .filter(|fps| *fps > 0.0)
                .or_else(|| rational_to_f32(stream.r_frame_rate).filter(|fps| *fps > 0.0))
                .unwrap_or(0.0);
            let duration = if stream.duration > 0 {
                rational_to_f32(stream.time_base)
                    .map(|base| stream.duration as f32 * base)
                    .unwrap_or(0.0)
            } else if (*input.0).duration > 0 {
                (*input.0).duration as f32 / 1_000_000.0
            } else if fps > 0.0 {
                frames.len() as f32 / fps
            } else {
                0.0
            };

            Ok(Self {
                frames,
                fps,
                duration,
                description: None,
            })
        }
    }
}

impl ImageFormat {
    #[cfg(feature = "multimodal")]
    fn from_image_crate(format: image::ImageFormat) -> Self {
        match format {
            image::ImageFormat::Jpeg => ImageFormat::Jpeg,
            image::ImageFormat::Png => ImageFormat::Png,
            image::ImageFormat::WebP => ImageFormat::WebP,
            _ => ImageFormat::Rgb,
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
const FFMPEG_AVERROR_EOF: i32 = -541_478_725;

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
#[cfg(unix)]
const FFMPEG_AVERROR_AGAIN: i32 = -libc::EAGAIN;

#[cfg(all(feature = "multimodal", feature = "format-conversion", not(unix)))]
const FFMPEG_AVERROR_AGAIN: i32 = -11;

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
struct AvFormatInput(*mut ffmpeg_sys_next::AVFormatContext);

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl Drop for AvFormatInput {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffmpeg_sys_next::avformat_close_input(&mut self.0);
            }
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
struct AvCodecContextHandle(*mut ffmpeg_sys_next::AVCodecContext);

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl AvCodecContextHandle {
    unsafe fn new(
        codec: *const ffmpeg_sys_next::AVCodec,
        path: &std::path::Path,
    ) -> Result<Self, MullamaError> {
        let context = ffmpeg_sys_next::avcodec_alloc_context3(codec);
        if context.is_null() {
            Err(MullamaError::MultimodalError(format!(
                "Failed to allocate FFmpeg codec context for {}",
                path.display()
            )))
        } else {
            Ok(Self(context))
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl Drop for AvCodecContextHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffmpeg_sys_next::avcodec_free_context(&mut self.0);
            }
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
struct AvPacketHandle(*mut ffmpeg_sys_next::AVPacket);

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl AvPacketHandle {
    unsafe fn new(path: &std::path::Path) -> Result<Self, MullamaError> {
        let packet = ffmpeg_sys_next::av_packet_alloc();
        if packet.is_null() {
            Err(MullamaError::MultimodalError(format!(
                "Failed to allocate FFmpeg packet for {}",
                path.display()
            )))
        } else {
            Ok(Self(packet))
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl Drop for AvPacketHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffmpeg_sys_next::av_packet_free(&mut self.0);
            }
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
struct AvFrameHandle(*mut ffmpeg_sys_next::AVFrame);

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl AvFrameHandle {
    unsafe fn new(path: &std::path::Path, label: &str) -> Result<Self, MullamaError> {
        let frame = ffmpeg_sys_next::av_frame_alloc();
        if frame.is_null() {
            Err(MullamaError::MultimodalError(format!(
                "Failed to allocate FFmpeg {} for {}",
                label,
                path.display()
            )))
        } else {
            Ok(Self(frame))
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl Drop for AvFrameHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffmpeg_sys_next::av_frame_free(&mut self.0);
            }
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
struct VideoScaler {
    context: *mut ffmpeg_sys_next::SwsContext,
    frame: *mut ffmpeg_sys_next::AVFrame,
    width: i32,
    height: i32,
    source_format: ffmpeg_sys_next::AVPixelFormat,
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl VideoScaler {
    unsafe fn new(
        width: i32,
        height: i32,
        source_format: ffmpeg_sys_next::AVPixelFormat,
        path: &std::path::Path,
    ) -> Result<Self, MullamaError> {
        let context = ffmpeg_sys_next::sws_getContext(
            width,
            height,
            source_format,
            width,
            height,
            ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
            ffmpeg_sys_next::SwsFlags::SWS_BILINEAR as i32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        );
        if context.is_null() {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to create FFmpeg scaler for {}",
                path.display()
            )));
        }

        let frame = ffmpeg_sys_next::av_frame_alloc();
        if frame.is_null() {
            ffmpeg_sys_next::sws_freeContext(context);
            return Err(MullamaError::MultimodalError(format!(
                "Failed to allocate RGB frame for {}",
                path.display()
            )));
        }

        (*frame).format = ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24 as i32;
        (*frame).width = width;
        (*frame).height = height;
        ffmpeg_check(
            ffmpeg_sys_next::av_frame_get_buffer(frame, 1),
            &format!("allocate RGB frame buffer for {}", path.display()),
        )?;

        Ok(Self {
            context,
            frame,
            width,
            height,
            source_format,
        })
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
impl Drop for VideoScaler {
    fn drop(&mut self) {
        unsafe {
            if !self.frame.is_null() {
                ffmpeg_sys_next::av_frame_free(&mut self.frame);
            }
            if !self.context.is_null() {
                ffmpeg_sys_next::sws_freeContext(self.context);
            }
        }
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
unsafe fn drain_video_frames(
    codec_context: *mut ffmpeg_sys_next::AVCodecContext,
    decoded_frame: *mut ffmpeg_sys_next::AVFrame,
    scaler: &mut Option<VideoScaler>,
    frames: &mut Vec<ImageInput>,
    path: &std::path::Path,
) -> Result<(), MullamaError> {
    loop {
        let result = ffmpeg_sys_next::avcodec_receive_frame(codec_context, decoded_frame);
        if result == FFMPEG_AVERROR_AGAIN || result == FFMPEG_AVERROR_EOF {
            return Ok(());
        }
        ffmpeg_check(
            result,
            &format!("decode video frame from {}", path.display()),
        )?;

        let scaler = ensure_video_scaler(codec_context, decoded_frame, scaler, path)?;
        ffmpeg_check(
            ffmpeg_sys_next::av_frame_make_writable(scaler.frame),
            &format!("prepare RGB frame for {}", path.display()),
        )?;

        let scaled_height = ffmpeg_sys_next::sws_scale(
            scaler.context,
            (*decoded_frame).data.as_ptr() as *const *const u8,
            (*decoded_frame).linesize.as_ptr(),
            0,
            (*decoded_frame).height,
            (*scaler.frame).data.as_ptr(),
            (*scaler.frame).linesize.as_ptr(),
        );
        if scaled_height <= 0 {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to scale decoded frame from {}",
                path.display()
            )));
        }

        frames.push(ImageInput {
            data: copy_rgb_frame(scaler.frame)?,
            dimensions: (scaler.width as u32, scaler.height as u32),
            format: ImageFormat::Rgb,
            caption: None,
        });

        ffmpeg_sys_next::av_frame_unref(decoded_frame);
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
unsafe fn ensure_video_scaler<'a>(
    codec_context: *mut ffmpeg_sys_next::AVCodecContext,
    decoded_frame: *mut ffmpeg_sys_next::AVFrame,
    scaler: &'a mut Option<VideoScaler>,
    path: &std::path::Path,
) -> Result<&'a mut VideoScaler, MullamaError> {
    let width = (*decoded_frame).width;
    let height = (*decoded_frame).height;
    let source_format = (*codec_context).pix_fmt;
    if width <= 0 || height <= 0 {
        return Err(MullamaError::MultimodalError(format!(
            "Decoded invalid FFmpeg frame dimensions for {}",
            path.display()
        )));
    }

    let needs_rebuild = scaler.as_ref().map_or(true, |scaler| {
        scaler.width != width
            || scaler.height != height
            || scaler.source_format as i32 != source_format as i32
    });
    if needs_rebuild {
        *scaler = Some(VideoScaler::new(width, height, source_format, path)?);
    }

    Ok(scaler.as_mut().expect("video scaler should be initialized"))
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
unsafe fn copy_rgb_frame(frame: *mut ffmpeg_sys_next::AVFrame) -> Result<Vec<u8>, MullamaError> {
    let width = (*frame).width;
    let height = (*frame).height;
    let buffer_size = ffmpeg_sys_next::av_image_get_buffer_size(
        ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
        width,
        height,
        1,
    );
    if buffer_size < 0 {
        return Err(ffmpeg_error(buffer_size, "calculate RGB frame buffer size"));
    }

    let mut output = vec![0u8; buffer_size as usize];
    let copied = ffmpeg_sys_next::av_image_copy_to_buffer(
        output.as_mut_ptr(),
        buffer_size,
        (*frame).data.as_ptr() as *const *const u8,
        (*frame).linesize.as_ptr(),
        ffmpeg_sys_next::AVPixelFormat::AV_PIX_FMT_RGB24,
        width,
        height,
        1,
    );
    if copied < 0 {
        return Err(ffmpeg_error(copied, "copy RGB frame data"));
    }

    output.truncate((width as usize) * (height as usize) * 3);
    Ok(output)
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
fn rational_to_f32(rational: ffmpeg_sys_next::AVRational) -> Option<f32> {
    if rational.den == 0 {
        None
    } else {
        Some(rational.num as f32 / rational.den as f32)
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
fn ffmpeg_check(code: i32, action: &str) -> Result<(), MullamaError> {
    if code < 0 {
        Err(ffmpeg_error(code, action))
    } else {
        Ok(())
    }
}

#[cfg(all(feature = "multimodal", feature = "format-conversion"))]
fn ffmpeg_error(code: i32, action: &str) -> MullamaError {
    use std::ffi::CStr;

    let mut buffer = [0i8; 256];
    let message = unsafe {
        if ffmpeg_sys_next::av_strerror(code, buffer.as_mut_ptr(), buffer.len()) == 0 {
            CStr::from_ptr(buffer.as_ptr())
                .to_string_lossy()
                .into_owned()
        } else {
            format!("FFmpeg error {}", code)
        }
    };

    MullamaError::MultimodalError(format!("Failed to {}: {}", action, message))
}

#[cfg(all(test, feature = "multimodal"))]
mod tests {
    use super::*;
    use image::{Delay, Frame, RgbImage, RgbaImage};
    use tempfile::tempdir;

    #[cfg(feature = "format-conversion")]
    fn ffmpeg_available() -> bool {
        use std::process::Command;

        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(feature = "format-conversion")]
    fn write_test_video(path: &std::path::Path, output_args: &[&str]) {
        use std::process::Command;

        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=2x2:rate=2:duration=1",
            ])
            .args(output_args)
            .arg(path)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn image_input_loads_png_from_disk() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.png");

        let mut image = RgbImage::new(2, 1);
        image.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        image.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        image.save(&path).unwrap();

        let loaded = ImageInput::from_path(&path).unwrap();
        assert_eq!(loaded.dimensions, (2, 1));
        assert_eq!(loaded.format, ImageFormat::Png);
        assert_eq!(loaded.data.len(), 6);
    }

    #[test]
    fn video_input_loads_gif_frames_from_disk() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.gif");

        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = image::codecs::gif::GifEncoder::new(file);
        let mut first = RgbaImage::new(1, 1);
        first.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        let mut second = RgbaImage::new(1, 1);
        second.put_pixel(0, 0, image::Rgba([0, 0, 255, 255]));
        encoder
            .encode_frame(Frame::from_parts(
                first,
                0,
                0,
                Delay::from_numer_denom_ms(100, 1),
            ))
            .unwrap();
        encoder
            .encode_frame(Frame::from_parts(
                second,
                0,
                0,
                Delay::from_numer_denom_ms(100, 1),
            ))
            .unwrap();
        drop(encoder);

        let loaded = VideoInput::from_path(&path).unwrap();
        assert_eq!(loaded.frames.len(), 2);
        assert_eq!(loaded.frames[0].dimensions, (1, 1));
        assert!(loaded.duration > 0.0);
    }

    #[cfg(feature = "format-conversion")]
    #[test]
    fn video_input_loads_mp4_frames_from_disk() {
        if !ffmpeg_available() {
            return;
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.mp4");
        write_test_video(&path, &["-pix_fmt", "yuv420p"]);

        let loaded = VideoInput::from_path(&path).unwrap();
        assert!(!loaded.frames.is_empty());
        assert_eq!(loaded.frames[0].dimensions, (2, 2));
        assert!(loaded.duration >= 0.0);
    }

    #[cfg(feature = "format-conversion")]
    #[test]
    fn video_input_loads_webm_frames_from_disk() {
        if !ffmpeg_available() {
            return;
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.webm");
        write_test_video(&path, &["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p"]);

        let loaded = VideoInput::from_path(&path).unwrap();
        assert!(!loaded.frames.is_empty());
        assert_eq!(loaded.frames[0].dimensions, (2, 2));
        assert!(loaded.duration >= 0.0);
    }
}
