//! Audio format conversion utilities

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use tokio::{fs, sync::Semaphore};

#[cfg(feature = "format-conversion")]
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

#[cfg(feature = "format-conversion")]
use symphonia::{
    core::{
        audio::{AudioBufferRef, Signal},
        codecs::{DecoderOptions, CODEC_TYPE_NULL},
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    },
    default::get_probe,
};

use crate::MullamaError;

use super::{AudioConversionResult, AudioFormatType, ConversionConfig};

/// Comprehensive audio format converter
#[cfg(feature = "format-conversion")]
pub struct AudioConverter {
    config: AudioConverterConfig,
    conversion_cache: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>,
    semaphore: Arc<Semaphore>,
}

/// Audio converter configuration
#[cfg(feature = "format-conversion")]
#[derive(Debug, Clone)]
pub struct AudioConverterConfig {
    /// Maximum concurrent conversions
    pub max_concurrent: usize,
    /// Enable caching of converted files
    pub enable_cache: bool,
    /// Default output quality (0.0 to 1.0)
    pub default_quality: f32,
    /// Default sample rate for conversions
    pub default_sample_rate: u32,
    /// Default number of channels
    pub default_channels: u16,
    /// Temporary directory for conversion files
    pub temp_dir: Option<PathBuf>,
}

#[cfg(feature = "format-conversion")]
impl AudioConverter {
    /// Create a new audio converter
    pub fn new() -> Self {
        Self::with_config(AudioConverterConfig::default())
    }

    /// Create audio converter with custom configuration
    pub fn with_config(config: AudioConverterConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            conversion_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Convert MP3 to WAV format
    pub async fn mp3_to_wav(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        self.convert_audio(
            input_path,
            AudioFormatType::Mp3,
            AudioFormatType::Wav,
            config,
        )
        .await
    }

    /// Convert WAV to MP3 format
    pub async fn wav_to_mp3(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        self.convert_audio(
            input_path,
            AudioFormatType::Wav,
            AudioFormatType::Mp3,
            config,
        )
        .await
    }

    /// Convert FLAC to WAV format
    pub async fn flac_to_wav(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        self.convert_audio(
            input_path,
            AudioFormatType::Flac,
            AudioFormatType::Wav,
            config,
        )
        .await
    }

    /// Convert between any supported audio formats
    pub async fn convert_audio(
        &self,
        input_path: impl AsRef<Path>,
        input_format: AudioFormatType,
        output_format: AudioFormatType,
        config: ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            MullamaError::ConfigError("Failed to acquire conversion semaphore".to_string())
        })?;

        let input_path = input_path.as_ref();
        let cache_key = format!(
            "{:?}_{:?}_{}",
            input_path.display(),
            output_format,
            serde_json::to_string(&config).unwrap_or_default()
        );

        // Check cache first
        if self.config.enable_cache {
            let cache = self.conversion_cache.read().await;
            if let Some(cached_data) = cache.get(&cache_key) {
                return self
                    .create_audio_result(cached_data.clone(), output_format, &config)
                    .await;
            }
        }

        // Perform conversion
        let result = match (input_format, output_format) {
            (AudioFormatType::Mp3, AudioFormatType::Wav) => {
                self.decode_and_encode_audio(input_path, output_format, &config)
                    .await?
            }
            (AudioFormatType::Wav, AudioFormatType::Mp3) => {
                self.encode_wav_to_mp3(input_path, &config).await?
            }
            (AudioFormatType::Flac, AudioFormatType::Wav) => {
                self.decode_and_encode_audio(input_path, output_format, &config)
                    .await?
            }
            _ => {
                // Generic conversion using symphonia
                self.decode_and_encode_audio(input_path, output_format, &config)
                    .await?
            }
        };

        // Cache result
        if self.config.enable_cache {
            let mut cache = self.conversion_cache.write().await;
            cache.insert(cache_key, result.data.clone());
        }

        Ok(result)
    }

    /// Convert audio from bytes
    pub async fn convert_audio_bytes(
        &self,
        input_data: &[u8],
        input_format: AudioFormatType,
        output_format: AudioFormatType,
        config: ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            MullamaError::ConfigError("Failed to acquire conversion semaphore".to_string())
        })?;

        match (input_format, output_format) {
            (AudioFormatType::Wav, AudioFormatType::Mp3) => {
                self.wav_bytes_to_mp3(input_data, &config).await
            }
            (AudioFormatType::Mp3, AudioFormatType::Wav) => {
                self.mp3_bytes_to_wav(input_data, &config).await
            }
            _ => {
                self.generic_audio_conversion(input_data, input_format, output_format, &config)
                    .await
            }
        }
    }

    /// Batch convert multiple audio files
    pub async fn batch_convert_audio(
        &self,
        conversions: Vec<(PathBuf, AudioFormatType, AudioFormatType, ConversionConfig)>,
    ) -> Result<Vec<AudioConversionResult>, MullamaError> {
        let mut results = Vec::new();

        for (path, input_fmt, output_fmt, config) in conversions {
            let result = self
                .convert_audio(&path, input_fmt, output_fmt, config)
                .await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Resample audio to different sample rate
    pub async fn resample_audio(
        &self,
        input_data: &[f32],
        input_rate: u32,
        output_rate: u32,
        channels: u16,
    ) -> Result<Vec<f32>, MullamaError> {
        if input_rate == output_rate {
            return Ok(input_data.to_vec());
        }

        #[cfg(feature = "format-conversion")]
        {
            use rubato::{
                Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
                WindowFunction,
            };

            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };

            let mut resampler = SincFixedIn::<f32>::new(
                output_rate as f64 / input_rate as f64,
                2.0,
                params,
                input_data.len(),
                channels as usize,
            )
            .map_err(|e| MullamaError::ConfigError(format!("Resampler error: {}", e)))?;

            // Convert to channel-interleaved format expected by rubato
            let mut channel_data =
                vec![vec![0.0f32; input_data.len() / channels as usize]; channels as usize];

            for (i, sample) in input_data.iter().enumerate() {
                let channel = i % channels as usize;
                let frame = i / channels as usize;
                if frame < channel_data[channel].len() {
                    channel_data[channel][frame] = *sample;
                }
            }

            let output_channels = resampler
                .process(&channel_data, None)
                .map_err(|e| MullamaError::ConfigError(format!("Resampling failed: {}", e)))?;

            // Convert back to interleaved format
            let mut output = Vec::new();
            let output_len = output_channels[0].len();

            for frame in 0..output_len {
                for channel in 0..channels as usize {
                    output.push(output_channels[channel][frame]);
                }
            }

            Ok(output)
        }

        #[cfg(not(feature = "format-conversion"))]
        {
            // Simple linear interpolation fallback
            let ratio = output_rate as f64 / input_rate as f64;
            let output_len = (input_data.len() as f64 * ratio) as usize;
            let mut output = Vec::with_capacity(output_len);

            for i in 0..output_len {
                let src_idx = (i as f64 / ratio) as usize;
                if src_idx < input_data.len() {
                    output.push(input_data[src_idx]);
                }
            }

            Ok(output)
        }
    }

    // Private helper methods
    async fn decode_and_encode_audio(
        &self,
        input_path: &Path,
        output_format: AudioFormatType,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        // Read input file
        let input_data = fs::read(input_path)
            .await
            .map_err(|e| MullamaError::ConfigError(format!("Failed to read audio file: {}", e)))?;

        // Decode using symphonia
        let audio_data = self.decode_with_symphonia(&input_data).await?;

        // Encode to target format
        let output_data = self
            .encode_audio_data(&audio_data, output_format, config)
            .await?;

        Ok(output_data)
    }

    async fn decode_with_symphonia(&self, data: &[u8]) -> Result<DecodedAudio, MullamaError> {
        let cursor = Cursor::new(data.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let hint = Hint::new();
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| {
                MullamaError::ConfigError(format!("Failed to probe audio format: {}", e))
            })?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| {
                MullamaError::ConfigError("No supported audio track found".to_string())
            })?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 44100;
        let mut channels = 2;

        // Decode all packets
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(_) => break,
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    sample_rate = audio_buf.spec().rate;
                    channels = audio_buf.spec().channels.count() as u16;

                    match audio_buf {
                        AudioBufferRef::F32(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample);
                            }
                        }
                        AudioBufferRef::U8(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push((sample as f32 - 128.0) / 128.0);
                            }
                        }
                        AudioBufferRef::U16(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push((sample as f32 - 32768.0) / 32768.0);
                            }
                        }
                        AudioBufferRef::U24(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push((sample.inner() as f32 - 8388608.0) / 8388608.0);
                            }
                        }
                        AudioBufferRef::U32(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push((sample as f32 - 2147483648.0) / 2147483648.0);
                            }
                        }
                        AudioBufferRef::S8(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample as f32 / 128.0);
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample as f32 / 32768.0);
                            }
                        }
                        AudioBufferRef::S24(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample.inner() as f32 / 8388608.0);
                            }
                        }
                        AudioBufferRef::S32(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample as f32 / 2147483648.0);
                            }
                        }
                        AudioBufferRef::F64(buf) => {
                            for &sample in buf.chan(0) {
                                samples.push(sample as f32);
                            }
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(DecodedAudio {
            samples,
            sample_rate,
            channels,
        })
    }

    async fn encode_audio_data(
        &self,
        audio: &DecodedAudio,
        format: AudioFormatType,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        match format {
            AudioFormatType::Wav => self.encode_to_wav(audio, config).await,
            AudioFormatType::Mp3 => self.encode_to_mp3(audio, config).await,
            AudioFormatType::Flac => self.encode_to_flac(audio, config).await,
            _ => Err(MullamaError::ConfigError(format!(
                "Unsupported output format: {:?}",
                format
            ))),
        }
    }

    async fn encode_to_wav(
        &self,
        audio: &DecodedAudio,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let sample_rate = config.sample_rate.unwrap_or(audio.sample_rate);
        let channels = config.channels.unwrap_or(audio.channels);

        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| {
                MullamaError::ConfigError(format!("Failed to create WAV writer: {}", e))
            })?;

            // Resample if needed
            let samples = if sample_rate != audio.sample_rate {
                self.resample_audio(&audio.samples, audio.sample_rate, sample_rate, channels)
                    .await?
            } else {
                audio.samples.clone()
            };

            // Convert to i16 and write
            for sample in samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(sample_i16).map_err(|e| {
                    MullamaError::ConfigError(format!("Failed to write WAV sample: {}", e))
                })?;
            }

            writer
                .finalize()
                .map_err(|e| MullamaError::ConfigError(format!("Failed to finalize WAV: {}", e)))?;
        }

        let duration = Duration::from_secs_f32(
            audio.samples.len() as f32 / sample_rate as f32 / channels as f32,
        );

        Ok(AudioConversionResult {
            data: cursor.into_inner(),
            format: AudioFormatType::Wav,
            sample_rate,
            channels,
            duration,
            metadata: HashMap::new(),
        })
    }

    async fn encode_to_mp3(
        &self,
        audio: &DecodedAudio,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        // For MP3 encoding, we'd typically use a library like minimp3 or integrate with FFmpeg
        // For now, this is a placeholder that returns the original as WAV
        self.encode_to_wav(audio, config).await
    }

    async fn encode_to_flac(
        &self,
        audio: &DecodedAudio,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        // For FLAC encoding, we'd use a FLAC encoder library
        // For now, this is a placeholder that returns the original as WAV
        self.encode_to_wav(audio, config).await
    }

    async fn wav_bytes_to_mp3(
        &self,
        wav_data: &[u8],
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let cursor = Cursor::new(wav_data);
        let mut reader = WavReader::new(cursor)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to read WAV: {}", e)))?;

        let spec = reader.spec();
        let samples: Result<Vec<f32>, _> = reader
            .samples::<i16>()
            .map(|s| s.map(|sample| sample as f32 / 32768.0))
            .collect();

        let samples = samples
            .map_err(|e| MullamaError::ConfigError(format!("Failed to read WAV samples: {}", e)))?;

        let audio = DecodedAudio {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        };

        self.encode_to_mp3(&audio, config).await
    }

    async fn mp3_bytes_to_wav(
        &self,
        mp3_data: &[u8],
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let audio = self.decode_with_symphonia(mp3_data).await?;
        self.encode_to_wav(&audio, config).await
    }

    async fn generic_audio_conversion(
        &self,
        input_data: &[u8],
        _input_format: AudioFormatType,
        output_format: AudioFormatType,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let audio = self.decode_with_symphonia(input_data).await?;
        self.encode_audio_data(&audio, output_format, config).await
    }

    async fn encode_wav_to_mp3(
        &self,
        input_path: &Path,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        let wav_data = fs::read(input_path)
            .await
            .map_err(|e| MullamaError::ConfigError(format!("Failed to read WAV file: {}", e)))?;

        self.wav_bytes_to_mp3(&wav_data, config).await
    }

    async fn create_audio_result(
        &self,
        data: Vec<u8>,
        format: AudioFormatType,
        config: &ConversionConfig,
    ) -> Result<AudioConversionResult, MullamaError> {
        // Extract basic info from the cached data
        Ok(AudioConversionResult {
            data,
            format,
            sample_rate: config.sample_rate.unwrap_or(44100),
            channels: config.channels.unwrap_or(2),
            duration: Duration::from_secs(1), // Placeholder
            metadata: HashMap::new(),
        })
    }
}

/// Decoded audio data
#[derive(Debug, Clone)]
struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

// Default implementations
#[cfg(feature = "format-conversion")]
impl Default for AudioConverterConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            enable_cache: true,
            default_quality: 0.8,
            default_sample_rate: 44100,
            default_channels: 2,
            temp_dir: None,
        }
    }
}
