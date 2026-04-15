use std::collections::HashMap;
use std::path::Path;

use crate::error::MullamaError;

use super::{AudioFeatures, AudioFormat, AudioInput, AudioProcessingConfig};

pub async fn load_audio_from_path(path: impl AsRef<Path>) -> Result<AudioInput, MullamaError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(MullamaError::ConfigError(format!(
            "Audio file not found: {}",
            path.display()
        )));
    }

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (samples, sample_rate, channels, duration) = match extension.as_str() {
        "wav" | "wave" => load_wav(path)?,
        "mp3" | "flac" | "ogg" | "aac" | "m4a" => load_with_symphonia(path)?,
        _ => return Err(MullamaError::ConfigError(format!(
            "Unsupported audio format: {}",
            extension
        ))),
    };

    let format = AudioFormat {
        container: extension.clone(),
        codec: if extension == "wav" {
            "pcm".to_string()
        } else {
            extension.clone()
        },
        bit_depth: 16,
        bitrate: None,
    };

    Ok(AudioInput {
        samples,
        sample_rate,
        channels: channels as u32,
        duration,
        format,
        transcript: None,
        metadata: HashMap::new(),
    })
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32, usize, f32), MullamaError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| MullamaError::ConfigError(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    let duration = samples.len() as f32 / (sample_rate as f32 * channels as f32);

    Ok((samples, sample_rate, channels, duration))
}

fn load_with_symphonia(path: &Path) -> Result<(Vec<f32>, u32, usize, f32), MullamaError> {
    use symphonia::core::audio::Signal;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| MullamaError::ConfigError(format!("Failed to open audio file: {}", e)))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| {
            MullamaError::ConfigError(format!(
                "Failed to probe audio file {}: {}",
                path.display(),
                e
            ))
        })?;

    let mut format_reader = probed.format;

    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| MullamaError::ConfigError("No audio track found".to_string()))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels_count = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| {
            MullamaError::ConfigError(format!("Failed to create audio decoder: {}", e))
        })?;

    let mut all_samples = Vec::new();

    loop {
        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => break,
        };

        let ch_count = decoded.spec().channels.count();

        match decoded {
            symphonia::core::audio::AudioBufferRef::F32(ref buf) => {
                for ch in 0..ch_count {
                    if ch == 0 {
                        all_samples.extend((**buf).chan(ch).iter().copied());
                    }
                }
            }
            symphonia::core::audio::AudioBufferRef::F64(ref buf) => {
                for ch in 0..ch_count {
                    if ch == 0 {
                        all_samples.extend((**buf).chan(ch).iter().map(|&s| s as f32));
                    }
                }
            }
            symphonia::core::audio::AudioBufferRef::S16(ref buf) => {
                for ch in 0..ch_count {
                    if ch == 0 {
                        all_samples
                            .extend((**buf).chan(ch).iter().map(|&s| s as f32 / i16::MAX as f32));
                    }
                }
            }
            symphonia::core::audio::AudioBufferRef::S32(ref buf) => {
                for ch in 0..ch_count {
                    if ch == 0 {
                        all_samples
                            .extend((**buf).chan(ch).iter().map(|&s| s as f32 / i32::MAX as f32));
                    }
                }
            }
            symphonia::core::audio::AudioBufferRef::U16(ref buf) => {
                for ch in 0..ch_count {
                    if ch == 0 {
                        all_samples.extend(
                            (**buf)
                                .chan(ch)
                                .iter()
                                .map(|&s| (s as f32 - 32768.0) / 32768.0),
                        );
                    }
                }
            }
            _ => continue,
        }
    }

    let duration = if sample_rate > 0 {
        all_samples.len() as f32 / sample_rate as f32
    } else {
        0.0
    };

    Ok((all_samples, sample_rate, channels_count, duration))
}

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

pub fn convert_audio_format(
    input: &AudioInput,
    target_format: &AudioFormat,
) -> Result<AudioInput, MullamaError> {
    let mut output = input.clone();
    output.format = target_format.clone();
    Ok(output)
}

pub fn extract_audio_features(audio: &AudioInput) -> AudioFeatures {
    AudioFeatures {
        duration: audio.duration,
        energy: calculate_energy(&audio.samples),
        zero_crossing_rate: calculate_zero_crossing_rate(&audio.samples),
        spectral_centroid: calculate_spectral_centroid(&audio.samples, audio.sample_rate),
        mfcc: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        pitch: detect_pitch(&audio.samples, audio.sample_rate),
        tempo: detect_tempo(&audio.samples, audio.sample_rate),
        has_speech: detect_speech(&audio.samples),
    }
}

fn apply_noise_reduction(samples: &mut [f32]) {
    for sample in samples.iter_mut() {
        if sample.abs() < 0.01 {
            *sample = 0.0;
        }
    }
}

fn apply_automatic_gain_control(samples: &mut [f32]) {
    let max_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    if max_amplitude > 0.0 {
        let gain = 0.8 / max_amplitude;
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, MullamaError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

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

fn calculate_spectral_centroid(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    let n = samples.len().min(4096);
    let frame = &samples[..n];

    let mut magnitudes = Vec::with_capacity(n / 2);
    let mut weighted_sum = 0.0f32;
    let mut total = 0.0f32;

    let freq_resolution = sample_rate as f32 / n as f32;

    for i in 0..n / 2 {
        let real = frame.get(i).copied().unwrap_or(0.0);
        let imag = if i + n / 2 < n {
            frame[i + n / 2]
        } else {
            0.0
        };
        let magnitude = (real * real + imag * imag).sqrt();
        let freq = i as f32 * freq_resolution;
        weighted_sum += freq * magnitude;
        total += magnitude;
        magnitudes.push(magnitude);
    }

    if total > 0.0 {
        weighted_sum / total
    } else {
        0.0
    }
}

fn detect_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    let min_period = (sample_rate as f32 / 2000.0) as usize;
    let max_period = (sample_rate as f32 / 50.0).min(samples.len() as f32 / 2.0) as usize;

    if max_period <= min_period || min_period == 0 {
        return 0.0;
    }

    let frame_len = samples.len().min(4096);
    let frame = &samples[..frame_len];

    let mut best_lag = min_period;
    let mut best_corr = f32::NEG_INFINITY;

    for lag in min_period..max_period {
        let mut corr = 0.0f32;
        let mut energy = 0.0f32;
        for i in 0..frame_len - lag {
            corr += frame[i] * frame[i + lag];
            energy += frame[i] * frame[i];
        }
        if energy > 0.0 {
            corr /= energy;
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_corr < 0.2 {
        return 0.0;
    }

    sample_rate as f32 / best_lag as f32
}

fn detect_tempo(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    let window_ms = 10;
    let window_samples = (sample_rate as f32 * window_ms as f32 / 1000.0) as usize;
    let hop_samples = window_samples / 2;

    let mut onset_envelope = Vec::new();
    let mut prev_energy = 0.0f32;

    let mut pos = 0;
    while pos + window_samples <= samples.len() {
        let energy: f32 = samples[pos..pos + window_samples]
            .iter()
            .map(|s| s * s)
            .sum();
        let diff = (energy - prev_energy).max(0.0);
        onset_envelope.push(diff);
        prev_energy = energy;
        pos += hop_samples;
    }

    if onset_envelope.is_empty() {
        return 0.0;
    }

    let mean = onset_envelope.iter().sum::<f32>() / onset_envelope.len() as f32;
    if mean == 0.0 {
        return 0.0;
    }

    let peak_threshold = mean * 2.0;
    let min_gap_frames = (60.0 / 200.0 * 1000.0 / window_ms as f32) as usize;

    let mut peaks = Vec::new();
    let mut last_peak = 0;
    for i in 1..onset_envelope.len().saturating_sub(1) {
        if onset_envelope[i] > peak_threshold
            && onset_envelope[i] > onset_envelope[i - 1]
            && onset_envelope[i] >= onset_envelope[i + 1]
            && i - last_peak >= min_gap_frames
        {
            peaks.push(i);
            last_peak = i;
        }
    }

    if peaks.len() < 2 {
        return 0.0;
    }

    let total_frames = peaks.last().unwrap() - peaks.first().unwrap();
    let intervals = peaks.len() - 1;
    let frames_per_interval = total_frames as f32 / intervals as f32;
    let seconds_per_interval = frames_per_interval * window_ms as f32 / 1000.0;

    if seconds_per_interval > 0.0 {
        60.0 / seconds_per_interval
    } else {
        0.0
    }
}

fn detect_speech(samples: &[f32]) -> bool {
    let energy = calculate_energy(samples);
    let zcr = calculate_zero_crossing_rate(samples);
    energy > 0.01 && zcr > 0.1 && zcr < 0.4
}
