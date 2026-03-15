# Format Conversion

Convert audio and image files between formats with sample rate conversion, resizing, quality control, and batch processing support.

!!! info "Feature Gate"
    This feature requires the `format-conversion` feature flag, which transitively enables `multimodal`.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["format-conversion"] }
    ```

## Overview

The format conversion module provides:

- **AudioConverter** - Convert between WAV, MP3, FLAC formats
- **ImageConverter** - Convert between JPEG, PNG, WebP, TIFF, BMP formats
- **Audio resampling** and normalization
- **Image resizing** with quality settings
- **Pipeline integration** (capture -> convert -> process)
- **Batch conversion** for processing multiple files
- **ConversionConfig** for unified parameter configuration

---

## System Dependencies

!!! warning "FFmpeg Required for Audio Conversion"
    Audio format conversion depends on FFmpeg libraries. Install them before building.

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
    sudo apt install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev
    ```

=== "macOS"

    ```bash
    brew install ffmpeg libpng jpeg webp
    ```

=== "Windows"

    Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH. Image libraries are typically bundled.

---

## AudioConverter

Convert audio files between formats with optional resampling and channel conversion.

### Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| WAV | Yes | Yes | Uncompressed PCM, best quality |
| MP3 | Yes | Yes | Lossy compression |
| FLAC | Yes | Yes | Lossless compression |

### Basic Usage

=== "Node.js"

    ```javascript
    const { AudioConverter } = require('mullama');

    const converter = new AudioConverter();

    // Simple format conversion
    const result = await converter.convert('input.mp3', {
      outputFormat: 'wav',
      sampleRate: 16000
    });

    console.log(`Converted: ${result.inputSize} -> ${result.outputSize} bytes`);
    ```

=== "Python"

    ```python
    from mullama import AudioConverter, ConversionConfig

    converter = AudioConverter()

    # Simple format conversion
    config = ConversionConfig(sample_rate=16000)
    result = await converter.convert("input.mp3", output_format="wav", config=config)

    print(f"Converted: {result.input_size} -> {result.output_size} bytes")
    ```

=== "Rust"

    ```rust
    use mullama::{AudioConverter, ConversionConfig};

    let converter = AudioConverter::new();

    // Simple format conversion
    let config = ConversionConfig::new().sample_rate(16000);
    let result = converter.convert_audio(
        "input.mp3",
        AudioFormatType::MP3,
        AudioFormatType::WAV,
        config
    ).await?;

    println!("Converted: {} bytes -> {} bytes", result.input_size, result.output_size);
    ```

=== "CLI"

    ```bash
    # Convert audio format
    mullama convert audio input.mp3 --output-format wav --sample-rate 16000

    # Batch conversion
    mullama convert audio *.mp3 --output-format wav --sample-rate 16000
    ```

### Resampling

Convert audio to a different sample rate, commonly needed for model input preparation.

=== "Node.js"

    ```javascript
    // Resample from 44.1kHz to 16kHz mono (common for speech models)
    const result = await converter.convert('music.mp3', {
      outputFormat: 'wav',
      sampleRate: 16000,
      channels: 1
    });

    // Or resample raw samples directly
    const resampled = await converter.resample(inputSamples, {
      inputRate: 44100,
      outputRate: 16000,
      channels: 1
    });
    ```

=== "Python"

    ```python
    # Resample from 44.1kHz to 16kHz mono
    config = ConversionConfig(sample_rate=16000, channels=1)
    result = await converter.convert("music.mp3", output_format="wav", config=config)

    # Or resample raw samples directly
    resampled = await converter.resample(
        input_samples,
        input_rate=44100,
        output_rate=16000,
        channels=1
    )
    ```

=== "Rust"

    ```rust
    // Resample from 44.1kHz to 16kHz mono
    let config = ConversionConfig::new().sample_rate(16000);
    let result = converter.convert_audio(
        "music.mp3",
        AudioFormatType::MP3,
        AudioFormatType::WAV,
        config
    ).await?;

    // Or resample raw samples directly
    let resampled = converter.resample_audio(
        &input_samples,
        44100,   // Input sample rate
        16000,   // Output sample rate
        1        // Mono
    ).await?;
    ```

### Audio Normalization

```rust
// Normalize audio levels after conversion
let mut samples = converted.samples.clone();
let max_amplitude = samples.iter()
    .map(|s| s.abs())
    .fold(0.0f32, f32::max);

if max_amplitude > 0.0 {
    let scale = 0.95 / max_amplitude;
    for sample in &mut samples {
        *sample *= scale;
    }
}
```

### Batch Audio Conversion

=== "Node.js"

    ```javascript
    const results = await converter.batchConvert([
      { input: 'file1.mp3', outputFormat: 'wav' },
      { input: 'file2.flac', outputFormat: 'wav' },
      { input: 'file3.mp3', outputFormat: 'wav' }
    ], { sampleRate: 16000 });

    results.forEach(r => console.log(`${r.inputPath} -> ${r.outputPath}`));
    ```

=== "Python"

    ```python
    config = ConversionConfig(sample_rate=16000)
    results = await converter.batch_convert([
        ("file1.mp3", "wav"),
        ("file2.flac", "wav"),
        ("file3.mp3", "wav")
    ], config=config)

    for r in results:
        print(f"{r.input_path} -> {r.output_path}")
    ```

=== "Rust"

    ```rust
    let config = ConversionConfig::new().sample_rate(16000);

    let conversions = vec![
        (PathBuf::from("file1.mp3"), AudioFormatType::MP3, AudioFormatType::WAV, config.clone()),
        (PathBuf::from("file2.flac"), AudioFormatType::FLAC, AudioFormatType::WAV, config.clone()),
        (PathBuf::from("file3.mp3"), AudioFormatType::MP3, AudioFormatType::WAV, config.clone()),
    ];

    let results = converter.batch_convert(conversions).await?;

    for result in &results {
        println!("{} -> {}", result.input_path.display(), result.output_path.display());
    }
    ```

---

## ImageConverter

Convert images between formats with resizing, quality control, and color space handling.

### Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| JPEG | Yes | Yes | Lossy, configurable quality (1-100) |
| PNG | Yes | Yes | Lossless, with alpha channel |
| WebP | Yes | Yes | Modern format, lossy or lossless |
| TIFF | Yes | Yes | High-quality, large files |
| BMP | Yes | Yes | Uncompressed bitmap |

### Basic Usage

=== "Node.js"

    ```javascript
    const { ImageConverter } = require('mullama');

    const converter = new ImageConverter();

    // Convert JPEG to PNG
    const result = await converter.convert('photo.jpg', { outputFormat: 'png' });

    // Convert with quality setting
    const webp = await converter.convert('image.png', {
      outputFormat: 'webp',
      quality: 85
    });
    ```

=== "Python"

    ```python
    from mullama import ImageConverter, ConversionConfig

    converter = ImageConverter()

    # Convert JPEG to PNG
    result = await converter.convert("photo.jpg", output_format="png")

    # Convert with quality setting
    config = ConversionConfig(quality=85)
    webp = await converter.convert("image.png", output_format="webp", config=config)
    ```

=== "Rust"

    ```rust
    use mullama::{ImageConverter, ConversionConfig};

    let converter = ImageConverter::new();

    // Convert JPEG to PNG
    let config = ConversionConfig::new();
    let result = converter.jpeg_to_png("photo.jpg", config).await?;

    // Convert PNG to WebP with quality setting
    let config = ConversionConfig::new().quality(85);
    let result = converter.convert_image(
        "image.png",
        ImageFormatType::PNG,
        ImageFormatType::WebP,
        config
    ).await?;
    ```

=== "CLI"

    ```bash
    # Convert image format
    mullama convert image photo.jpg --output-format png

    # Resize and convert
    mullama convert image photo.jpg --output-format webp \
      --width 512 --height 512 --quality 85
    ```

### Resizing

=== "Node.js"

    ```javascript
    // Resize to fit within bounds, preserving aspect ratio
    const resized = await converter.resize('large_photo.jpg', {
      width: 512,
      height: 512,
      filter: 'lanczos3'
    });

    // Resize for vision model input (224x224)
    const modelInput = await converter.convert('photo.jpg', {
      outputFormat: 'png',
      width: 224,
      height: 224,
      quality: 95
    });
    ```

=== "Python"

    ```python
    # Resize to fit within bounds, preserving aspect ratio
    resized = await converter.resize("large_photo.jpg", width=512, height=512)

    # Resize for vision model input (224x224)
    config = ConversionConfig(dimensions=(224, 224), quality=95)
    model_input = await converter.convert("photo.jpg", output_format="png", config=config)
    ```

=== "Rust"

    ```rust
    use image::imageops::FilterType;

    let converter = ImageConverter::new();

    // Resize to fit within 512x512, preserving aspect ratio
    let result = converter.resize_image(
        "large_photo.jpg",
        (512, 512),
        FilterType::Lanczos3
    ).await?;

    // Resize for vision model input
    let config = ConversionConfig::new()
        .dimensions(224, 224)
        .quality(95);

    let result = converter.convert_image(
        "photo.jpg",
        ImageFormatType::JPEG,
        ImageFormatType::PNG,
        config
    ).await?;
    ```

### Quality and Compression

```rust
// JPEG quality (1-100)
let config = ConversionConfig::new().quality(85);

// WebP quality with specific mode
let config = ConversionConfig::new()
    .quality(80)
    .option("webp_mode", "lossy");  // or "lossless"

// PNG compression level
let config = ConversionConfig::new()
    .option("png_compression", "best");  // "fast", "default", "best"
```

### Converting from Bytes

Process images from memory without filesystem access.

```rust
let converter = ImageConverter::new();

// Convert raw bytes (e.g., from HTTP response)
let jpeg_bytes: Vec<u8> = download_image().await?;

let config = ConversionConfig::new()
    .dimensions(512, 512)
    .quality(90);

let result = converter.convert_image_bytes(
    &jpeg_bytes,
    ImageFormatType::JPEG,
    ImageFormatType::PNG,
    config
).await?;

// result.data contains the PNG bytes
```

---

## ConversionConfig

Unified configuration for both audio and image conversion.

```rust
let config = ConversionConfig::new()
    .quality(85)                    // Compression quality (1-100)
    .sample_rate(16000)             // Audio: target sample rate
    .dimensions(512, 512)           // Image: target dimensions
    .preserve_metadata(true)        // Keep EXIF/metadata
    .option("channels", "1")        // Custom format-specific options
    .option("bit_depth", "16");
```

### Configuration Parameters

| Parameter | Applies To | Description | Default |
|-----------|-----------|-------------|---------|
| `quality` | Image, Audio | Compression quality (1-100) | Format default |
| `sample_rate` | Audio | Target sample rate in Hz | Preserve original |
| `dimensions` | Image | Target width x height | Preserve original |
| `preserve_metadata` | Both | Keep EXIF/metadata | false |
| `options` | Both | Format-specific key-value pairs | Empty |

---

## Pipeline Integration

Combine format conversion with audio capture and model processing.

=== "Node.js"

    ```javascript
    const { AudioConverter, StreamingAudioProcessor, loadModel } = require('mullama');

    const model = await loadModel('whisper-model.gguf');
    const converter = new AudioConverter();

    // Capture -> Convert -> Process pipeline
    async function processAudioFile(filePath) {
      // Step 1: Convert to model-compatible format
      const converted = await converter.convert(filePath, {
        outputFormat: 'wav',
        sampleRate: 16000,
        channels: 1
      });

      // Step 2: Process with model
      const transcript = await model.transcribe(converted.samples);
      return transcript;
    }
    ```

=== "Python"

    ```python
    from mullama import AudioConverter, ConversionConfig, load_model

    model = await load_model("whisper-model.gguf")
    converter = AudioConverter()

    async def process_audio_file(file_path):
        # Step 1: Convert to model-compatible format
        config = ConversionConfig(sample_rate=16000, channels=1)
        converted = await converter.convert(file_path, output_format="wav", config=config)

        # Step 2: Process with model
        transcript = await model.transcribe(converted.samples)
        return transcript
    ```

=== "Rust"

    ```rust
    use mullama::{AudioConverter, ConversionConfig, Model, Context, ContextParams};
    use std::sync::Arc;
    use std::path::Path;

    async fn process_audio_file(
        audio_path: &Path,
        model: &Arc<Model>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let converter = AudioConverter::new();

        // Step 1: Convert to model-compatible format
        let config = ConversionConfig::new()
            .sample_rate(16000)
            .option("channels", "1");

        let audio_format = detect_format(audio_path)?;
        let converted = converter.convert_audio(
            audio_path, audio_format, AudioFormatType::WAV, config
        ).await?;

        // Step 2: Process with model
        let model = model.clone();
        let samples = converted.samples.clone();
        let transcript = tokio::task::spawn_blocking(move || {
            let mut ctx = Context::new(model, ContextParams::default())?;
            ctx.process_audio(&samples)
        }).await??;

        Ok(transcript)
    }
    ```

---

## Performance Tips

!!! tip "Optimization Strategies"

    1. **Batch operations** - Use `batch_convert` instead of sequential conversions
    2. **Reuse converters** - Create `AudioConverter` and `ImageConverter` once and reuse
    3. **Match model requirements** - Convert to the exact format your model expects
    4. **Avoid unnecessary conversions** - Skip conversion if input is already compatible
    5. **Parallel processing** - Combine with the `parallel` feature for CPU-bound batch work

---

## See Also

- [Streaming Audio](streaming-audio.md) - Real-time audio capture
- [Multimodal Guide](../guide/multimodal.md) - Multimodal processing
- [Parallel Processing](parallel.md) - Batch processing with parallelism
