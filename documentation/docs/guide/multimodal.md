# Multimodal

Process images and audio alongside text using vision-language models (VLMs) and audio-language models. Mullama supports multimodal inference with a unified API across image and audio inputs.

!!! abstract "Feature Gate"
    In Rust, enable the `multimodal` feature flag:

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["multimodal"] }

    # For audio processing, also enable streaming-audio
    mullama = { version = "0.1", features = ["multimodal", "streaming-audio"] }
    ```

    Node.js and Python include multimodal support by default.

## MultimodalProcessor Overview

The `MultimodalProcessor` is the central API for processing text, images, and audio together. It handles:

- Loading and preprocessing images into the format expected by vision encoders
- Converting audio into the format expected by audio encoders
- Combining multimodal inputs with text prompts
- Routing to the appropriate encoder based on input type

=== "Node.js"

    ```javascript
    import { Model, MultimodalProcessor } from 'mullama';

    const model = await Model.load('./llava-v1.6.gguf');
    const processor = new MultimodalProcessor(model);

    const response = await processor.generate({
      text: "Describe this image in detail.",
      images: ['./photo.jpg'],
    });
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, MultimodalProcessor

    model = Model.load("./llava-v1.6.gguf")
    processor = MultimodalProcessor(model)

    response = processor.generate(
        text="Describe this image in detail.",
        images=["./photo.jpg"],
    )
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::{Model, MultimodalProcessor};
    use std::sync::Arc;

    let model = Arc::new(Model::load("llava-v1.6.gguf")?);
    let mut processor = MultimodalProcessor::new(model)?;

    let response = processor.generate_with_image(
        "Describe this image in detail.",
        "photo.jpg",
        200
    )?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    mullama run llava:13b "Describe this image in detail." --image ./photo.jpg
    ```

## Image Input

### From File Path

The simplest way to process an image:

=== "Node.js"

    ```javascript
    const response = await processor.generate({
      text: "What objects are in this image?",
      images: ['./photo.jpg'],
    });
    ```

=== "Python"

    ```python
    response = processor.generate(
        text="What objects are in this image?",
        images=["./photo.jpg"],
    )
    ```

=== "Rust"

    ```rust
    let response = processor.generate_with_image(
        "What objects are in this image?",
        "photo.jpg",
        200
    )?;
    ```

=== "CLI"

    ```bash
    mullama run llava:13b "What objects are in this image?" --image ./photo.jpg
    ```

### From Buffer

Process images from memory (useful for web applications receiving uploads):

=== "Node.js"

    ```javascript
    import { readFileSync } from 'fs';

    const imageBuffer = readFileSync('./photo.jpg');

    const response = await processor.generate({
      text: "Describe this image.",
      imageBuffers: [imageBuffer],
    });
    ```

=== "Python"

    ```python
    with open("./photo.jpg", "rb") as f:
        image_buffer = f.read()

    response = processor.generate(
        text="Describe this image.",
        image_buffers=[image_buffer],
    )
    ```

=== "Rust"

    ```rust
    let image_data = std::fs::read("photo.jpg")?;

    let response = processor.generate_with_image_buffer(
        "Describe this image.",
        &image_data,
        200
    )?;
    ```

=== "CLI"

    ```bash
    # Pipe image data via stdin
    cat photo.jpg | mullama run llava:13b "Describe this image." --image -
    ```

### From Raw Pixels

Process raw pixel data (useful for video frames or generated images):

=== "Node.js"

    ```javascript
    import { RawImage } from 'mullama';

    // Raw RGBA pixels
    const pixels = new Uint8Array(width * height * 4);
    const image = new RawImage(pixels, width, height, 'rgba');

    const response = await processor.generate({
      text: "What do you see?",
      rawImages: [image],
    });
    ```

=== "Python"

    ```python
    import numpy as np
    from mullama import RawImage

    # Raw RGBA pixels (numpy array)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    image = RawImage(pixels, width, height, "rgba")

    response = processor.generate(
        text="What do you see?",
        raw_images=[image],
    )
    ```

=== "Rust"

    ```rust
    use mullama::multimodal::RawImage;

    let pixels: Vec<u8> = vec![0; width * height * 4];
    let image = RawImage::new(&pixels, width, height, 4)?;

    let response = processor.generate_with_raw_image(
        "What do you see?",
        &image,
        200
    )?;
    ```

=== "CLI"

    ```bash
    # Raw pixel input is not available via CLI
    # Use file path or buffer instead
    mullama run llava:13b "What do you see?" --image ./image.png
    ```

## Supported Image Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JPEG | `.jpg`, `.jpeg` | Most common, lossy compression |
| PNG | `.png` | Lossless, supports transparency |
| WebP | `.webp` | Modern format, good compression |
| BMP | `.bmp` | Uncompressed bitmap |
| GIF | `.gif` | First frame extracted for static analysis |
| TIFF | `.tiff`, `.tif` | High-quality, large files |

!!! info "Image Preprocessing"
    Images are automatically resized and normalized to match the vision encoder's expected input dimensions. The original aspect ratio is preserved with padding. No manual preprocessing is required.

## Vision Encoder Types

Mullama supports multiple vision encoder architectures:

| Encoder | Description | Models |
|---------|-------------|--------|
| CLIP | Contrastive Language-Image Pre-training | LLaVA, InternVL |
| DINOv2 | Self-supervised vision transformer | Some research models |
| SigLIP | Sigmoid-based CLIP variant | PaliGemma |

The correct encoder is automatically selected based on the model's metadata. No configuration is needed.

## Audio Input and Processing

Process audio alongside text for speech understanding tasks:

=== "Node.js"

    ```javascript
    import { Model, MultimodalProcessor } from 'mullama';

    const model = await Model.load('./audio-model.gguf');
    const processor = new MultimodalProcessor(model);

    const response = await processor.generate({
      text: "Transcribe this audio.",
      audio: ['./recording.wav'],
    });
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, MultimodalProcessor

    model = Model.load("./audio-model.gguf")
    processor = MultimodalProcessor(model)

    response = processor.generate(
        text="Transcribe this audio.",
        audio=["./recording.wav"],
    )
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::{Model, MultimodalProcessor};

    let model = Arc::new(Model::load("audio-model.gguf")?);
    let mut processor = MultimodalProcessor::new(model)?;

    let response = processor.generate_with_audio(
        "Transcribe this audio.",
        "recording.wav",
        500
    )?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    mullama run audio-model "Transcribe this audio." --audio ./recording.wav
    ```

### Audio Formats

| Format | Description | Sample Rates |
|--------|-------------|--------------|
| PCM16 | 16-bit signed integer | Any |
| PCM32 | 32-bit signed integer | Any |
| Float32 | 32-bit float | Any |
| WAV | Waveform Audio | Any (auto-detected) |
| MP3 | MPEG Layer 3 | Any (auto-decoded) |
| FLAC | Free Lossless Audio | Any (auto-decoded) |

!!! info "Audio Preprocessing"
    Audio is automatically resampled to the model's expected sample rate (typically 16 kHz) and converted to mono if needed. The `format-conversion` feature handles all necessary format transformations.

### Raw Audio Buffers

Process audio from memory (e.g., from a microphone stream):

=== "Node.js"

    ```javascript
    import { AudioBuffer } from 'mullama';

    // Float32 PCM samples at 16kHz
    const samples = new Float32Array(16000 * 5); // 5 seconds
    const audio = new AudioBuffer(samples, 16000, 1);

    const response = await processor.generate({
      text: "What was said?",
      audioBuffers: [audio],
    });
    ```

=== "Python"

    ```python
    import numpy as np
    from mullama import AudioBuffer

    # Float32 PCM samples at 16kHz
    samples = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds
    audio = AudioBuffer(samples, sample_rate=16000, channels=1)

    response = processor.generate(
        text="What was said?",
        audio_buffers=[audio],
    )
    ```

=== "Rust"

    ```rust
    use mullama::multimodal::AudioBuffer;

    let samples: Vec<f32> = vec![0.0; 16000 * 5]; // 5 seconds
    let audio = AudioBuffer::new(&samples, 16000, 1)?;

    let response = processor.generate_with_audio_buffer(
        "What was said?",
        &audio,
        500
    )?;
    ```

=== "CLI"

    ```bash
    # Record and process
    mullama run audio-model "What was said?" --audio-device default --duration 5
    ```

## Batch Multimodal Processing

Process multiple multimodal inputs efficiently:

=== "Node.js"

    ```javascript
    const images = [
      './image1.jpg',
      './image2.jpg',
      './image3.jpg',
    ];

    const results = await processor.generateBatch(
      images.map(img => ({
        text: "Describe this image briefly.",
        images: [img],
      }))
    );

    results.forEach((result, i) => {
      console.log(`Image ${i+1}: ${result}`);
    });
    ```

=== "Python"

    ```python
    images = ["./image1.jpg", "./image2.jpg", "./image3.jpg"]

    inputs = [
        {"text": "Describe this image briefly.", "images": [img]}
        for img in images
    ]
    results = processor.generate_batch(inputs)

    for i, result in enumerate(results):
        print(f"Image {i+1}: {result}")
    ```

=== "Rust"

    ```rust
    let images = vec!["image1.jpg", "image2.jpg", "image3.jpg"];

    let results = processor.generate_batch(
        images.iter().map(|img| {
            ("Describe this image briefly.", img.to_string())
        }).collect(),
        200
    )?;

    for (i, result) in results.iter().enumerate() {
        println!("Image {}: {}", i + 1, result);
    }
    ```

=== "CLI"

    ```bash
    # Process multiple images
    for img in image1.jpg image2.jpg image3.jpg; do
      mullama run llava:13b "Describe this image briefly." --image "$img"
    done
    ```

## Vision-Language Examples

### Describe an Image

=== "Node.js"

    ```javascript
    const response = await processor.generate({
      text: "Provide a detailed description of this image including colors, objects, and composition.",
      images: ['./landscape.jpg'],
      maxTokens: 500,
    });
    ```

=== "Python"

    ```python
    response = processor.generate(
        text="Provide a detailed description of this image including colors, objects, and composition.",
        images=["./landscape.jpg"],
        max_tokens=500,
    )
    ```

=== "Rust"

    ```rust
    let response = processor.generate_with_image(
        "Provide a detailed description of this image including colors, objects, and composition.",
        "landscape.jpg",
        500
    )?;
    ```

=== "CLI"

    ```bash
    mullama run llava:13b \
      "Provide a detailed description of this image including colors, objects, and composition." \
      --image ./landscape.jpg --max-tokens 500
    ```

### Answer Questions About an Image

=== "Node.js"

    ```javascript
    const questions = [
      "How many people are in this image?",
      "What is the dominant color?",
      "Is this indoors or outdoors?",
    ];

    for (const question of questions) {
      const answer = await processor.generate({
        text: question,
        images: ['./photo.jpg'],
        maxTokens: 100,
      });
      console.log(`Q: ${question}\nA: ${answer}\n`);
    }
    ```

=== "Python"

    ```python
    questions = [
        "How many people are in this image?",
        "What is the dominant color?",
        "Is this indoors or outdoors?",
    ]

    for question in questions:
        answer = processor.generate(
            text=question,
            images=["./photo.jpg"],
            max_tokens=100,
        )
        print(f"Q: {question}\nA: {answer}\n")
    ```

=== "Rust"

    ```rust
    let questions = vec![
        "How many people are in this image?",
        "What is the dominant color?",
        "Is this indoors or outdoors?",
    ];

    for question in &questions {
        let answer = processor.generate_with_image(question, "photo.jpg", 100)?;
        println!("Q: {}\nA: {}\n", question, answer);
    }
    ```

=== "CLI"

    ```bash
    mullama run llava:13b "How many people are in this image?" --image ./photo.jpg
    mullama run llava:13b "What is the dominant color?" --image ./photo.jpg
    mullama run llava:13b "Is this indoors or outdoors?" --image ./photo.jpg
    ```

### OCR and Text Extraction

=== "Node.js"

    ```javascript
    const response = await processor.generate({
      text: "Extract all text visible in this image. Format it as a list.",
      images: ['./document.png'],
      maxTokens: 1000,
    });
    console.log(response);
    ```

=== "Python"

    ```python
    response = processor.generate(
        text="Extract all text visible in this image. Format it as a list.",
        images=["./document.png"],
        max_tokens=1000,
    )
    print(response)
    ```

=== "Rust"

    ```rust
    let response = processor.generate_with_image(
        "Extract all text visible in this image. Format it as a list.",
        "document.png",
        1000
    )?;
    ```

=== "CLI"

    ```bash
    mullama run llava:13b \
      "Extract all text visible in this image. Format it as a list." \
      --image ./document.png --max-tokens 1000
    ```

## Audio-Language Examples

### Speech Transcription

=== "Node.js"

    ```javascript
    const transcription = await processor.generate({
      text: "Transcribe the following audio word for word.",
      audio: ['./speech.wav'],
      maxTokens: 1000,
    });
    console.log(transcription);
    ```

=== "Python"

    ```python
    transcription = processor.generate(
        text="Transcribe the following audio word for word.",
        audio=["./speech.wav"],
        max_tokens=1000,
    )
    print(transcription)
    ```

=== "Rust"

    ```rust
    let transcription = processor.generate_with_audio(
        "Transcribe the following audio word for word.",
        "speech.wav",
        1000
    )?;
    ```

=== "CLI"

    ```bash
    mullama run audio-model "Transcribe the following audio word for word." \
      --audio ./speech.wav --max-tokens 1000
    ```

### Audio Understanding

=== "Node.js"

    ```javascript
    const response = await processor.generate({
      text: "Describe the sounds in this audio clip. What is happening?",
      audio: ['./environment.wav'],
      maxTokens: 200,
    });
    ```

=== "Python"

    ```python
    response = processor.generate(
        text="Describe the sounds in this audio clip. What is happening?",
        audio=["./environment.wav"],
        max_tokens=200,
    )
    ```

=== "Rust"

    ```rust
    let response = processor.generate_with_audio(
        "Describe the sounds in this audio clip. What is happening?",
        "environment.wav",
        200
    )?;
    ```

=== "CLI"

    ```bash
    mullama run audio-model \
      "Describe the sounds in this audio clip. What is happening?" \
      --audio ./environment.wav
    ```

## Supported Multimodal Models

| Model | Modality | Description |
|-------|----------|-------------|
| LLaVA 1.5/1.6 | Vision | Image understanding with CLIP |
| BakLLaVA | Vision | High-resolution image understanding |
| Obsidian | Vision | Efficient vision-language model |
| MiniCPM-V | Vision | Compact vision model |
| Qwen2-VL | Vision | Qwen's vision-language model |

!!! warning "Model Compatibility"
    Multimodal models require both a text model and a vision/audio projector. Ensure you download the complete model with all components.

## See Also

- [Loading Models](models.md) -- Loading multimodal model files
- [Streaming](streaming.md) -- Streaming multimodal generation output
- [Tutorials: Vision Assistant](../examples/voice-assistant.md) -- Building a vision-language assistant
- [API Reference: Multimodal](../api/multimodal.md) -- Complete Multimodal API documentation
