---
title: "Tutorial: Multimodal Processing"
description: Process images and text together using vision-language models with Mullama for captioning, visual QA, and batch image analysis.
---

# Multimodal Processing

Process images alongside text using vision-language models (VLMs). This tutorial covers image captioning, visual question answering, supported formats, image preprocessing, and batch processing.

---

## What You'll Build

A multimodal processing pipeline that:

- Loads and processes images in various formats (JPEG, PNG, WebP)
- Generates image captions and descriptions
- Answers questions about image content
- Preprocesses images for optimal model performance
- Processes multiple images in batch
- Handles vision model loading and configuration

---

## Prerequisites

- Mullama with `multimodal` feature enabled
- A vision-capable GGUF model (e.g., LLaVA, BakLLaVA)
- Image processing dependencies:

=== "Linux (Ubuntu/Debian)"
    ```bash
    sudo apt install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev
    ```

=== "macOS"
    ```bash
    brew install libpng jpeg-turbo webp
    ```

- Rust toolchain (multimodal is primarily a Rust-native feature)
- Features: `multimodal`, optionally `format-conversion`

```bash
# Pull a vision-capable model
mullama pull llava:7b
```

---

## Vision Model Requirements

Not all models support multimodal input. You need a vision-language model:

| Model | Size | Description |
|-------|------|-------------|
| `llava:7b` | 4.7 GB | LLaVA 1.5 -- general-purpose vision |
| `llava:13b` | 8.1 GB | LLaVA 1.5 -- higher quality |
| `bakllava:7b` | 4.7 GB | BakLLaVA -- improved architecture |
| `llava-llama3:8b` | 5.0 GB | LLaVA with Llama 3 backbone |

Vision models consist of two components:

- **Vision encoder** -- Processes images into embeddings (e.g., CLIP ViT)
- **Language model** -- Generates text conditioned on vision embeddings

---

## Step 1: Load a Vision Model

```rust
use mullama::{MultimodalProcessor, ImageInput};
use std::path::Path;

// Initialize the multimodal processor with a vision model
let processor = MultimodalProcessor::new()
    .model_path("./llava-7b.Q4_K_M.gguf")
    .enable_image_processing()
    .n_ctx(2048)
    .n_gpu_layers(-1)  // GPU acceleration
    .build()?;

println!("Vision model loaded: {}", processor.model_name());
println!("Image support: {}", processor.supports_images());
```

---

## Step 2: Image Loading and Preprocessing

Load images and prepare them for the vision encoder.

```rust
use mullama::{ImageInput, ImageFormat};

// Load image from file
let image = ImageInput::from_file("./photo.jpg")?;
println!("Image size: {}x{}", image.width(), image.height());
println!("Format: {:?}", image.format());

// Load from bytes (e.g., from HTTP request)
let bytes = std::fs::read("./photo.png")?;
let image = ImageInput::from_bytes(&bytes, ImageFormat::Png)?;

// Load from URL (requires format-conversion feature)
let image = ImageInput::from_url("https://example.com/image.jpg").await?;

// Preprocessing: resize for optimal model input
let preprocessed = image
    .resize(336, 336)         // Standard CLIP input size
    .normalize()              // Normalize pixel values
    .to_rgb()?;               // Ensure RGB format

println!("Preprocessed: {}x{} RGB", preprocessed.width(), preprocessed.height());
```

### Supported Image Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JPEG | `.jpg`, `.jpeg` | Lossy, most common |
| PNG | `.png` | Lossless, supports transparency |
| WebP | `.webp` | Modern format, good compression |
| TIFF | `.tif`, `.tiff` | High quality, scientific imaging |
| BMP | `.bmp` | Uncompressed, large files |

---

## Step 3: Image Captioning

Generate descriptive captions for images.

```rust
use mullama::{MultimodalProcessor, ImageInput, SamplerParams};

fn caption_image(processor: &MultimodalProcessor, image_path: &str) -> Result<String, MullamaError> {
    let image = ImageInput::from_file(image_path)?;

    // Simple captioning prompt
    let caption = processor.generate_with_image(
        &image,
        "Describe this image in detail.",
        200,  // max_tokens
        Some(SamplerParams {
            temperature: 0.3,    // Low temperature for factual descriptions
            top_p: 0.9,
            ..Default::default()
        }),
    )?;

    println!("Caption: {}", caption.trim());
    Ok(caption.trim().to_string())
}

// Usage
let caption = caption_image(&processor, "./sunset.jpg")?;
```

---

## Step 4: Visual Question Answering

Ask questions about image content.

```rust
fn visual_qa(
    processor: &MultimodalProcessor,
    image_path: &str,
    question: &str,
) -> Result<String, MullamaError> {
    let image = ImageInput::from_file(image_path)?;

    let answer = processor.generate_with_image(
        &image,
        question,
        150,
        Some(SamplerParams {
            temperature: 0.2,    // Very focused for factual answers
            top_k: 20,
            ..Default::default()
        }),
    )?;

    println!("Q: {}", question);
    println!("A: {}", answer.trim());
    Ok(answer.trim().to_string())
}

// Ask different questions about the same image
let image_path = "./street_scene.jpg";
visual_qa(&processor, image_path, "How many people are in this image?")?;
visual_qa(&processor, image_path, "What is the weather like?")?;
visual_qa(&processor, image_path, "What colors are prominent?")?;
visual_qa(&processor, image_path, "Is this indoors or outdoors?")?;
```

---

## Step 5: Streaming Image Responses

Stream generated text for longer descriptions.

```rust
use mullama::{MultimodalProcessor, ImageInput, StreamConfig, TokenStream};
use futures::StreamExt;
use std::io::Write;

async fn stream_image_description(
    processor: &MultimodalProcessor,
    image_path: &str,
) -> Result<String, MullamaError> {
    let image = ImageInput::from_file(image_path)?;

    let config = StreamConfig::default()
        .max_tokens(300)
        .temperature(0.5);

    let prompt = "Provide a detailed description of this image, including colors, \
                  objects, people, setting, and mood.";

    let mut stream = processor.stream_with_image(&image, prompt, config).await?;
    let mut response = String::new();

    print!("Description: ");
    while let Some(token) = stream.next().await {
        let token = token?;
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
        response.push_str(&token.text);
        if token.is_final { break; }
    }
    println!();

    Ok(response)
}
```

---

## Step 6: Batch Image Processing

Process multiple images efficiently.

```rust
use mullama::{MultimodalProcessor, ImageInput, MullamaError};
use std::path::PathBuf;

fn batch_caption_images(
    processor: &MultimodalProcessor,
    image_paths: &[PathBuf],
    prompt: &str,
) -> Vec<Result<(PathBuf, String), MullamaError>> {
    let start = std::time::Instant::now();
    let mut results = Vec::new();

    for (i, path) in image_paths.iter().enumerate() {
        print!("[{}/{}] Processing: {} ... ",
            i + 1, image_paths.len(), path.display());
        std::io::stdout().flush().unwrap();

        let result = (|| {
            let image = ImageInput::from_file(path)?;
            let caption = processor.generate_with_image(
                &image, prompt, 100,
                Some(SamplerParams { temperature: 0.3, ..Default::default() }),
            )?;
            Ok((path.clone(), caption.trim().to_string()))
        })();

        match &result {
            Ok((_, caption)) => println!("\"{}\"", &caption[..caption.len().min(60)]),
            Err(e) => println!("Error: {}", e),
        }
        results.push(result);
    }

    let elapsed = start.elapsed();
    let successful = results.iter().filter(|r| r.is_ok()).count();
    println!("\nProcessed: {}/{} images in {:.2}s",
        successful, image_paths.len(), elapsed.as_secs_f64());

    results
}

// Usage
let paths: Vec<PathBuf> = std::fs::read_dir("./images/")?
    .filter_map(|e| e.ok())
    .filter(|e| {
        let ext = e.path().extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(ext, "jpg" | "jpeg" | "png" | "webp")
    })
    .map(|e| e.path())
    .collect();

let results = batch_caption_images(&processor, &paths, "Describe this image briefly:");
```

---

## Complete Working Example

```rust
use mullama::prelude::*;
use mullama::{MultimodalProcessor, ImageInput, SamplerParams, MullamaError};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mullama Multimodal Demo");
    println!("=======================\n");

    #[cfg(feature = "multimodal")]
    {
        let model_path = std::env::args().nth(1)
            .unwrap_or_else(|| "llava-7b.Q4_K_M.gguf".to_string());
        let image_path = std::env::args().nth(2)
            .unwrap_or_else(|| "sample.jpg".to_string());

        // Initialize processor
        println!("Loading vision model...");
        let processor = MultimodalProcessor::new()
            .model_path(&model_path)
            .enable_image_processing()
            .n_ctx(2048)
            .n_gpu_layers(-1)
            .build()?;
        println!("Model ready!\n");

        // Load image
        let image = ImageInput::from_file(&image_path)?;
        println!("Image: {} ({}x{})\n", image_path, image.width(), image.height());

        // Caption
        println!("--- Captioning ---");
        let caption = processor.generate_with_image(
            &image,
            "Describe this image in one sentence.",
            100,
            Some(SamplerParams { temperature: 0.3, ..Default::default() }),
        )?;
        println!("Caption: {}\n", caption.trim());

        // Visual QA
        println!("--- Visual QA ---");
        let questions = [
            "What objects can you see?",
            "What colors are prominent?",
            "Describe the mood or atmosphere.",
        ];

        for question in &questions {
            let answer = processor.generate_with_image(
                &image, question, 80,
                Some(SamplerParams { temperature: 0.2, ..Default::default() }),
            )?;
            println!("Q: {}", question);
            println!("A: {}\n", answer.trim());
        }

        // Detailed description
        println!("--- Detailed Description ---");
        let description = processor.generate_with_image(
            &image,
            "Provide a detailed, structured description of this image.",
            300,
            Some(SamplerParams { temperature: 0.5, ..Default::default() }),
        )?;
        println!("{}", description.trim());
    }

    #[cfg(not(feature = "multimodal"))]
    {
        println!("This example requires the 'multimodal' feature.");
        println!("Run with: cargo run --example multimodal --features multimodal");
    }

    Ok(())
}
```

---

## Python Bindings (Experimental)

Multimodal support in Python bindings is under development. Currently you can use the daemon for multimodal processing:

```python
import requests
import base64

# Using the Mullama daemon's multimodal endpoint
def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = requests.post("http://localhost:8080/v1/chat/completions", json={
        "model": "llava:7b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "max_tokens": 200
    })

    return response.json()["choices"][0]["message"]["content"]

# Usage
caption = describe_image("./photo.jpg")
print(f"Caption: {caption}")
```

---

## Image Preprocessing Tips

| Parameter | Recommendation | Reason |
|-----------|---------------|--------|
| Resolution | 336x336 or 672x672 | Matches CLIP encoder training |
| Format | RGB | Vision encoders expect 3 channels |
| Normalization | ImageNet mean/std | Standard for CLIP-based models |
| Aspect ratio | Preserve with padding | Avoids distortion |
| File size | < 10 MB | Memory efficiency during loading |

!!! warning "Model Compatibility"
    Not all GGUF models support multimodal input. You need specifically trained vision-language models. Standard text-only models will produce errors when given image input. Check model documentation or metadata for `vision` or `multimodal` capabilities.

---

## What's Next

- [Voice Assistant](voice-assistant.md) -- Combine vision with audio processing
- [Batch Processing](batch.md) -- Process image collections at scale
- [API Server](api-server.md) -- Serve multimodal endpoints over HTTP
- [Guide: Multimodal](../guide/multimodal.md) -- In-depth multimodal architecture
