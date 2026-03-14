# Multimodal Examples

Examples using vision-language and audio models.

## Image Description

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load VLM model
    let model = Arc::new(Model::load("llava-v1.5-7b-q4.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;

    // Create multimodal context
    let mut mtmd = MtmdContext::new(
        "llava-v1.5-7b-mmproj-f16.gguf",
        &model,
        MtmdParams::default()
    )?;

    // Load image
    let image = mtmd.bitmap_from_file("photo.jpg")?;
    println!("Image size: {}x{}", image.width(), image.height());

    // Create prompt with image
    let chunks = mtmd.tokenize(
        "<|im_start|>user\nDescribe this image in detail.<__media__><|im_end|>\n<|im_start|>assistant\n",
        &[&image]
    )?;

    // Evaluate
    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    // Generate description
    print!("Description: ");
    ctx.generate_streaming_continue(n_past, 256, |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        true
    })?;
    println!();

    Ok(())
}
```

## Image Q&A

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;
use std::io::{self, Write};

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    // Load image once
    let image = mtmd.bitmap_from_file("image.jpg")?;

    loop {
        print!("Question (or 'quit'): ");
        io::stdout().flush()?;

        let mut question = String::new();
        io::stdin().read_line(&mut question)?;
        let question = question.trim();

        if question.is_empty() || question == "quit" {
            break;
        }

        // Build prompt
        let prompt = format!(
            "<|im_start|>user\n{}<__media__><|im_end|>\n<|im_start|>assistant\n",
            question
        );

        // Tokenize and evaluate
        let chunks = mtmd.tokenize(&prompt, &[&image])?;
        let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

        // Generate answer
        print!("Answer: ");
        ctx.generate_streaming_continue(n_past, 200, |token| {
            print!("{}", token);
            io::stdout().flush().ok();
            true
        })?;
        println!("\n");

        ctx.clear()?;
    }

    Ok(())
}
```

## Compare Two Images

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    // Load both images
    let image1 = mtmd.bitmap_from_file("before.jpg")?;
    let image2 = mtmd.bitmap_from_file("after.jpg")?;

    // Compare with two media markers
    let prompt = "<|im_start|>user\nCompare these two images. First image:<__media__>\nSecond image:<__media__>\nWhat are the differences?<|im_end|>\n<|im_start|>assistant\n";

    let chunks = mtmd.tokenize(prompt, &[&image1, &image2])?;
    println!("Created {} chunks", chunks.len());

    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    let response = ctx.generate_continue(n_past, 300)?;
    println!("Comparison:\n{}", response);

    Ok(())
}
```

## OCR / Text Extraction

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn extract_text(image_path: &str) -> Result<String, mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    let image = mtmd.bitmap_from_file(image_path)?;

    let chunks = mtmd.tokenize(
        "<|im_start|>user\nExtract all text from this image. Output only the text, nothing else.<__media__><|im_end|>\n<|im_start|>assistant\n",
        &[&image]
    )?;

    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;
    ctx.generate_continue(n_past, 500)
}

fn main() -> Result<(), mullama::MullamaError> {
    let text = extract_text("document.png")?;
    println!("Extracted text:\n{}", text);
    Ok(())
}
```

## Image Classification

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn classify_image(image_path: &str, categories: &[&str]) -> Result<String, mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    let image = mtmd.bitmap_from_file(image_path)?;

    let categories_str = categories.join(", ");
    let prompt = format!(
        "<|im_start|>user\nClassify this image into one of these categories: {}. Respond with only the category name.<__media__><|im_end|>\n<|im_start|>assistant\n",
        categories_str
    );

    let chunks = mtmd.tokenize(&prompt, &[&image])?;
    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    ctx.generate_continue(n_past, 20)
}

fn main() -> Result<(), mullama::MullamaError> {
    let categories = ["cat", "dog", "bird", "car", "building", "landscape"];
    let result = classify_image("photo.jpg", &categories)?;
    println!("Classification: {}", result.trim());
    Ok(())
}
```

## Process Multiple Images

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;
use std::path::Path;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mtmd_params = MtmdParams::default();

    let image_dir = Path::new("./images");
    let mut descriptions = Vec::new();

    for entry in std::fs::read_dir(image_dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            // Create fresh context for each image
            let mut ctx = Context::new(model.clone(), ContextParams::default())?;
            let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, mtmd_params.clone())?;

            let image = mtmd.bitmap_from_file(path.to_str().unwrap())?;
            let chunks = mtmd.tokenize(
                "Describe this image briefly: <__media__>",
                &[&image]
            )?;

            let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;
            let desc = ctx.generate_continue(n_past, 100)?;

            descriptions.push((path.file_name().unwrap().to_string_lossy().to_string(), desc));
            println!("Processed: {}", path.display());
        }
    }

    println!("\n--- Results ---");
    for (file, desc) in descriptions {
        println!("{}: {}", file, desc.trim());
    }

    Ok(())
}
```

## Load Image from URL

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    // Download image
    let response = reqwest::blocking::get("https://example.com/image.jpg")?;
    let image_data = response.bytes()?;

    // Load from buffer
    let image = mtmd.bitmap_from_buffer(&image_data)?;

    let chunks = mtmd.tokenize("What's in this image? <__media__>", &[&image])?;
    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    let response = ctx.generate_continue(n_past, 200)?;
    println!("{}", response);

    Ok(())
}
```

## Custom Image Processing

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams, Bitmap};
use std::sync::Arc;
use image::{self, GenericImageView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("llava-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;

    // Load and preprocess with image crate
    let img = image::open("photo.jpg")?;

    // Resize
    let img = img.resize(800, 600, image::imageops::FilterType::Lanczos3);

    // Convert to RGB bytes
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels: Vec<u8> = rgb.into_raw();

    // Create bitmap from raw pixels
    let bitmap = Bitmap::from_rgb(width, height, &pixels)?;

    // Process normally
    let chunks = mtmd.tokenize("Describe this image: <__media__>", &[&bitmap])?;
    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    let response = ctx.generate_continue(n_past, 200)?;
    println!("{}", response);

    Ok(())
}
```

## Check Model Capabilities

```rust
use mullama::{Model, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("multimodal-model.gguf")?);
    let mtmd = MtmdContext::new("mmproj.gguf", &model, MtmdParams::default())?;

    println!("Model capabilities:");
    println!("  Vision: {}", mtmd.supports_vision());
    println!("  Audio: {}", mtmd.supports_audio());

    if let Some(rate) = mtmd.audio_bitrate() {
        println!("  Audio sample rate: {} Hz", rate);
    }

    println!("  Uses M-RoPE: {}", mtmd.uses_mrope());
    println!("  Needs non-causal: {}", mtmd.needs_non_causal());

    Ok(())
}
```
