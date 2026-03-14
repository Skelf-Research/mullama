# Multimodal Support

Mullama supports vision-language models (VLMs) and audio-language models through the `multimodal` feature.

## Enabling Multimodal

Add the feature to your `Cargo.toml`:

```toml
[dependencies]
mullama = { version = "0.1", features = ["multimodal"] }
```

## Vision-Language Models

### Supported Models

| Model | Size | Description |
|-------|------|-------------|
| NanoLLaVA | 0.5B | Tiny VLM for testing |
| LLaVA 1.5 | 7B/13B | High quality image understanding |
| LLaVA 1.6 | 7B/13B/34B | Improved visual reasoning |
| Qwen-VL | 7B | Strong multilingual support |
| InternVL | Various | State-of-the-art performance |

### Basic Usage

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load the text model
    let model = Arc::new(Model::load("llava-v1.5-7b-q4.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams::default())?;

    // Create multimodal context with projector
    let mtmd_params = MtmdParams::default();
    let mut mtmd = MtmdContext::new(
        "llava-v1.5-7b-mmproj-f16.gguf",
        &model,
        mtmd_params
    )?;

    println!("Supports vision: {}", mtmd.supports_vision());
    println!("Supports audio: {}", mtmd.supports_audio());

    // Load image
    let image = mtmd.bitmap_from_file("photo.jpg")?;
    println!("Image size: {}x{}", image.width(), image.height());

    // Tokenize with image
    let chunks = mtmd.tokenize(
        "Describe this image in detail: <__media__>",
        &[&image]
    )?;

    println!("Created {} chunks", chunks.len());

    // Evaluate chunks
    let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;

    // Generate response
    let response = context.generate_continue(n_past, 256)?;
    println!("{}", response);

    Ok(())
}
```

## Image Loading

### From File

```rust
let image = mtmd.bitmap_from_file("image.jpg")?;
```

Supported formats: JPEG, PNG, BMP, GIF, WebP

### From Buffer

```rust
let image_data = std::fs::read("image.png")?;
let image = mtmd.bitmap_from_buffer(&image_data)?;
```

### From Raw Pixels

```rust
use mullama::Bitmap;

// RGB data: width * height * 3 bytes
let pixels: Vec<u8> = generate_image();
let image = Bitmap::from_rgb(width, height, &pixels)?;
```

## Multiple Images

Include multiple images in a single prompt:

```rust
let image1 = mtmd.bitmap_from_file("photo1.jpg")?;
let image2 = mtmd.bitmap_from_file("photo2.jpg")?;

// Use multiple markers
let chunks = mtmd.tokenize(
    "Compare these two images: <__media__> and <__media__>",
    &[&image1, &image2]
)?;
```

## Custom Media Markers

Change the default `<__media__>` marker:

```rust
let params = MtmdParams {
    media_marker: Some("<image>".to_string()),
    ..Default::default()
};

let mtmd = MtmdContext::new(mmproj_path, &model, params)?;

// Now use <image> instead of <__media__>
let chunks = mtmd.tokenize(
    "What's in this picture? <image>",
    &[&image]
)?;
```

## Audio Support

Some models support audio input:

```rust
// Check audio support
if mtmd.supports_audio() {
    let sample_rate = mtmd.audio_bitrate().unwrap();
    println!("Audio sample rate: {} Hz", sample_rate);

    // Load audio file
    let audio = mtmd.bitmap_from_file("speech.wav")?;

    let chunks = mtmd.tokenize(
        "Transcribe this audio: <__media__>",
        &[&audio]
    )?;

    // Process as usual
    let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;
}
```

## MtmdParams Configuration

```rust
let params = MtmdParams {
    use_gpu: true,              // Use GPU for vision encoder
    print_timings: false,       // Print processing times
    n_threads: 4,               // CPU threads
    media_marker: None,         // Custom marker (None = default)
    warmup: true,               // Warmup encode on init
    image_min_tokens: None,     // Min tokens per image
    image_max_tokens: None,     // Max tokens per image
    ..Default::default()
};
```

## Understanding Chunks

Multimodal input is split into chunks:

```rust
let chunks = mtmd.tokenize(prompt, &[&image])?;

for (i, chunk) in chunks.iter().enumerate() {
    match chunk.chunk_type() {
        ChunkType::Text => {
            println!("Chunk {}: Text ({} tokens)", i, chunk.n_tokens());
        }
        ChunkType::Image => {
            println!("Chunk {}: Image ({} tokens)", i, chunk.n_tokens());
        }
        ChunkType::Audio => {
            println!("Chunk {}: Audio ({} tokens)", i, chunk.n_tokens());
        }
    }
}
```

## Chat with Images

Build a conversation with images:

```rust
// First turn with image
let image = mtmd.bitmap_from_file("chart.png")?;
let chunks = mtmd.tokenize(
    "<|im_start|>user\nWhat does this chart show? <__media__><|im_end|>\n<|im_start|>assistant\n",
    &[&image]
)?;

let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;
let response = context.generate_continue(n_past, 256)?;
println!("Assistant: {}", response);

// Follow-up (text only)
let follow_up = format!(
    "<|im_end|>\n<|im_start|>user\nWhat's the trend?<|im_end|>\n<|im_start|>assistant\n"
);
let tokens = model.tokenize(&follow_up, false, true)?;
context.decode(&tokens)?;
let response = context.generate_continue(context.n_past(), 256)?;
println!("Assistant: {}", response);
```

## Image Description Example

Complete example for describing images:

```rust
use mullama::{Model, Context, ContextParams, MtmdContext, MtmdParams};
use std::sync::Arc;

fn describe_image(image_path: &str) -> Result<String, mullama::MullamaError> {
    let model = Arc::new(Model::load("llava-v1.5-7b-q4.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams {
        n_ctx: 4096,
        ..Default::default()
    })?;

    let mut mtmd = MtmdContext::new(
        "llava-v1.5-7b-mmproj-f16.gguf",
        &model,
        MtmdParams::default()
    )?;

    let image = mtmd.bitmap_from_file(image_path)?;

    let chunks = mtmd.tokenize(
        "<|im_start|>user\nDescribe this image in detail.<__media__><|im_end|>\n<|im_start|>assistant\n",
        &[&image]
    )?;

    let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;
    let description = context.generate_continue(n_past, 512)?;

    Ok(description)
}
```

## Performance Tips

1. **Use GPU** - Vision encoders benefit greatly from GPU acceleration
2. **Batch images** - Process multiple images together when possible
3. **Resize large images** - Very large images use more tokens
4. **Warmup** - First inference is slower due to memory allocation
5. **Use quantized projectors** - F16 projectors can be quantized for speed

## Error Handling

```rust
match mtmd.bitmap_from_file(path) {
    Ok(image) => process_image(image),
    Err(MullamaError::MultimodalError(msg)) => {
        eprintln!("Failed to load image: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}

match mtmd.tokenize(prompt, &images) {
    Ok(chunks) => evaluate(chunks),
    Err(MullamaError::InvalidInput(msg)) => {
        // Wrong number of images for markers
        eprintln!("Input error: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```
