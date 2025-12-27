//! Multimodal example showing how to use vision-language models (VLMs)
//!
//! This example demonstrates:
//! 1. Loading a VLM model (e.g., LLaVA, Qwen-VL)
//! 2. Creating a multimodal context with an mmproj file
//! 3. Loading an image and generating a description
//!
//! To run this example, you need:
//! - A VLM model file (e.g., llava-v1.5-7b-q4_k_m.gguf)
//! - The corresponding mmproj file (e.g., llava-v1.5-7b-mmproj-f16.gguf)
//! - An image file to describe
//!
//! ```bash
//! cargo run --example multimodal --features multimodal -- \
//!     --model path/to/model.gguf \
//!     --mmproj path/to/mmproj.gguf \
//!     --image path/to/image.jpg
//! ```

use mullama::{Context, ContextParams, Model, MullamaError};
use std::sync::Arc;

#[cfg(feature = "multimodal")]
use mullama::{MtmdContext, MtmdParams};

fn main() -> Result<(), MullamaError> {
    println!("Mullama Multimodal Example");
    println!("==========================\n");

    #[cfg(not(feature = "multimodal"))]
    {
        println!("This example requires the 'multimodal' feature.");
        println!("Run with: cargo run --example multimodal --features multimodal");
        return Ok(());
    }

    #[cfg(feature = "multimodal")]
    {
        run_multimodal_example()
    }
}

#[cfg(feature = "multimodal")]
fn run_multimodal_example() -> Result<(), MullamaError> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_path = get_arg(&args, "--model");
    let mmproj_path = get_arg(&args, "--mmproj");
    let image_path = get_arg(&args, "--image");

    // If no args provided, show usage
    if model_path.is_none() || mmproj_path.is_none() || image_path.is_none() {
        print_usage();
        println!("\n--- API Demonstration Mode ---\n");
        demonstrate_api();
        return Ok(());
    }

    let model_path = model_path.unwrap();
    let mmproj_path = mmproj_path.unwrap();
    let image_path = image_path.unwrap();

    println!("Loading model: {}", model_path);
    let model = Arc::new(Model::load(&model_path)?);
    println!("Model loaded successfully!");
    println!("  Vocabulary size: {}", model.vocab_size());
    println!("  Training context: {} tokens\n", model.n_ctx_train());

    // Create context for text generation
    let ctx_params = ContextParams::default();
    let mut context = Context::new(model.clone(), ctx_params)?;
    println!("Created inference context\n");

    // Create multimodal context
    println!("Loading multimodal projector: {}", mmproj_path);
    let mtmd_params = MtmdParams::default();
    let mut mtmd = MtmdContext::new(&mmproj_path, &model, mtmd_params)?;
    println!("Multimodal context created!");
    println!("  Supports vision: {}", mtmd.supports_vision());
    println!("  Supports audio: {}", mtmd.supports_audio());
    if let Some(rate) = mtmd.audio_bitrate() {
        println!("  Audio bitrate: {} Hz", rate);
    }
    println!();

    // Load image
    println!("Loading image: {}", image_path);
    let image = mtmd.bitmap_from_file(&image_path)?;
    println!("Image loaded: {}x{}", image.width(), image.height());
    println!();

    // Prepare prompt with image marker
    let prompt = "Describe this image in detail: <__media__>";
    println!("Prompt: {}\n", prompt);

    // Tokenize the prompt with the image
    println!("Tokenizing prompt with image...");
    let chunks = mtmd.tokenize(prompt, &[&image])?;
    println!("Created {} input chunks", chunks.len());

    // Print chunk information
    for (i, chunk) in chunks.iter().enumerate() {
        let type_str = match chunk.chunk_type() {
            mullama::ChunkType::Text => "text",
            mullama::ChunkType::Image => "image",
            mullama::ChunkType::Audio => "audio",
        };
        println!("  Chunk {}: {} ({} tokens)", i, type_str, chunk.n_tokens());
    }
    println!();

    // Evaluate the chunks
    println!("Evaluating multimodal input...");
    let n_past = mtmd.eval_chunks(
        &mut context,
        &chunks,
        0,    // n_past: starting position
        0,    // seq_id
        512,  // n_batch
        true, // logits_last
    )?;
    println!("Evaluated {} tokens\n", n_past);

    // Generate response
    println!("Generating response...");
    println!("---");

    // Use the context to generate text (simplified - actual implementation
    // would use proper sampling and token generation)
    let max_tokens = 256;
    let mut generated = String::new();
    let mut n_decoded = 0;

    // Note: This is a simplified generation loop. In practice, you'd use
    // the full sampling infrastructure from the Context.
    for _ in 0..max_tokens {
        // Get logits and sample next token
        let logits = context.get_logits();
        if logits.is_empty() {
            break;
        }

        // Simple greedy sampling (in practice, use proper sampler)
        let token = argmax(logits);

        // Check for end of sequence
        if model.token_is_eog(token) {
            break;
        }

        // Decode token to string
        // lstrip=0 means no leading whitespace removal, special=false for normal tokens
        if let Ok(text) = model.token_to_str(token, 0, false) {
            print!("{}", text);
            generated.push_str(&text);
        }

        // Decode next token
        n_decoded += 1;
        if context.decode(&[token]).is_err() {
            break;
        }
    }

    println!("\n---");
    println!("\nGeneration complete! ({} tokens)", n_decoded);

    Ok(())
}

#[cfg(feature = "multimodal")]
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

#[cfg(feature = "multimodal")]
fn print_usage() {
    println!("Usage: cargo run --example multimodal --features multimodal -- \\");
    println!("    --model <model.gguf> \\");
    println!("    --mmproj <mmproj.gguf> \\");
    println!("    --image <image.jpg>");
    println!();
    println!("Required files:");
    println!("  --model   Path to VLM model (e.g., llava-v1.5-7b-q4_k_m.gguf)");
    println!("  --mmproj  Path to multimodal projector (e.g., llava-v1.5-7b-mmproj-f16.gguf)");
    println!("  --image   Path to image file (jpg, png, bmp, gif)");
}

#[cfg(feature = "multimodal")]
fn demonstrate_api() {
    println!("The multimodal API provides:");
    println!();
    println!("1. MtmdContext - Main multimodal processing context");
    println!("   let mtmd = MtmdContext::new(mmproj_path, &model, params)?;");
    println!();
    println!("2. Bitmap - Image or audio data container");
    println!("   let image = mtmd.bitmap_from_file(\"image.jpg\")?;");
    println!("   let audio = mtmd.bitmap_from_file(\"audio.wav\")?;");
    println!();
    println!("3. InputChunks - Tokenized multimodal input");
    println!("   let chunks = mtmd.tokenize(\"Describe: <__media__>\", &[&image])?;");
    println!();
    println!("4. Evaluation in LLM context");
    println!("   let n_past = mtmd.eval_chunks(&mut context, &chunks, 0, 0, 512, true)?;");
    println!();
    println!("Supported models:");
    println!("  - LLaVA (various versions)");
    println!("  - Qwen-VL");
    println!("  - InternVL");
    println!("  - Other llama.cpp compatible VLMs");
}

#[cfg(feature = "multimodal")]
fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}
