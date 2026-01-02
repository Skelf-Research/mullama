# Basic Examples

Common usage patterns for Mullama.

## Simple Text Generation

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Load model
    let model = Arc::new(Model::load("model.gguf")?);
    println!("Loaded model with {} vocab", model.vocab_size());

    // Create context
    let mut ctx = Context::new(model, ContextParams::default())?;

    // Generate
    let response = ctx.generate("Hello, my name is", 50)?;
    println!("{}", response);

    Ok(())
}
```

## Chat Conversation

```rust
use mullama::{Model, Context, ContextParams, ChatMessage};
use std::sync::Arc;
use std::io::{self, Write};

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("chat-model.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;

    let system_prompt = "You are a helpful AI assistant.";

    loop {
        // Get user input
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() || input == "quit" {
            break;
        }

        // Format as chat
        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(input),
        ];

        let prompt = model.apply_chat_template(&messages)?;

        // Generate response
        print!("Assistant: ");
        ctx.generate_streaming(&prompt, 500, |token| {
            print!("{}", token);
            io::stdout().flush().ok();
            true
        })?;
        println!("\n");

        // Clear for next turn
        ctx.clear()?;
    }

    Ok(())
}
```

## Streaming Output

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut ctx = Context::new(model, ContextParams::default())?;

    let start = Instant::now();
    let mut token_count = 0;

    println!("Generating...\n");

    ctx.generate_streaming("Write a short story about a robot:", 300, |token| {
        print!("{}", token);
        std::io::stdout().flush().ok();
        token_count += 1;
        true
    })?;

    let elapsed = start.elapsed();
    println!("\n\n---");
    println!("Generated {} tokens in {:.2}s", token_count, elapsed.as_secs_f32());
    println!("Speed: {:.1} tokens/sec", token_count as f32 / elapsed.as_secs_f32());

    Ok(())
}
```

## Custom Sampling

```rust
use mullama::{Model, Context, ContextParams, SamplingParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut ctx = Context::new(model, ContextParams::default())?;

    // Creative writing settings
    let creative = SamplingParams {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 100,
        repeat_penalty: 1.2,
        ..Default::default()
    };

    let story = ctx.generate_with_params(
        "Once upon a time in a magical forest,",
        200,
        creative
    )?;
    println!("Creative:\n{}\n", story);

    ctx.clear()?;

    // Factual settings
    let factual = SamplingParams {
        temperature: 0.3,
        top_p: 0.8,
        top_k: 20,
        ..Default::default()
    };

    let facts = ctx.generate_with_params(
        "The capital of France is",
        50,
        factual
    )?;
    println!("Factual:\n{}", facts);

    Ok(())
}
```

## Batch Processing

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);

    let prompts = vec![
        "Translate to Spanish: Hello",
        "Translate to Spanish: Goodbye",
        "Translate to Spanish: Thank you",
    ];

    let mut results = Vec::new();

    for prompt in &prompts {
        let mut ctx = Context::new(model.clone(), ContextParams::default())?;
        let response = ctx.generate(prompt, 20)?;
        results.push(response);
    }

    for (prompt, result) in prompts.iter().zip(results.iter()) {
        println!("{} -> {}", prompt, result.trim());
    }

    Ok(())
}
```

## Embeddings and Similarity

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("embedding-model.gguf")?);
    let params = ContextParams { embeddings: true, ..Default::default() };
    let ctx = Context::new(model, params)?;

    // Get embeddings
    let query = ctx.get_embeddings("machine learning")?;
    let doc1 = ctx.get_embeddings("artificial intelligence and neural networks")?;
    let doc2 = ctx.get_embeddings("cooking recipes and ingredients")?;

    // Compare
    let sim1 = cosine_similarity(&query, &doc1);
    let sim2 = cosine_similarity(&query, &doc2);

    println!("Query: 'machine learning'");
    println!("Similarity to AI doc: {:.3}", sim1);
    println!("Similarity to cooking doc: {:.3}", sim2);

    Ok(())
}
```

## GPU Configuration

```rust
use mullama::{ModelBuilder, Context, ContextParams};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    // Offload 35 layers to GPU
    let model = Arc::new(
        ModelBuilder::new("model.gguf")
            .with_n_gpu_layers(35)
            .build()?
    );

    // Use flash attention
    let params = ContextParams {
        flash_attn: true,
        ..Default::default()
    };

    let mut ctx = Context::new(model, params)?;
    let response = ctx.generate("Hello!", 50)?;
    println!("{}", response);

    Ok(())
}
```

## Error Handling

```rust
use mullama::{Model, Context, ContextParams, MullamaError};
use std::sync::Arc;

fn main() {
    match run() {
        Ok(()) => println!("Success!"),
        Err(e) => handle_error(e),
    }
}

fn run() -> Result<(), MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut ctx = Context::new(model, ContextParams::default())?;
    let _ = ctx.generate("Hello", 50)?;
    Ok(())
}

fn handle_error(e: MullamaError) {
    match e {
        MullamaError::ModelLoadError(msg) => {
            eprintln!("Failed to load model: {}", msg);
            eprintln!("Check that the file exists and is a valid GGUF file.");
        }
        MullamaError::ContextError(msg) => {
            eprintln!("Context error: {}", msg);
            eprintln!("You may need more RAM or a smaller context size.");
        }
        MullamaError::GenerationError(msg) => {
            eprintln!("Generation failed: {}", msg);
        }
        MullamaError::MemoryError(msg) => {
            eprintln!("Out of memory: {}", msg);
            eprintln!("Try a smaller model or reduce batch size.");
        }
        _ => eprintln!("Error: {}", e),
    }
}
```

## Configuration from File

```rust
use mullama::{Model, Context, ContextParams, SamplingParams};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct Config {
    model_path: String,
    context_size: u32,
    temperature: f32,
    max_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load config
    let config: Config = serde_json::from_str(&std::fs::read_to_string("config.json")?)?;

    // Apply config
    let model = Arc::new(Model::load(&config.model_path)?);

    let ctx_params = ContextParams {
        n_ctx: config.context_size,
        ..Default::default()
    };

    let mut ctx = Context::new(model, ctx_params)?;

    let sampling = SamplingParams {
        temperature: config.temperature,
        ..Default::default()
    };

    let response = ctx.generate_with_params("Hello", config.max_tokens, sampling)?;
    println!("{}", response);

    Ok(())
}
```
