# Text Generation

Learn how to generate text with Mullama's inference engine.

## Basic Generation

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("model.gguf")?);
let mut context = Context::new(model.clone(), ContextParams::default())?;

let response = context.generate("Hello, how are you?", 100)?;
println!("{}", response);
```

## Context Parameters

Configure the inference context:

```rust
let params = ContextParams {
    n_ctx: 4096,        // Context window size
    n_batch: 512,       // Tokens processed per batch
    n_ubatch: 512,      // Micro-batch size
    n_threads: 8,       // CPU threads for generation
    n_threads_batch: 8, // CPU threads for batch processing
    rope_freq_base: 0.0,    // RoPE frequency base (0 = auto)
    rope_freq_scale: 0.0,   // RoPE frequency scale (0 = auto)
    flash_attn: true,   // Use flash attention if available
    ..Default::default()
};

let mut context = Context::new(model, params)?;
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_ctx` | Maximum context length | Model default |
| `n_batch` | Prompt processing batch size | 2048 |
| `n_threads` | CPU threads | System cores |
| `flash_attn` | Enable flash attention | true |

## Sampling Parameters

Control the randomness and quality of generation:

```rust
use mullama::SamplingParams;

let sampling = SamplingParams {
    temperature: 0.7,       // Randomness (0 = deterministic)
    top_k: 40,              // Consider top K tokens
    top_p: 0.9,             // Nucleus sampling threshold
    min_p: 0.05,            // Minimum probability threshold
    repeat_penalty: 1.1,    // Penalize repetition
    repeat_last_n: 64,      // Window for repeat penalty
    ..Default::default()
};

let response = context.generate_with_params("prompt", 100, sampling)?;
```

### Sampling Strategies

#### Temperature

Controls randomness. Lower = more focused, higher = more creative.

```rust
// Deterministic (best for code, facts)
let params = SamplingParams { temperature: 0.0, ..Default::default() };

// Balanced (good for most tasks)
let params = SamplingParams { temperature: 0.7, ..Default::default() };

// Creative (stories, brainstorming)
let params = SamplingParams { temperature: 1.2, ..Default::default() };
```

#### Top-K Sampling

Limits choices to the K most likely tokens:

```rust
let params = SamplingParams {
    top_k: 40,  // Only consider top 40 tokens
    ..Default::default()
};
```

#### Top-P (Nucleus) Sampling

Dynamically selects tokens until cumulative probability reaches P:

```rust
let params = SamplingParams {
    top_p: 0.9,  // Consider tokens until 90% probability mass
    ..Default::default()
};
```

#### Min-P Sampling

Filters tokens below a minimum probability relative to the top token:

```rust
let params = SamplingParams {
    min_p: 0.05,  // Exclude tokens < 5% of top token probability
    ..Default::default()
};
```

### Repetition Control

Prevent repetitive output:

```rust
let params = SamplingParams {
    repeat_penalty: 1.1,     // Penalty multiplier for repeated tokens
    repeat_last_n: 64,       // Look back window
    presence_penalty: 0.0,   // Penalize any token that appeared
    frequency_penalty: 0.0,  // Penalize based on frequency
    ..Default::default()
};
```

## Streaming Generation

Get tokens as they're generated:

```rust
context.generate_streaming("Tell me a story:", 500, |token| {
    print!("{}", token);
    std::io::Write::flush(&mut std::io::stdout()).ok();
    true  // Return false to stop early
})?;
```

### With Progress Tracking

```rust
let mut token_count = 0;

context.generate_streaming("Write an essay:", 1000, |token| {
    token_count += 1;
    print!("{}", token);

    if token_count % 100 == 0 {
        eprintln!("\n[{} tokens generated]", token_count);
    }

    true
})?;
```

## Chat Format

Format messages for chat models:

```rust
// Manual formatting
let prompt = format!(
    "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}|im_end|>
<|im_start|>assistant
",
    user_message
);

let response = context.generate(&prompt, 500)?;
```

### Using Chat Templates

```rust
use mullama::ChatMessage;

let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("What is the capital of France?"),
];

let prompt = model.apply_chat_template(&messages)?;
let response = context.generate(&prompt, 100)?;
```

## Stop Sequences

Stop generation at specific strings:

```rust
let params = SamplingParams {
    stop_sequences: vec![
        "\n\n".to_string(),
        "User:".to_string(),
    ],
    ..Default::default()
};

let response = context.generate_with_params(prompt, 500, params)?;
```

## Managing Context

### Clearing Context

Reset the context for a new conversation:

```rust
context.clear()?;
```

### Context Window Management

When context fills up:

```rust
// Check current position
let n_past = context.n_past();
let n_ctx = context.n_ctx();

if n_past > n_ctx - 100 {
    // Context nearly full, need to handle
    context.clear()?;
    // Re-inject system prompt, summarize history, etc.
}
```

## Batch Processing

Process multiple prompts efficiently:

```rust
use mullama::Batch;

let prompts = vec![
    "Translate to French: Hello",
    "Translate to French: Goodbye",
    "Translate to French: Thank you",
];

// Create batch
let mut batch = Batch::new(512, prompts.len());

for (i, prompt) in prompts.iter().enumerate() {
    let tokens = model.tokenize(prompt, true, false)?;
    batch.add_sequence(i, &tokens)?;
}

// Process batch
context.decode_batch(&batch)?;
```

## Error Handling

```rust
use mullama::MullamaError;

match context.generate(prompt, max_tokens) {
    Ok(response) => println!("{}", response),
    Err(MullamaError::ContextError(msg)) => {
        eprintln!("Context error: {}", msg);
    }
    Err(MullamaError::GenerationError(msg)) => {
        eprintln!("Generation failed: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Performance Tips

1. **Use appropriate batch sizes** - Larger batches for long prompts
2. **Enable flash attention** - Significant speedup for long contexts
3. **Tune thread count** - Match to physical cores, not hyperthreads
4. **Reuse contexts** - Creating contexts is expensive
5. **Use streaming** - Better UX and allows early stopping
