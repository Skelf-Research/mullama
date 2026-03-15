---
title: Streaming API
description: Real-time token streaming with backpressure handling and cancellation
---

# Streaming API

The streaming module provides real-time token generation with the `Stream` trait, enabling incremental text output, backpressure handling, and integration with web response streams.

!!! info "Feature Gate"
    This module requires the `streaming` feature flag (which automatically enables `async`):
    ```toml
    mullama = { version = "0.1", features = ["streaming"] }
    ```

## TokenStream

The primary streaming interface, implementing the `futures::Stream` trait for async iteration over generated tokens.

```rust
pub struct TokenStream {
    // Internal state for token generation and buffering
}

impl TokenStream {
    /// Get next token (async, waits for generation)
    pub async fn next(&mut self) -> Option<Result<TokenData, MullamaError>>;

    /// Try to get next token without waiting (non-blocking)
    pub fn try_next(&mut self) -> Option<Result<TokenData, MullamaError>>;

    /// Get stream configuration
    pub fn config(&self) -> &StreamConfig;

    /// Check if stream has completed
    pub fn is_finished(&self) -> bool;

    /// Get count of tokens generated so far
    pub fn tokens_generated(&self) -> usize;

    /// Cancel the stream (stops generation immediately)
    pub fn cancel(&mut self);
}

// Implements the futures Stream trait for use with StreamExt combinators
impl Stream for TokenStream {
    type Item = Result<TokenData, MullamaError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>>;
}
```

### TokenData

Individual token data emitted by the stream.

```rust
#[derive(Debug, Clone)]
pub struct TokenData {
    pub text: String,        // Decoded text for this token
    pub token_id: TokenId,   // Numeric token ID
    pub is_final: bool,      // Whether this is the last token (EOS or max reached)
    pub logprob: f32,        // Log probability of this token selection
}
```

## StreamConfig

Configuration for streaming behavior, using a builder pattern.

```rust
pub struct StreamConfig {
    // Internal fields
}

impl StreamConfig {
    pub fn new() -> Self;
    pub fn max_tokens(mut self, max: usize) -> Self;
    pub fn temperature(mut self, temp: f32) -> Self;
    pub fn top_k(mut self, k: u32) -> Self;
    pub fn top_p(mut self, p: f32) -> Self;
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self;
    pub fn stream_delay_ms(mut self, delay: u64) -> Self;
    pub fn buffer_size(mut self, size: usize) -> Self;
    pub fn timeout_ms(mut self, timeout: u64) -> Self;
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_tokens` | `usize` | `256` | Maximum number of tokens to generate |
| `temperature` | `f32` | `0.8` | Sampling temperature (higher = more random) |
| `top_k` | `u32` | `40` | Top-K sampling parameter |
| `top_p` | `f32` | `0.95` | Nucleus sampling threshold |
| `stop_sequences` | `Vec<String>` | `[]` | Text sequences that trigger early stopping |
| `stream_delay_ms` | `u64` | `0` | Artificial delay between tokens (for testing/throttling) |
| `buffer_size` | `usize` | `32` | Internal buffer size for backpressure management |
| `timeout_ms` | `u64` | `30000` | Maximum time for the entire stream (0 = no timeout) |

**Example:**

```rust
let config = StreamConfig::new()
    .max_tokens(500)
    .temperature(0.7)
    .top_p(0.9)
    .stop_sequences(vec!["\n\n".to_string(), "END".to_string()])
    .buffer_size(64)
    .timeout_ms(60000);
```

## Creating Streams

Streams are created from `AsyncModel` or `AsyncContext`:

```rust
use mullama::{AsyncModel, StreamConfig};

let model = AsyncModel::load("model.gguf").await?;

let config = StreamConfig::new()
    .max_tokens(200)
    .temperature(0.7);

let stream = model.generate_stream("Tell me about Rust", config).await?;
```

## Token-by-Token Iteration

### Using `while let` with `next()`

The most common pattern for consuming a token stream:

```rust
use mullama::{AsyncModel, StreamConfig};

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    let model = AsyncModel::load("model.gguf").await?;

    let config = StreamConfig::new()
        .max_tokens(200)
        .temperature(0.7);

    let mut stream = model.generate_stream("Tell me about AI", config).await?;

    print!("AI: ");
    while let Some(token_result) = stream.next().await {
        let token_data = token_result?;
        print!("{}", token_data.text);
        // Flush stdout for real-time display
    }
    println!();

    println!("Generated {} tokens", stream.tokens_generated());
    Ok(())
}
```

### Using `StreamExt` combinators

```rust
use tokio_stream::StreamExt;

let mut stream = model.generate_stream("Hello", config).await?;

// Collect all tokens into a string
let tokens: Vec<String> = stream
    .map(|r| r.map(|t| t.text))
    .collect::<Result<Vec<_>, _>>()
    .await?;

let full_text = tokens.join("");
println!("Full response: {}", full_text);
```

### Taking a limited number of tokens

```rust
use tokio_stream::StreamExt;

let mut stream = model.generate_stream("Count:", config).await?;

// Take only first 10 tokens
let first_10: Vec<_> = stream
    .take(10)
    .collect::<Vec<_>>()
    .await;
```

## Error Handling in Streams

Errors can occur during streaming for various reasons:

```rust
let mut stream = model.generate_stream("prompt", config).await?;

while let Some(token_result) = stream.next().await {
    match token_result {
        Ok(token_data) => {
            print!("{}", token_data.text);
        }
        Err(MullamaError::StreamingError(msg)) => {
            eprintln!("Stream error: {}", msg);
            break;
        }
        Err(MullamaError::GenerationError(msg)) => {
            eprintln!("Generation failed: {}", msg);
            break;
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            break;
        }
    }
}
```

## Backpressure

The stream uses an internal buffer to handle backpressure when the consumer is slower than the producer:

1. Tokens are buffered up to `buffer_size`
2. When the buffer is full, generation pauses until the consumer reads
3. This prevents unbounded memory growth

```rust
// Configure small buffer for demonstration
let config = StreamConfig::new()
    .max_tokens(1000)
    .buffer_size(16); // Generation pauses when 16 tokens are buffered

let mut stream = model.generate_stream("Write a long essay", config).await?;

while let Some(token_result) = stream.next().await {
    let token_data = token_result?;
    print!("{}", token_data.text);

    // Simulate slow processing -- backpressure kicks in automatically
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
}
```

## Cancellation and Early Stopping

### Manual Cancellation

```rust
let mut stream = model.generate_stream("Count:", config).await?;
let mut count = 0;

while let Some(token_result) = stream.next().await {
    let token_data = token_result?;
    print!("{}", token_data.text);

    count += 1;
    if count >= 50 {
        stream.cancel(); // Immediately stops generation
        break;
    }
}

println!("\nStopped after {} tokens", count);
```

### Stop Sequences

Configured stop sequences automatically end the stream when generated:

```rust
let config = StreamConfig::new()
    .max_tokens(500)
    .stop_sequences(vec![
        "\n\n".to_string(),     // Stop on double newline
        "```".to_string(),      // Stop on code fence
        "END".to_string(),      // Stop on explicit marker
    ]);

let mut stream = model.generate_stream("Write a function:", config).await?;
// Stream automatically stops when any stop sequence is generated
```

### Timeout

```rust
let config = StreamConfig::new()
    .max_tokens(10000)
    .timeout_ms(5000); // 5 second timeout

let mut stream = model.generate_stream("Write forever", config).await?;
// Stream will end with StreamingError after 5 seconds
```

## Integration with Web Responses

Streams integrate naturally with web frameworks for Server-Sent Events (SSE):

```rust
use axum::{response::sse::{Event, Sse}, extract::State};
use futures::stream::Stream;
use std::convert::Infallible;

async fn stream_handler(
    State(model): State<Arc<AsyncModel>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let config = StreamConfig::new()
        .max_tokens(500)
        .temperature(0.7);

    let token_stream = model
        .generate_stream("Hello", config)
        .await
        .unwrap();

    let event_stream = token_stream.map(|result| {
        match result {
            Ok(token_data) => Ok(Event::default().data(token_data.text)),
            Err(_) => Ok(Event::default().data("[ERROR]")),
        }
    });

    Sse::new(event_stream)
}
```

## Complete Example

```rust
use mullama::{AsyncModel, StreamConfig};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    let model = AsyncModel::load("model.gguf").await?;

    // Interactive streaming generation with multiple prompts
    let prompts = vec![
        "What is Rust?",
        "Explain ownership in one paragraph:",
        "Write a haiku about programming:",
    ];

    for prompt in prompts {
        let config = StreamConfig::new()
            .max_tokens(150)
            .temperature(0.7)
            .stop_sequences(vec!["\n\n".to_string()]);

        println!("\n> {}", prompt);
        print!("  ");

        let mut stream = model.generate_stream(prompt, config).await?;
        let mut total_tokens = 0;

        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(token_data) => {
                    print!("{}", token_data.text);
                    total_tokens += 1;
                }
                Err(e) => {
                    eprintln!("\nStream error: {}", e);
                    break;
                }
            }
        }

        println!("\n  [{} tokens generated]", total_tokens);
    }

    Ok(())
}
```
