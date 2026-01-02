# Streaming Generation

Stream tokens in real-time as they're generated for responsive applications.

## Basic Streaming

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;
use std::io::Write;

let model = Arc::new(Model::load("model.gguf")?);
let mut context = Context::new(model, ContextParams::default())?;

context.generate_streaming("Write a poem:", 200, |token| {
    print!("{}", token);
    std::io::stdout().flush().ok();
    true  // Continue generating
})?;
```

## Early Stopping

Return `false` from the callback to stop generation:

```rust
let mut total_tokens = 0;

context.generate_streaming("Generate text:", 1000, |token| {
    print!("{}", token);
    total_tokens += 1;

    // Stop after 100 tokens
    if total_tokens >= 100 {
        return false;
    }

    // Stop on specific content
    if token.contains("THE END") {
        return false;
    }

    true
})?;
```

## Collecting Output

Build the complete response while streaming:

```rust
let mut response = String::new();

context.generate_streaming("Hello:", 200, |token| {
    print!("{}", token);
    std::io::stdout().flush().ok();
    response.push_str(&token);
    true
})?;

println!("\n\nFull response: {}", response);
```

## Progress Tracking

Track generation progress:

```rust
use std::time::Instant;

let start = Instant::now();
let mut token_count = 0;

context.generate_streaming("Write an essay:", 500, |token| {
    token_count += 1;

    // Print progress every 50 tokens
    if token_count % 50 == 0 {
        let elapsed = start.elapsed().as_secs_f32();
        let tokens_per_sec = token_count as f32 / elapsed;
        eprintln!("\n[{} tokens, {:.1} t/s]", token_count, tokens_per_sec);
    }

    print!("{}", token);
    std::io::stdout().flush().ok();
    true
})?;

let total_time = start.elapsed();
println!("\n\nGenerated {} tokens in {:.2}s", token_count, total_time.as_secs_f32());
```

## With Channels

Send tokens to another thread:

```rust
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel();

// Consumer thread
let handle = thread::spawn(move || {
    let mut full_response = String::new();
    while let Ok(token) = rx.recv() {
        full_response.push_str(&token);
        // Process token (e.g., send to websocket)
    }
    full_response
});

// Producer
context.generate_streaming("Hello:", 200, |token| {
    tx.send(token.to_string()).ok();
    true
})?;

drop(tx);  // Signal end
let response = handle.join().unwrap();
```

## Async Streaming

With the `async` feature:

```rust
use mullama::{Model, Context, ContextParams};
use tokio::sync::mpsc;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model, ContextParams::default())?;

    let (tx, mut rx) = mpsc::channel(32);

    // Spawn generation task
    let gen_handle = tokio::task::spawn_blocking(move || {
        context.generate_streaming("Hello:", 200, |token| {
            tx.blocking_send(token.to_string()).is_ok()
        })
    });

    // Process tokens asynchronously
    while let Some(token) = rx.recv().await {
        print!("{}", token);
        tokio::io::stdout().flush().await.ok();
    }

    gen_handle.await??;
    Ok(())
}
```

## Server-Sent Events

Stream to web clients:

```rust
use axum::{
    response::sse::{Event, Sse},
    routing::post,
    Router,
};
use futures::stream;
use std::sync::mpsc;

async fn generate_sse(prompt: String) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel();

    // Start generation in background
    tokio::task::spawn_blocking(move || {
        let model = Model::load("model.gguf").unwrap();
        let mut context = Context::new(Arc::new(model), ContextParams::default()).unwrap();

        context.generate_streaming(&prompt, 200, |token| {
            tx.send(token.to_string()).is_ok()
        }).ok();
    });

    // Convert to SSE stream
    let stream = stream::unfold(rx, |rx| async move {
        rx.recv().ok().map(|token| {
            let event = Event::default().data(token);
            (Ok(event), rx)
        })
    });

    Sse::new(stream)
}
```

## WebSocket Streaming

Real-time bidirectional streaming:

```rust
use tokio_tungstenite::tungstenite::Message;

async fn handle_websocket(ws: WebSocket) {
    let (mut sender, mut receiver) = ws.split();

    while let Some(msg) = receiver.next().await {
        if let Ok(Message::Text(prompt)) = msg {
            // Stream response back
            let (tx, mut rx) = mpsc::channel(32);

            tokio::task::spawn_blocking(move || {
                context.generate_streaming(&prompt, 200, |token| {
                    tx.blocking_send(token.to_string()).is_ok()
                })
            });

            while let Some(token) = rx.recv().await {
                sender.send(Message::Text(token)).await.ok();
            }
        }
    }
}
```

## Buffered Streaming

Buffer tokens for smoother output:

```rust
use std::collections::VecDeque;

let mut buffer: VecDeque<String> = VecDeque::new();
let buffer_size = 5;

context.generate_streaming("Hello:", 200, |token| {
    buffer.push_back(token.to_string());

    // Flush when buffer is full
    if buffer.len() >= buffer_size {
        while let Some(t) = buffer.pop_front() {
            print!("{}", t);
        }
        std::io::stdout().flush().ok();
    }

    true
})?;

// Flush remaining
for t in buffer {
    print!("{}", t);
}
```

## Error Handling

```rust
match context.generate_streaming("Hello:", 200, |token| {
    print!("{}", token);
    true
}) {
    Ok(()) => println!("\nGeneration complete"),
    Err(mullama::MullamaError::GenerationError(msg)) => {
        eprintln!("\nGeneration failed: {}", msg);
    }
    Err(e) => eprintln!("\nError: {}", e),
}
```

## Best Practices

1. **Flush output** - Call `flush()` after each token for immediate display
2. **Use channels** - For cross-thread streaming
3. **Handle early stopping** - Return `false` to stop gracefully
4. **Track progress** - Log tokens/second for debugging
5. **Buffer when needed** - Smooth output for slow consumers
