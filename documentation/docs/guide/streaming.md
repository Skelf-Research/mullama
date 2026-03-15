# Streaming

Stream tokens in real-time as they are generated for responsive user experiences. Streaming reduces perceived latency by showing output incrementally rather than waiting for the complete response.

!!! abstract "Feature Gate"
    Basic callback-based streaming is available without any feature flags. For async `TokenStream` with backpressure, enable the `streaming` feature in Rust:

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["streaming"] }
    ```

## Why Streaming Matters

Without streaming, users must wait for the entire response before seeing any output. With streaming:

- **Time-to-first-token** is typically under 100ms, giving immediate feedback
- Users can read output as it arrives, improving perceived responsiveness
- Applications can implement early stopping to save compute
- Server-sent events (SSE) enable real-time web interfaces

## Basic Streaming

The simplest way to stream tokens as they are generated:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    // Async iterator pattern
    for await (const token of context.generateStream("Write a poem:", 200)) {
      process.stdout.write(token.text);
    }
    console.log();
    ```

=== "Python"

    ```python
    from mullama import Model, Context

    model = Model.load("./model.gguf")
    context = Context(model)

    # Generator pattern
    for token in context.generate_stream("Write a poem:", max_tokens=200):
        print(token.text, end="", flush=True)
    print()
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;
    use std::io::Write;

    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model, ContextParams::default())?;

    context.generate_streaming("Write a poem:", 200, |token| {
        print!("{}", token);
        std::io::stdout().flush().ok();
        true  // Return true to continue, false to stop
    })?;
    println!();
    ```

=== "CLI"

    ```bash
    # CLI streams by default
    mullama run llama3.2:1b "Write a poem:"

    # Disable streaming (wait for full response)
    mullama run llama3.2:1b "Write a poem:" --no-stream
    ```

## Stream Configuration

Configure streaming behavior with buffer sizes, timeouts, and sampling parameters:

=== "Node.js"

    ```javascript
    import { Model, Context, StreamConfig } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    const config = new StreamConfig({
      maxTokens: 500,
      temperature: 0.7,
      topK: 40,
      topP: 0.9,
      stopSequences: ["END"],
      bufferSize: 64,
      timeoutMs: 30000,
    });

    for await (const token of context.generateStream("Hello:", config)) {
      process.stdout.write(token.text);
    }
    ```

=== "Python"

    ```python
    from mullama import Model, Context, StreamConfig

    model = Model.load("./model.gguf")
    context = Context(model)

    config = StreamConfig(
        max_tokens=500,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        stop_sequences=["END"],
        buffer_size=64,
        timeout_ms=30000,
    )

    for token in context.generate_stream("Hello:", config=config):
        print(token.text, end="", flush=True)
    ```

=== "Rust"

    ```rust
    use mullama::StreamConfig;

    let config = StreamConfig::new()
        .max_tokens(500)
        .temperature(0.7)
        .top_k(40)
        .top_p(0.9)
        .stop_sequences(vec!["END".to_string()])
        .stream_delay_ms(0);

    let mut stream = model.generate_stream("Hello:", config).await?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello:" \
      --temperature 0.7 \
      --top-k 40 \
      --top-p 0.9 \
      --stop "END"
    ```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxTokens` | `usize` | 256 | Maximum number of tokens to generate |
| `temperature` | `f32` | 1.0 | Sampling temperature |
| `topK` | `u32` | 40 | Top-K candidates |
| `topP` | `f32` | 0.9 | Nucleus sampling threshold |
| `stopSequences` | `Vec<String>` | `[]` | Stop generation at these strings |
| `bufferSize` | `usize` | 64 | Internal token buffer size |
| `timeoutMs` | `u64` | 30000 | Maximum time for generation (milliseconds) |

## Async TokenStream

With the `streaming` feature enabled (Rust) or in Node.js/Python, use async streaming with proper backpressure handling:

=== "Node.js"

    ```javascript
    import { AsyncModel, StreamConfig } from 'mullama';

    const model = await AsyncModel.load('./model.gguf');

    const config = new StreamConfig({ maxTokens: 200, temperature: 0.7 });
    const stream = model.generateStream("Tell me about Rust:", config);

    let tokenCount = 0;
    for await (const token of stream) {
      process.stdout.write(token.text);
      tokenCount++;
    }

    console.log(`\nGenerated ${tokenCount} tokens`);
    console.log(`Tokens/sec: ${stream.tokensPerSecond().toFixed(1)}`);
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, StreamConfig

    async def main():
        model = await AsyncModel.load("./model.gguf")

        config = StreamConfig(max_tokens=200, temperature=0.7)

        token_count = 0
        async for token in model.generate_stream("Tell me about Rust:", config=config):
            print(token.text, end="", flush=True)
            token_count += 1

        print(f"\nGenerated {token_count} tokens")

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, StreamConfig};
    use tokio_stream::StreamExt;

    #[tokio::main]
    async fn main() -> Result<(), mullama::MullamaError> {
        let model = AsyncModel::load("model.gguf").await?;

        let config = StreamConfig::new()
            .max_tokens(200)
            .temperature(0.7);

        let mut stream = model.generate_stream("Tell me about Rust:", config).await?;

        while let Some(token_result) = stream.next().await {
            let token_data = token_result?;
            print!("{}", token_data.text);
        }

        println!("\nTokens generated: {}", stream.tokens_generated());
        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # CLI streams by default with performance stats
    mullama run llama3.2:1b "Tell me about Rust:" --verbose
    ```

## Backpressure Handling

When the consumer is slower than the producer, the stream automatically pauses generation to prevent unbounded memory growth:

=== "Node.js"

    ```javascript
    // Slow consumer -- generator pauses automatically
    for await (const token of context.generateStream("Generate text:", config)) {
      process.stdout.write(token.text);

      // Simulate slow processing
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    ```

=== "Python"

    ```python
    import asyncio

    # Slow consumer -- generator pauses automatically
    async for token in model.generate_stream("Generate text:", config=config):
        print(token.text, end="", flush=True)

        # Simulate slow processing
        await asyncio.sleep(0.05)
    ```

=== "Rust"

    ```rust
    use tokio::time::{sleep, Duration};

    let mut stream = model.generate_stream("Generate text:", config).await?;

    while let Some(token_result) = stream.next().await {
        let token_data = token_result?;
        print!("{}", token_data.text);

        // Slow consumer -- generator will pause automatically
        sleep(Duration::from_millis(50)).await;
    }
    ```

=== "CLI"

    ```bash
    # Pipe to a slow consumer; backpressure handled automatically
    mullama run llama3.2:1b "Generate text:" | while read -r line; do
      echo "$line"
      sleep 0.1
    done
    ```

!!! info "Buffer Behavior"
    The internal buffer prevents unbounded memory growth. When the buffer is full, the generator blocks until the consumer catches up. The default buffer size of 64 tokens is sufficient for most use cases.

## Error Recovery in Streams

Handle errors gracefully during streaming:

=== "Node.js"

    ```javascript
    try {
      for await (const token of context.generateStream("Hello:", config)) {
        process.stdout.write(token.text);
      }
    } catch (error) {
      if (error.code === 'STREAM_TIMEOUT') {
        console.error('\nGeneration timed out');
      } else if (error.code === 'GENERATION_ERROR') {
        console.error(`\nGeneration failed: ${error.message}`);
      } else {
        throw error;
      }
    }
    ```

=== "Python"

    ```python
    from mullama import MullamaError, StreamTimeoutError

    try:
        for token in context.generate_stream("Hello:", config=config):
            print(token.text, end="", flush=True)
    except StreamTimeoutError:
        print("\nGeneration timed out")
    except MullamaError as e:
        print(f"\nGeneration failed: {e}")
    ```

=== "Rust"

    ```rust
    let mut stream = model.generate_stream("Hello:", config).await?;

    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token_data) => print!("{}", token_data.text),
            Err(mullama::MullamaError::StreamingError(msg)) => {
                eprintln!("\nStreaming error: {}", msg);
                break;
            }
            Err(e) => {
                eprintln!("\nUnexpected error: {}", e);
                break;
            }
        }
    }
    ```

=== "CLI"

    ```bash
    # CLI handles errors automatically with descriptive messages
    mullama run llama3.2:1b "Hello:" --timeout 30000
    ```

## Early Stopping

Stop generation before reaching max tokens:

=== "Node.js"

    ```javascript
    let output = '';
    let tokenCount = 0;

    for await (const token of context.generateStream("Generate a list:", 1000)) {
      output += token.text;
      tokenCount++;
      process.stdout.write(token.text);

      // Stop after 5 list items
      const itemCount = (output.match(/\n-/g) || []).length;
      if (itemCount >= 5) {
        break;  // Breaking the loop stops generation
      }
    }
    ```

=== "Python"

    ```python
    output = ""
    token_count = 0

    for token in context.generate_stream("Generate a list:", max_tokens=1000):
        output += token.text
        token_count += 1
        print(token.text, end="", flush=True)

        # Stop after 5 list items
        if output.count("\n-") >= 5:
            break  # Breaking the loop stops generation
    ```

=== "Rust"

    ```rust
    let mut output = String::new();
    let mut total_tokens = 0;

    context.generate_streaming("Generate a list:", 1000, |token| {
        output.push_str(&token);
        total_tokens += 1;

        // Stop after 5 list items
        let item_count = output.matches("\n-").count();
        if item_count >= 5 {
            return false;  // Stop generation
        }

        print!("{}", token);
        true
    })?;
    ```

=== "CLI"

    ```bash
    # Ctrl+C to stop generation
    mullama run llama3.2:1b "Generate a list:" --max-tokens 1000
    ```

## SSE (Server-Sent Events) Integration

Stream tokens to web clients using Server-Sent Events:

=== "Node.js"

    ```javascript
    import express from 'express';
    import { Model, Context, StreamConfig } from 'mullama';

    const app = express();
    const model = await Model.load('./model.gguf');

    app.post('/generate', async (req, res) => {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const context = new Context(model);
      const config = new StreamConfig({ maxTokens: 500 });

      for await (const token of context.generateStream(req.body.prompt, config)) {
        res.write(`data: ${JSON.stringify({ text: token.text })}\n\n`);
      }

      res.write('data: [DONE]\n\n');
      res.end();
    });

    app.listen(3000);
    ```

=== "Python"

    ```python
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from mullama import Model, Context, StreamConfig

    app = FastAPI()
    model = Model.load("./model.gguf")

    @app.post("/generate")
    async def generate(prompt: str):
        async def event_stream():
            context = Context(model)
            config = StreamConfig(max_tokens=500)

            for token in context.generate_stream(prompt, config=config):
                yield f"data: {{'text': '{token.text}'}}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    ```

=== "Rust"

    ```rust
    use axum::{
        response::sse::{Event, Sse},
        routing::post,
        Router, Json,
    };
    use futures::stream;
    use std::sync::mpsc;

    async fn generate_sse(
        Json(body): Json<GenerateRequest>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
        let (tx, rx) = mpsc::channel();

        tokio::task::spawn_blocking(move || {
            let model = Model::load("model.gguf").unwrap();
            let mut context = Context::new(Arc::new(model), ContextParams::default()).unwrap();

            context.generate_streaming(&body.prompt, 500, |token| {
                tx.send(token.to_string()).is_ok()
            }).ok();
        });

        let stream = stream::unfold(rx, |rx| async move {
            rx.recv().ok().map(|token| {
                let event = Event::default().data(token);
                (Ok(event), rx)
            })
        });

        Sse::new(stream)
    }
    ```

=== "CLI"

    ```bash
    # Start the daemon with OpenAI-compatible streaming API
    mullama serve --model llama3.2:1b

    # Test with curl (SSE format)
    curl -X POST http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
      }'
    ```

## Combining Streaming with Web Frameworks

### WebSocket Streaming

For bidirectional real-time communication:

=== "Node.js"

    ```javascript
    import { WebSocketServer } from 'ws';
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const wss = new WebSocketServer({ port: 8080 });

    wss.on('connection', (ws) => {
      const context = new Context(model);

      ws.on('message', async (data) => {
        const prompt = data.toString();

        for await (const token of context.generateStream(prompt, 500)) {
          ws.send(JSON.stringify({ type: 'token', text: token.text }));
        }

        ws.send(JSON.stringify({ type: 'done' }));
      });
    });
    ```

=== "Python"

    ```python
    import asyncio
    import websockets
    import json
    from mullama import Model, Context

    model = Model.load("./model.gguf")

    async def handler(websocket):
        context = Context(model)

        async for message in websocket:
            for token in context.generate_stream(message, max_tokens=500):
                await websocket.send(json.dumps({
                    "type": "token", "text": token.text
                }))
            await websocket.send(json.dumps({"type": "done"}))

    asyncio.run(websockets.serve(handler, "localhost", 8080))
    ```

=== "Rust"

    ```rust
    use tokio::sync::mpsc;

    async fn handle_websocket(ws: WebSocket) {
        let (mut sender, mut receiver) = ws.split();

        while let Some(msg) = receiver.next().await {
            if let Ok(Message::Text(prompt)) = msg {
                let (tx, mut rx) = mpsc::channel(32);

                tokio::task::spawn_blocking(move || {
                    context.generate_streaming(&prompt, 500, |token| {
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

=== "CLI"

    ```bash
    # Start daemon with WebSocket support
    mullama serve --model llama3.2:1b --websocket

    # Test with websocat
    echo "Hello!" | websocat ws://localhost:8080/ws
    ```

## Best Practices

1. **Always flush output** -- Call `flush()` after each token for immediate display
2. **Handle early stopping** -- Break from loops or return `false` from callbacks to stop gracefully
3. **Set buffer sizes appropriately** -- 32-128 tokens is typically sufficient for the internal buffer
4. **Track throughput** -- Log tokens/second to identify performance bottlenecks
5. **Use SSE for web apps** -- Standard, well-supported by all browsers and HTTP clients
6. **Implement timeouts** -- Prevent runaway generation with the `timeoutMs` configuration

!!! tip "Performance Monitoring"
    Stream objects expose `tokensPerSecond()` / `tokens_per_second()` after generation completes, making it easy to track inference performance.

## See Also

- [Async Support](async.md) -- Non-blocking generation with async/await
- [Text Generation](generation.md) -- Core generation parameters and configuration
- [Advanced: WebSockets](../advanced/websockets.md) -- Production WebSocket integration
- [API Reference: Streaming](../api/streaming.md) -- Complete Streaming API documentation
