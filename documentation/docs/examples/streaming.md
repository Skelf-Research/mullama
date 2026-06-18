---
title: "Tutorial: Streaming Generation"
description: Real-time token-by-token text generation with console streaming, SSE, WebSocket patterns, backpressure handling, and performance measurement.
---

# Streaming Generation

Display generated text token-by-token as it is produced. This tutorial covers console streaming, Server-Sent Events (SSE), WebSocket streaming, cancellation, backpressure, and measuring performance.

---

## What You'll Build

Streaming patterns that:

- Display tokens in real-time on the console
- Stream responses via SSE for web clients
- Handle WebSocket bidirectional streaming
- Implement backpressure for slow consumers
- Support cancellation mid-generation
- Measure tokens-per-second performance

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- A GGUF model file
- Node.js 16+ or Python 3.8+

```bash
mullama pull llama3.2:1b
```

---

## Console Streaming

The most basic streaming pattern: print each token as it arrives.

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');

    const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 2048 });

    const prompt = 'The future of artificial intelligence is';
    console.log(`Prompt: ${prompt}`);
    process.stdout.write('Output: ');

    // generateStream returns an array of token pieces
    const pieces = ctx.generateStream(prompt, 100, { temperature: 0.8 });

    for (const piece of pieces) {
        process.stdout.write(piece);  // No newline between tokens
    }
    console.log();  // Final newline
    ```

=== "Python"
    ```python
    from mullama import Model, Context, SamplerParams

    model = Model.load("./model.gguf", n_gpu_layers=-1)
    ctx = Context(model, n_ctx=2048)

    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    print("Output: ", end="", flush=True)

    # generate_stream returns a list of token pieces
    pieces = ctx.generate_stream(prompt, max_tokens=100,
                                  params=SamplerParams(temperature=0.8))

    for piece in pieces:
        print(piece, end="", flush=True)  # No newline between tokens
    print()  # Final newline
    ```

---

## Measuring Tokens Per Second

Track generation speed for performance benchmarking.

=== "Node.js"
    ```javascript
    function streamWithStats(ctx, prompt, maxTokens = 200, params = {}) {
        const startTime = Date.now();
        const pieces = ctx.generateStream(prompt, maxTokens, params);

        let tokenCount = 0;
        let output = '';

        process.stdout.write('> ');
        for (const piece of pieces) {
            process.stdout.write(piece);
            output += piece;
            tokenCount++;
        }
        console.log();

        const elapsed = (Date.now() - startTime) / 1000;
        const tokensPerSec = tokenCount / elapsed;

        console.log(`--- Stats ---`);
        console.log(`Tokens: ${tokenCount}`);
        console.log(`Time: ${elapsed.toFixed(2)}s`);
        console.log(`Speed: ${tokensPerSec.toFixed(1)} tok/s`);
        console.log(`Characters: ${output.length}`);

        return { output, tokenCount, elapsed, tokensPerSec };
    }

    // Usage
    const stats = streamWithStats(ctx, 'Explain quantum computing:', 150);
    ```

=== "Python"
    ```python
    import time

    def stream_with_stats(ctx, prompt, max_tokens=200, params=None):
        start_time = time.time()
        pieces = ctx.generate_stream(prompt, max_tokens=max_tokens, params=params)

        token_count = 0
        output = ""

        print("> ", end="", flush=True)
        for piece in pieces:
            print(piece, end="", flush=True)
            output += piece
            token_count += 1
        print()

        elapsed = time.time() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

        print(f"--- Stats ---")
        print(f"Tokens: {token_count}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {tokens_per_sec:.1f} tok/s")
        print(f"Characters: {len(output)}")

        return {"output": output, "tokens": token_count,
                "elapsed": elapsed, "tok_per_sec": tokens_per_sec}

    # Usage
    stats = stream_with_stats(ctx, "Explain quantum computing:", max_tokens=150)
    ```

---

## Server-Sent Events (SSE) Streaming

Stream tokens to web clients using the SSE protocol.

=== "Node.js"
    ```javascript
    const express = require('express');
    const { JsModel, JsContext } = require('mullama');

    const app = express();
    app.use(express.json());

    const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 4096 });

    app.post('/stream', (req, res) => {
        const { prompt, max_tokens = 200 } = req.body;

        // Set SSE headers
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');  // Disable nginx buffering

        const pieces = ctx.generateStream(prompt, max_tokens, { temperature: 0.7 });
        let tokenIndex = 0;

        for (const piece of pieces) {
            const data = JSON.stringify({
                token: piece,
                index: tokenIndex++,
                done: false
            });
            res.write(`data: ${data}\n\n`);
        }

        // Send completion event
        res.write(`data: ${JSON.stringify({ token: '', index: tokenIndex, done: true })}\n\n`);
        res.end();
        ctx.clearCache();
    });

    app.listen(3000, () => console.log('SSE server on :3000'));
    ```

=== "Python"
    ```python
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from mullama import Model, Context, SamplerParams
    import json

    app = FastAPI()
    model = Model.load("./model.gguf", n_gpu_layers=-1)
    ctx = Context(model, n_ctx=4096)

    class StreamRequest(BaseModel):
        prompt: str
        max_tokens: int = 200

    @app.post("/stream")
    async def stream(req: StreamRequest):
        async def generate():
            pieces = ctx.generate_stream(
                req.prompt, max_tokens=req.max_tokens,
                params=SamplerParams(temperature=0.7)
            )
            for i, piece in enumerate(pieces):
                data = json.dumps({"token": piece, "index": i, "done": False})
                yield f"data: {data}\n\n"

            yield f'data: {json.dumps({"token": "", "index": i+1, "done": True})}\n\n'
            ctx.clear_cache()

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Run: uvicorn server:app --port 3000
    ```

### Client-Side SSE Consumption

```javascript
// Browser JavaScript
const eventSource = new EventSource('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: 'Hello!', max_tokens: 100 })
});

// Using fetch for POST-based SSE
async function streamChat(prompt) {
    const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 200 })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();  // Keep incomplete chunk

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.done) return;
                document.getElementById('output').textContent += data.token;
            }
        }
    }
}
```

---

## WebSocket Streaming

Bidirectional streaming for interactive chat applications.

=== "Node.js"
    ```javascript
    const WebSocket = require('ws');
    const { JsModel, JsContext } = require('mullama');

    const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 4096 });

    const wss = new WebSocket.Server({ port: 8765 });

    wss.on('connection', (ws) => {
        console.log('Client connected');

        ws.on('message', (data) => {
            const { prompt, max_tokens = 200 } = JSON.parse(data);

            // Stream tokens over WebSocket
            const pieces = ctx.generateStream(prompt, max_tokens, { temperature: 0.7 });
            for (const piece of pieces) {
                if (ws.readyState !== WebSocket.OPEN) break;  // Client disconnected
                ws.send(JSON.stringify({ type: 'token', content: piece }));
            }

            ws.send(JSON.stringify({ type: 'done' }));
            ctx.clearCache();
        });

        ws.on('close', () => console.log('Client disconnected'));
    });

    console.log('WebSocket server on ws://localhost:8765');
    ```

=== "Python"
    ```python
    import asyncio, json, websockets
    from mullama import Model, Context, SamplerParams

    model = Model.load("./model.gguf", n_gpu_layers=-1)
    ctx = Context(model, n_ctx=4096)

    async def handle_client(websocket):
        print("Client connected")
        async for message in websocket:
            data = json.loads(message)
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 200)

            pieces = ctx.generate_stream(
                prompt, max_tokens=max_tokens,
                params=SamplerParams(temperature=0.7)
            )

            for piece in pieces:
                await websocket.send(json.dumps({"type": "token", "content": piece}))

            await websocket.send(json.dumps({"type": "done"}))
            ctx.clear_cache()

    async def main():
        async with websockets.serve(handle_client, "localhost", 8765):
            print("WebSocket server on ws://localhost:8765")
            await asyncio.Future()  # Run forever

    asyncio.run(main())
    ```

---

## Backpressure Handling

Handle slow consumers that cannot keep up with generation speed.

=== "Node.js"
    ```javascript
    function streamWithBackpressure(ctx, prompt, ws, maxTokens = 200) {
        const pieces = ctx.generateStream(prompt, maxTokens, { temperature: 0.7 });
        let buffered = 0;
        const HIGH_WATER_MARK = 16;  // Max buffered messages

        for (const piece of pieces) {
            if (ws.readyState !== 1) break;  // WebSocket.OPEN

            // Check if send buffer is too full
            buffered = ws.bufferedAmount;
            if (buffered > HIGH_WATER_MARK * 1024) {
                // Back off: wait for buffer to drain
                // In production, use a proper async pattern
                console.warn(`Backpressure: ${buffered} bytes buffered`);
            }

            ws.send(JSON.stringify({ type: 'token', content: piece }));
        }
    }
    ```

=== "Python"
    ```python
    async def stream_with_backpressure(ctx, prompt, websocket, max_tokens=200):
        pieces = ctx.generate_stream(
            prompt, max_tokens=max_tokens,
            params=SamplerParams(temperature=0.7)
        )
        HIGH_WATER_MARK = 64 * 1024  # 64KB

        for piece in pieces:
            # Check write buffer size
            if hasattr(websocket, 'transport'):
                buffer_size = websocket.transport.get_write_buffer_size()
                if buffer_size > HIGH_WATER_MARK:
                    # Wait for buffer to drain
                    await asyncio.sleep(0.01)

            await websocket.send(json.dumps({"type": "token", "content": piece}))
    ```

---

## Cancellation

Stop generation mid-stream when the user cancels or navigates away.

=== "Node.js"
    ```javascript
    function streamWithCancellation(ctx, prompt, maxTokens = 500) {
        const pieces = ctx.generateStream(prompt, maxTokens, { temperature: 0.7 });
        let cancelled = false;
        let output = '';

        // Set up cancellation (e.g., from user pressing Ctrl+C)
        const handler = () => { cancelled = true; };
        process.on('SIGINT', handler);

        process.stdout.write('> ');
        for (const piece of pieces) {
            if (cancelled) {
                console.log('\n[Generation cancelled]');
                break;
            }
            process.stdout.write(piece);
            output += piece;
        }
        if (!cancelled) console.log();

        process.removeListener('SIGINT', handler);
        return output;
    }

    // For HTTP: check if client disconnected
    app.post('/stream', (req, res) => {
        let clientDisconnected = false;
        req.on('close', () => { clientDisconnected = true; });

        const pieces = ctx.generateStream(req.body.prompt, 200);
        res.setHeader('Content-Type', 'text/event-stream');

        for (const piece of pieces) {
            if (clientDisconnected) break;  // Stop generating
            res.write(`data: ${JSON.stringify({ token: piece })}\n\n`);
        }
        if (!clientDisconnected) res.end();
    });
    ```

=== "Python"
    ```python
    import signal

    def stream_with_cancellation(ctx, prompt, max_tokens=500):
        cancelled = False

        def handle_cancel(sig, frame):
            nonlocal cancelled
            cancelled = True

        old_handler = signal.signal(signal.SIGINT, handle_cancel)

        pieces = ctx.generate_stream(prompt, max_tokens=max_tokens,
                                      params=SamplerParams(temperature=0.7))
        output = ""
        print("> ", end="", flush=True)

        for piece in pieces:
            if cancelled:
                print("\n[Generation cancelled]")
                break
            print(piece, end="", flush=True)
            output += piece

        if not cancelled:
            print()

        signal.signal(signal.SIGINT, old_handler)
        return output

    # For FastAPI: check if client disconnected
    @app.post("/stream")
    async def stream(req: StreamRequest, raw: Request):
        async def generate():
            pieces = ctx.generate_stream(req.prompt, max_tokens=req.max_tokens)
            for piece in pieces:
                if await raw.is_disconnected():
                    break  # Client left
                yield f"data: {json.dumps({'token': piece})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    ```

---

## Complete Working Example

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');

    const MODEL_PATH = process.env.MODEL_PATH || './model.gguf';
    const model = JsModel.load(MODEL_PATH, { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 2048 });

    console.log('Mullama Streaming Demo');
    console.log('======================\n');

    // --- Basic streaming ---
    console.log('--- Basic Streaming ---');
    const prompt = 'The three laws of robotics are:';
    process.stdout.write(`${prompt}`);
    const pieces = ctx.generateStream(prompt, 100, { temperature: 0.7 });
    for (const p of pieces) process.stdout.write(p);
    console.log('\n');

    // --- With performance stats ---
    console.log('--- Performance Measurement ---');
    const start = Date.now();
    const pieces2 = ctx.generateStream('Explain gravity in one paragraph:', 150);
    let count = 0;
    process.stdout.write('> ');
    for (const p of pieces2) {
        process.stdout.write(p);
        count++;
    }
    const elapsed = (Date.now() - start) / 1000;
    console.log(`\n[${count} tokens in ${elapsed.toFixed(2)}s = ${(count/elapsed).toFixed(1)} tok/s]\n`);

    // --- Different temperatures ---
    console.log('--- Temperature Comparison ---');
    for (const temp of [0.1, 0.7, 1.2]) {
        process.stdout.write(`temp=${temp}: `);
        const p = ctx.generateStream('Once upon a time', 30, { temperature: temp });
        for (const t of p) process.stdout.write(t);
        console.log();
        ctx.clearCache();
    }

    // --- Early stopping ---
    console.log('\n--- Early Stopping (stop at period) ---');
    const pieces3 = ctx.generateStream('The meaning of life is', 200);
    let text = '';
    process.stdout.write('> ');
    for (const p of pieces3) {
        process.stdout.write(p);
        text += p;
        if (text.includes('.') && text.length > 50) {
            console.log(' [stopped]');
            break;
        }
    }
    console.log();
    ```

=== "Python"
    ```python
    import time
    from mullama import Model, Context, SamplerParams

    MODEL_PATH = "./model.gguf"
    model = Model.load(MODEL_PATH, n_gpu_layers=-1)
    ctx = Context(model, n_ctx=2048)

    print("Mullama Streaming Demo")
    print("======================\n")

    # --- Basic streaming ---
    print("--- Basic Streaming ---")
    prompt = "The three laws of robotics are:"
    print(prompt, end="", flush=True)
    pieces = ctx.generate_stream(prompt, max_tokens=100,
                                  params=SamplerParams(temperature=0.7))
    for p in pieces:
        print(p, end="", flush=True)
    print("\n")

    # --- With performance stats ---
    print("--- Performance Measurement ---")
    start = time.time()
    pieces = ctx.generate_stream("Explain gravity in one paragraph:", max_tokens=150)
    count = 0
    print("> ", end="", flush=True)
    for p in pieces:
        print(p, end="", flush=True)
        count += 1
    elapsed = time.time() - start
    print(f"\n[{count} tokens in {elapsed:.2f}s = {count/elapsed:.1f} tok/s]\n")

    # --- Different temperatures ---
    print("--- Temperature Comparison ---")
    for temp in [0.1, 0.7, 1.2]:
        print(f"temp={temp}: ", end="", flush=True)
        pieces = ctx.generate_stream("Once upon a time", max_tokens=30,
                                      params=SamplerParams(temperature=temp))
        for p in pieces:
            print(p, end="", flush=True)
        print()
        ctx.clear_cache()

    # --- Early stopping ---
    print("\n--- Early Stopping (stop at period) ---")
    pieces = ctx.generate_stream("The meaning of life is", max_tokens=200)
    text = ""
    print("> ", end="", flush=True)
    for p in pieces:
        print(p, end="", flush=True)
        text += p
        if "." in text and len(text) > 50:
            print(" [stopped]")
            break
    else:
        print()
    ```

=== "Rust"
    ```rust
    use mullama::prelude::*;
    use mullama::{AsyncModel, StreamConfig, TokenStream};
    use futures::StreamExt;
    use std::io::Write;
    use std::time::Instant;

    #[tokio::main]
    async fn main() -> Result<(), MullamaError> {
        let model = AsyncModel::load("path/to/model.gguf").await?;

        // Basic streaming
        let config = StreamConfig::default().max_tokens(100).temperature(0.8);
        let mut stream = TokenStream::new(model.clone(), "Hello world", config).await?;
        let start = Instant::now();
        let mut count = 0;

        while let Some(token) = stream.next().await {
            let token = token?;
            print!("{}", token.text);
            std::io::stdout().flush().unwrap();
            count += 1;
            if token.is_final { break; }
        }

        let elapsed = start.elapsed();
        println!("\n[{} tokens in {:.2}s = {:.1} tok/s]",
            count, elapsed.as_secs_f64(), count as f64 / elapsed.as_secs_f64());

        Ok(())
    }
    ```

---

## What's Next

- [Build a Chatbot](chatbot.md) -- Use streaming in a conversational app
- [API Server](api-server.md) -- Serve streaming responses over HTTP
- [Batch Processing](batch.md) -- Process multiple prompts efficiently
- [Guide: Streaming](../guide/streaming.md) -- In-depth streaming architecture
