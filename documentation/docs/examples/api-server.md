---
title: "Tutorial: API Server"
description: Build a production API server with OpenAI-compatible endpoints, streaming SSE responses, rate limiting, and Docker deployment using Mullama.
---

# API Server

Build a production-ready API server with OpenAI-compatible endpoints for chat completions and embeddings. Includes streaming Server-Sent Events (SSE), rate limiting, CORS, and Docker deployment.

---

## What You'll Build

A complete API server that:

- Serves `/v1/chat/completions` (OpenAI-compatible)
- Serves `/v1/embeddings` for vector generation
- Provides a `/health` endpoint for monitoring
- Streams responses via Server-Sent Events (SSE)
- Handles concurrent requests
- Includes rate limiting and CORS
- Deploys via Docker

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- A GGUF model file
- Node.js 16+ (with Express) or Python 3.8+ (with FastAPI and uvicorn)

```bash
# Node.js dependencies
npm install mullama express cors

# Python dependencies
pip install mullama fastapi uvicorn
```

---

## Architecture Overview

```
Client Request --> Rate Limiter --> Route Handler --> Mullama Inference --> Response
                                         |
                                         |--> /v1/chat/completions (streaming SSE)
                                         |--> /v1/embeddings (batch)
                                         |--> /health (status)
```

---

## Step 1: Basic Server Setup

=== "Node.js"
    ```javascript
    const express = require('express');
    const cors = require('cors');
    const { JsModel, JsContext, JsEmbeddingGenerator } = require('mullama');

    const app = express();
    app.use(cors());
    app.use(express.json({ limit: '10mb' }));

    // Load model at startup
    const MODEL_PATH = process.env.MODEL_PATH || './llama3.2-1b-instruct.Q4_K_M.gguf';
    console.log(`Loading model: ${MODEL_PATH}`);
    const model = JsModel.load(MODEL_PATH, { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 4096, nBatch: 512 });
    const embedGen = new JsEmbeddingGenerator(model, 512, true);

    console.log(`Model loaded: ${model.name || 'unknown'}`);
    console.log(`Parameters: ${model.nParams?.toLocaleString()}`);
    ```

=== "Python"
    ```python
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from mullama import Model, Context, SamplerParams, EmbeddingGenerator
    import os, time, json

    app = FastAPI(title="Mullama API Server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # Load model at startup
    MODEL_PATH = os.environ.get("MODEL_PATH", "./llama3.2-1b-instruct.Q4_K_M.gguf")
    print(f"Loading model: {MODEL_PATH}")
    model = Model.load(MODEL_PATH, n_gpu_layers=-1)
    ctx = Context(model, n_ctx=4096, n_batch=512)
    embed_gen = EmbeddingGenerator(model, n_ctx=512, normalize=True)

    print(f"Model loaded: {model.name or 'unknown'}")
    print(f"Parameters: {model.n_params:,}")
    ```

---

## Step 2: Health Endpoint

=== "Node.js"
    ```javascript
    app.get('/health', (req, res) => {
        res.json({
            status: 'ok',
            model: model.name || 'unknown',
            architecture: model.architecture,
            parameters: model.nParams,
            context_size: ctx.nCtx,
            uptime: process.uptime(),
        });
    });
    ```

=== "Python"
    ```python
    start_time = time.time()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": model.name or "unknown",
            "architecture": model.architecture,
            "parameters": model.n_params,
            "context_size": ctx.n_ctx,
            "uptime": time.time() - start_time,
        }
    ```

---

## Step 3: Chat Completions Endpoint

Implement the OpenAI-compatible `/v1/chat/completions` endpoint with streaming support.

=== "Node.js"
    ```javascript
    app.post('/v1/chat/completions', (req, res) => {
        const { messages, max_tokens = 512, temperature = 0.7,
                top_p = 0.9, stream = false } = req.body;

        if (!messages || !Array.isArray(messages)) {
            return res.status(400).json({ error: { message: 'messages is required' } });
        }

        // Format messages using chat template
        const formattedMessages = messages.map(m => [m.role, m.content]);
        const prompt = model.applyChatTemplate(formattedMessages);

        const params = { temperature, topP: top_p, penaltyRepeat: 1.1 };
        const requestId = `chatcmpl-${Date.now()}`;

        if (stream) {
            // Streaming SSE response
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            const pieces = ctx.generateStream(prompt, max_tokens, params);
            for (const piece of pieces) {
                const chunk = {
                    id: requestId,
                    object: 'chat.completion.chunk',
                    created: Math.floor(Date.now() / 1000),
                    model: model.name || 'mullama',
                    choices: [{
                        index: 0,
                        delta: { content: piece },
                        finish_reason: null
                    }]
                };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }

            // Send final chunk
            const finalChunk = {
                id: requestId,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: model.name || 'mullama',
                choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
            };
            res.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
        } else {
            // Non-streaming response
            const text = ctx.generate(prompt, max_tokens, params);
            const promptTokens = model.tokenize(prompt, false).length;
            const completionTokens = model.tokenize(text, false).length;

            res.json({
                id: requestId,
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model: model.name || 'mullama',
                choices: [{
                    index: 0,
                    message: { role: 'assistant', content: text.trim() },
                    finish_reason: 'stop'
                }],
                usage: {
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens,
                    total_tokens: promptTokens + completionTokens
                }
            });
        }
        ctx.clearCache();
    });
    ```

=== "Python"
    ```python
    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        messages: list[ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Format messages using chat template
        formatted = [(m.role, m.content) for m in request.messages]
        prompt = model.apply_chat_template(formatted)

        params = SamplerParams(
            temperature=request.temperature,
            top_p=request.top_p,
            penalty_repeat=1.1
        )
        request_id = f"chatcmpl-{int(time.time() * 1000)}"

        if request.stream:
            return StreamingResponse(
                stream_chat(prompt, request.max_tokens, params, request_id),
                media_type="text/event-stream"
            )

        # Non-streaming response
        text = ctx.generate(prompt, max_tokens=request.max_tokens, params=params)
        prompt_tokens = len(model.tokenize(prompt, add_bos=False))
        completion_tokens = len(model.tokenize(text, add_bos=False))
        ctx.clear_cache()

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model.name or "mullama",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text.strip()},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    async def stream_chat(prompt, max_tokens, params, request_id):
        pieces = ctx.generate_stream(prompt, max_tokens=max_tokens, params=params)
        for piece in pieces:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model.name or "mullama",
                "choices": [{
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model.name or "mullama",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
        ctx.clear_cache()
    ```

---

## Step 4: Embeddings Endpoint

=== "Node.js"
    ```javascript
    app.post('/v1/embeddings', (req, res) => {
        const { input, model: modelName } = req.body;

        if (!input) {
            return res.status(400).json({ error: { message: 'input is required' } });
        }

        const texts = Array.isArray(input) ? input : [input];
        const embeddings = embedGen.embedBatch(texts);
        const totalTokens = texts.reduce((sum, t) =>
            sum + model.tokenize(t, false).length, 0);

        res.json({
            object: 'list',
            data: embeddings.map((emb, i) => ({
                object: 'embedding',
                index: i,
                embedding: emb
            })),
            model: modelName || model.name || 'mullama',
            usage: { prompt_tokens: totalTokens, total_tokens: totalTokens }
        });
    });
    ```

=== "Python"
    ```python
    class EmbeddingRequest(BaseModel):
        input: str | list[str]
        model: str = "mullama"

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingRequest):
        texts = request.input if isinstance(request.input, list) else [request.input]
        embs = embed_gen.embed_batch(texts)
        total_tokens = sum(len(model.tokenize(t, add_bos=False)) for t in texts)

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": emb.tolist()}
                for i, emb in enumerate(embs)
            ],
            "model": request.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
        }
    ```

---

## Step 5: Rate Limiting

=== "Node.js"
    ```javascript
    // Simple in-memory rate limiter
    const rateLimits = new Map();

    function rateLimit(windowMs = 60000, maxRequests = 30) {
        return (req, res, next) => {
            const key = req.ip || req.connection.remoteAddress;
            const now = Date.now();
            const windowStart = now - windowMs;

            if (!rateLimits.has(key)) rateLimits.set(key, []);
            const requests = rateLimits.get(key).filter(t => t > windowStart);
            rateLimits.set(key, requests);

            if (requests.length >= maxRequests) {
                return res.status(429).json({
                    error: { message: 'Rate limit exceeded. Try again later.' }
                });
            }

            requests.push(now);
            next();
        };
    }

    // Apply rate limiting to inference endpoints
    app.use('/v1/chat/completions', rateLimit(60000, 20));
    app.use('/v1/embeddings', rateLimit(60000, 60));
    ```

=== "Python"
    ```python
    from collections import defaultdict

    # Simple in-memory rate limiter
    rate_limits: dict[str, list[float]] = defaultdict(list)

    async def check_rate_limit(request: Request, max_requests: int = 30,
                                window_seconds: int = 60):
        client_ip = request.client.host
        now = time.time()
        window_start = now - window_seconds

        # Clean old entries
        rate_limits[client_ip] = [t for t in rate_limits[client_ip] if t > window_start]

        if len(rate_limits[client_ip]) >= max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        rate_limits[client_ip].append(now)

    # Apply in endpoints:
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw: Request):
        await check_rate_limit(raw, max_requests=20)
        # ... rest of handler
    ```

---

## Step 6: Error Handling

=== "Node.js"
    ```javascript
    // Global error handler
    app.use((err, req, res, next) => {
        console.error('Server error:', err.message);
        res.status(500).json({
            error: {
                message: 'Internal server error',
                type: 'server_error',
                code: 500
            }
        });
    });

    // Request validation middleware
    function validateRequest(req, res, next) {
        if (req.headers['content-type'] &&
            !req.headers['content-type'].includes('application/json')) {
            return res.status(415).json({
                error: { message: 'Content-Type must be application/json' }
            });
        }
        next();
    }
    app.use('/v1', validateRequest);
    ```

=== "Python"
    ```python
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "code": 500
                }
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "code": exc.status_code}}
        )
    ```

---

## Complete Working Example

=== "Node.js"
    ```javascript
    const express = require('express');
    const cors = require('cors');
    const { JsModel, JsContext, JsEmbeddingGenerator } = require('mullama');

    const app = express();
    app.use(cors());
    app.use(express.json({ limit: '10mb' }));

    // --- Model Loading ---
    const MODEL_PATH = process.env.MODEL_PATH || './model.gguf';
    const model = JsModel.load(MODEL_PATH, { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 4096, nBatch: 512 });
    const embedGen = new JsEmbeddingGenerator(model, 512, true);

    // --- Rate Limiting ---
    const limits = new Map();
    function rateLimit(max = 30) {
        return (req, res, next) => {
            const key = req.ip;
            const now = Date.now();
            const reqs = (limits.get(key) || []).filter(t => t > now - 60000);
            if (reqs.length >= max) return res.status(429).json({ error: { message: 'Rate limited' } });
            reqs.push(now);
            limits.set(key, reqs);
            next();
        };
    }

    // --- Endpoints ---
    app.get('/health', (req, res) => {
        res.json({ status: 'ok', model: model.name, uptime: process.uptime() });
    });

    app.post('/v1/chat/completions', rateLimit(20), (req, res) => {
        const { messages, max_tokens = 512, temperature = 0.7, stream = false } = req.body;
        if (!messages) return res.status(400).json({ error: { message: 'messages required' } });

        const prompt = model.applyChatTemplate(messages.map(m => [m.role, m.content]));
        const params = { temperature, penaltyRepeat: 1.1 };
        const id = `chatcmpl-${Date.now()}`;

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            const pieces = ctx.generateStream(prompt, max_tokens, params);
            for (const piece of pieces) {
                const chunk = { id, object: 'chat.completion.chunk',
                    choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] };
                res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
            res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk',
                choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
        } else {
            const text = ctx.generate(prompt, max_tokens, params);
            res.json({ id, object: 'chat.completion',
                choices: [{ index: 0, message: { role: 'assistant', content: text.trim() },
                    finish_reason: 'stop' }] });
        }
        ctx.clearCache();
    });

    app.post('/v1/embeddings', rateLimit(60), (req, res) => {
        const texts = Array.isArray(req.body.input) ? req.body.input : [req.body.input];
        const embs = embedGen.embedBatch(texts);
        res.json({ object: 'list',
            data: embs.map((e, i) => ({ object: 'embedding', index: i, embedding: e })) });
    });

    const PORT = process.env.PORT || 8080;
    app.listen(PORT, () => console.log(`Mullama API server on port ${PORT}`));
    ```

=== "Python"
    ```python
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    from mullama import Model, Context, SamplerParams, EmbeddingGenerator
    from collections import defaultdict
    import os, time, json

    app = FastAPI(title="Mullama API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # --- Model Loading ---
    MODEL_PATH = os.environ.get("MODEL_PATH", "./model.gguf")
    model = Model.load(MODEL_PATH, n_gpu_layers=-1)
    ctx = Context(model, n_ctx=4096, n_batch=512)
    embed_gen = EmbeddingGenerator(model, n_ctx=512, normalize=True)
    start_time = time.time()

    # --- Rate Limiting ---
    limits: dict[str, list[float]] = defaultdict(list)
    async def check_limit(req: Request, max_req: int = 30):
        ip = req.client.host
        now = time.time()
        limits[ip] = [t for t in limits[ip] if t > now - 60]
        if len(limits[ip]) >= max_req:
            raise HTTPException(429, "Rate limited")
        limits[ip].append(now)

    # --- Schemas ---
    class Msg(BaseModel):
        role: str
        content: str

    class ChatReq(BaseModel):
        messages: list[Msg]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False

    class EmbedReq(BaseModel):
        input: str | list[str]

    # --- Endpoints ---
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model.name, "uptime": time.time() - start_time}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatReq, raw: Request):
        await check_limit(raw, 20)
        prompt = model.apply_chat_template([(m.role, m.content) for m in req.messages])
        params = SamplerParams(temperature=req.temperature, penalty_repeat=1.1)
        rid = f"chatcmpl-{int(time.time()*1000)}"

        if req.stream:
            async def generate():
                for piece in ctx.generate_stream(prompt, max_tokens=req.max_tokens, params=params):
                    chunk = {"id": rid, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f'data: {json.dumps({"id": rid, "choices": [{{"delta": {{}}, "finish_reason": "stop"}}]})}\n\n'
                yield "data: [DONE]\n\n"
                ctx.clear_cache()
            return StreamingResponse(generate(), media_type="text/event-stream")

        text = ctx.generate(prompt, max_tokens=req.max_tokens, params=params)
        ctx.clear_cache()
        return {"id": rid, "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text.strip()},
                             "finish_reason": "stop"}]}

    @app.post("/v1/embeddings")
    async def embed(req: EmbedReq, raw: Request):
        await check_limit(raw, 60)
        texts = req.input if isinstance(req.input, list) else [req.input]
        embs = embed_gen.embed_batch(texts)
        return {"object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": e.tolist()}
                         for i, e in enumerate(embs)]}

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    ```

---

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install mullama fastapi uvicorn

COPY server.py .
COPY models/ ./models/

ENV MODEL_PATH=/app/models/llama3.2-1b-instruct.Q4_K_M.gguf
ENV PORT=8080

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Build and run
docker build -t mullama-api .
docker run -p 8080:8080 -v ./models:/app/models mullama-api
```

---

## Testing the API

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'

# Streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Tell me a joke"}], "stream": true}'

# Embeddings
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Goodbye world"]}'
```

---

## Alternative: Built-in Daemon

For simpler deployments, Mullama includes a built-in daemon with the same API compatibility:

```bash
# Start the daemon (includes all endpoints automatically)
mullama serve --model llama3.2:1b --port 8080

# It is immediately compatible with OpenAI client libraries
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

See [Daemon REST API](../daemon/rest-api.md) for full documentation.

---

## What's Next

- [Streaming Generation](streaming.md) -- Deep dive into streaming patterns
- [Batch Processing](batch.md) -- Optimize throughput for multiple requests
- [Daemon: REST API](../daemon/rest-api.md) -- Built-in OpenAI-compatible server
- [Daemon: Deployment](../daemon/deployment.md) -- Production deployment guide
