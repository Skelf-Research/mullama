---
title: REST API
description: Management, generation, health, and monitoring endpoints
---

# REST API

The Mullama daemon exposes a REST API for model management, text generation, system monitoring, and health checks. These endpoints complement the [OpenAI](openai-api.md) and [Anthropic](anthropic-api.md) compatibility APIs.

**Base URL:** `http://localhost:8080`
**Content-Type:** `application/json`

---

## Management Endpoints

### List Models

Retrieve all models known to the daemon (loaded and available).

```
GET /api/models
```

**Example:**

```bash
curl http://localhost:8080/api/models
```

**Response:**

```json
{
  "models": [
    {
      "alias": "llama3.2:1b",
      "path": "/home/user/.cache/mullama/models/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
      "parameters": 1236000000,
      "context_size": 4096,
      "gpu_layers": 35,
      "is_default": true,
      "active_requests": 0
    },
    {
      "alias": "qwen2.5:7b",
      "path": "/home/user/.cache/mullama/models/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf",
      "parameters": 7615000000,
      "context_size": 4096,
      "gpu_layers": 0,
      "is_default": false,
      "active_requests": 1
    }
  ]
}
```

---

### Get Model Details

Retrieve detailed information about a specific loaded model.

```
GET /api/models/:name
```

**Example:**

```bash
curl http://localhost:8080/api/models/llama3.2:1b
```

**Response:**

```json
{
  "alias": "llama3.2:1b",
  "path": "/home/user/.cache/mullama/models/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "parameters": 1236000000,
  "context_size": 4096,
  "gpu_layers": 35,
  "is_default": true,
  "active_requests": 0,
  "architecture": "LlamaForCausalLM",
  "vocab_size": 128256,
  "capabilities": {
    "vision": false,
    "tools": false,
    "thinking": false,
    "json": true
  }
}
```

---

### Pull Model

Download a model from HuggingFace. Supports streaming progress updates.

```
POST /api/models/pull
```

**Request Body:**

```json
{
  "name": "llama3.2:1b",
  "stream": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Model alias or HuggingFace spec |
| `stream` | boolean | No | Stream progress updates (default: false) |

**Example:**

```bash
curl -X POST http://localhost:8080/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2:1b", "stream": true}'
```

**Non-Streaming Response:**

```json
{
  "status": "success",
  "model": "llama3.2:1b",
  "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
  "size_bytes": 858993459
}
```

**Streaming Response (NDJSON):**

```
{"status":"downloading","progress":0.0,"total_bytes":858993459,"downloaded_bytes":0}
{"status":"downloading","progress":0.25,"total_bytes":858993459,"downloaded_bytes":214748364}
{"status":"downloading","progress":0.50,"total_bytes":858993459,"downloaded_bytes":429496729}
{"status":"downloading","progress":0.75,"total_bytes":858993459,"downloaded_bytes":644245094}
{"status":"downloading","progress":1.0,"total_bytes":858993459,"downloaded_bytes":858993459}
{"status":"verifying"}
{"status":"success","model":"llama3.2:1b"}
```

---

### Load Model

Load a model into memory for inference.

```
POST /api/models/load
```

**Request Body:**

```json
{
  "alias": "my-model",
  "path": "/path/to/model.gguf",
  "gpu_layers": 35,
  "context_size": 4096,
  "set_default": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | string | Yes | Name/alias for the model |
| `path` | string | No | Path to GGUF file (resolves alias if omitted) |
| `gpu_layers` | integer | No | GPU layers to offload (default: 0) |
| `context_size` | integer | No | Context window size (default: 4096) |
| `mmproj` | string | No | Path to vision projector |
| `set_default` | boolean | No | Set as default model |

**Example:**

```bash
curl -X POST http://localhost:8080/api/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "my-model",
    "path": "/opt/models/custom.gguf",
    "gpu_layers": 35,
    "context_size": 8192
  }'
```

**Response:**

```json
{
  "status": "loaded",
  "alias": "my-model",
  "parameters": 7000000000,
  "context_size": 8192,
  "gpu_layers": 35
}
```

---

### Unload Model

Unload a model from memory, freeing resources.

```
POST /api/models/:name/unload
```

**Example:**

```bash
curl -X POST http://localhost:8080/api/models/llama3.2:1b/unload
```

**Response:**

```json
{
  "status": "unloaded",
  "alias": "llama3.2:1b"
}
```

---

### Delete Model

Remove a model from disk entirely.

```
DELETE /api/models/:name
```

**Example:**

```bash
curl -X DELETE http://localhost:8080/api/models/my-model
```

**Response:**

```json
{
  "status": "deleted",
  "alias": "my-model"
}
```

---

## Generation Endpoints

### Text Generation

Generate text from a prompt. Supports streaming.

```
POST /api/generate
```

**Request Body:**

```json
{
  "model": "llama3.2:1b",
  "prompt": "Explain quantum computing in simple terms",
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | daemon default | Model to use |
| `prompt` | string | Yes | -- | Input text |
| `max_tokens` | integer | No | 512 | Maximum tokens |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `top_p` | float | No | 0.9 | Nucleus sampling |
| `top_k` | integer | No | 40 | Top-k sampling |
| `stop` | array | No | -- | Stop sequences |
| `stream` | boolean | No | false | Enable streaming |
| `system` | string | No | -- | System prompt |

**Example:**

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "What is the capital of France?",
    "max_tokens": 256
  }'
```

**Non-Streaming Response:**

```json
{
  "model": "llama3.2:1b",
  "response": "The capital of France is Paris.",
  "done": true,
  "total_duration_ms": 450,
  "prompt_tokens": 8,
  "completion_tokens": 7,
  "tokens_per_second": 15.5
}
```

**Streaming Response (NDJSON):**

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "prompt": "Hello!", "stream": true}'
```

```
{"model":"llama3.2:1b","response":"Hello","done":false}
{"model":"llama3.2:1b","response":"!","done":false}
{"model":"llama3.2:1b","response":" How","done":false}
{"model":"llama3.2:1b","response":" can","done":false}
{"model":"llama3.2:1b","response":" I","done":false}
{"model":"llama3.2:1b","response":" help","done":false}
{"model":"llama3.2:1b","response":"?","done":false}
{"model":"llama3.2:1b","response":"","done":true,"total_duration_ms":320,"prompt_tokens":3,"completion_tokens":7,"tokens_per_second":21.8}
```

---

### Chat Completion

Generate a chat completion from conversation history. Supports streaming.

```
POST /api/chat
```

**Request Body:**

```json
{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 256,
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | daemon default | Model to use |
| `messages` | array | Yes | -- | Conversation messages |
| `max_tokens` | integer | No | 512 | Maximum tokens |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `stream` | boolean | No | false | Enable streaming |

**Example:**

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "What is the meaning of life?"}
    ]
  }'
```

**Response:**

```json
{
  "model": "llama3.2:1b",
  "message": {
    "role": "assistant",
    "content": "The meaning of life is a profound philosophical question..."
  },
  "done": true,
  "total_duration_ms": 890,
  "prompt_tokens": 12,
  "completion_tokens": 45
}
```

---

### Generate Embeddings

Generate vector embeddings for input text.

```
POST /api/embeddings
```

**Request Body:**

```json
{
  "model": "nomic-embed",
  "input": "Hello, world!"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | No | Model to use (default: daemon default) |
| `input` | string or array | Yes | Text(s) to embed |

**Example (single text):**

```bash
curl -X POST http://localhost:8080/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed",
    "input": "Hello, world!"
  }'
```

**Example (batch):**

```bash
curl -X POST http://localhost:8080/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed",
    "input": ["Hello, world!", "How are you?", "Goodbye!"]
  }'
```

**Response:**

```json
{
  "model": "nomic-embed",
  "embeddings": [
    [0.0023, -0.0091, 0.0152, 0.0284, -0.0037, ...],
  ],
  "dimensions": 768,
  "total_tokens": 4
}
```

---

## System Endpoints

### System Status

Get comprehensive system status including uptime, loaded models, and statistics.

```
GET /api/system/status
```

**Example:**

```bash
curl http://localhost:8080/api/system/status
```

**Response:**

```json
{
  "version": "0.1.1",
  "uptime_secs": 3600,
  "models_loaded": 2,
  "http_endpoint": "http://0.0.0.0:8080",
  "ipc_endpoint": "ipc:///tmp/mullama.sock",
  "default_model": "llama3.2:1b",
  "stats": {
    "requests_total": 150,
    "tokens_generated": 45000,
    "active_requests": 1,
    "gpu_available": true,
    "memory_used_bytes": 4500000000,
    "memory_total_bytes": 16000000000
  }
}
```

---

### Health Check

Simple health check that returns 200 if the daemon is running and healthy.

```
GET /health
```

**Example:**

```bash
curl http://localhost:8080/health
```

**Response:**

```json
{
  "status": "ok"
}
```

HTTP status codes:

- `200` -- Daemon is healthy
- `503` -- Daemon is starting up or shutting down

---

### Detailed Status

Returns more detailed status information about the daemon state.

```
GET /status
```

**Example:**

```bash
curl http://localhost:8080/status
```

**Response:**

```json
{
  "status": "ok",
  "version": "0.1.1",
  "uptime_secs": 3600,
  "models_loaded": 2,
  "active_requests": 0
}
```

---

### List Default Models

Get the list of pre-configured default models available for quick setup.

```
GET /api/defaults
```

**Example:**

```bash
curl http://localhost:8080/api/defaults
```

**Response:**

```json
[
  {
    "name": "llama3.2-1b",
    "description": "Meta Llama 3.2 1B - Fast and lightweight",
    "size_hint": "1B",
    "tags": ["chat", "instruct", "fast", "lightweight"],
    "from": "hf:bartowski/Llama-3.2-1B-Instruct-GGUF",
    "has_thinking": false,
    "has_vision": false,
    "has_tools": false
  },
  {
    "name": "deepseek-r1-7b",
    "description": "DeepSeek R1 7B - Advanced reasoning model",
    "size_hint": "7B",
    "tags": ["reasoning", "thinking", "chain-of-thought"],
    "from": "hf:bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    "has_thinking": true,
    "has_vision": false,
    "has_tools": false
  }
]
```

---

### Use Default Model

Download and load one of the pre-configured default models.

```
POST /api/defaults/:name/use
```

**Example:**

```bash
curl -X POST http://localhost:8080/api/defaults/llama3.2-1b/use
```

**Response:**

```json
{
  "status": "loading",
  "model": "llama3.2-1b",
  "message": "Downloading and loading model..."
}
```

---

### Prometheus Metrics

Exposes metrics in Prometheus text exposition format for monitoring integration.

```
GET /metrics
```

**Example:**

```bash
curl http://localhost:8080/metrics
```

**Response:**

```
# HELP mullama_info Mullama daemon information
# TYPE mullama_info gauge
mullama_info{version="0.1.1"} 1

# HELP mullama_uptime_seconds Daemon uptime in seconds
# TYPE mullama_uptime_seconds counter
mullama_uptime_seconds 3600

# HELP mullama_models_loaded Number of currently loaded models
# TYPE mullama_models_loaded gauge
mullama_models_loaded 2

# HELP mullama_requests_total Total number of requests processed
# TYPE mullama_requests_total counter
mullama_requests_total{endpoint="chat"} 120
mullama_requests_total{endpoint="generate"} 25
mullama_requests_total{endpoint="embeddings"} 5

# HELP mullama_requests_active Currently active requests
# TYPE mullama_requests_active gauge
mullama_requests_active 1

# HELP mullama_tokens_generated_total Total tokens generated
# TYPE mullama_tokens_generated_total counter
mullama_tokens_generated_total 45000

# HELP mullama_tokens_per_second Average tokens per second
# TYPE mullama_tokens_per_second gauge
mullama_tokens_per_second{model="llama3.2:1b"} 35.2

# HELP mullama_prompt_tokens_total Total prompt tokens processed
# TYPE mullama_prompt_tokens_total counter
mullama_prompt_tokens_total 12500

# HELP mullama_model_parameters Model parameter count
# TYPE mullama_model_parameters gauge
mullama_model_parameters{model="llama3.2:1b"} 1236000000

# HELP mullama_model_context_size Model context size
# TYPE mullama_model_context_size gauge
mullama_model_context_size{model="llama3.2:1b"} 4096

# HELP mullama_model_gpu_layers Model GPU layers
# TYPE mullama_model_gpu_layers gauge
mullama_model_gpu_layers{model="llama3.2:1b"} 35

# HELP mullama_memory_used_bytes Memory used by models
# TYPE mullama_memory_used_bytes gauge
mullama_memory_used_bytes 4500000000

# HELP mullama_request_duration_seconds Request processing duration
# TYPE mullama_request_duration_seconds histogram
mullama_request_duration_seconds_bucket{le="0.1"} 10
mullama_request_duration_seconds_bucket{le="0.5"} 85
mullama_request_duration_seconds_bucket{le="1.0"} 130
mullama_request_duration_seconds_bucket{le="5.0"} 148
mullama_request_duration_seconds_bucket{le="+Inf"} 150
mullama_request_duration_seconds_count 150
mullama_request_duration_seconds_sum 95.5
```

---

## Streaming Format

Streaming responses use **Newline-Delimited JSON (NDJSON)**. Each line is a complete JSON object followed by a newline character (`\n`).

```
Content-Type: application/x-ndjson
```

The client reads line-by-line and parses each line as JSON. The final line contains `"done": true` with summary statistics.

!!! tip "curl with streaming"
    Use `--no-buffer` or `-N` with curl to see streaming output in real-time:
    ```bash
    curl -N -X POST http://localhost:8080/api/generate \
      -H "Content-Type: application/json" \
      -d '{"model": "llama3.2:1b", "prompt": "Hello!", "stream": true}'
    ```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "not_found",
    "code": "model_not_found"
  }
}
```

**Common HTTP Status Codes:**

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `200` | Success | Request completed |
| `400` | Bad request | Invalid JSON, missing required fields |
| `404` | Not found | Model not loaded, invalid endpoint |
| `409` | Conflict | Model already loaded/unloaded |
| `413` | Payload too large | Request body exceeds limit |
| `429` | Too many requests | Rate limit exceeded (if configured) |
| `500` | Internal error | Model inference failed |
| `503` | Service unavailable | Daemon starting up, model loading |

**Error Types:**

| Type | Description |
|------|-------------|
| `invalid_request` | Malformed request body |
| `not_found` | Requested resource does not exist |
| `model_error` | Model loading or inference failure |
| `conflict` | Conflicting operation |
| `internal_error` | Unexpected server error |
| `overloaded` | Too many concurrent requests |

---

## CORS

The API server enables CORS with permissive settings by default:

- **Allowed Origins:** Any (`*`)
- **Allowed Methods:** Any
- **Allowed Headers:** Any

This allows browser-based applications and the embedded Web UI to communicate with the API without restrictions. Configure allowed origins in the [configuration](configuration.md) for production deployments.

---

## Route Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List all loaded models |
| `GET` | `/api/models/:name` | Get model details |
| `POST` | `/api/models/pull` | Download model from HuggingFace |
| `POST` | `/api/models/load` | Load model into memory |
| `POST` | `/api/models/:name/unload` | Unload model from memory |
| `DELETE` | `/api/models/:name` | Delete model from disk |
| `POST` | `/api/generate` | Text generation |
| `POST` | `/api/chat` | Chat completion |
| `POST` | `/api/embeddings` | Generate embeddings |
| `GET` | `/api/system/status` | System status |
| `GET` | `/api/defaults` | List default models |
| `POST` | `/api/defaults/:name/use` | Load a default model |
| `GET` | `/health` | Health check |
| `GET` | `/status` | Detailed status |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/chat/completions` | [OpenAI chat](openai-api.md) |
| `POST` | `/v1/completions` | [OpenAI completions](openai-api.md) |
| `POST` | `/v1/embeddings` | [OpenAI embeddings](openai-api.md) |
| `GET` | `/v1/models` | [OpenAI models list](openai-api.md) |
| `POST` | `/v1/messages` | [Anthropic messages](anthropic-api.md) |
