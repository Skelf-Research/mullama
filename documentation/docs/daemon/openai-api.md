---
title: OpenAI API Compatibility
description: Drop-in replacement for OpenAI Chat Completions, Completions, Embeddings, and Models endpoints
---

# OpenAI API Compatibility

The Mullama daemon provides a drop-in replacement for the OpenAI API. Applications built for OpenAI can connect to Mullama by simply changing the base URL, enabling local LLM inference with no code changes.

## Overview

**Base URL:** `http://localhost:8080/v1`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with streaming |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |

---

## Using with OpenAI SDKs

=== "Python"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="unused"  # Required by SDK but not validated
    )

    # Chat completion
    response = client.chat.completions.create(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=256
    )
    print(response.choices[0].message.content)
    ```

=== "Node.js"

    ```javascript
    import OpenAI from 'openai';

    const client = new OpenAI({
      baseURL: 'http://localhost:8080/v1',
      apiKey: 'unused'
    });

    const response = await client.chat.completions.create({
      model: 'llama3.2:1b',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' }
      ],
      temperature: 0.7,
      max_tokens: 256
    });
    console.log(response.choices[0].message.content);
    ```

=== "curl"

    ```bash
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3.2:1b",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
      }'
    ```

---

## Chat Completions

### POST /v1/chat/completions

Generate a chat completion from a conversation history.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | daemon default | Model identifier |
| `messages` | array | Yes | -- | Conversation messages |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `max_tokens` | integer | No | 512 | Maximum tokens to generate |
| `stream` | boolean | No | false | Enable SSE streaming |
| `stop` | string/array | No | -- | Stop sequence(s) |
| `n` | integer | No | 1 | Number of completions |
| `frequency_penalty` | float | No | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | No | 0.0 | Presence penalty (-2.0 to 2.0) |
| `seed` | integer | No | -- | Random seed for reproducibility |
| `user` | string | No | -- | End-user identifier (logged, not used) |

### Message Format

```json
{
  "role": "system" | "user" | "assistant",
  "content": "Message text"
}
```

### Non-Streaming Example

**Request:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1706000000,
  "model": "llama3.2:1b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations that would be impractical for classical computers."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 28,
    "total_tokens": 53
  }
}
```

### Streaming Example

**Request:**

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

**Response (SSE):**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{"content":" time"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama3.2:1b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Streaming with SDKs

=== "Python"

    ```python
    stream = client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    ```

=== "Node.js"

    ```javascript
    const stream = await client.chat.completions.create({
      model: 'llama3.2:1b',
      messages: [{ role: 'user', content: 'Tell me a story' }],
      stream: true
    });
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      process.stdout.write(content);
    }
    ```

### Multi-Turn Conversation

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "system", "content": "You are a math tutor."},
      {"role": "user", "content": "What is 15 * 23?"},
      {"role": "assistant", "content": "15 * 23 = 345"},
      {"role": "user", "content": "Now divide that by 5"}
    ]
  }'
```

---

## Text Completions

### POST /v1/completions

Generate a text completion from a prompt (legacy completions API).

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | daemon default | Model identifier |
| `prompt` | string | Yes | -- | Input text |
| `max_tokens` | integer | No | 512 | Maximum tokens |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `top_p` | float | No | 0.9 | Nucleus sampling |
| `stream` | boolean | No | false | Enable SSE streaming |
| `stop` | string/array | No | -- | Stop sequence(s) |
| `echo` | boolean | No | false | Include prompt in response |
| `suffix` | string | No | -- | Text after completion (fill-in-middle) |

**Example:**

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1706000000,
  "model": "llama3.2:1b",
  "choices": [
    {
      "text": " jumped over the lazy dog. This classic pangram contains every letter of the English alphabet.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 18,
    "total_tokens": 23
  }
}
```

---

## Embeddings

### POST /v1/embeddings

Generate vector embeddings for input text.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | No | Model to use (default: daemon default) |
| `input` | string/array | Yes | Text(s) to embed |
| `encoding_format` | string | No | `float` or `base64` (default: float) |

**Example (single text):**

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed",
    "input": "Hello, world!"
  }'
```

**Example (batch):**

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed",
    "input": [
      "The cat sat on the mat",
      "A dog played in the park",
      "Machine learning is fascinating"
    ]
  }'
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0091, 0.0152, 0.0284, -0.0037],
      "index": 0
    }
  ],
  "model": "nomic-embed",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

### Embedding SDK Usage

=== "Python"

    ```python
    response = client.embeddings.create(
        model="nomic-embed",
        input=["Hello, world!", "How are you?"]
    )
    for item in response.data:
        print(f"Index {item.index}: {len(item.embedding)} dimensions")
    ```

=== "Node.js"

    ```javascript
    const response = await client.embeddings.create({
      model: 'nomic-embed',
      input: ['Hello, world!', 'How are you?']
    });
    response.data.forEach(item => {
      console.log(`Index ${item.index}: ${item.embedding.length} dimensions`);
    });
    ```

---

## Models

### GET /v1/models

List all models available for inference.

**Example:**

```bash
curl http://localhost:8080/v1/models
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2:1b",
      "object": "model",
      "created": 1706000000,
      "owned_by": "local"
    },
    {
      "id": "qwen2.5:7b",
      "object": "model",
      "created": 1706000000,
      "owned_by": "local"
    }
  ]
}
```

---

## Streaming Format

When `stream: true` is set, the response uses Server-Sent Events (SSE):

- Each event begins with `data: ` followed by a JSON object
- Events are separated by double newlines (`\n\n`)
- The stream terminates with `data: [DONE]`
- Content-Type: `text/event-stream`

**Parsing example (Python):**

```python
import requests
import json

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="")
```

---

## Supported Parameters

### Full Parameter Support

| Parameter | Supported | Notes |
|-----------|:---------:|-------|
| `model` | Yes | Uses local model aliases |
| `messages` | Yes | system, user, assistant roles |
| `temperature` | Yes | 0.0 - 2.0 |
| `top_p` | Yes | 0.0 - 1.0 |
| `max_tokens` | Yes | Model-dependent maximum |
| `stream` | Yes | SSE format |
| `stop` | Yes | String or array |
| `n` | Yes | Number of completions |
| `frequency_penalty` | Yes | -2.0 to 2.0 |
| `presence_penalty` | Yes | -2.0 to 2.0 |
| `seed` | Yes | For reproducibility |
| `user` | Accepted | Logged but not used for billing |

### Unsupported Parameters

| Parameter | Status | Notes |
|-----------|--------|-------|
| `tools` / `functions` | Partial | Depends on model and Modelfile config |
| `tool_choice` | Partial | Basic support |
| `response_format` | Partial | `json_object` mode via system prompt |
| `logprobs` | No | Not implemented |
| `top_logprobs` | No | Not implemented |
| `logit_bias` | No | Not implemented |

---

## Function/Tool Calling

Tool calling is supported for models configured with `TOOLFORMAT` in their Modelfile.

**Request:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

**Response (when model decides to call a tool):**

```json
{
  "id": "chatcmpl-abc123",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"Paris\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

!!! info "Tool Calling Requirements"
    Tool calling requires a model with tool calling capabilities configured in its Modelfile. See [Modelfile Format](modelfile.md#toolformat) for configuration details.

---

## Vision (Multimodal)

For models with vision capabilities, images can be included in messages:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava:7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQ..."
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```

!!! note "Image Format"
    Images must be base64-encoded and included as data URLs. The supported formats are JPEG, PNG, and WebP.

---

## Differences from OpenAI API

| Feature | OpenAI | Mullama |
|---------|--------|---------|
| **Authentication** | API key required | Optional (configurable) |
| **Model names** | `gpt-4`, `gpt-3.5-turbo` | Local aliases (`llama3.2:1b`, etc.) |
| **Rate limiting** | Per-key quotas | Optional per-IP limiting |
| **Billing** | Pay per token | Free (local inference) |
| **logprobs** | Supported | Not implemented |
| **logit_bias** | Supported | Not implemented |
| **response_format** | Full JSON mode | Partial (via system prompt) |
| **Tool calling** | Full support | Model-dependent |
| **Vision** | GPT-4V | Requires vision model (LLaVA, etc.) |
| **Embeddings** | Multiple models | Single loaded model |
| **Moderation** | Available | Not implemented |
| **Fine-tuning** | API-based | Via Modelfile/LoRA |
| **Batch API** | Available | Not implemented |

---

## Error Responses

Errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| HTTP Code | Error Type | Common Causes |
|-----------|-----------|---------------|
| 400 | `invalid_request_error` | Malformed request, missing fields |
| 401 | `authentication_error` | Invalid API key (when auth enabled) |
| 404 | `not_found_error` | Model not loaded |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `internal_error` | Model inference failure |
| 503 | `overloaded_error` | Server overloaded |
