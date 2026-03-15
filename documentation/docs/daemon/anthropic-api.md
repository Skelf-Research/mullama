---
title: Anthropic API Compatibility
description: Anthropic Claude Messages API compatibility layer
---

# Anthropic API Compatibility

The Mullama daemon provides an Anthropic Claude-compatible Messages API endpoint, allowing applications built for the Anthropic API to work with local models.

## Endpoint

```
POST /v1/messages
```

This single endpoint handles both streaming and non-streaming message generation, matching the Anthropic Messages API specification.

---

## Request Format

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | daemon default | Model to use |
| `max_tokens` | integer | Yes | -- | Maximum tokens to generate |
| `messages` | array | Yes | -- | Conversation messages |
| `system` | string | No | -- | System prompt |
| `stream` | boolean | No | `false` | Enable streaming |
| `temperature` | float | No | `1.0` | Sampling temperature (0-1) |
| `top_p` | float | No | -- | Nucleus sampling threshold |
| `top_k` | integer | No | -- | Top-k sampling |
| `stop_sequences` | array | No | -- | Stop sequences |
| `metadata` | object | No | -- | Metadata (accepted but ignored) |

### Message Format

Messages use the Anthropic role/content format:

```json
{
  "role": "user",
  "content": "Hello!"
}
```

Content can also be an array of content blocks for multimodal input:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/4AAQ..."
      }
    }
  ]
}
```

**Supported Content Block Types:**

| Type | Description |
|------|-------------|
| `text` | Text content |
| `image` | Base64-encoded image |
| `tool_use` | Tool/function call |
| `tool_result` | Tool execution result |

---

## Non-Streaming Response

**Example Request:**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What is the capital of Japan?"}
    ]
  }'
```

**Response:**

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "The capital of Japan is Tokyo."
    }
  ],
  "model": "llama3.2:1b",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 14,
    "output_tokens": 9
  }
}
```

---

## Streaming Response

When `stream: true` is set, the response uses Server-Sent Events (SSE) with Anthropic's streaming protocol.

**Example Request:**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

**Response (SSE Events):**

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01abc","type":"message","role":"assistant","content":[],"model":"llama3.2:1b","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Once"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" upon"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" time"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":45}}

event: message_stop
data: {"type":"message_stop"}
```

### Stream Event Types

| Event | Description |
|-------|-------------|
| `message_start` | Initial message metadata |
| `content_block_start` | Start of a content block |
| `content_block_delta` | Token-by-token content |
| `content_block_stop` | End of a content block |
| `message_delta` | Final message metadata (stop reason, usage) |
| `message_stop` | Stream complete |
| `ping` | Keep-alive ping |
| `error` | Error occurred |

---

## System Prompt

The system prompt can be specified either as a top-level field or as the first message with role "system":

=== "Top-level system field"

    ```json
    {
      "model": "llama3.2:1b",
      "max_tokens": 1024,
      "system": "You are a helpful coding assistant.",
      "messages": [
        {"role": "user", "content": "Write a Python function to sort a list"}
      ]
    }
    ```

=== "System message"

    ```json
    {
      "model": "llama3.2:1b",
      "max_tokens": 1024,
      "messages": [
        {"role": "user", "content": "Write a Python function to sort a list"}
      ]
    }
    ```

---

## Multi-Turn Conversations

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 1024,
    "system": "You are a math tutor.",
    "messages": [
      {"role": "user", "content": "What is 15 * 23?"},
      {"role": "assistant", "content": "15 * 23 = 345"},
      {"role": "user", "content": "Now divide that by 5"}
    ]
  }'
```

---

## Extended Thinking Support

For models configured with thinking tokens (e.g., DeepSeek-R1), the response can include separate thinking content. This is controlled by the model's Modelfile configuration with `THINKING` directives.

When a model has thinking enabled, streaming responses may include thinking content in the delta:

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "thinking_delta",
    "thinking": "Let me work through this step by step..."
  }
}
```

---

## Using with SDKs

### Python (anthropic package)

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8080",
    api_key="unused"  # Required by SDK but not validated
)

# Non-streaming
message = client.messages.create(
    model="llama3.2:1b",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(message.content[0].text)

# Streaming
with client.messages.stream(
    model="llama3.2:1b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

### Node.js (anthropic package)

```javascript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  baseURL: 'http://localhost:8080',
  apiKey: 'unused'
});

// Non-streaming
const message = await client.messages.create({
  model: 'llama3.2:1b',
  max_tokens: 1024,
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});
console.log(message.content[0].text);

// Streaming
const stream = client.messages.stream({
  model: 'llama3.2:1b',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Tell me a story' }]
});
for await (const event of stream) {
  if (event.type === 'content_block_delta') {
    process.stdout.write(event.delta.text);
  }
}
```

### Python (requests)

```python
import requests
import json

# Non-streaming
response = requests.post(
    "http://localhost:8080/v1/messages",
    json={
        "model": "llama3.2:1b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
data = response.json()
print(data["content"][0]["text"])

# Streaming
response = requests.post(
    "http://localhost:8080/v1/messages",
    json={
        "model": "llama3.2:1b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)
for line in response.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data: "):
            event_data = json.loads(line[6:])
            if event_data["type"] == "content_block_delta":
                print(event_data["delta"]["text"], end="")
```

---

## Error Responses

Errors follow the Anthropic error format:

```json
{
  "type": "error",
  "error": {
    "type": "not_found_error",
    "message": "Model 'unknown-model' not found"
  }
}
```

**Error Types:**

| Type | HTTP Code | Description |
|------|-----------|-------------|
| `invalid_request_error` | 400 | Malformed request |
| `not_found_error` | 404 | Model not found |
| `overloaded_error` | 529 | Server overloaded |
| `api_error` | 500 | Internal error |

---

## Differences from Anthropic API

- **Authentication**: No API key validation. Headers like `x-api-key` and `anthropic-version` are accepted but not checked.
- **Model names**: Use local model aliases instead of Anthropic model names (e.g., `llama3.2:1b` instead of `claude-3-opus`).
- **Rate limiting**: No built-in rate limiting.
- **Tool use**: Supported via Modelfile TOOLFORMAT configuration.
- **Vision**: Supported when the model has multimodal capabilities.
- **Metadata**: The `metadata` field is accepted for compatibility but ignored.
- **Max tokens**: The `max_tokens` field is required (matching Anthropic's spec).
