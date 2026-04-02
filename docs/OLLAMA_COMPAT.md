# Ollama API Compatibility

Mullama provides Ollama-compatible REST API endpoints alongside the OpenAI-compatible API.

## Endpoint Status

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/api/generate` | POST | Supported | Text generation with NDJSON streaming |
| `/api/chat` | POST | Supported | Chat completion with NDJSON streaming |
| `/api/tags` | GET | Supported | List local models |
| `/api/show` | POST | Supported | Show model info |
| `/api/pull` | POST | Supported | Pull model (via `/api/models/pull`) |
| `/api/delete` | DELETE | Supported | Delete model |
| `/api/copy` | POST | Supported | Copy/alias model |
| `/api/embeddings` | POST | Supported | Generate embeddings |
| `/api/ps` | GET | Supported | List running models |
| `/api/version` | GET | Supported | Version info |
| `/api/create` | POST | Partial | Create from Modelfile |

## Usage Examples

### Generate

```bash
curl http://localhost:8080/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Why is the sky blue?"
}'
```

### Chat

```bash
curl http://localhost:8080/api/chat -d '{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### List Models

```bash
curl http://localhost:8080/api/tags
```

### Embeddings

```bash
curl http://localhost:8080/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "The quick brown fox"
}'
```

### Show Model Info

```bash
curl http://localhost:8080/api/show -d '{"name": "llama3.2:1b"}'
```

### Running Models

```bash
curl http://localhost:8080/api/ps
```

### Version

```bash
curl http://localhost:8080/api/version
```

## CLI Compatibility

| Ollama Command | Mullama Equivalent |
|---------------|-------------------|
| `ollama run model "prompt"` | `mullama run model "prompt"` |
| `ollama serve` | `mullama serve` |
| `ollama list` | `mullama list` |
| `ollama pull model` | `mullama pull model` |
| `ollama rm model` | `mullama rm model` |
| `ollama cp src dest` | `mullama cp src dest` |
| `ollama ps` | `mullama ps` |
| `ollama show model` | `mullama show model` |
| `ollama create model -f Modelfile` | `mullama create model -f Modelfile` |

## Differences from Ollama

1. **Streaming format**: Mullama returns complete JSON responses by default. NDJSON streaming support is available via `"stream": true`.
2. **Additional APIs**: Mullama also supports OpenAI-compatible (`/v1/`) and Anthropic-compatible (`/v1/messages`) endpoints.
3. **Model sources**: Mullama supports HuggingFace models directly (`hf:user/repo:file.gguf`) in addition to Ollama-format models.
