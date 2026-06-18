---
title: Configuration
description: Server settings, environment variables, config file, GPU, logging, and resource management
---

# Configuration

The Mullama daemon can be configured through a configuration file, command-line flags, and environment variables. This page covers all available configuration options and their interactions.

## Configuration Precedence

Configuration sources are applied in this order (highest precedence first):

1. **CLI flags** -- Command-line arguments override everything
2. **Environment variables** -- Override config file values
3. **Configuration file** -- Base configuration
4. **Built-in defaults** -- Used when nothing else is specified

---

## Configuration File

### Location

The configuration file is loaded from:

| Priority | Path |
|----------|------|
| 1 (highest) | `MULLAMA_CONFIG` environment variable |
| 2 | `~/.mullama/config.yaml` |
| 3 | `~/.config/mullama/config.yaml` (Linux XDG) |
| 4 | Built-in defaults |

### Format

The configuration file uses YAML format. Create `~/.mullama/config.yaml`:

```yaml
# Server configuration
server:
  host: "0.0.0.0"
  port: 8080
  socket: "ipc:///tmp/mullama.sock"
  max_connections: 100
  request_timeout: 300  # seconds

# Model defaults
models:
  default_model: "llama3.2:1b"
  gpu_layers: 35
  context_size: 4096
  threads: 8
  models_dir: "~/.mullama/models"
  cache_dir: "~/.cache/mullama/models"
  flash_attention: false

# Logging
logging:
  level: "info"           # trace, debug, info, warn, error
  file: "/tmp/mullamad.log"
  format: "text"          # text, json

# Metrics
metrics:
  enabled: true
  endpoint: "/metrics"

# Security
security:
  allowed_origins:        # CORS origins (empty = allow all)
    - "http://localhost:8080"
    - "http://localhost:5173"
  api_key: ""             # Empty = no auth required

# Memory management
memory:
  poll_interval_ms: 5000
  system_threshold: 0.90
  gpu_threshold: 0.90
  enable_recovery: true
```

### Complete Configuration Reference

```yaml
# ============================================================================
# Mullama Daemon Configuration
# ============================================================================

server:
  # HTTP server bind address
  # Values: IP address or "0.0.0.0" for all interfaces
  host: "0.0.0.0"

  # HTTP server port (0 to disable HTTP)
  port: 8080

  # IPC socket address for local communication
  socket: "ipc:///tmp/mullama.sock"

  # Maximum concurrent HTTP connections
  max_connections: 100

  # Request timeout in seconds (0 = no timeout)
  request_timeout: 300

  # Maximum request body size in bytes (for image uploads)
  max_body_size: 52428800  # 50 MB

models:
  # Default model to use when none specified in requests
  default_model: ""

  # Default GPU layers for newly loaded models
  gpu_layers: 0

  # Default context window size
  context_size: 4096

  # CPU threads per model (0 = auto-detect, typically num_cpus / 2)
  threads: 0

  # Custom model storage directory
  models_dir: "~/.mullama/models"

  # Downloaded model cache directory
  cache_dir: ""  # Empty = platform default

  # Enable flash attention for all models
  flash_attention: false

  # Auto-load these models on daemon start
  auto_load: []
  # auto_load:
  #   - alias: "llama3.2:1b"
  #     gpu_layers: 35
  #     context_size: 4096
  #   - alias: "qwen2.5:7b"
  #     gpu_layers: 20
  #     context_size: 8192

logging:
  # Log level: trace, debug, info, warn, error
  level: "info"

  # Log file path (used in daemon/background mode)
  file: "/tmp/mullamad.log"

  # Log format: text (human-readable), json (structured)
  format: "text"

  # Include timestamps in log output
  timestamps: true

  # Log individual token generation (very verbose)
  log_tokens: false

metrics:
  # Enable Prometheus metrics endpoint
  enabled: true

  # Metrics endpoint path
  endpoint: "/metrics"

security:
  # CORS allowed origins (empty array = allow all)
  allowed_origins: []

  # API key for authentication (empty = no auth)
  api_key: ""

  # Rate limiting (requests per minute per IP, 0 = disabled)
  rate_limit: 0

  # TLS certificate path (empty = no TLS)
  tls_cert: ""

  # TLS private key path
  tls_key: ""

memory:
  # Memory polling interval in milliseconds
  poll_interval_ms: 5000

  # System RAM usage threshold for warnings (0.0-1.0)
  system_threshold: 0.90

  # GPU VRAM usage threshold for warnings (0.0-1.0)
  gpu_threshold: 0.90

  # Enable automatic OOM recovery (unload LRU models)
  enable_recovery: true

  # Maximum total memory for loaded models (0 = unlimited)
  max_model_memory: 0  # bytes

tui:
  # TUI color theme: auto, dark, light
  theme: "auto"

  # Show thinking tokens for reasoning models
  show_thinking: true

  # Auto-save sessions on exit
  auto_save: true

  # Maximum messages per session
  history_limit: 1000
```

---

## Server Settings

### host

The network interface to bind the HTTP server to.

| Value | Meaning |
|-------|---------|
| `0.0.0.0` | Listen on all interfaces (default) |
| `127.0.0.1` | Localhost only (no external access) |
| `192.168.1.100` | Specific network interface |

```bash
# CLI flag
mullama serve --http-addr 127.0.0.1

# Environment variable
export MULLAMA_HOST=127.0.0.1
```

### port

HTTP server port. Set to `0` to disable the HTTP server entirely (IPC-only mode).

```bash
# CLI flag
mullama serve --http-port 9090

# Environment variable
export MULLAMA_PORT=9090
```

### socket

IPC socket address for local CLI and TUI communication. Uses NNG REQ/REP pattern.

```bash
# CLI flag
mullama serve --socket ipc:///var/run/mullama.sock

# Environment variable
export MULLAMA_SOCKET=ipc:///var/run/mullama.sock
```

| Platform | Default Socket |
|----------|---------------|
| Linux/macOS | `ipc:///tmp/mullama.sock` |
| Windows | `ipc://mullama` (named pipe) |

### max_connections

Maximum concurrent HTTP connections. Requests beyond this limit receive `503 Service Unavailable`.

### request_timeout

Maximum time in seconds for a single request. Long-running generation requests may need higher values.

---

## Model Settings

### default_model

The model used when API requests do not specify a `model` field.

```bash
# CLI flag (set on startup)
mullama serve --model llama3.2:1b  # First model becomes default

# Or load and set as default
mullama load llama3.2:1b --default
```

### gpu_layers

Default number of model layers to offload to GPU. Set per-model when loading for fine-grained control.

```bash
# CLI flag
mullama serve --gpu-layers 35

# Environment variable
export MULLAMA_GPU_LAYERS=35
```

**Guidelines:**

| Model Size | Recommended GPU Layers | VRAM Required |
|------------|----------------------|---------------|
| 1B | 24 | ~1.5 GB |
| 3B | 28 | ~3 GB |
| 7B | 35 | ~5 GB |
| 14B | 40 | ~10 GB |
| 32B | 64 | ~20 GB |
| 70B | 80 | ~40 GB |

### context_size

Default context window size in tokens. Larger contexts require more memory.

```bash
# CLI flag
mullama serve --context-size 8192

# Environment variable
export MULLAMA_CONTEXT_SIZE=8192
```

**Memory impact:** Roughly `context_size * 0.5 MB` additional memory per model.

### threads

CPU threads allocated per model for inference. Defaults to `num_cpus / 2`.

```bash
# CLI flag
mullama serve --threads 8

# Auto-detect
mullama serve  # Uses num_cpus / 2
```

!!! tip "Thread Tuning"
    - For single-model deployments: use `num_cpus - 1`
    - For multi-model deployments: divide available cores among models
    - Hyperthreading: using physical core count often performs better than total threads

### models_dir

Directory for storing custom model configurations (created with `mullama create`).

```bash
export MULLAMA_MODELS_DIR="/opt/mullama/models"
```

### cache_dir

Directory for HuggingFace model downloads. Platform-specific defaults:

| Platform | Default |
|----------|---------|
| Linux | `~/.cache/mullama/models` |
| macOS | `~/Library/Caches/mullama/models` |
| Windows | `%LOCALAPPDATA%\mullama\models` |

```bash
export MULLAMA_CACHE_DIR="/mnt/fast-storage/models"
```

### auto_load

Models to automatically load when the daemon starts. Configured in the YAML file:

```yaml
models:
  auto_load:
    - alias: "llama3.2:1b"
      gpu_layers: 35
      context_size: 4096
    - alias: "nomic-embed"
      gpu_layers: 0
      context_size: 2048
```

Equivalent to CLI:

```bash
mullama serve --model llama3.2:1b --model nomic-embed
```

---

## Logging Configuration

### Log Levels

| Level | Description |
|-------|-------------|
| `trace` | Very verbose, includes internal state changes |
| `debug` | Detailed debugging information |
| `info` | Standard operational messages (default) |
| `warn` | Warnings that may need attention |
| `error` | Errors that require investigation |

```bash
export MULLAMA_LOG_LEVEL=debug
```

### Log Output

| Mode | Default Output | Override |
|------|---------------|---------|
| Foreground (`serve`) | stderr | -- |
| Background (`daemon start`) | `/tmp/mullamad.log` | `logging.file` in config |

### Viewing Logs

```bash
# View daemon logs
mullama daemon logs

# Follow in real-time
mullama daemon logs -f

# Last 200 lines
mullama daemon logs -n 200

# View with journalctl (systemd)
journalctl -u mullama -f
```

---

## Metrics

### Prometheus Endpoint

When enabled, metrics are exposed at the configured endpoint (default: `/metrics`).

```yaml
metrics:
  enabled: true
  endpoint: "/metrics"
```

Disable metrics:

```yaml
metrics:
  enabled: false
```

See the [REST API](rest-api.md#prometheus-metrics) page for the full list of exposed metrics.

---

## Security

### CORS (Cross-Origin Resource Sharing)

By default, the daemon allows requests from any origin. Restrict to specific origins for production:

```yaml
security:
  allowed_origins:
    - "https://your-app.example.com"
    - "http://localhost:5173"  # Vite dev server
```

### API Key Authentication

Set a required API key:

```yaml
security:
  api_key: "your-secret-key"
```

Or via environment variable:

```bash
export MULLAMA_API_KEY="your-secret-key"
```

When set, all API requests must include the key:

```bash
# As Bearer token
curl -H "Authorization: Bearer your-secret-key" http://localhost:8080/v1/models

# As x-api-key header
curl -H "x-api-key: your-secret-key" http://localhost:8080/v1/models
```

!!! warning "API Key Security"
    The API key is transmitted in plain text. Always use TLS (HTTPS) in production when API key authentication is enabled.

### Rate Limiting

Limit requests per minute per IP address:

```yaml
security:
  rate_limit: 60  # 60 requests per minute per IP
```

Set to `0` to disable rate limiting (default).

---

## Memory Monitoring

The daemon includes a background memory monitor that tracks system and GPU memory usage.

### Configuration

```yaml
memory:
  poll_interval_ms: 5000    # Check every 5 seconds
  system_threshold: 0.90    # Warn at 90% RAM usage
  gpu_threshold: 0.90       # Warn at 90% VRAM usage
  enable_recovery: true     # Auto-unload LRU models under pressure
```

### Behavior

The memory monitor:

1. Periodically checks system RAM and GPU VRAM usage
2. Logs warnings when usage exceeds thresholds
3. When recovery is enabled, unloads least-recently-used models to free memory
4. Reports memory status via `/api/system/status`

---

## Environment Variables

Complete environment variable reference:

| Variable | Config Key | Default | Description |
|----------|-----------|---------|-------------|
| `MULLAMA_HOST` | `server.host` | `0.0.0.0` | HTTP bind address |
| `MULLAMA_PORT` | `server.port` | `8080` | HTTP port |
| `MULLAMA_SOCKET` | `server.socket` | `ipc:///tmp/mullama.sock` | IPC socket |
| `MULLAMA_GPU_LAYERS` | `models.gpu_layers` | `0` | Default GPU layers |
| `MULLAMA_CONTEXT_SIZE` | `models.context_size` | `4096` | Default context size |
| `MULLAMA_MODELS_DIR` | `models.models_dir` | `~/.mullama/models` | Custom models dir |
| `MULLAMA_CACHE_DIR` | `models.cache_dir` | Platform-specific | Download cache dir |
| `MULLAMA_BIN` | -- | Auto-detected | Binary path for auto-spawn |
| `MULLAMA_CONFIG` | -- | `~/.mullama/config.yaml` | Config file path |
| `MULLAMA_LOG_LEVEL` | `logging.level` | `info` | Log level |
| `MULLAMA_API_KEY` | `security.api_key` | -- | API authentication key |
| `HF_TOKEN` | -- | -- | HuggingFace API token |

---

## GPU Acceleration

### Build-Time GPU Selection

Set environment variables before building:

```bash
# NVIDIA CUDA
export LLAMA_CUDA=1

# Apple Silicon (Metal)
export LLAMA_METAL=1

# AMD ROCm
export LLAMA_HIPBLAS=1

# OpenCL
export LLAMA_CLBLAST=1
```

Then build:

```bash
cargo build --release --features daemon
```

### Runtime GPU Configuration

```bash
# Offload 35 layers to GPU
mullama serve --model llama3.2:1b --gpu-layers 35

# Full offload (all layers)
mullama serve --model llama3.2:1b --gpu-layers 99

# Per-model GPU configuration when loading
mullama load my-model:./model.gguf --gpu-layers 20
```

### GPU Memory Estimation

| Model Size | Q4_K_M VRAM | Q8_0 VRAM | F16 VRAM |
|------------|-------------|-----------|----------|
| 1B | ~1 GB | ~1.5 GB | ~2.5 GB |
| 3B | ~2 GB | ~3.5 GB | ~6 GB |
| 7B | ~4.5 GB | ~7.5 GB | ~14 GB |
| 14B | ~9 GB | ~15 GB | ~28 GB |
| 32B | ~20 GB | ~33 GB | ~64 GB |

---

## Cache Locations

### Default Paths

| Platform | Cache (Downloads) | Models (Custom) | Sessions | Config |
|----------|-------------------|-----------------|----------|--------|
| Linux | `~/.cache/mullama/models` | `~/.mullama/models` | `~/.mullama/sessions` | `~/.mullama/config.yaml` |
| macOS | `~/Library/Caches/mullama/models` | `~/.mullama/models` | `~/.mullama/sessions` | `~/.mullama/config.yaml` |
| Windows | `%LOCALAPPDATA%\mullama\models` | `%USERPROFILE%\.mullama\models` | `%USERPROFILE%\.mullama\sessions` | `%USERPROFILE%\.mullama\config.yaml` |

### Cache Management

```bash
# Show cache path
mullama cache path

# Show total size
mullama cache size

# List cached models
mullama cache list --verbose

# Clear all cached models
mullama cache clear --force
```

---

## Auto-Spawn Configuration

When CLI commands auto-spawn the daemon, the following defaults are used:

| Setting | Value |
|---------|-------|
| HTTP Port | `8080` (or `MULLAMA_PORT`) |
| IPC Socket | `ipc:///tmp/mullama.sock` (or `MULLAMA_SOCKET`) |
| Log File | `/tmp/mullamad.log` |
| Background | `true` |
| GPU Layers | `0` (or `MULLAMA_GPU_LAYERS`) |
| Context Size | `4096` (or `MULLAMA_CONTEXT_SIZE`) |

Override by starting the daemon explicitly:

```bash
mullama daemon start --http-port 9090 --gpu-layers 35 --context-size 8192
```

---

## Per-Model Configuration

Models can have individual configurations when loaded:

```bash
# Via CLI
mullama load llama3.2:1b -g 35 -c 4096
mullama load qwen2.5:7b -g 20 -c 8192

# Via API
curl -X POST http://localhost:8080/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"alias": "llama3.2:1b", "gpu_layers": 35, "context_size": 4096}'
```

Per-model settings in Modelfile:

```dockerfile
FROM llama3.2:1b
PARAMETER num_ctx 8192
GPU_LAYERS 35
FLASH_ATTENTION true
```

---

## Example Configurations

### Development (localhost, single model)

```yaml
server:
  host: "127.0.0.1"
  port: 8080
models:
  gpu_layers: 0
  context_size: 4096
logging:
  level: "debug"
security:
  allowed_origins: []  # Allow all for dev
```

### Production (secured, multi-model)

```yaml
server:
  host: "127.0.0.1"  # Behind reverse proxy
  port: 8080
  max_connections: 200
  request_timeout: 120
models:
  gpu_layers: 35
  context_size: 8192
  threads: 16
  auto_load:
    - alias: "llama3.2:1b"
      gpu_layers: 35
    - alias: "qwen2.5:7b"
      gpu_layers: 35
logging:
  level: "warn"
  format: "json"
metrics:
  enabled: true
security:
  allowed_origins:
    - "https://app.example.com"
  api_key: "${MULLAMA_API_KEY}"
  rate_limit: 120
memory:
  enable_recovery: true
  system_threshold: 0.85
```

### Edge Device (minimal resources)

```yaml
server:
  host: "0.0.0.0"
  port: 8080
models:
  gpu_layers: 0
  context_size: 2048
  threads: 2
logging:
  level: "warn"
metrics:
  enabled: false
memory:
  system_threshold: 0.80
  enable_recovery: true
```
