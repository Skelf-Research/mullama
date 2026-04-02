# Production Deployment Guide

## Docker

### Quick Start

```bash
# Build and run with Docker Compose
docker compose up -d

# Pull and run a model
docker exec mullama mullama pull llama3.2:1b
docker exec mullama mullama run llama3.2:1b "Hello!"
```

### Custom Build

```bash
# CPU-only
docker build -t mullama .

# NVIDIA CUDA (GPU acceleration)
docker build -f Dockerfile.cuda -t mullama-cuda .
```

### Model Storage

Mount a host directory for persistent model storage:

```yaml
# docker-compose.yml
services:
  mullama:
    volumes:
      - ./models:/models
```

## HTTPS/TLS

Enable TLS for production deployments:

```bash
mullama serve --tls-cert /path/to/cert.pem --tls-key /path/to/key.pem
```

Or with environment variables and a reverse proxy (recommended):

```bash
# Behind nginx/caddy reverse proxy
mullama serve --host 127.0.0.1 --port 8080
```

## Logging

Control log verbosity with environment variables:

```bash
# Set log level
MULLAMA_LOG=info mullama serve

# Detailed debugging
MULLAMA_LOG=debug mullama serve

# Module-specific levels
RUST_LOG=mullama=debug,tower_http=info mullama serve
```

## API Key Authentication

```bash
# Auto-generate a secure key
mullama serve --require-api-key

# Use a specific key
mullama serve --api-key "your-secret-key"

# Client usage
curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8080/v1/models
```

## GPU Acceleration

```bash
# NVIDIA CUDA
export LLAMA_CUDA=1
mullama serve --gpu-layers -1  # offload all layers

# Apple Silicon (Metal)
export LLAMA_METAL=1
mullama serve --gpu-layers -1
```

## Resource Limits

```bash
mullama serve \
  --max-tokens-limit 4096 \
  --max-concurrent-requests 32 \
  --max-requests-per-second 100 \
  --max-request-body-mb 10 \
  --context-pool-size 2
```

## Health Checks

```bash
# Health endpoint
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics

# Ollama-compatible status
curl http://localhost:8080/api/ps
```
