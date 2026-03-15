---
title: CLI Reference
description: Complete command-line reference for the mullama binary
---

# CLI Reference

The `mullama` binary serves as both the daemon server and the client CLI. All commands follow the pattern:

```
mullama <command> [options] [arguments]
```

## Commands Overview

### By Category

=== "Running"

    | Command | Description |
    |---------|-------------|
    | `run` | One-shot text generation (auto-spawns daemon) |
    | `chat` | Interactive TUI chat client |

=== "Serving"

    | Command | Description |
    |---------|-------------|
    | `serve` | Start the daemon server (foreground) |

=== "Model Management"

    | Command | Description |
    |---------|-------------|
    | `pull` | Download a model from HuggingFace |
    | `list` | List all local models |
    | `show` | Show model details |
    | `create` | Create a custom model from a Modelfile |
    | `cp` | Copy/rename a model |
    | `rm` | Remove a model from disk |

=== "Daemon Lifecycle"

    | Command | Description |
    |---------|-------------|
    | `daemon start` | Start daemon as background process |
    | `daemon stop` | Stop the running daemon |
    | `daemon status` | Show daemon status |
    | `daemon restart` | Restart the daemon |
    | `daemon logs` | View daemon log output |

=== "Monitoring"

    | Command | Description |
    |---------|-------------|
    | `ps` | Show running (loaded) models |
    | `status` | Show daemon status |
    | `cache` | Manage the model cache |
    | `ping` | Ping the daemon |

=== "Utility"

    | Command | Description |
    |---------|-------------|
    | `load` | Load a model into the daemon |
    | `unload` | Unload a model from memory |
    | `search` | Search HuggingFace for models |
    | `info` | Show HuggingFace repository details |
    | `tokenize` | Tokenize text using a model |
    | `embed` | Generate embeddings for text |
    | `stop` | Shutdown the daemon |

---

## Server Commands

### `mullama serve`

Start the daemon server in the foreground. The server provides both IPC and HTTP interfaces.

```bash
mullama serve [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <SPEC>` | -- | Model(s) to load on startup (repeatable) |
| `--mmproj <PATH>` | -- | Multimodal projector path (for vision models) |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket address |
| `-p, --http-port <PORT>` | `8080` | HTTP port (0 to disable HTTP) |
| `--http-addr <ADDR>` | `0.0.0.0` | HTTP bind address |
| `-g, --gpu-layers <N>` | `0` | Default GPU layers to offload |
| `-c, --context-size <N>` | `4096` | Default context size (tokens) |
| `-t, --threads <N>` | `num_cpus / 2` | CPU threads per model |
| `-v, --verbose` | -- | Verbose output |

**Aliases:** `start`

**Examples:**

```bash
# Start with a single model
mullama serve --model llama3.2:1b

# Start with GPU acceleration and custom port
mullama serve --model llama3.2:1b -g 35 -p 9090

# Start with multiple models
mullama serve --model llama3.2:1b --model qwen2.5:7b --model deepseek-r1:7b

# Start a vision model with projector
mullama serve --model llava:7b --mmproj ./mmproj-model-f16.gguf

# Start with a local GGUF file (alias:path format)
mullama serve --model my-model:./path/to/model.gguf

# Start with HuggingFace model
mullama serve --model hf:bartowski/Llama-3.2-1B-Instruct-GGUF

# Localhost-only binding (no external access)
mullama serve --model llama3.2:1b --http-addr 127.0.0.1

# IPC-only mode (no HTTP server)
mullama serve --model llama3.2:1b --http-port 0

# Large context with many threads
mullama serve --model qwen2.5:7b --context-size 32768 --threads 16
```

!!! info "Foreground vs Background"
    `mullama serve` runs in the foreground and logs to stderr. Use `mullama daemon start` for background operation with log file output.

---

## Generation Commands

### `mullama run`

One-shot text generation. Auto-spawns the daemon if it is not running.

```bash
mullama run [MODEL] <PROMPT> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL` | No | Model alias or path (uses daemon default if omitted) |
| `PROMPT` | Yes | The prompt text to send |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <MODEL>` | daemon default | Model to use (alternative to positional) |
| `-n, --max-tokens <N>` | `256` | Maximum tokens to generate |
| `-t, --temperature <F>` | `0.7` | Sampling temperature (0.0-2.0) |
| `--top-p <F>` | -- | Nucleus sampling threshold (0.0-1.0) |
| `--top-k <N>` | -- | Top-k sampling candidates |
| `--repeat-penalty <F>` | -- | Repetition penalty factor |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket address |
| `-i, --image <PATH>` | -- | Image file for vision models |
| `--http-port <PORT>` | `8080` | HTTP port for vision requests |
| `--system <PROMPT>` | -- | System prompt |
| `--stats` | -- | Show generation statistics |
| `--no-stream` | -- | Disable streaming (wait for full response) |
| `--json` | -- | Output response as JSON |

**Examples:**

```bash
# Basic generation with model specified
mullama run llama3.2:1b "What is the capital of France?"

# Using default model
mullama run "Explain photosynthesis"

# With specific model and parameters
mullama run --model qwen2.5:7b -n 512 -t 0.9 "Write a poem about Rust"

# With system prompt
mullama run llama3.2:1b --system "You are a pirate" "Tell me about the ocean"

# Vision model with image
mullama run --model llava:7b --image photo.jpg "Describe this image in detail"

# Low temperature for deterministic output
mullama run --model deepseek-r1:7b -t 0.1 "What is 15 * 23?"

# Show performance stats after generation
mullama run --stats llama3.2:1b "Hello, world!"

# JSON output for scripting
mullama run --json llama3.2:1b "List 3 colors" | jq .content
```

!!! tip "Auto-Spawn"
    If the daemon is not running, `mullama run` will automatically start it in the background before sending the request.

### `mullama chat`

Launch the interactive TUI chat client. See the [TUI Chat](tui.md) page for full details.

```bash
mullama chat [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <MODEL>` | daemon default | Initial model to use |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket address |
| `-t, --timeout <SECS>` | `10` | Connection timeout in seconds |
| `--system <PROMPT>` | -- | Initial system prompt |

**Aliases:** `tui`

**Examples:**

```bash
# Launch with default model
mullama chat

# Launch with specific model
mullama chat --model deepseek-r1:7b

# With custom system prompt
mullama chat --model qwen2.5:7b --system "You are a helpful coding assistant"

# Connect to non-default daemon
mullama chat --socket ipc:///var/run/mullama.sock
```

---

## Model Management Commands

### `mullama pull`

Download a model from HuggingFace.

```bash
mullama pull <SPEC> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `SPEC` | Model alias, HuggingFace spec (`hf:org/repo`), or `hf:org/repo:file.gguf` |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-q, --quiet` | -- | Suppress progress bar |
| `--force` | -- | Re-download even if cached |

**Aliases:** `download`

**Examples:**

```bash
# Download using alias
mullama pull llama3.2:1b
mullama pull qwen2.5:7b
mullama pull deepseek-r1:7b

# Download from HuggingFace (auto-detect best GGUF)
mullama pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF

# Download specific file from HuggingFace
mullama pull hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf

# Download specific quantization
mullama pull hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q5_K_M.gguf

# Quiet mode (for scripts)
mullama pull -q llama3.2:1b

# Force re-download
mullama pull --force llama3.2:1b
```

### `mullama list`

List all local models (cached HuggingFace downloads and custom models).

```bash
mullama list [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-v, --verbose` | -- | Show detailed information (path, repository) |
| `--json` | -- | Output as JSON |

**Aliases:** `ls`

**Example output:**

```
NAME                                      SIZE       QUANTIZATION   MODIFIED
Llama-3.2-1B-Instruct-GGUF:Q4_K_M        0.8 GB    Q4_K_M         2 days ago
qwen2.5-7b-instruct-q4_k_m               4.7 GB    Q4_K_M         5 hours ago
deepseek-r1-7b                            4.9 GB    Q4_K_M         1 hour ago
my-assistant                              0.8 GB    Q4_K_M         30 minutes ago

4 model(s), 11.2 GB total
```

**Verbose output:**

```
NAME                                      SIZE       PATH
Llama-3.2-1B-Instruct-GGUF:Q4_K_M        0.8 GB    ~/.cache/mullama/models/bartowski/...
  Repo: bartowski/Llama-3.2-1B-Instruct-GGUF
  Quantization: Q4_K_M
  Modified: 2025-01-20 14:30:00

qwen2.5-7b-instruct-q4_k_m               4.7 GB    ~/.cache/mullama/models/Qwen/...
  Repo: Qwen/Qwen2.5-7B-Instruct-GGUF
  Quantization: Q4_K_M
  Modified: 2025-01-23 09:15:00
```

### `mullama ps`

Show models currently loaded in the daemon.

```bash
mullama ps [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--json` | -- | Output as JSON |

**Example output:**

```
NAME                 SIZE       GPU          CONTEXT    ACTIVE
*llama3.2:1b         1236M      35 layers    4096       -
 qwen2.5:7b          7615M      CPU          4096       1 req
 deepseek-r1:7b      4900M      20 layers    8192       -

* = default model
3 model(s) loaded, 13.7 GB total
```

### `mullama show`

Show detailed information about a model.

```bash
mullama show <NAME> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--modelfile` | -- | Show the Modelfile/Mullamafile content |
| `--parameters` | -- | Show model parameters only |
| `--license` | -- | Show license information |
| `--json` | -- | Output as JSON |

**Examples:**

```bash
# Show model details
mullama show llama3.2:1b

# Show the Modelfile
mullama show my-assistant --modelfile

# Show parameters
mullama show my-assistant --parameters

# JSON output
mullama show llama3.2:1b --json
```

**Example output:**

```
Model: llama3.2:1b
  Family:       llama
  Parameters:   1.24B
  Quantization: Q4_K_M
  Size:         0.8 GB
  Context:      4096
  Repository:   bartowski/Llama-3.2-1B-Instruct-GGUF
  Path:         ~/.cache/mullama/models/bartowski/Llama-3.2-1B-Instruct-GGUF/...
  Modified:     2025-01-20 14:30:00
```

### `mullama create`

Create a custom model configuration from a Modelfile or Mullamafile.

```bash
mullama create <NAME> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --file <PATH>` | `./Mullamafile` or `./Modelfile` | Path to Modelfile |
| `--download` | `true` | Download base model if not cached |
| `-q, --quiet` | -- | Suppress progress output |

**Examples:**

```bash
# Create from Modelfile in current directory
mullama create my-assistant

# Create from specific file
mullama create my-coder -f ./coding-assistant.modelfile

# Create without auto-downloading base model
mullama create my-model -f Modelfile --download=false
```

### `mullama cp`

Copy or rename a custom model.

```bash
mullama cp <SOURCE> <DESTINATION>
```

**Aliases:** `copy`

**Examples:**

```bash
# Copy a model
mullama cp my-assistant my-assistant-v2

# Rename by copying and removing
mullama cp old-name new-name && mullama rm old-name
```

### `mullama rm`

Remove a model from disk.

```bash
mullama rm <NAME> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --force` | -- | Skip confirmation prompt |

**Aliases:** `delete`, `remove`

**Examples:**

```bash
# Remove with confirmation
mullama rm my-old-model

# Force remove (no confirmation)
mullama rm -f my-old-model
```

### `mullama load`

Load a model into the running daemon.

```bash
mullama load <SPEC> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-g, --gpu-layers <N>` | `0` | GPU layers to offload |
| `-c, --context-size <N>` | `4096` | Context size |
| `--mmproj <PATH>` | -- | Vision projector path |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--default` | -- | Set as default model |

**Examples:**

```bash
# Load by alias:path format
mullama load llama:./models/llama.gguf

# Load with GPU acceleration
mullama load llama:./models/llama.gguf -g 35

# Load with custom context
mullama load qwen:./models/qwen.gguf -c 8192

# Load and set as default
mullama load llama3.2:1b --default

# Load a vision model
mullama load llava:./llava.gguf --mmproj ./mmproj.gguf -g 35
```

### `mullama unload`

Unload a model from the daemon, freeing memory.

```bash
mullama unload <ALIAS> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |

**Examples:**

```bash
# Unload a model
mullama unload qwen2.5:7b

# Unload from custom socket
mullama unload llama3.2:1b -s ipc:///var/run/mullama.sock
```

---

## Discovery Commands

### `mullama search`

Search HuggingFace for GGUF models.

```bash
mullama search <QUERY> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --limit <N>` | `10` | Maximum results to show |
| `--all` | -- | Show all models (not just GGUF) |
| `-f, --files` | -- | Show available GGUF files per repository |

**Aliases:** `find`

**Examples:**

```bash
# Search for Llama models
mullama search "llama 7b"

# Show available quantizations for matches
mullama search "mistral instruct" --files

# Include non-GGUF results
mullama search "phi" --all

# Limit results
mullama search "deepseek" -n 5
```

**Example output:**

```
REPOSITORY                                    DOWNLOADS    UPDATED
bartowski/Llama-3.2-1B-Instruct-GGUF          125,432      3 days ago
bartowski/Llama-3.2-3B-Instruct-GGUF           89,201      3 days ago
bartowski/Meta-Llama-3.1-8B-Instruct-GGUF     312,445      2 weeks ago

3 result(s)
```

### `mullama info`

Show details about a HuggingFace repository.

```bash
mullama info <REPO>
```

**Examples:**

```bash
mullama info bartowski/Llama-3.2-1B-Instruct-GGUF
mullama info TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

**Example output:**

```
Repository: bartowski/Llama-3.2-1B-Instruct-GGUF
Description: GGUF quantizations of Meta Llama 3.2 1B Instruct
Downloads: 125,432
Last Updated: 2025-01-20

Available Files:
  Llama-3.2-1B-Instruct-Q2_K.gguf         (0.5 GB)
  Llama-3.2-1B-Instruct-Q3_K_M.gguf       (0.6 GB)
  Llama-3.2-1B-Instruct-Q4_K_M.gguf       (0.8 GB)  [recommended]
  Llama-3.2-1B-Instruct-Q5_K_M.gguf       (0.9 GB)
  Llama-3.2-1B-Instruct-Q6_K.gguf         (1.0 GB)
  Llama-3.2-1B-Instruct-Q8_0.gguf         (1.3 GB)
  Llama-3.2-1B-Instruct-f16.gguf          (2.5 GB)
```

---

## Daemon Management

### `mullama daemon start`

Start the daemon as a background process.

```bash
mullama daemon start [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <SPEC>` | -- | Model(s) to load (repeatable) |
| `-p, --http-port <PORT>` | `8080` | HTTP port |
| `--http-addr <ADDR>` | `0.0.0.0` | HTTP bind address |
| `-g, --gpu-layers <N>` | `0` | Default GPU layers |
| `-c, --context-size <N>` | `4096` | Default context size |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |

**Examples:**

```bash
# Start with defaults
mullama daemon start

# Start with a model pre-loaded
mullama daemon start --model llama3.2:1b

# Start with GPU and custom port
mullama daemon start --model llama3.2:1b -g 35 -p 9090

# Start with multiple models
mullama daemon start --model llama3.2:1b --model qwen2.5:7b
```

### `mullama daemon stop`

Stop the running daemon gracefully.

```bash
mullama daemon stop [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `-f, --force` | -- | Force stop (SIGKILL instead of graceful shutdown) |
| `--timeout <SECS>` | `5` | Graceful shutdown timeout |

**Examples:**

```bash
# Graceful stop
mullama daemon stop

# Force stop
mullama daemon stop --force

# Stop non-default daemon
mullama daemon stop -s ipc:///var/run/mullama.sock
```

### `mullama daemon restart`

Restart the daemon (stop then start with the same configuration).

```bash
mullama daemon restart [OPTIONS]
```

**Options:** Same as `daemon start`.

### `mullama daemon status`

Show daemon status information.

```bash
mullama daemon status [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--json` | -- | Output as JSON |

**Example output:**

```
Mullama Daemon Status
=====================
Running:     Yes
Version:     0.1.1
Uptime:      1h 23m 45s
Models:      2 loaded
Socket:      ipc:///tmp/mullama.sock
HTTP:        http://0.0.0.0:8080
Logs:        /tmp/mullamad.log
GPU:         CUDA available (35 layers offloaded)
Memory:      4.2 GB / 16.0 GB (26%)
Requests:    1,523 total
Tokens:      456,789 generated
```

### `mullama daemon logs`

View daemon log output.

```bash
mullama daemon logs [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --lines <N>` | `50` | Number of lines to show |
| `-f, --follow` | -- | Follow log output (like `tail -f`) |

**Examples:**

```bash
# Show last 50 lines
mullama daemon logs

# Show last 200 lines
mullama daemon logs -n 200

# Follow log output in real-time
mullama daemon logs -f
```

---

## Monitoring Commands

### `mullama status`

Show daemon status (shorthand for `daemon status`).

```bash
mullama status [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--json` | -- | Output as JSON |

### `mullama ping`

Ping the daemon to check connectivity.

```bash
mullama ping [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |

**Example output:**

```
Daemon is running (response time: 0.3ms)
```

### `mullama cache`

Manage the local model cache.

```bash
mullama cache <ACTION> [OPTIONS]
```

**Actions:**

| Action | Description |
|--------|-------------|
| `list [--verbose]` | List cached models with sizes |
| `show` | Show cache details (alias for `list --verbose`) |
| `path` | Show cache directory path |
| `size` | Show total cache size |
| `remove <REPO_ID> [--filename FILE]` | Remove specific cached model(s) |
| `clear [--force]` | Clear entire cache |

**Examples:**

```bash
# List cached models
mullama cache list

# Show detailed cache info
mullama cache show

# Show cache directory
mullama cache path

# Show total size
mullama cache size

# Remove specific model
mullama cache remove bartowski/Llama-3.2-1B-Instruct-GGUF

# Remove specific file
mullama cache remove bartowski/Llama-3.2-1B-Instruct-GGUF --filename Q4_K_M.gguf

# Clear all (with confirmation)
mullama cache clear

# Clear all (no confirmation)
mullama cache clear --force
```

---

## Utility Commands

### `mullama tokenize`

Tokenize text using a loaded model's tokenizer.

```bash
mullama tokenize <TEXT> [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <MODEL>` | daemon default | Model to use |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--json` | -- | Output as JSON |

**Example:**

```bash
$ mullama tokenize "Hello, world!"
Tokens: [15496, 11, 1917, 0]
Count: 4
```

### `mullama embed`

Generate embeddings for one or more texts.

```bash
mullama embed <TEXT>... [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <MODEL>` | daemon default | Model to use |
| `-s, --socket <ADDR>` | `ipc:///tmp/mullama.sock` | IPC socket |
| `--json` | -- | Output as JSON |
| `--dimensions` | -- | Show only the embedding dimensions |

**Examples:**

```bash
# Generate embeddings
mullama embed "Hello, world!"

# Multiple texts
mullama embed "First text" "Second text"

# JSON output
mullama embed --json "Hello" | jq .embedding[:5]

# Show dimensions only
mullama embed --dimensions "Hello"
```

### `mullama stop`

Shutdown the daemon (alias for `daemon stop`).

```bash
mullama stop [OPTIONS]
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MULLAMA_HOST` | Default HTTP bind address | `0.0.0.0` |
| `MULLAMA_PORT` | Default HTTP port | `8080` |
| `MULLAMA_MODELS_DIR` | Custom model storage directory | `~/.mullama/models` |
| `MULLAMA_CACHE_DIR` | Override model cache directory | Platform-specific |
| `MULLAMA_GPU_LAYERS` | Default GPU layers | `0` |
| `MULLAMA_CONTEXT_SIZE` | Default context size | `4096` |
| `MULLAMA_SOCKET` | Default IPC socket path | `ipc:///tmp/mullama.sock` |
| `MULLAMA_BIN` | Path to mullama binary (for auto-spawn) | Auto-detected |
| `MULLAMA_CONFIG` | Path to configuration file | `~/.mullama/config.yaml` |
| `MULLAMA_LOG_LEVEL` | Logging level (trace, debug, info, warn, error) | `info` |
| `HF_TOKEN` | HuggingFace API token for gated models | -- |

**Examples:**

```bash
# Set persistent environment
export MULLAMA_PORT=9090
export MULLAMA_GPU_LAYERS=35
export MULLAMA_MODELS_DIR=/mnt/models
export HF_TOKEN="hf_your_token_here"

# Then use normally
mullama serve --model llama3.2:1b
# Equivalent to: mullama serve --model llama3.2:1b --http-port 9090 --gpu-layers 35
```

!!! note "Precedence"
    CLI flags take precedence over environment variables, which take precedence over config file values.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Connection failed (daemon not reachable) |
| `3` | Model not found |
| `4` | Invalid arguments |
| `5` | Download failed |
| `6` | Timeout |
| `7` | Permission denied |

---

## Global Behavior

- **Auto-spawn**: Commands that require a daemon connection (`run`, `chat`, `ps`, `load`, `unload`, `tokenize`, `embed`) will automatically start the daemon if it is not running.
- **JSON output**: Most commands support `--json` for machine-readable output suitable for scripting.
- **IPC socket**: All client commands accept `-s, --socket` to connect to a non-default daemon instance.
- **Quiet mode**: Commands with progress output support `-q, --quiet` for script-friendly operation.
- **Tab completion**: Install shell completions with `mullama completions bash|zsh|fish`.

---

## Shell Completions

Generate shell completion scripts:

```bash
# Bash
mullama completions bash > /etc/bash_completion.d/mullama

# Zsh
mullama completions zsh > ~/.zfunc/_mullama

# Fish
mullama completions fish > ~/.config/fish/completions/mullama.fish
```

---

## Usage Patterns

### Scripting

```bash
#!/bin/bash
# Generate summaries for multiple files
for file in docs/*.md; do
    content=$(cat "$file")
    summary=$(mullama run --json --no-stream llama3.2:1b "Summarize: $content" | jq -r .content)
    echo "$file: $summary"
done
```

### Pipeline Integration

```bash
# Pipe input to mullama
echo "Translate to French: Hello, how are you?" | mullama run llama3.2:1b -

# Use with jq for structured output
mullama run --json qwen2.5:7b "List 5 programming languages as JSON array" | jq .content
```

### Multi-Instance

```bash
# Run two daemon instances on different ports
mullama daemon start --http-port 8080 --socket ipc:///tmp/mullama-1.sock --model llama3.2:1b
mullama daemon start --http-port 8081 --socket ipc:///tmp/mullama-2.sock --model qwen2.5:7b

# Send requests to specific instances
mullama run -s ipc:///tmp/mullama-1.sock "Hello from instance 1"
mullama run -s ipc:///tmp/mullama-2.sock "Hello from instance 2"
```
