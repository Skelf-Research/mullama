---
title: "Tutorial: Edge Deployment"
description: Deploy Mullama on resource-constrained devices like Raspberry Pi, Jetson Nano, and Intel NUC with optimized configuration and system integration.
---

# Edge Deployment

Deploy Mullama on resource-constrained devices for offline, private AI inference. This tutorial covers hardware selection, model optimization, memory management, service configuration, and monitoring.

---

## What You'll Build

An edge inference system that:

- Runs quantized models on limited hardware (2-8 GB RAM)
- Optimizes CPU-only inference with tuned thread counts
- Minimizes memory usage with small contexts and KV cache quantization
- Starts automatically via systemd service
- Serves an API for local network access
- Monitors resource usage on constrained hardware

---

## Prerequisites

- A supported edge device (see hardware table below)
- Linux (Ubuntu/Debian or Raspberry Pi OS)
- Python 3.8+ (`pip install mullama`)
- A small quantized GGUF model (Q4_K_M or Q2_K)

---

## Supported Hardware

| Device | RAM | CPU | Expected Speed | Best Model Size |
|--------|-----|-----|----------------|-----------------|
| Raspberry Pi 5 | 8 GB | Cortex-A76 (4c) | 3-8 tok/s | 1B-3B Q4_K_M |
| Raspberry Pi 4 | 4-8 GB | Cortex-A72 (4c) | 1-4 tok/s | 1B Q4_K_M |
| Jetson Nano | 4 GB | Cortex-A57 (4c) + GPU | 5-15 tok/s | 1B-3B Q4_K_M |
| Intel NUC i5 | 16 GB | i5-1240P (12c) | 15-30 tok/s | 3B-7B Q4_K_M |
| Intel NUC i3 | 8 GB | i3-1115G4 (4c) | 8-15 tok/s | 1B-3B Q4_K_M |
| Orange Pi 5 | 8 GB | RK3588 (8c) | 4-10 tok/s | 1B-3B Q4_K_M |

---

## Step 1: Model Selection for Edge

Choose the right model size and quantization for your hardware.

### Quantization Levels

| Quantization | Size (1B model) | Size (3B model) | Quality | Speed |
|--------------|-----------------|-----------------|---------|-------|
| Q2_K | 0.4 GB | 1.2 GB | Low | Fastest |
| Q3_K_M | 0.5 GB | 1.5 GB | Medium-Low | Fast |
| Q4_K_M | 0.7 GB | 2.0 GB | Medium | Good |
| Q5_K_M | 0.8 GB | 2.4 GB | Medium-High | Moderate |
| Q6_K | 0.9 GB | 2.8 GB | High | Slower |
| Q8_0 | 1.1 GB | 3.3 GB | Very High | Slowest |

### Model Size Guidelines

```
Available RAM - OS overhead (1 GB) - Context memory = Max model size

Example: Raspberry Pi 5 (8 GB)
  8 GB - 1 GB (OS) - 0.5 GB (context) = 6.5 GB max model
  Recommended: 1B-3B models in Q4_K_M (0.7-2.0 GB)
```

=== "Python"
    ```python
    # Download a small, optimized model for edge
    # Option 1: Via daemon
    # mullama pull llama3.2:1b

    # Option 2: Direct download (if daemon not available)
    import urllib.request
    model_url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    model_path = "/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf"

    print(f"Downloading model to {model_path}...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Done!")
    ```

=== "Bash"
    ```bash
    # Create model directory
    sudo mkdir -p /opt/mullama/models

    # Download a small model (Q4_K_M quantization, ~700 MB for 1B model)
    wget -O /opt/mullama/models/llama3.2-1b-Q4_K_M.gguf \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    # For extremely constrained devices, use Q2_K (~400 MB)
    wget -O /opt/mullama/models/llama3.2-1b-Q2_K.gguf \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q2_K.gguf"
    ```

---

## Step 2: Memory Optimization

Configure Mullama for minimal memory footprint.

=== "Python"
    ```python
    from mullama import Model, Context, SamplerParams

    # Edge-optimized model loading
    model = Model.load(
        "/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf",
        n_gpu_layers=0,      # CPU only (no GPU on most edge devices)
        use_mmap=True,        # Memory-map the model file (reduces RSS)
        use_mlock=False,      # Don't lock in RAM (allow OS to page)
    )

    # Small context to minimize KV cache memory
    ctx = Context(
        model,
        n_ctx=512,            # Small context window (saves ~200 MB vs 4096)
        n_batch=256,          # Smaller batch for lower memory peak
        n_threads=4,          # Match physical core count
    )

    print(f"Model: {model.name}")
    print(f"Size: {model.size / 1e6:.0f} MB")
    print(f"Context: {ctx.n_ctx} tokens")
    ```

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');

    // Edge-optimized model loading
    const model = JsModel.load('/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf', {
        nGpuLayers: 0,       // CPU only
        useMmap: true,        // Memory-map the model file
        useMlock: false,      // Don't lock in RAM
    });

    // Small context to minimize KV cache memory
    const ctx = new JsContext(model, {
        nCtx: 512,            // Small context window
        nBatch: 256,          // Smaller batch for lower memory peak
        nThreads: 4,          // Match physical core count
    });

    console.log(`Model: ${model.name}`);
    console.log(`Size: ${(model.size / 1e6).toFixed(0)} MB`);
    console.log(`Context: ${ctx.nCtx} tokens`);
    ```

### Memory Budget Breakdown

| Component | 512 ctx | 2048 ctx | 4096 ctx |
|-----------|---------|----------|----------|
| Model (1B Q4) | 700 MB | 700 MB | 700 MB |
| KV Cache | ~50 MB | ~200 MB | ~400 MB |
| Working memory | ~50 MB | ~100 MB | ~150 MB |
| **Total** | **~800 MB** | **~1000 MB** | **~1250 MB** |

---

## Step 3: CPU Optimization

Tune inference for CPU-bound execution.

=== "Python"
    ```python
    import os
    import multiprocessing

    def get_optimal_threads():
        """Determine optimal thread count for edge device."""
        physical_cores = multiprocessing.cpu_count()

        # Use physical cores only (not hyperthreads)
        # On ARM (RPi), all cores are physical
        # On x86, divide by 2 for hyperthreading
        import platform
        if platform.machine() in ('x86_64', 'AMD64'):
            physical_cores = max(1, physical_cores // 2)

        return physical_cores

    def create_edge_context(model, context_size=512):
        """Create an optimized context for edge inference."""
        n_threads = get_optimal_threads()

        ctx = Context(
            model,
            n_ctx=context_size,
            n_batch=min(context_size, 256),  # Batch <= context
            n_threads=n_threads,
        )

        print(f"Edge context: {context_size} ctx, {n_threads} threads, "
              f"batch={min(context_size, 256)}")
        return ctx

    # Usage
    ctx = create_edge_context(model, context_size=512)
    ```

### Thread Count Guidelines

| Device | Physical Cores | Recommended Threads | Notes |
|--------|----------------|---------------------|-------|
| RPi 5 | 4 | 4 | All cores, no HT |
| RPi 4 | 4 | 4 | All cores, no HT |
| Jetson Nano | 4 | 4 | CPU cores only |
| Intel NUC i5 | 4P+8E | 4-8 | Performance cores preferred |
| Intel NUC i3 | 2+2HT | 2 | Physical cores only |

!!! warning "Over-threading"
    Setting threads higher than physical core count usually hurts performance on edge devices due to cache thrashing and context switching overhead. Always benchmark with different thread counts.

---

## Step 4: Systemd Service

Set up automatic start on boot.

=== "Bash"
    ```bash
    # Create a dedicated user
    sudo useradd -r -s /bin/false mullama

    # Set ownership
    sudo chown -R mullama:mullama /opt/mullama

    # Create the service file
    sudo tee /etc/systemd/system/mullama-edge.service << 'EOF'
    [Unit]
    Description=Mullama Edge Inference Server
    After=network.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=mullama
    Group=mullama
    WorkingDirectory=/opt/mullama

    # Environment
    Environment=MODEL_PATH=/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf
    Environment=PORT=8080
    Environment=N_CTX=512
    Environment=N_THREADS=4

    # Run the server
    ExecStart=/usr/bin/python3 /opt/mullama/server.py

    # Resource limits
    MemoryMax=2G
    CPUQuota=100%

    # Restart on failure
    Restart=on-failure
    RestartSec=10

    # Security hardening
    NoNewPrivileges=yes
    ProtectSystem=strict
    ReadWritePaths=/opt/mullama/logs

    [Install]
    WantedBy=multi-user.target
    EOF

    # Enable and start
    sudo systemctl daemon-reload
    sudo systemctl enable mullama-edge
    sudo systemctl start mullama-edge

    # Check status
    sudo systemctl status mullama-edge
    ```

---

## Step 5: Edge API Server

A lightweight API server optimized for edge hardware.

=== "Python"
    ```python
    #!/usr/bin/env python3
    """Mullama Edge API Server - optimized for resource-constrained devices."""

    import os, time, json
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from mullama import Model, Context, SamplerParams

    # --- Configuration from environment ---
    MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf")
    PORT = int(os.environ.get("PORT", 8080))
    N_CTX = int(os.environ.get("N_CTX", 512))
    N_THREADS = int(os.environ.get("N_THREADS", 4))

    # --- Load model ---
    print(f"Loading model: {MODEL_PATH}")
    model = Model.load(MODEL_PATH, n_gpu_layers=0, use_mmap=True)
    ctx = Context(model, n_ctx=N_CTX, n_batch=256, n_threads=N_THREADS)
    print(f"Ready: {model.name} | ctx={N_CTX} | threads={N_THREADS}")

    class EdgeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {
                    "status": "ok", "model": model.name or "unknown",
                    "context_size": N_CTX, "threads": N_THREADS,
                })
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path == "/generate":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}

                prompt = body.get("prompt", "")
                max_tokens = min(body.get("max_tokens", 100), N_CTX - 50)
                temperature = body.get("temperature", 0.7)

                if not prompt:
                    return self.send_json(400, {"error": "prompt required"})

                start = time.time()
                params = SamplerParams(temperature=temperature)
                text = ctx.generate(prompt, max_tokens=max_tokens, params=params)
                elapsed = time.time() - start
                tokens = len(model.tokenize(text, add_bos=False))
                ctx.clear_cache()

                self.send_json(200, {
                    "text": text.strip(),
                    "tokens": tokens,
                    "time_ms": int(elapsed * 1000),
                    "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0,
                })
            else:
                self.send_json(404, {"error": "Not found"})

        def send_json(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format, *args):
            pass  # Suppress default logging for performance

    if __name__ == "__main__":
        server = HTTPServer(("0.0.0.0", PORT), EdgeHandler)
        print(f"Edge server listening on port {PORT}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
    ```

=== "Node.js"
    ```javascript
    const http = require('http');
    const { JsModel, JsContext } = require('mullama');

    const MODEL_PATH = process.env.MODEL_PATH || '/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf';
    const PORT = parseInt(process.env.PORT || '8080');
    const N_CTX = parseInt(process.env.N_CTX || '512');
    const N_THREADS = parseInt(process.env.N_THREADS || '4');

    console.log(`Loading model: ${MODEL_PATH}`);
    const model = JsModel.load(MODEL_PATH, { nGpuLayers: 0, useMmap: true });
    const ctx = new JsContext(model, { nCtx: N_CTX, nBatch: 256, nThreads: N_THREADS });
    console.log(`Ready: ${model.name} | ctx=${N_CTX} | threads=${N_THREADS}`);

    const server = http.createServer((req, res) => {
        if (req.method === 'GET' && req.url === '/health') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'ok', model: model.name, context: N_CTX }));
            return;
        }

        if (req.method === 'POST' && req.url === '/generate') {
            let body = '';
            req.on('data', chunk => body += chunk);
            req.on('end', () => {
                const { prompt, max_tokens = 100, temperature = 0.7 } = JSON.parse(body || '{}');
                if (!prompt) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'prompt required' }));
                    return;
                }

                const start = Date.now();
                const text = ctx.generate(prompt, Math.min(max_tokens, N_CTX - 50), { temperature });
                const elapsed = (Date.now() - start) / 1000;
                const tokens = model.tokenize(text, false).length;
                ctx.clearCache();

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    text: text.trim(), tokens, time_ms: Math.round(elapsed * 1000),
                    tokens_per_sec: tokens / elapsed
                }));
            });
            return;
        }

        res.writeHead(404);
        res.end('Not found');
    });

    server.listen(PORT, () => console.log(`Edge server on port ${PORT}`));
    ```

---

## Step 6: Power and Thermal Considerations

Edge devices have power and thermal limits. Configure accordingly.

| Device | TDP | Idle Power | Inference Power | Notes |
|--------|-----|-----------|-----------------|-------|
| RPi 5 | 12W | 3W | 8-10W | Active cooling recommended |
| RPi 4 | 6W | 2.5W | 5-6W | Passive cooling sufficient |
| Jetson Nano | 10W | 3W | 8-10W | 5W mode available |
| Intel NUC | 28W | 5W | 15-25W | Fan-cooled |

```bash
# Monitor CPU temperature (Linux)
watch -n 1 cat /sys/class/thermal/thermal_zone0/temp

# Limit CPU frequency if overheating (RPi)
echo 1500000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

# Monitor power usage (if supported)
vcgencmd measure_volts core
vcgencmd get_throttled
```

---

## Step 7: Monitoring

Monitor resource usage on constrained hardware.

=== "Python"
    ```python
    import os, time, threading

    class EdgeMonitor:
        def __init__(self, interval=5):
            self.interval = interval
            self.running = False
            self._thread = None

        def start(self):
            self.running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

        def stop(self):
            self.running = False

        def _monitor_loop(self):
            while self.running:
                stats = self.get_stats()
                if stats["memory_percent"] > 85:
                    print(f"WARNING: Memory at {stats['memory_percent']:.0f}%")
                if stats.get("cpu_temp", 0) > 80:
                    print(f"WARNING: CPU temp at {stats['cpu_temp']:.0f}C")
                time.sleep(self.interval)

        def get_stats(self):
            # Memory
            with open("/proc/meminfo") as f:
                meminfo = dict(line.split(":") for line in f.read().strip().split("\n"))
            total = int(meminfo["MemTotal"].strip().split()[0]) / 1024
            available = int(meminfo["MemAvailable"].strip().split()[0]) / 1024
            used = total - available

            stats = {
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_percent": used / total * 100,
            }

            # CPU temperature
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    stats["cpu_temp"] = int(f.read().strip()) / 1000
            except FileNotFoundError:
                pass

            # Load average
            load1, load5, load15 = os.getloadavg()
            stats["load_1m"] = load1
            stats["load_5m"] = load5

            return stats

    # Usage
    monitor = EdgeMonitor(interval=10)
    monitor.start()
    ```

=== "Bash"
    ```bash
    #!/bin/bash
    # edge-monitor.sh - Simple resource monitoring for edge devices

    while true; do
        TIMESTAMP=$(date +%H:%M:%S)
        MEM_USED=$(free -m | awk '/Mem:/ {print $3}')
        MEM_TOTAL=$(free -m | awk '/Mem:/ {print $2}')
        MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
        LOAD=$(cat /proc/loadavg | cut -d' ' -f1)

        # CPU temperature (RPi/ARM)
        TEMP="N/A"
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            TEMP=$(echo "scale=1; $(cat /sys/class/thermal/thermal_zone0/temp)/1000" | bc)
        fi

        echo "[$TIMESTAMP] Mem: ${MEM_USED}/${MEM_TOTAL}MB (${MEM_PCT}%) | Load: ${LOAD} | Temp: ${TEMP}C"

        # Alert on high usage
        if [ $MEM_PCT -gt 90 ]; then
            echo "ALERT: Memory critical!"
        fi

        sleep 5
    done
    ```

---

## Complete Deployment Script

```bash
#!/bin/bash
# deploy-edge.sh - Complete edge deployment setup

set -e

echo "=== Mullama Edge Deployment ==="

# 1. Install dependencies
echo "[1/6] Installing dependencies..."
sudo apt update -qq
sudo apt install -y python3 python3-pip
pip3 install mullama --break-system-packages 2>/dev/null || pip3 install mullama

# 2. Create directories
echo "[2/6] Setting up directories..."
sudo mkdir -p /opt/mullama/{models,logs}

# 3. Download model
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
MODEL_PATH="/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[3/6] Downloading model (this may take a while)..."
    wget -q --show-progress -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "[3/6] Model already exists, skipping download."
fi

# 4. Deploy server script
echo "[4/6] Deploying server..."
cat > /opt/mullama/server.py << 'PYEOF'
#!/usr/bin/env python3
import os, time, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from mullama import Model, Context, SamplerParams

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/mullama/models/llama3.2-1b-Q4_K_M.gguf")
PORT = int(os.environ.get("PORT", 8080))
N_CTX = int(os.environ.get("N_CTX", 512))
N_THREADS = int(os.environ.get("N_THREADS", 4))

model = Model.load(MODEL_PATH, n_gpu_layers=0, use_mmap=True)
ctx = Context(model, n_ctx=N_CTX, n_batch=256, n_threads=N_THREADS)
print(f"Ready: {model.name} | ctx={N_CTX} | threads={N_THREADS} | port={PORT}")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ok", "model": model.name})
        else:
            self.respond(404, {"error": "not found"})
    def do_POST(self):
        if self.path == "/generate":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = body.get("prompt", "")
            if not prompt:
                return self.respond(400, {"error": "prompt required"})
            start = time.time()
            text = ctx.generate(prompt, max_tokens=min(body.get("max_tokens", 100), N_CTX-50),
                                params=SamplerParams(temperature=body.get("temperature", 0.7)))
            elapsed = time.time() - start
            ctx.clear_cache()
            self.respond(200, {"text": text.strip(), "time_ms": int(elapsed*1000)})
        else:
            self.respond(404, {"error": "not found"})
    def respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    def log_message(self, *args): pass

HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
PYEOF

# 5. Create systemd service
echo "[5/6] Creating systemd service..."
sudo tee /etc/systemd/system/mullama-edge.service > /dev/null << EOF
[Unit]
Description=Mullama Edge Server
After=network.target

[Service]
Type=simple
User=$(whoami)
Environment=MODEL_PATH=$MODEL_PATH
Environment=PORT=8080
Environment=N_CTX=512
Environment=N_THREADS=$(nproc)
ExecStart=/usr/bin/python3 /opt/mullama/server.py
Restart=on-failure
RestartSec=10
MemoryMax=2G

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mullama-edge
sudo systemctl start mullama-edge

# 6. Verify
echo "[6/6] Verifying deployment..."
sleep 3
if curl -s http://localhost:8080/health | grep -q "ok"; then
    echo ""
    echo "=== Deployment successful! ==="
    echo "  Health: curl http://localhost:8080/health"
    echo "  Generate: curl -X POST http://localhost:8080/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello!\"}'"
    echo "  Logs: journalctl -u mullama-edge -f"
else
    echo "ERROR: Server not responding. Check: journalctl -u mullama-edge"
    exit 1
fi
```

---

## Latency Expectations

Realistic performance for different configurations:

| Config | Prompt (50 tok) | Generation (100 tok) | Total |
|--------|-----------------|---------------------|-------|
| RPi 5, 1B Q4 | ~2s | ~15s | ~17s |
| RPi 5, 1B Q2 | ~1.5s | ~12s | ~13.5s |
| Jetson Nano, 1B Q4 | ~1s | ~10s | ~11s |
| NUC i5, 3B Q4 | ~1s | ~5s | ~6s |
| NUC i5, 1B Q4 | ~0.5s | ~3s | ~3.5s |

!!! tip "Reducing Latency"
    - Use Q2_K for faster inference (slight quality trade-off)
    - Reduce `max_tokens` to minimum needed
    - Keep prompts short (fewer tokens to process)
    - Use greedy sampling (`temperature=0`) to avoid sampling overhead
    - Pre-warm the model with a dummy generation on startup

---

## What's Next

- [API Server](api-server.md) -- More advanced server patterns
- [Streaming Generation](streaming.md) -- Stream responses for perceived speed
- [Batch Processing](batch.md) -- Optimize throughput on edge
- [Guide: Models](../guide/models.md) -- Model selection and quantization deep dive
