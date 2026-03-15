---
title: Getting Started
description: Get up and running with Mullama, the high-performance LLM inference engine with Node.js, Python, Rust, Go, PHP, and C/C++ bindings, plus an Ollama-compatible CLI and daemon.
---

# Getting Started

Get up and running with Mullama in under 5 minutes. Choose the path that matches your use case.

---

## Choose Your Path

<div class="grid cards" markdown>

-   :material-code-braces: **Library Developer**

    ---

    Use Mullama from Node.js, Python, Rust, Go, PHP, or C/C++ for native LLM inference in your applications.

    **1.** Install the package for your language<br>
    **2.** Download a model (`mullama pull llama3.2:1b`)<br>
    **3.** Load and generate in 5 lines of code

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-console: **CLI / Server User**

    ---

    Use Mullama as a local AI server with an OpenAI-compatible API, model management, and TUI chat.

    **1.** Install the daemon (`cargo install mullama --features daemon`)<br>
    **2.** Pull a model (`mullama pull llama3.2:1b`)<br>
    **3.** Start serving (`mullama serve --port 8080`)

    [:octicons-arrow-right-24: Daemon & CLI](../daemon/index.md)

-   :material-swap-horizontal: **Migrating from Ollama**

    ---

    Mullama is a drop-in replacement for Ollama with better performance, embeddable library bindings, and full API compatibility.

    **1.** Install Mullama (same commands, same API)<br>
    **2.** Use your existing Modelfiles and scripts<br>
    **3.** Access native bindings for deeper integration

    [:octicons-arrow-right-24: Migration Guide](../comparison/migration-from-ollama.md)

</div>

---

## Quick Start

=== "Node.js"

    ```bash
    npm install mullama
    ```

    ```javascript title="index.js"
    const { Model } = require('mullama');

    async function main() {
        const model = new Model('models/llama3.2-1b.gguf', { contextSize: 2048 });
        const response = await model.generate('Explain quantum computing in one sentence.', {
            maxTokens: 100
        });
        console.log(response);
    }

    main();
    ```

=== "Python"

    ```bash
    pip install mullama
    ```

    ```python title="main.py"
    from mullama import Model

    model = Model("models/llama3.2-1b.gguf", context_size=2048)
    response = model.generate("Explain quantum computing in one sentence.", max_tokens=100)
    print(response)
    ```

=== "Rust"

    ```toml title="Cargo.toml"
    [dependencies]
    mullama = { version = "0.1.1", features = ["async", "streaming"] }
    tokio = { version = "1", features = ["full"] }
    ```

    ```rust title="src/main.rs"
    use mullama::prelude::*;

    #[tokio::main]
    async fn main() -> Result<(), MullamaError> {
        let model = ModelBuilder::new()
            .path("models/llama3.2-1b.gguf")
            .context_size(2048)
            .build().await?;

        let response = model.generate("Explain quantum computing in one sentence.", 100).await?;
        println!("{}", response);
        Ok(())
    }
    ```

=== "Go"

    ```bash
    go get github.com/neul-labs/mullama-go
    ```

    ```go title="main.go"
    package main

    import (
        "fmt"
        "github.com/neul-labs/mullama-go"
    )

    func main() {
        model, _ := mullama.NewModel("models/llama3.2-1b.gguf", mullama.WithContextSize(2048))
        defer model.Close()

        response, _ := model.Generate("Explain quantum computing in one sentence.", mullama.MaxTokens(100))
        fmt.Println(response)
    }
    ```

=== "PHP"

    ```bash
    composer require neul-labs/mullama
    ```

    ```php title="main.php"
    <?php
    require_once 'vendor/autoload.php';
    use Mullama\Model;

    $model = new Model('models/llama3.2-1b.gguf', contextSize: 2048);
    $response = $model->generate('Explain quantum computing in one sentence.', maxTokens: 100);
    echo $response . "\n";
    ```

=== "C/C++"

    ```c title="main.c"
    #include "mullama.h"

    int main() {
        mullama_config_t config = mullama_config_default();
        config.context_size = 2048;

        mullama_model_t* model = mullama_model_load("models/llama3.2-1b.gguf", &config);
        char* response = mullama_generate(model, "Explain quantum computing in one sentence.", 100);
        printf("%s\n", response);

        mullama_free(response);
        mullama_model_free(model);
        return 0;
    }
    ```

=== "CLI"

    ```bash
    # Install the daemon
    cargo install mullama --features daemon

    # Pull and run
    mullama pull llama3.2:1b
    mullama run llama3.2:1b "Explain quantum computing in one sentence."
    ```

---

## Download a Model

You need a GGUF model file to use Mullama. There are several ways to obtain one.

### Using the CLI (Recommended)

```bash
# Pull a model by alias (auto-downloads from registry)
mullama pull llama3.2:1b

# List downloaded models
mullama list

# Show model details
mullama show llama3.2:1b --modelfile
```

### From HuggingFace

Download GGUF models directly from HuggingFace repositories:

```bash
# Using wget
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -O models/llama3.2-1b.gguf

# Using the HuggingFace CLI
pip install huggingface-hub
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models/
```

### Direct Download

For air-gapped environments, download the `.gguf` file directly and place it in your models directory:

```bash
mkdir -p models/
# Copy or download your .gguf file into this directory
ls models/*.gguf
```

---

## Model Aliases

The Mullama daemon supports convenient aliases that map to specific model files and quantizations:

| Alias | Model | Quantization | Size | Best For |
|-------|-------|-------------|------|----------|
| `llama3.2:1b` | Llama 3.2 1B Instruct | Q4_K_M | ~1.3 GB | Fast prototyping, edge devices |
| `llama3.2:3b` | Llama 3.2 3B Instruct | Q4_K_M | ~2.0 GB | Balanced speed/quality |
| `qwen2.5:7b` | Qwen 2.5 7B Instruct | Q4_K_M | ~4.4 GB | Multilingual, coding |
| `deepseek-r1:7b` | DeepSeek R1 7B | Q4_K_M | ~4.5 GB | Reasoning, chain-of-thought |
| `phi-4:14b` | Phi-4 14B | Q4_K_M | ~8.4 GB | Strong reasoning, compact |
| `mistral:7b` | Mistral 7B Instruct | Q4_K_M | ~4.1 GB | General purpose |
| `codellama:7b` | Code Llama 7B | Q4_K_M | ~4.1 GB | Code generation |
| `llama3.1:8b` | Llama 3.1 8B Instruct | Q4_K_M | ~4.9 GB | Long context (128K) |
| `gemma2:9b` | Gemma 2 9B | Q4_K_M | ~5.5 GB | Google's latest |
| `llama3.3:70b` | Llama 3.3 70B Instruct | Q4_K_M | ~40 GB | Maximum quality |

!!! tip "Quantization Levels"

    GGUF models come in various quantization levels that trade quality for size:

    - **Q4_K_M** -- Best balance of quality and size (recommended for most users)
    - **Q5_K_M** -- Higher quality, ~20% larger than Q4
    - **Q6_K** -- Near-original quality, good for critical applications
    - **Q8_0** -- Highest practical quality, 2x size of Q4
    - **F16** -- Full precision, largest files (usually unnecessary)

!!! info "Custom Models"

    You can create custom model configurations using Modelfiles:

    ```bash
    mullama create my-assistant -f Modelfile
    mullama run my-assistant "Hello!"
    ```

    See the [Modelfile Format](../daemon/modelfile.md) documentation for details.

---

## Prerequisites

!!! info "System Requirements"

    - **RAM:** 8 GB minimum (16 GB recommended for 7B models)
    - **Disk:** 2-50 GB depending on model size
    - **OS:** Linux (x86_64, aarch64), macOS (Apple Silicon, Intel), Windows (x86_64)
    - **Rust:** 1.75+ (for building from source or using the Rust library)
    - **CMake:** 3.12+ (for building llama.cpp)
    - **C++ Compiler:** GCC 9+, Clang 10+, or MSVC 2019+ (C++17 support required)

---

## Next Steps

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    ---

    Detailed installation for all languages, feature flags, and build options.

-   :material-monitor: **[Platform Setup](platform-setup.md)**

    ---

    Install OS-specific dependencies for audio, image, and video processing.

-   :material-gpu: **[GPU Acceleration](gpu.md)**

    ---

    Configure CUDA, Metal, ROCm, or OpenCL for faster inference.

-   :material-rocket-launch: **[Your First Project](first-project.md)**

    ---

    Build a complete chatbot from scratch with streaming and multi-turn conversation.

-   :material-book-open-variant: **[Library Guide](../guide/index.md)**

    ---

    Deep dive into models, contexts, sampling, streaming, and more.

-   :material-swap-horizontal: **[Ollama Migration](../comparison/migration-from-ollama.md)**

    ---

    Switch from Ollama with zero code changes using API compatibility.

</div>
