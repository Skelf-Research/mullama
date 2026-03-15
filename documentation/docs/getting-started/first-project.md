---
title: Your First Project
description: Build a complete chatbot with Mullama from scratch. Step-by-step tutorial covering model loading, text generation, streaming output, multi-turn conversation, and deploying as an API server.
---

# Your First Project

Build a working chatbot from scratch using Mullama. This tutorial covers model loading, text generation, streaming output, multi-turn conversation, and deploying as an API.

---

## Prerequisites

Before starting, ensure you have:

- [x] Mullama installed ([Installation Guide](installation.md))
- [x] Platform dependencies installed ([Platform Setup](platform-setup.md))
- [x] A model downloaded (we will do this in Step 1)

!!! info "Time to Complete"

    This tutorial takes approximately 10-15 minutes to complete.

---

## Step 1: Download a Model

First, download a small model for development. The Llama 3.2 1B model is fast and requires minimal resources.

=== "Using the CLI"

    ```bash
    # Install the daemon if you have not already
    cargo install mullama --features daemon

    # Pull a small, fast model
    mullama pull llama3.2:1b

    # Verify it downloaded
    mullama list
    ```

=== "Direct Download"

    ```bash
    mkdir -p models/
    wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
        -O models/llama3.2-1b.gguf
    ```

!!! tip "Model Path"

    Throughout this tutorial, we use `models/llama3.2-1b.gguf` as the model path. If you used `mullama pull`, the model is stored in `~/.mullama/models/` -- adjust the path accordingly or create a symlink:

    ```bash
    ln -s ~/.mullama/models/llama3.2-1b.gguf models/llama3.2-1b.gguf
    ```

---

## Step 2: Create Your Project

=== "Node.js"

    ```bash
    mkdir my-chatbot && cd my-chatbot
    npm init -y
    npm install mullama
    ```

=== "Python"

    ```bash
    mkdir my-chatbot && cd my-chatbot
    python -m venv .venv
    source .venv/bin/activate
    pip install mullama
    ```

=== "Rust"

    ```bash
    cargo new my-chatbot && cd my-chatbot
    cargo add mullama --features "async,streaming"
    cargo add tokio --features full
    ```

---

## Step 3: Simple Generation

Start with a basic one-shot generation to verify everything works.

=== "Node.js"

    ```javascript title="index.js"
    const { Model } = require('mullama');

    async function main() {
        // Load the model
        console.log('Loading model...');
        const model = new Model('models/llama3.2-1b.gguf', {
            contextSize: 2048,
            nGpuLayers: -1  // Use GPU if available
        });
        console.log('Model loaded!');

        // Generate a response
        const response = await model.generate(
            'What is the capital of France? Answer in one sentence.',
            { maxTokens: 100 }
        );

        console.log('Response:', response);

        // Clean up
        model.close();
    }

    main().catch(console.error);
    ```

    Run it:

    ```bash
    node index.js
    ```

=== "Python"

    ```python title="main.py"
    from mullama import Model

    def main():
        # Load the model
        print("Loading model...")
        model = Model("models/llama3.2-1b.gguf",
            context_size=2048,
            n_gpu_layers=-1  # Use GPU if available
        )
        print("Model loaded!")

        # Generate a response
        response = model.generate(
            "What is the capital of France? Answer in one sentence.",
            max_tokens=100
        )

        print(f"Response: {response}")

    if __name__ == "__main__":
        main()
    ```

    Run it:

    ```bash
    python main.py
    ```

=== "Rust"

    ```rust title="src/main.rs"
    use mullama::prelude::*;

    #[tokio::main]
    async fn main() -> Result<(), MullamaError> {
        // Load the model
        println!("Loading model...");
        let model = ModelBuilder::new()
            .path("models/llama3.2-1b.gguf")
            .context_size(2048)
            .n_gpu_layers(-1)  // Use GPU if available
            .build().await?;
        println!("Model loaded!");

        // Generate a response
        let response = model.generate(
            "What is the capital of France? Answer in one sentence.",
            100
        ).await?;

        println!("Response: {}", response);
        Ok(())
    }
    ```

    Run it:

    ```bash
    cargo run
    ```

Expected output:

```
Loading model...
Model loaded!
Response: The capital of France is Paris, a city renowned for its art, culture, and history.
```

---

## Step 4: Add Streaming Output

Instead of waiting for the entire response, stream tokens as they are generated for a more interactive experience.

=== "Node.js"

    ```javascript title="streaming.js"
    const { Model } = require('mullama');

    async function main() {
        const model = new Model('models/llama3.2-1b.gguf', {
            contextSize: 2048,
            nGpuLayers: -1
        });

        console.log('Generating with streaming...\n');

        // Stream tokens one by one
        const stream = model.stream(
            'Write a haiku about programming.',
            { maxTokens: 100 }
        );

        for await (const token of stream) {
            process.stdout.write(token);
        }

        console.log('\n\nDone!');
        model.close();
    }

    main().catch(console.error);
    ```

    Run it:

    ```bash
    node streaming.js
    ```

=== "Python"

    ```python title="streaming.py"
    from mullama import Model

    def main():
        model = Model("models/llama3.2-1b.gguf",
            context_size=2048,
            n_gpu_layers=-1
        )

        print("Generating with streaming...\n")

        # Stream tokens one by one
        for token in model.stream(
            "Write a haiku about programming.",
            max_tokens=100
        ):
            print(token, end="", flush=True)

        print("\n\nDone!")

    if __name__ == "__main__":
        main()
    ```

    Run it:

    ```bash
    python streaming.py
    ```

=== "Rust"

    ```rust title="src/main.rs"
    use mullama::prelude::*;
    use tokio_stream::StreamExt;

    #[tokio::main]
    async fn main() -> Result<(), MullamaError> {
        let model = ModelBuilder::new()
            .path("models/llama3.2-1b.gguf")
            .context_size(2048)
            .n_gpu_layers(-1)
            .build().await?;

        println!("Generating with streaming...\n");

        // Stream tokens one by one
        let mut stream = model.stream(
            "Write a haiku about programming.",
            100
        ).await?;

        while let Some(token) = stream.next().await {
            let token = token?;
            print!("{}", token);
        }

        println!("\n\nDone!");
        Ok(())
    }
    ```

    Run it:

    ```bash
    cargo run
    ```

You should see tokens appearing one by one in your terminal, creating a typewriter effect.

---

## Step 5: Multi-Turn Conversation

Build an interactive chatbot that maintains conversation history across multiple turns.

=== "Node.js"

    ```javascript title="chatbot.js"
    const { Model } = require('mullama');
    const readline = require('readline');

    async function main() {
        const model = new Model('models/llama3.2-1b.gguf', {
            contextSize: 4096,
            nGpuLayers: -1
        });

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        // Conversation history
        const messages = [
            { role: 'system', content: 'You are a helpful assistant. Be concise and friendly.' }
        ];

        console.log('Chatbot ready! Type "quit" to exit.\n');

        const askQuestion = () => {
            rl.question('You: ', async (input) => {
                if (input.toLowerCase() === 'quit') {
                    console.log('Goodbye!');
                    model.close();
                    rl.close();
                    return;
                }

                // Add user message to history
                messages.push({ role: 'user', content: input });

                // Generate response with streaming
                process.stdout.write('Assistant: ');
                let fullResponse = '';

                const stream = model.chat(messages, {
                    maxTokens: 500,
                    stream: true
                });

                for await (const token of stream) {
                    process.stdout.write(token);
                    fullResponse += token;
                }

                console.log('\n');

                // Add assistant response to history
                messages.push({ role: 'assistant', content: fullResponse });

                askQuestion();
            });
        };

        askQuestion();
    }

    main().catch(console.error);
    ```

    Run it:

    ```bash
    node chatbot.js
    ```

=== "Python"

    ```python title="chatbot.py"
    from mullama import Model

    def main():
        model = Model("models/llama3.2-1b.gguf",
            context_size=4096,
            n_gpu_layers=-1
        )

        # Conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise and friendly."}
        ]

        print("Chatbot ready! Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Generate response with streaming
            print("Assistant: ", end="", flush=True)
            full_response = ""

            for token in model.chat(messages, max_tokens=500, stream=True):
                print(token, end="", flush=True)
                full_response += token

            print("\n")

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})

    if __name__ == "__main__":
        main()
    ```

    Run it:

    ```bash
    python chatbot.py
    ```

=== "Rust"

    ```rust title="src/main.rs"
    use mullama::prelude::*;
    use tokio_stream::StreamExt;
    use std::io::{self, Write, BufRead};

    #[tokio::main]
    async fn main() -> Result<(), MullamaError> {
        let model = ModelBuilder::new()
            .path("models/llama3.2-1b.gguf")
            .context_size(4096)
            .n_gpu_layers(-1)
            .build().await?;

        // Conversation history
        let mut messages = vec![
            ChatMessage::system("You are a helpful assistant. Be concise and friendly.")
        ];

        println!("Chatbot ready! Type \"quit\" to exit.\n");

        let stdin = io::stdin();
        loop {
            print!("You: ");
            io::stdout().flush()?;

            let mut input = String::new();
            stdin.lock().read_line(&mut input)?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("quit") {
                println!("Goodbye!");
                break;
            }

            // Add user message to history
            messages.push(ChatMessage::user(input));

            // Generate response with streaming
            print!("Assistant: ");
            io::stdout().flush()?;

            let mut full_response = String::new();
            let mut stream = model.chat_stream(&messages, 500).await?;

            while let Some(token) = stream.next().await {
                let token = token?;
                print!("{}", token);
                io::stdout().flush()?;
                full_response.push_str(&token);
            }

            println!("\n");

            // Add assistant response to history
            messages.push(ChatMessage::assistant(&full_response));
        }

        Ok(())
    }
    ```

    Run it:

    ```bash
    cargo run
    ```

Example conversation:

```
Chatbot ready! Type "quit" to exit.

You: What is Rust?
Assistant: Rust is a systems programming language focused on safety, speed, and concurrency.
It prevents common bugs like null pointer dereferences and data races at compile time.

You: How does it compare to C++?
Assistant: Compared to C++, Rust offers memory safety without a garbage collector through
its ownership system. It has a steeper learning curve initially but catches more bugs at
compile time. Performance is comparable to C++ in most benchmarks.

You: quit
Goodbye!
```

---

## Step 6: Deploy as an API Server

Turn your chatbot into a REST API that other applications can consume.

=== "Node.js"

    ```bash
    npm install express
    ```

    ```javascript title="server.js"
    const express = require('express');
    const { Model } = require('mullama');

    const app = express();
    app.use(express.json());

    // Load model once at startup
    const model = new Model('models/llama3.2-1b.gguf', {
        contextSize: 4096,
        nGpuLayers: -1
    });

    // Simple generation endpoint
    app.post('/api/generate', async (req, res) => {
        const { prompt, maxTokens = 200 } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: 'prompt is required' });
        }

        try {
            const response = await model.generate(prompt, { maxTokens });
            res.json({ response });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Chat endpoint with conversation history
    app.post('/api/chat', async (req, res) => {
        const { messages, maxTokens = 500 } = req.body;

        if (!messages || !Array.isArray(messages)) {
            return res.status(400).json({ error: 'messages array is required' });
        }

        try {
            const response = await model.chat(messages, { maxTokens });
            res.json({
                message: { role: 'assistant', content: response }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Streaming endpoint using Server-Sent Events
    app.post('/api/chat/stream', async (req, res) => {
        const { messages, maxTokens = 500 } = req.body;

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        try {
            const stream = model.chat(messages, {
                maxTokens,
                stream: true
            });

            for await (const token of stream) {
                res.write(`data: ${JSON.stringify({ token })}\n\n`);
            }

            res.write('data: [DONE]\n\n');
            res.end();
        } catch (error) {
            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
            res.end();
        }
    });

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`Mullama API server running on http://localhost:${PORT}`);
        console.log('Endpoints:');
        console.log('  POST /api/generate  - Simple text generation');
        console.log('  POST /api/chat      - Chat with conversation history');
        console.log('  POST /api/chat/stream - Streaming chat (SSE)');
    });
    ```

    Run the server:

    ```bash
    node server.js
    ```

    Test it:

    ```bash
    # Simple generation
    curl -X POST http://localhost:3000/api/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What is 2+2?", "maxTokens": 50}'

    # Chat with history
    curl -X POST http://localhost:3000/api/chat \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "user", "content": "Hello, who are you?"}
            ]
        }'
    ```

=== "Python"

    ```bash
    pip install fastapi uvicorn
    ```

    ```python title="server.py"
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from mullama import Model
    import json

    app = FastAPI(title="Mullama Chat API")

    # Load model once at startup
    model = Model("models/llama3.2-1b.gguf",
        context_size=4096,
        n_gpu_layers=-1
    )

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 200

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: list[Message]
        max_tokens: int = 500
        stream: bool = False

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        """Simple text generation."""
        response = model.generate(req.prompt, max_tokens=req.max_tokens)
        return {"response": response}

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        """Chat with conversation history."""
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        if req.stream:
            # Return Server-Sent Events
            async def event_stream():
                for token in model.chat(messages, max_tokens=req.max_tokens, stream=True):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream"
            )

        response = model.chat(messages, max_tokens=req.max_tokens)
        return {"message": {"role": "assistant", "content": response}}

    if __name__ == "__main__":
        import uvicorn
        print("Mullama API server running on http://localhost:8000")
        print("Docs available at http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

    Run the server:

    ```bash
    python server.py
    ```

    Test it:

    ```bash
    # Simple generation
    curl -X POST http://localhost:8000/api/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "What is 2+2?", "max_tokens": 50}'

    # Chat with history
    curl -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "user", "content": "Hello, who are you?"}
            ]
        }'

    # Interactive docs
    open http://localhost:8000/docs
    ```

=== "Rust (Axum)"

    ```toml title="Cargo.toml"
    [dependencies]
    mullama = { version = "0.1.1", features = ["web", "streaming"] }
    tokio = { version = "1", features = ["full"] }
    axum = "0.7"
    serde = { version = "1", features = ["derive"] }
    serde_json = "1"
    tower-http = { version = "0.5", features = ["cors"] }
    ```

    ```rust title="src/main.rs"
    use axum::{
        extract::State,
        response::sse::{Event, Sse},
        routing::post,
        Json, Router,
    };
    use mullama::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokio_stream::StreamExt;

    #[derive(Clone)]
    struct AppState {
        model: Arc<Model>,
    }

    #[derive(Deserialize)]
    struct GenerateRequest {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
    }

    #[derive(Deserialize)]
    struct ChatRequest {
        messages: Vec<ChatMsg>,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
    }

    #[derive(Deserialize, Serialize)]
    struct ChatMsg {
        role: String,
        content: String,
    }

    #[derive(Serialize)]
    struct GenerateResponse {
        response: String,
    }

    fn default_max_tokens() -> usize { 200 }

    async fn generate(
        State(state): State<AppState>,
        Json(req): Json<GenerateRequest>,
    ) -> Json<GenerateResponse> {
        let response = state.model
            .generate(&req.prompt, req.max_tokens)
            .await
            .unwrap_or_else(|e| format!("Error: {}", e));

        Json(GenerateResponse { response })
    }

    async fn chat_stream(
        State(state): State<AppState>,
        Json(req): Json<ChatRequest>,
    ) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
        let messages: Vec<ChatMessage> = req.messages.iter().map(|m| {
            match m.role.as_str() {
                "system" => ChatMessage::system(&m.content),
                "assistant" => ChatMessage::assistant(&m.content),
                _ => ChatMessage::user(&m.content),
            }
        }).collect();

        let stream = state.model
            .chat_stream(&messages, req.max_tokens)
            .await
            .unwrap();

        let event_stream = stream.map(|token| {
            let token = token.unwrap_or_default();
            Ok(Event::default().data(token))
        });

        Sse::new(event_stream)
    }

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading model...");
        let model = ModelBuilder::new()
            .path("models/llama3.2-1b.gguf")
            .context_size(4096)
            .n_gpu_layers(-1)
            .build().await?;

        let state = AppState {
            model: Arc::new(model),
        };

        let app = Router::new()
            .route("/api/generate", post(generate))
            .route("/api/chat/stream", post(chat_stream))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
        println!("Mullama API server running on http://localhost:3000");
        axum::serve(listener, app).await?;
        Ok(())
    }
    ```

    Run the server:

    ```bash
    cargo run --release
    ```

---

## Complete Working Examples

Here are the complete, production-ready versions of everything we built.

=== "Node.js"

    ```javascript title="complete-chatbot.js"
    const { Model } = require('mullama');
    const readline = require('readline');

    async function main() {
        // Configuration
        const MODEL_PATH = process.env.MODEL_PATH || 'models/llama3.2-1b.gguf';
        const CONTEXT_SIZE = parseInt(process.env.CONTEXT_SIZE || '4096');
        const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || '500');

        // Load model
        console.log(`Loading model from ${MODEL_PATH}...`);
        const model = new Model(MODEL_PATH, {
            contextSize: CONTEXT_SIZE,
            nGpuLayers: -1
        });
        console.log('Model loaded successfully!\n');

        // Setup readline
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        // System prompt
        const messages = [{
            role: 'system',
            content: 'You are a helpful, concise assistant. Answer questions clearly and directly.'
        }];

        console.log('=== Mullama Chatbot ===');
        console.log('Commands: "quit" to exit, "clear" to reset history\n');

        const prompt = () => {
            rl.question('You: ', async (input) => {
                const trimmed = input.trim();

                if (!trimmed) { prompt(); return; }
                if (trimmed === 'quit') { model.close(); rl.close(); return; }
                if (trimmed === 'clear') {
                    messages.length = 1;  // Keep system prompt
                    console.log('[History cleared]\n');
                    prompt();
                    return;
                }

                messages.push({ role: 'user', content: trimmed });

                process.stdout.write('Assistant: ');
                let response = '';

                try {
                    for await (const token of model.chat(messages, {
                        maxTokens: MAX_TOKENS,
                        stream: true
                    })) {
                        process.stdout.write(token);
                        response += token;
                    }
                } catch (err) {
                    console.error(`\n[Error: ${err.message}]`);
                }

                console.log('\n');
                if (response) {
                    messages.push({ role: 'assistant', content: response });
                }

                prompt();
            });
        };

        prompt();
    }

    main().catch(console.error);
    ```

=== "Python"

    ```python title="complete_chatbot.py"
    import os
    import sys
    from mullama import Model

    def main():
        # Configuration
        model_path = os.environ.get("MODEL_PATH", "models/llama3.2-1b.gguf")
        context_size = int(os.environ.get("CONTEXT_SIZE", "4096"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "500"))

        # Load model
        print(f"Loading model from {model_path}...")
        model = Model(model_path,
            context_size=context_size,
            n_gpu_layers=-1
        )
        print("Model loaded successfully!\n")

        # System prompt
        messages = [{
            "role": "system",
            "content": "You are a helpful, concise assistant. Answer questions clearly and directly."
        }]

        print("=== Mullama Chatbot ===")
        print('Commands: "quit" to exit, "clear" to reset history\n')

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input == "quit":
                print("Goodbye!")
                break
            if user_input == "clear":
                messages = messages[:1]  # Keep system prompt
                print("[History cleared]\n")
                continue

            messages.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            response = ""

            try:
                for token in model.chat(messages, max_tokens=max_tokens, stream=True):
                    print(token, end="", flush=True)
                    response += token
            except Exception as e:
                print(f"\n[Error: {e}]")

            print("\n")
            if response:
                messages.append({"role": "assistant", "content": response})

    if __name__ == "__main__":
        main()
    ```

---

## What You Built

In this tutorial you learned how to:

1. **Download and load** a GGUF model with Mullama
2. **Generate text** with configurable parameters
3. **Stream tokens** for real-time output
4. **Maintain conversation history** for multi-turn chat
5. **Deploy as an API** using Express (Node.js), FastAPI (Python), or Axum (Rust)

---

## What's Next

<div class="grid cards" markdown>

-   :material-message-text: **[Streaming Guide](../guide/streaming.md)**

    ---

    Advanced streaming with backpressure, cancellation, and custom callbacks.

-   :material-tune: **[Sampling Strategies](../guide/sampling.md)**

    ---

    Control generation quality with temperature, top-p, top-k, and repetition penalties.

-   :material-image-multiple: **[Multimodal](../guide/multimodal.md)**

    ---

    Process images and audio alongside text for vision and voice models.

-   :material-database: **[Embeddings](../guide/embeddings.md)**

    ---

    Generate text embeddings for semantic search and RAG pipelines.

-   :material-web: **[Web Framework](../advanced/web-framework.md)**

    ---

    Build production REST APIs with Axum integration, middleware, and auth.

-   :material-swap-horizontal: **[Language Bindings](../bindings/index.md)**

    ---

    Deep dive into Node.js, Python, Go, PHP, and C/C++ binding APIs.

</div>

!!! tip "Explore More Examples"

    Check out the [Tutorials & Examples](../examples/index.md) section for complete projects including RAG pipelines, voice assistants, batch processing, and edge deployment.
