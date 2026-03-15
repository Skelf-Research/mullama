---
title: "Tutorial: Build a Chatbot"
description: Build a multi-turn conversational chatbot with streaming responses, conversation memory, and chat template formatting using Mullama.
---

# Build a Chatbot

Build a multi-turn conversational chatbot with streaming responses, conversation memory management, and proper chat template formatting.

---

## What You'll Build

A complete chatbot application that:

- Loads a local LLM model for private, offline inference
- Maintains multi-turn conversation history
- Formats prompts using ChatML chat templates
- Streams responses token-by-token for responsive UX
- Manages context window limits with token counting
- Supports system prompts for personality customization

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- A chat-capable GGUF model (e.g., `llama3.2-1b-instruct`)
- Node.js 16+ or Python 3.8+

```bash
# Pull a model via the daemon
mullama pull llama3.2:1b
```

---

## Step 1: Load the Model

Start by loading a GGUF model and creating an inference context.

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');

    // Load model with GPU acceleration if available
    const model = JsModel.load('./llama3.2-1b-instruct.Q4_K_M.gguf', {
        nGpuLayers: -1  // Offload all layers to GPU (-1 = all, 0 = CPU only)
    });

    // Create inference context with 4096 token window
    const ctx = new JsContext(model, {
        nCtx: 4096,
        nBatch: 512
    });

    console.log(`Model loaded: ${model.name}`);
    console.log(`Context size: ${ctx.nCtx} tokens`);
    ```

=== "Python"
    ```python
    from mullama import Model, Context, SamplerParams

    # Load model with GPU acceleration if available
    model = Model.load("./llama3.2-1b-instruct.Q4_K_M.gguf", n_gpu_layers=-1)

    # Create inference context with 4096 token window
    ctx = Context(model, n_ctx=4096, n_batch=512)

    print(f"Model loaded: {model.name}")
    print(f"Context size: {ctx.n_ctx} tokens")
    ```

---

## Step 2: Chat Template Formatting

Chat models expect a specific prompt format. Use the built-in chat template support for proper formatting.

=== "Node.js"
    ```javascript
    // Messages are [role, content] tuples
    const messages = [
        ['system', 'You are a helpful, concise AI assistant.'],
        ['user', 'What is Rust?'],
    ];

    // Apply the model's native chat template (ChatML, Llama-3, etc.)
    const prompt = model.applyChatTemplate(messages);
    console.log(prompt);
    // Output varies by model. For ChatML:
    // <|im_start|>system
    // You are a helpful, concise AI assistant.<|im_end|>
    // <|im_start|>user
    // What is Rust?<|im_end|>
    // <|im_start|>assistant
    ```

=== "Python"
    ```python
    # Messages are (role, content) tuples
    messages = [
        ("system", "You are a helpful, concise AI assistant."),
        ("user", "What is Rust?"),
    ]

    # Apply the model's native chat template
    prompt = model.apply_chat_template(messages)
    print(prompt)
    # Output varies by model. For ChatML:
    # <|im_start|>system
    # You are a helpful, concise AI assistant.<|im_end|>
    # <|im_start|>user
    # What is Rust?<|im_end|>
    # <|im_start|>assistant
    ```

!!! tip "Chat Template Formats"
    Different models use different chat formats. The `applyChatTemplate` / `apply_chat_template` method automatically uses the correct format for the loaded model. Common formats include ChatML, Llama-3, Mistral, and Phi-3.

---

## Step 3: Streaming Responses

Stream tokens as they are generated for a responsive chat experience.

=== "Node.js"
    ```javascript
    function streamResponse(ctx, prompt, params) {
        const pieces = ctx.generateStream(prompt, 512, params);
        let response = '';

        for (const piece of pieces) {
            process.stdout.write(piece);
            response += piece;
        }
        process.stdout.write('\n');
        return response;
    }

    // Usage
    const params = { temperature: 0.7, topP: 0.9, penaltyRepeat: 1.1 };
    const response = streamResponse(ctx, prompt, params);
    ```

=== "Python"
    ```python
    def stream_response(ctx, prompt, params):
        pieces = ctx.generate_stream(prompt, max_tokens=512, params=params)
        response = ""

        for piece in pieces:
            print(piece, end="", flush=True)
            response += piece

        print()  # Final newline
        return response

    # Usage
    params = SamplerParams(temperature=0.7, top_p=0.9, penalty_repeat=1.1)
    response = stream_response(ctx, prompt, params)
    ```

---

## Step 4: Conversation Memory

Track conversation history and manage context window limits.

=== "Node.js"
    ```javascript
    class ChatHistory {
        constructor(model, maxMessages = 20) {
            this.model = model;
            this.messages = [];
            this.systemPrompt = 'You are a helpful AI assistant.';
            this.maxMessages = maxMessages;
            this.maxContextTokens = 3072; // Reserve 1024 for response
        }

        addMessage(role, content) {
            this.messages.push([role, content]);
            // Trim oldest messages if over limit
            while (this.messages.length > this.maxMessages) {
                this.messages.shift();
            }
        }

        getTokenCount() {
            const allText = this.messages.map(m => m[1]).join(' ');
            return this.model.tokenize(allText, false).length;
        }

        buildPrompt(userMessage) {
            this.addMessage('user', userMessage);

            // Trim history if context is too full
            while (this.getTokenCount() > this.maxContextTokens && this.messages.length > 2) {
                this.messages.shift();
            }

            const fullMessages = [
                ['system', this.systemPrompt],
                ...this.messages
            ];
            return this.model.applyChatTemplate(fullMessages);
        }

        addResponse(content) {
            this.addMessage('assistant', content);
        }
    }
    ```

=== "Python"
    ```python
    class ChatHistory:
        def __init__(self, model, max_messages=20):
            self.model = model
            self.messages: list[tuple[str, str]] = []
            self.system_prompt = "You are a helpful AI assistant."
            self.max_messages = max_messages
            self.max_context_tokens = 3072  # Reserve 1024 for response

        def add_message(self, role: str, content: str):
            self.messages.append((role, content))
            # Trim oldest messages if over limit
            while len(self.messages) > self.max_messages:
                self.messages.pop(0)

        def get_token_count(self) -> int:
            all_text = " ".join(content for _, content in self.messages)
            return len(self.model.tokenize(all_text, add_bos=False))

        def build_prompt(self, user_message: str) -> str:
            self.add_message("user", user_message)

            # Trim history if context is too full
            while self.get_token_count() > self.max_context_tokens and len(self.messages) > 2:
                self.messages.pop(0)

            full_messages = [("system", self.system_prompt)] + self.messages
            return self.model.apply_chat_template(full_messages)

        def add_response(self, content: str):
            self.add_message("assistant", content)
    ```

---

## Step 5: Token Counting

Monitor token usage to prevent context overflow and provide statistics.

=== "Node.js"
    ```javascript
    function getContextStats(model, history) {
        const tokenCount = history.getTokenCount();
        const maxTokens = 4096;
        const usagePercent = (tokenCount / maxTokens * 100).toFixed(1);

        return {
            messages: history.messages.length,
            tokens: tokenCount,
            maxTokens: maxTokens,
            usage: `${usagePercent}%`,
            remaining: maxTokens - tokenCount
        };
    }

    // Check before generating
    const stats = getContextStats(model, history);
    if (stats.remaining < 200) {
        console.log('Warning: Context nearly full, trimming history...');
        history.messages.splice(0, 2); // Remove oldest exchange
    }
    ```

=== "Python"
    ```python
    def get_context_stats(model, history):
        token_count = history.get_token_count()
        max_tokens = 4096
        usage_percent = token_count / max_tokens * 100

        return {
            "messages": len(history.messages),
            "tokens": token_count,
            "max_tokens": max_tokens,
            "usage": f"{usage_percent:.1f}%",
            "remaining": max_tokens - token_count,
        }

    # Check before generating
    stats = get_context_stats(model, history)
    if stats["remaining"] < 200:
        print("Warning: Context nearly full, trimming history...")
        history.messages = history.messages[2:]  # Remove oldest exchange
    ```

---

## Complete Working Example

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');
    const readline = require('readline');

    // --- Configuration ---
    const MODEL_PATH = process.env.MODEL_PATH || './llama3.2-1b-instruct.Q4_K_M.gguf';
    const SYSTEM_PROMPT = 'You are a helpful, friendly AI assistant. Keep responses concise.';

    // --- Load Model ---
    console.log('Loading model...');
    const model = JsModel.load(MODEL_PATH, { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 4096, nBatch: 512 });
    console.log(`Model ready: ${model.name || 'unknown'}\n`);

    // --- Chat State ---
    const messages = [];
    const maxContextTokens = 3072;
    const samplerParams = { temperature: 0.7, topP: 0.9, penaltyRepeat: 1.1 };

    function trimHistory() {
        while (messages.length > 2) {
            const allText = messages.map(m => m[1]).join(' ');
            if (model.tokenize(allText, false).length <= maxContextTokens) break;
            messages.shift();
        }
    }

    function chat(userInput) {
        messages.push(['user', userInput]);
        trimHistory();

        const fullMessages = [['system', SYSTEM_PROMPT], ...messages];
        const prompt = model.applyChatTemplate(fullMessages);

        // Stream the response
        process.stdout.write('Assistant: ');
        const pieces = ctx.generateStream(prompt, 512, samplerParams);
        let response = '';
        for (const piece of pieces) {
            process.stdout.write(piece);
            response += piece;
        }
        console.log();

        messages.push(['assistant', response.trim()]);
        ctx.clearCache();
        return response.trim();
    }

    // --- Interactive Loop ---
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log('Mullama Chatbot - Type "quit" to exit, "/stats" for info\n');

    function promptUser() {
        rl.question('You: ', (input) => {
            const trimmed = input.trim();
            if (!trimmed) return promptUser();
            if (trimmed === 'quit' || trimmed === 'exit') {
                console.log('Goodbye!');
                rl.close();
                return;
            }
            if (trimmed === '/stats') {
                const tokens = model.tokenize(messages.map(m => m[1]).join(' '), false).length;
                console.log(`Messages: ${messages.length} | Tokens: ${tokens}/4096\n`);
                return promptUser();
            }
            if (trimmed === '/clear') {
                messages.length = 0;
                ctx.clearCache();
                console.log('History cleared.\n');
                return promptUser();
            }
            chat(trimmed);
            console.log();
            promptUser();
        });
    }
    promptUser();
    ```

=== "Python"
    ```python
    import sys
    from mullama import Model, Context, SamplerParams

    # --- Configuration ---
    MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./llama3.2-1b-instruct.Q4_K_M.gguf"
    SYSTEM_PROMPT = "You are a helpful, friendly AI assistant. Keep responses concise."

    # --- Load Model ---
    print("Loading model...")
    model = Model.load(MODEL_PATH, n_gpu_layers=-1)
    ctx = Context(model, n_ctx=4096, n_batch=512)
    print(f"Model ready: {model.name or 'unknown'}\n")

    # --- Chat State ---
    messages: list[tuple[str, str]] = []
    max_context_tokens = 3072
    params = SamplerParams(temperature=0.7, top_p=0.9, penalty_repeat=1.1)

    def trim_history():
        while len(messages) > 2:
            all_text = " ".join(content for _, content in messages)
            if len(model.tokenize(all_text, add_bos=False)) <= max_context_tokens:
                break
            messages.pop(0)

    def chat(user_input: str) -> str:
        messages.append(("user", user_input))
        trim_history()

        full_messages = [("system", SYSTEM_PROMPT)] + messages
        prompt = model.apply_chat_template(full_messages)

        # Stream the response
        print("Assistant: ", end="", flush=True)
        pieces = ctx.generate_stream(prompt, max_tokens=512, params=params)
        response = ""
        for piece in pieces:
            print(piece, end="", flush=True)
            response += piece
        print()

        messages.append(("assistant", response.strip()))
        ctx.clear_cache()
        return response.strip()

    # --- Interactive Loop ---
    print('Mullama Chatbot - Type "quit" to exit, "/stats" for info\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input == "/stats":
            all_text = " ".join(c for _, c in messages)
            tokens = len(model.tokenize(all_text, add_bos=False))
            print(f"Messages: {len(messages)} | Tokens: {tokens}/4096\n")
            continue
        if user_input == "/clear":
            messages.clear()
            ctx.clear_cache()
            print("History cleared.\n")
            continue

        chat(user_input)
        print()
    ```

=== "Rust"
    ```rust
    use mullama::{Context, ContextParams, Model, MullamaError, SamplerParams};
    use std::io::{self, Write};
    use std::sync::Arc;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let model_path = std::env::args().nth(1)
            .unwrap_or_else(|| "llama3.2-1b-instruct.Q4_K_M.gguf".into());

        println!("Loading model...");
        let model = Arc::new(Model::load(&model_path)?);

        let mut ctx_params = ContextParams::default();
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 512;
        let mut context = Context::new(model.clone(), ctx_params)?;

        let mut sampler_params = SamplerParams::default();
        sampler_params.temperature = 0.7;
        sampler_params.top_p = 0.9;
        sampler_params.penalty_repeat = 1.1;

        println!("Chatbot ready. Type 'quit' to exit.\n");

        let mut history = Vec::new();
        loop {
            print!("You: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input == "quit" || input == "exit" { break; }
            if input.is_empty() { continue; }

            history.push(format!("User: {}", input));
            let prompt = format!(
                "System: You are a helpful assistant.\n{}\nAssistant:",
                history.join("\n")
            );

            print!("Assistant: ");
            io::stdout().flush()?;
            let response = context.generate(&prompt, 512)?;
            println!("{}", response.trim());

            history.push(format!("Assistant: {}", response.trim()));
            println!();
        }
        Ok(())
    }
    ```

---

## Extension Ideas

### Custom System Prompts

=== "Node.js"
    ```javascript
    // Coding assistant
    const SYSTEM_PROMPT = `You are an expert programmer.
    Provide clear code examples with explanations.
    Always specify the programming language.`;

    // Creative writer
    const SYSTEM_PROMPT = `You are a creative writing assistant.
    Help craft compelling stories with vivid imagery.`;

    // Concise factual responder
    const SYSTEM_PROMPT = `Answer in 1-2 sentences maximum.
    Be precise and factual. No filler words.`;
    ```

=== "Python"
    ```python
    # Coding assistant
    SYSTEM_PROMPT = """You are an expert programmer.
    Provide clear code examples with explanations.
    Always specify the programming language."""

    # Creative writer
    SYSTEM_PROMPT = """You are a creative writing assistant.
    Help craft compelling stories with vivid imagery."""

    # Concise factual responder
    SYSTEM_PROMPT = """Answer in 1-2 sentences maximum.
    Be precise and factual. No filler words."""
    ```

### Save and Load Conversations

=== "Node.js"
    ```javascript
    const fs = require('fs');

    function saveConversation(messages, filename) {
        fs.writeFileSync(filename, JSON.stringify(messages, null, 2));
        console.log(`Saved ${messages.length} messages to ${filename}`);
    }

    function loadConversation(filename) {
        if (fs.existsSync(filename)) {
            return JSON.parse(fs.readFileSync(filename, 'utf-8'));
        }
        return [];
    }
    ```

=== "Python"
    ```python
    import json
    from pathlib import Path

    def save_conversation(messages, filename):
        Path(filename).write_text(json.dumps(messages, indent=2))
        print(f"Saved {len(messages)} messages to {filename}")

    def load_conversation(filename):
        path = Path(filename)
        if path.exists():
            return json.loads(path.read_text())
        return []
    ```

### Typing Indicator

=== "Node.js"
    ```javascript
    function showTypingIndicator() {
        const frames = ['|', '/', '-', '\\'];
        let i = 0;
        return setInterval(() => {
            process.stdout.write(`\rAssistant is typing ${frames[i++ % 4]} `);
        }, 100);
    }

    // Usage
    const indicator = showTypingIndicator();
    const response = chat(userInput);
    clearInterval(indicator);
    process.stdout.write('\r' + ' '.repeat(30) + '\r');
    ```

=== "Python"
    ```python
    import threading, time

    def typing_indicator(stop_event):
        frames = ["|", "/", "-", "\\"]
        i = 0
        while not stop_event.is_set():
            print(f"\rAssistant is typing {frames[i % 4]} ", end="", flush=True)
            i += 1
            time.sleep(0.1)

    # Usage
    stop = threading.Event()
    t = threading.Thread(target=typing_indicator, args=(stop,))
    t.start()
    response = chat(user_input)
    stop.set()
    t.join()
    print("\r" + " " * 30 + "\r", end="")
    ```

---

## What's Next

- [Streaming Generation](streaming.md) -- Deep dive into streaming patterns and SSE
- [API Server](api-server.md) -- Expose your chatbot as a REST API
- [RAG Pipeline](rag.md) -- Ground responses in your own documents
- [Voice Assistant](voice-assistant.md) -- Add voice input to your chatbot
