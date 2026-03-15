# Sessions & State

Save and restore inference state for conversation continuity, checkpointing, and efficient context resumption. Sessions persist the KV cache so you can resume generation without re-processing the entire conversation history.

## Overview

A session captures the complete inference state at a point in time, including:

- **KV cache** -- All computed attention key-value pairs
- **Token position** -- Current position in the sequence
- **Sequence state** -- Active sequence metadata

This enables several powerful use cases:

- Resume long conversations without re-encoding the full history
- Share pre-computed system prompts across many users
- Create checkpoints during multi-step reasoning
- Implement undo/redo in interactive applications

## Saving Session State

Save the current context state to a file:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model, { nCtx: 4096 });

    // Process some tokens
    const systemPrompt = "You are a helpful assistant specializing in Rust programming.";
    await context.generate(systemPrompt, 0);  // Encode without generating

    // Save state
    await context.saveSession('./session.bin');
    console.log('Session saved');
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")
    context = Context(model, ContextParams(n_ctx=4096))

    # Process some tokens
    system_prompt = "You are a helpful assistant specializing in Rust programming."
    context.generate(system_prompt, max_tokens=0)  # Encode without generating

    # Save state
    context.save_session("./session.bin")
    print("Session saved")
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model, ContextParams::default())?;

    // Process some tokens
    let system_prompt = "You are a helpful assistant specializing in Rust programming.";
    let tokens = model.tokenize(system_prompt, true, true)?;
    context.decode(&tokens)?;

    // Save state
    context.save_session("session.bin")?;
    println!("Session saved");
    ```

=== "CLI"

    ```bash
    # Save session during interactive mode
    mullama run llama3.2:1b --interactive --save-session ./session.bin \
      --system "You are a helpful assistant specializing in Rust programming."
    ```

## Restoring Session State

Load a previously saved session to resume from that point:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model, { nCtx: 4096 });

    // Restore saved state
    await context.loadSession('./session.bin');
    console.log(`Restored ${context.tokenCount()} tokens of context`);

    // Continue from where we left off
    const response = await context.generate("How do I use lifetimes?", 200);
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")
    context = Context(model, ContextParams(n_ctx=4096))

    # Restore saved state
    context.load_session("./session.bin")
    print(f"Restored {context.token_count()} tokens of context")

    # Continue from where we left off
    response = context.generate("How do I use lifetimes?", max_tokens=200)
    print(response)
    ```

=== "Rust"

    ```rust
    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model, ContextParams::default())?;

    // Restore saved state
    context.load_session("session.bin")?;
    println!("Restored {} tokens of context", context.n_past());

    // Continue from where we left off
    let response = context.generate("How do I use lifetimes?", 200)?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    # Resume from saved session
    mullama run llama3.2:1b "How do I use lifetimes?" \
      --load-session ./session.bin
    ```

## Use Cases

### Pre-Computed System Prompts

Encode a complex system prompt once and share it across users to save processing time:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');

    // One-time setup: encode the system prompt
    async function createSystemSession() {
      const context = new Context(model, { nCtx: 4096 });
      const systemPrompt = `You are an expert Python developer.
    You follow PEP 8 style guidelines.
    You write comprehensive docstrings.
    You include type hints in all function signatures.`;

      await context.generate(systemPrompt, 0);
      await context.saveSession('./python-expert.session');
    }

    // Per-request: load pre-computed session
    async function handleUserQuery(query) {
      const context = new Context(model, { nCtx: 4096 });
      await context.loadSession('./python-expert.session');
      return context.generate(query, 500);
    }

    const response = await handleUserQuery("Write a function to parse CSV files.");
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./model.gguf")

    # One-time setup: encode the system prompt
    def create_system_session():
        context = Context(model, ContextParams(n_ctx=4096))
        system_prompt = """You are an expert Python developer.
    You follow PEP 8 style guidelines.
    You write comprehensive docstrings.
    You include type hints in all function signatures."""

        context.generate(system_prompt, max_tokens=0)
        context.save_session("./python-expert.session")

    # Per-request: load pre-computed session
    def handle_user_query(query: str) -> str:
        context = Context(model, ContextParams(n_ctx=4096))
        context.load_session("./python-expert.session")
        return context.generate(query, max_tokens=500)

    response = handle_user_query("Write a function to parse CSV files.")
    print(response)
    ```

=== "Rust"

    ```rust
    // One-time setup
    fn create_system_session(model: &Arc<Model>) -> Result<(), MullamaError> {
        let mut context = Context::new(model.clone(), ContextParams::default())?;
        let system_prompt = "You are an expert Python developer...";
        let tokens = model.tokenize(system_prompt, true, true)?;
        context.decode(&tokens)?;
        context.save_session("python-expert.session")?;
        Ok(())
    }

    // Per-request
    fn handle_query(model: &Arc<Model>, query: &str) -> Result<String, MullamaError> {
        let mut context = Context::new(model.clone(), ContextParams::default())?;
        context.load_session("python-expert.session")?;
        context.generate(query, 500)
    }
    ```

=== "CLI"

    ```bash
    # Create the system session
    mullama run llama3.2:1b --save-session python-expert.session \
      --system "You are an expert Python developer." --max-tokens 0

    # Use it for each query
    mullama run llama3.2:1b "Write a CSV parser." \
      --load-session python-expert.session
    ```

### Conversation Checkpoints

Save checkpoints during long conversations to enable backtracking:

=== "Node.js"

    ```javascript
    const messages = [];
    let turnCount = 0;

    async function chat(userMessage) {
      messages.push({ role: 'user', content: userMessage });
      turnCount++;

      // Save checkpoint every 5 turns
      if (turnCount % 5 === 0) {
        await context.saveSession(`./checkpoints/turn-${turnCount}.session`);
      }

      const prompt = model.applyChatTemplate(messages);
      const response = await context.generate(prompt, 500);
      messages.push({ role: 'assistant', content: response });
      return response;
    }

    // Later: revert to a checkpoint
    async function revertToTurn(turn) {
      await context.loadSession(`./checkpoints/turn-${turn}.session`);
      messages.length = turn * 2;  // Remove messages after checkpoint
    }
    ```

=== "Python"

    ```python
    messages = []
    turn_count = 0

    def chat(user_message: str) -> str:
        global turn_count
        messages.append({"role": "user", "content": user_message})
        turn_count += 1

        # Save checkpoint every 5 turns
        if turn_count % 5 == 0:
            context.save_session(f"./checkpoints/turn-{turn_count}.session")

        prompt = model.apply_chat_template(messages)
        response = context.generate(prompt, max_tokens=500)
        messages.append({"role": "assistant", "content": response})
        return response

    # Later: revert to a checkpoint
    def revert_to_turn(turn: int):
        context.load_session(f"./checkpoints/turn-{turn}.session")
        del messages[turn * 2:]  # Remove messages after checkpoint
    ```

=== "Rust"

    ```rust
    fn save_checkpoint(context: &Context, turn: usize) -> Result<(), MullamaError> {
        let path = format!("checkpoints/turn-{}.session", turn);
        context.save_session(&path)
    }

    fn revert_to_turn(context: &mut Context, turn: usize) -> Result<(), MullamaError> {
        let path = format!("checkpoints/turn-{}.session", turn);
        context.load_session(&path)
    }
    ```

=== "CLI"

    ```bash
    # Save checkpoints in interactive mode
    mullama run llama3.2:1b --interactive \
      --checkpoint-dir ./checkpoints \
      --checkpoint-interval 5
    ```

## Session File Format

Session files store binary data in a format compatible with llama.cpp's state serialization:

| Field | Size | Description |
|-------|------|-------------|
| Magic number | 4 bytes | File format identifier |
| Version | 4 bytes | Format version |
| Token count | 4 bytes | Number of tokens processed |
| KV cache data | Variable | Serialized key-value cache |
| Sequence state | Variable | Active sequence metadata |

!!! warning "Compatibility"
    Session files are tied to a specific model architecture and context size. You cannot load a session saved with one model into a context using a different model. The context size (`n_ctx`) must also match.

## Managing Multiple Sessions

Manage a collection of sessions for different users or conversations:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';
    import { existsSync, mkdirSync } from 'fs';

    class SessionManager {
      constructor(model, sessionDir = './sessions') {
        this.model = model;
        this.sessionDir = sessionDir;
        if (!existsSync(sessionDir)) mkdirSync(sessionDir, { recursive: true });
      }

      sessionPath(sessionId) {
        return `${this.sessionDir}/${sessionId}.session`;
      }

      async createSession(sessionId, systemPrompt) {
        const context = new Context(this.model, { nCtx: 4096 });
        await context.generate(systemPrompt, 0);
        await context.saveSession(this.sessionPath(sessionId));
        return context;
      }

      async loadOrCreate(sessionId, systemPrompt) {
        const context = new Context(this.model, { nCtx: 4096 });
        const path = this.sessionPath(sessionId);

        if (existsSync(path)) {
          await context.loadSession(path);
        } else {
          await context.generate(systemPrompt, 0);
          await context.saveSession(path);
        }
        return context;
      }

      async save(sessionId, context) {
        await context.saveSession(this.sessionPath(sessionId));
      }
    }
    ```

=== "Python"

    ```python
    from pathlib import Path
    from mullama import Model, Context, ContextParams

    class SessionManager:
        def __init__(self, model, session_dir="./sessions"):
            self.model = model
            self.session_dir = Path(session_dir)
            self.session_dir.mkdir(parents=True, exist_ok=True)

        def session_path(self, session_id: str) -> Path:
            return self.session_dir / f"{session_id}.session"

        def create_session(self, session_id: str, system_prompt: str) -> Context:
            context = Context(self.model, ContextParams(n_ctx=4096))
            context.generate(system_prompt, max_tokens=0)
            context.save_session(str(self.session_path(session_id)))
            return context

        def load_or_create(self, session_id: str, system_prompt: str) -> Context:
            context = Context(self.model, ContextParams(n_ctx=4096))
            path = self.session_path(session_id)

            if path.exists():
                context.load_session(str(path))
            else:
                context.generate(system_prompt, max_tokens=0)
                context.save_session(str(path))
            return context

        def save(self, session_id: str, context: Context):
            context.save_session(str(self.session_path(session_id)))
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::path::PathBuf;
    use std::sync::Arc;

    struct SessionManager {
        model: Arc<Model>,
        session_dir: PathBuf,
    }

    impl SessionManager {
        fn new(model: Arc<Model>, session_dir: &str) -> Self {
            std::fs::create_dir_all(session_dir).ok();
            Self {
                model,
                session_dir: PathBuf::from(session_dir),
            }
        }

        fn load_or_create(&self, session_id: &str, system_prompt: &str)
            -> Result<Context, mullama::MullamaError>
        {
            let path = self.session_dir.join(format!("{}.session", session_id));
            let mut context = Context::new(self.model.clone(), ContextParams::default())?;

            if path.exists() {
                context.load_session(path.to_str().unwrap())?;
            } else {
                let tokens = self.model.tokenize(system_prompt, true, true)?;
                context.decode(&tokens)?;
                context.save_session(path.to_str().unwrap())?;
            }
            Ok(context)
        }
    }
    ```

=== "CLI"

    ```bash
    # Manage sessions via daemon
    mullama session list
    mullama session save my-chat
    mullama session load my-chat
    mullama session delete my-chat
    ```

## Memory Considerations

Session files can be large, as they contain the full KV cache:

| Context Size | KV Type | Session File Size |
|--------------|---------|-------------------|
| 2048 tokens | F16 | ~50-200 MB |
| 4096 tokens | F16 | ~100-400 MB |
| 8192 tokens | F16 | ~200-800 MB |
| 4096 tokens | Q8_0 | ~50-200 MB |
| 4096 tokens | Q4_0 | ~25-100 MB |

!!! tip "Reducing Session Size"
    Use quantized KV cache types (Q8_0 or Q4_0) to significantly reduce session file sizes. The quality impact is minimal for most use cases.

=== "Node.js"

    ```javascript
    // Use quantized KV cache for smaller session files
    const context = new Context(model, {
      nCtx: 4096,
      kvCacheType: 'q8_0',  // Much smaller session files
    });
    ```

=== "Python"

    ```python
    context = Context(model, ContextParams(
        n_ctx=4096,
        kv_cache_type="q8_0",  # Much smaller session files
    ))
    ```

=== "Rust"

    ```rust
    let params = ContextParams {
        n_ctx: 4096,
        kv_cache_type: KvCacheType::Q8_0,
        ..Default::default()
    };
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b --interactive \
      --kv-cache-type q8_0 \
      --save-session ./compact-session.bin
    ```

## See Also

- [Text Generation](generation.md) -- Context parameters and KV cache management
- [Memory Management](memory.md) -- Memory optimization strategies
- [Async Support](async.md) -- Async session operations
- [API Reference: Context](../api/context.md) -- Session save/load API documentation
