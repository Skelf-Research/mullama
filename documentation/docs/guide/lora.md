# LoRA Adapters

Load and manage LoRA (Low-Rank Adaptation) adapters for fine-tuned model behavior without reloading the full base model. LoRA enables task-specific customization with minimal memory overhead.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models. Instead of modifying all model weights, LoRA trains small "adapter" matrices that are applied on top of the frozen base model. This means:

- **Small file sizes** -- Adapters are typically 10-100 MB vs. multi-GB base models
- **Fast switching** -- Swap adapters without reloading the base model
- **Composable** -- Apply multiple adapters simultaneously
- **Memory efficient** -- One base model serves many fine-tuned variants

## Loading LoRA Adapters

Apply a LoRA adapter to a loaded model:

=== "Node.js"

    ```javascript
    import { Model, Context, LoRAAdapter } from 'mullama';

    const model = await Model.load('./base-model.gguf');

    // Load and apply a LoRA adapter
    const adapter = await LoRAAdapter.load('./adapter.gguf');
    model.applyAdapter(adapter);

    const context = new Context(model);
    const response = await context.generate("Hello!", 100);
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, LoRAAdapter

    model = Model.load("./base-model.gguf")

    # Load and apply a LoRA adapter
    adapter = LoRAAdapter.load("./adapter.gguf")
    model.apply_adapter(adapter)

    context = Context(model)
    response = context.generate("Hello!", max_tokens=100)
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams, LoRAAdapter};
    use std::sync::Arc;

    let model = Arc::new(Model::load("base-model.gguf")?);

    // Load and apply a LoRA adapter
    let adapter = LoRAAdapter::load("adapter.gguf")?;
    model.apply_adapter(&adapter)?;

    let mut context = Context::new(model, ContextParams::default())?;
    let response = context.generate("Hello!", 100)?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    # Apply LoRA adapter when running
    mullama run llama3.2:1b "Hello!" --lora ./adapter.gguf

    # Or specify in a Modelfile
    # FROM llama3.2:1b
    # ADAPTER ./adapter.gguf
    mullama create my-model -f Modelfile
    mullama run my-model "Hello!"
    ```

## Scale Adjustment

Control how strongly the adapter affects the output with a scale factor:

=== "Node.js"

    ```javascript
    // Full adapter strength (default: 1.0)
    model.applyAdapter(adapter, { scale: 1.0 });

    // Half strength -- blend between base and adapted
    model.applyAdapter(adapter, { scale: 0.5 });

    // Subtle influence
    model.applyAdapter(adapter, { scale: 0.1 });
    ```

=== "Python"

    ```python
    # Full adapter strength (default: 1.0)
    model.apply_adapter(adapter, scale=1.0)

    # Half strength -- blend between base and adapted
    model.apply_adapter(adapter, scale=0.5)

    # Subtle influence
    model.apply_adapter(adapter, scale=0.1)
    ```

=== "Rust"

    ```rust
    // Full adapter strength (default: 1.0)
    model.apply_adapter_scaled(&adapter, 1.0)?;

    // Half strength -- blend between base and adapted
    model.apply_adapter_scaled(&adapter, 0.5)?;

    // Subtle influence
    model.apply_adapter_scaled(&adapter, 0.1)?;
    ```

=== "CLI"

    ```bash
    # Full strength
    mullama run llama3.2:1b "Hello!" --lora adapter.gguf --lora-scale 1.0

    # Half strength
    mullama run llama3.2:1b "Hello!" --lora adapter.gguf --lora-scale 0.5
    ```

!!! tip "Scale Tuning"
    Start with scale 1.0 and reduce if the adapter's influence is too strong (e.g., if it causes repetition or loses general ability). A scale of 0.5-0.8 often provides a good balance between specialization and general capability.

## Multiple Adapters

Apply multiple LoRA adapters simultaneously for combined specialization:

=== "Node.js"

    ```javascript
    import { Model, LoRAAdapter, LoRAManager } from 'mullama';

    const model = await Model.load('./base-model.gguf');

    // Load multiple adapters
    const codeAdapter = await LoRAAdapter.load('./code-adapter.gguf');
    const styleAdapter = await LoRAAdapter.load('./style-adapter.gguf');

    // Apply with different scales
    const manager = new LoRAManager(model);
    manager.add(codeAdapter, { scale: 0.8, name: 'code' });
    manager.add(styleAdapter, { scale: 0.5, name: 'style' });

    const context = new Context(model);
    const response = await context.generate("Write a function:", 200);
    ```

=== "Python"

    ```python
    from mullama import Model, LoRAAdapter, LoRAManager

    model = Model.load("./base-model.gguf")

    # Load multiple adapters
    code_adapter = LoRAAdapter.load("./code-adapter.gguf")
    style_adapter = LoRAAdapter.load("./style-adapter.gguf")

    # Apply with different scales
    manager = LoRAManager(model)
    manager.add(code_adapter, scale=0.8, name="code")
    manager.add(style_adapter, scale=0.5, name="style")

    context = Context(model)
    response = context.generate("Write a function:", max_tokens=200)
    ```

=== "Rust"

    ```rust
    use mullama::{Model, LoRAAdapter, LoRAManager};

    let model = Arc::new(Model::load("base-model.gguf")?);

    let code_adapter = LoRAAdapter::load("code-adapter.gguf")?;
    let style_adapter = LoRAAdapter::load("style-adapter.gguf")?;

    let mut manager = LoRAManager::new(model.clone());
    manager.add("code", &code_adapter, 0.8)?;
    manager.add("style", &style_adapter, 0.5)?;
    ```

=== "CLI"

    ```bash
    # Multiple adapters with different scales
    mullama run llama3.2:1b "Write a function:" \
      --lora code-adapter.gguf --lora-scale 0.8 \
      --lora style-adapter.gguf --lora-scale 0.5
    ```

!!! warning "Adapter Compatibility"
    Multiple adapters must be compatible with the same base model architecture. Combining adapters trained for different base models will produce undefined behavior.

## Hot-Swapping Adapters

Switch between adapters without reloading the base model:

=== "Node.js"

    ```javascript
    const model = await Model.load('./base-model.gguf');
    const manager = new LoRAManager(model);

    // Start with code adapter
    const codeAdapter = await LoRAAdapter.load('./code-adapter.gguf');
    manager.add(codeAdapter, { name: 'code' });

    let context = new Context(model);
    const codeResponse = await context.generate("Write Python:", 200);

    // Hot-swap to chat adapter
    manager.remove('code');
    const chatAdapter = await LoRAAdapter.load('./chat-adapter.gguf');
    manager.add(chatAdapter, { name: 'chat' });

    // No need to reload model -- just create new context
    context = new Context(model);
    const chatResponse = await context.generate("Hello, how are you?", 200);
    ```

=== "Python"

    ```python
    model = Model.load("./base-model.gguf")
    manager = LoRAManager(model)

    # Start with code adapter
    code_adapter = LoRAAdapter.load("./code-adapter.gguf")
    manager.add(code_adapter, name="code")

    context = Context(model)
    code_response = context.generate("Write Python:", max_tokens=200)

    # Hot-swap to chat adapter
    manager.remove("code")
    chat_adapter = LoRAAdapter.load("./chat-adapter.gguf")
    manager.add(chat_adapter, name="chat")

    # New context picks up the adapter change
    context = Context(model)
    chat_response = context.generate("Hello, how are you?", max_tokens=200)
    ```

=== "Rust"

    ```rust
    let model = Arc::new(Model::load("base-model.gguf")?);
    let mut manager = LoRAManager::new(model.clone());

    // Start with code adapter
    let code_adapter = LoRAAdapter::load("code-adapter.gguf")?;
    manager.add("code", &code_adapter, 1.0)?;

    let mut context = Context::new(model.clone(), ContextParams::default())?;
    let code_response = context.generate("Write Python:", 200)?;

    // Hot-swap to chat adapter
    manager.remove("code")?;
    let chat_adapter = LoRAAdapter::load("chat-adapter.gguf")?;
    manager.add("chat", &chat_adapter, 1.0)?;

    let mut context = Context::new(model.clone(), ContextParams::default())?;
    let chat_response = context.generate("Hello!", 200)?;
    ```

=== "CLI"

    ```bash
    # Hot-swap via daemon REST API
    curl -X POST http://localhost:8080/v1/lora/load \
      -d '{"path": "./code-adapter.gguf", "name": "code", "scale": 1.0}'

    curl -X POST http://localhost:8080/v1/lora/unload \
      -d '{"name": "code"}'

    curl -X POST http://localhost:8080/v1/lora/load \
      -d '{"path": "./chat-adapter.gguf", "name": "chat", "scale": 1.0}'
    ```

## Use Cases

LoRA adapters enable specialization for specific tasks without the cost of full fine-tuning:

| Use Case | Description | Typical Scale |
|----------|-------------|---------------|
| Code generation | Fine-tuned on code repositories | 0.8-1.0 |
| Customer support | Trained on support conversations | 0.7-1.0 |
| Domain expert | Specialized in medical, legal, etc. | 0.8-1.0 |
| Style transfer | Writing in a specific author's style | 0.3-0.7 |
| Language pair | Translation between specific languages | 0.8-1.0 |
| Instruction following | Improved instruction adherence | 0.5-0.8 |

## Finding LoRA Adapters

Find pre-trained LoRA adapters on HuggingFace:

=== "Node.js"

    ```javascript
    import { daemon } from 'mullama';

    // Search for adapters
    const adapters = await daemon.searchAdapters('code llama');
    for (const adapter of adapters) {
      console.log(`${adapter.name} - ${adapter.description}`);
    }

    // Download an adapter
    await daemon.pullAdapter('username/code-llama-adapter');
    ```

=== "Python"

    ```python
    from mullama import daemon

    # Search for adapters on HuggingFace
    adapters = daemon.search_adapters("code llama")
    for adapter in adapters:
        print(f"{adapter.name} - {adapter.description}")

    # Download an adapter
    daemon.pull_adapter("username/code-llama-adapter")
    ```

=== "Rust"

    ```rust
    use mullama::daemon::DaemonClient;

    let client = DaemonClient::connect().await?;
    let adapters = client.search_adapters("code llama").await?;
    for adapter in &adapters {
        println!("{} - {}", adapter.name, adapter.description);
    }

    client.pull_adapter("username/code-llama-adapter").await?;
    ```

=== "CLI"

    ```bash
    # Browse adapters on HuggingFace
    # Look for GGUF-format LoRA files in model repositories

    # Download and use
    mullama pull username/code-llama-adapter
    mullama run llama3.2:1b "Hello!" --lora ~/.mullama/adapters/code-llama-adapter.gguf
    ```

!!! info "Adapter Format"
    Mullama supports LoRA adapters in GGUF format. If you find adapters in other formats (safetensors, PyTorch), they must be converted to GGUF first using tools like `convert-lora-to-gguf.py` from the llama.cpp project.

## Memory Considerations

LoRA adapters are memory-efficient compared to full model copies:

| Component | Memory | Notes |
|-----------|--------|-------|
| Base model (7B, Q4_K_M) | ~4 GB | Loaded once, shared |
| LoRA adapter (rank 16) | ~20-50 MB | Per adapter |
| LoRA adapter (rank 64) | ~80-200 MB | Higher quality |
| LoRA adapter (rank 128) | ~200-500 MB | Maximum quality |

The adapter rank determines both quality and memory usage. Higher ranks capture more detail from the fine-tuning but require more memory.

## See Also

- [Loading Models](models.md) -- Base model loading and configuration
- [Memory Management](memory.md) -- Memory optimization strategies
- [Daemon: Modelfile](../daemon/modelfile.md) -- Specifying adapters in Modelfiles
- [API Reference: Model](../api/model.md) -- LoRA API documentation
