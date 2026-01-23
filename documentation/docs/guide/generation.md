# Text Generation

Learn how to generate text with Mullama's inference engine, from basic completions to advanced chat interactions with fine-tuned sampling control.

## Basic Generation

The simplest way to generate text from a prompt:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    const response = await context.generate("Once upon a time,", 100);
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, Context

    model = Model.load("./model.gguf")
    context = Context(model)

    response = context.generate("Once upon a time,", max_tokens=100)
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams::default())?;

    let response = context.generate("Once upon a time,", 100)?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Once upon a time,"
    ```

## Context Parameters

Configure the inference context for your workload. Context parameters control the maximum sequence length, parallelism, and memory efficiency:

=== "Node.js"

    ```javascript
    const context = new Context(model, {
      nCtx: 4096,            // Context window size (max tokens)
      nBatch: 512,           // Tokens processed per prompt batch
      nThreads: 8,           // CPU threads for generation
      nThreadsBatch: 8,      // CPU threads for batch processing
      ropeFreqBase: 0.0,     // RoPE frequency base (0 = model default)
      ropeFreqScale: 0.0,    // RoPE frequency scale (0 = model default)
      flashAttn: true,       // Enable flash attention
      kvCacheType: 'f16',    // KV cache quantization
    });
    ```

=== "Python"

    ```python
    from mullama import Context, ContextParams

    context = Context(model, ContextParams(
        n_ctx=4096,              # Context window size (max tokens)
        n_batch=512,             # Tokens processed per prompt batch
        n_threads=8,             # CPU threads for generation
        n_threads_batch=8,       # CPU threads for batch processing
        rope_freq_base=0.0,      # RoPE frequency base (0 = model default)
        rope_freq_scale=0.0,     # RoPE frequency scale (0 = model default)
        flash_attn=True,         # Enable flash attention
        kv_cache_type="f16",     # KV cache quantization
    ))
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams, KvCacheType};

    let params = ContextParams {
        n_ctx: 4096,
        n_batch: 512,
        n_threads: 8,
        n_threads_batch: 8,
        rope_freq_base: 0.0,
        rope_freq_scale: 0.0,
        flash_attn: true,
        kv_cache_type: KvCacheType::F16,
        ..Default::default()
    };

    let mut context = Context::new(model, params)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Hello!" \
      --ctx-size 4096 \
      --batch-size 512 \
      --threads 8 \
      --flash-attn
    ```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_ctx` | `usize` | Model default | Maximum context length in tokens |
| `n_batch` | `usize` | 2048 | Prompt processing batch size |
| `n_threads` | `usize` | System cores | CPU threads for token generation |
| `n_threads_batch` | `usize` | System cores | CPU threads for batch prompt processing |
| `rope_freq_base` | `f32` | 0.0 (auto) | RoPE positional encoding frequency base |
| `rope_freq_scale` | `f32` | 0.0 (auto) | RoPE frequency scaling factor |
| `flash_attn` | `bool` | `true` | Enable flash attention for memory efficiency |
| `embeddings` | `bool` | `false` | Enable embedding extraction |
| `kv_cache_type` | `KvCacheType` | `F16` | KV cache quantization (F32/F16/BF16/Q8_0/Q4_0) |

!!! info "Context Size vs Training Length"
    Setting `n_ctx` larger than the model's training context may work with RoPE scaling but can degrade quality. For best results, stay within the trained context window.

## Sampling Parameters

Control the randomness and quality of generation with sampling parameters:

=== "Node.js"

    ```javascript
    const response = await context.generate("Write a poem:", 200, {
      temperature: 0.7,       // Controls randomness (0.0 = deterministic)
      topK: 40,               // Consider only top K candidates
      topP: 0.9,              // Nucleus sampling threshold
      minP: 0.05,             // Minimum probability relative to top token
      typicalP: 1.0,          // Typical sampling threshold
      penaltyRepeat: 1.1,     // Penalize repeated tokens
      penaltyFreq: 0.0,       // Frequency-based penalty
      penaltyPresent: 0.0,    // Presence-based penalty
      seed: 42,               // Random seed for reproducibility
    });
    ```

=== "Python"

    ```python
    from mullama import SamplerParams

    response = context.generate("Write a poem:", max_tokens=200, params=SamplerParams(
        temperature=0.7,         # Controls randomness (0.0 = deterministic)
        top_k=40,                # Consider only top K candidates
        top_p=0.9,               # Nucleus sampling threshold
        min_p=0.05,              # Minimum probability relative to top token
        typical_p=1.0,           # Typical sampling threshold
        penalty_repeat=1.1,      # Penalize repeated tokens
        penalty_freq=0.0,        # Frequency-based penalty
        penalty_present=0.0,     # Presence-based penalty
        seed=42,                 # Random seed for reproducibility
    ))
    ```

=== "Rust"

    ```rust
    use mullama::SamplerParams;

    let sampling = SamplerParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.05,
        typical_p: 1.0,
        penalty_repeat: 1.1,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        seed: 42,
        ..Default::default()
    };

    let response = context.generate_with_params("Write a poem:", 200, sampling)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write a poem:" \
      --temperature 0.7 \
      --top-k 40 \
      --top-p 0.9 \
      --min-p 0.05 \
      --repeat-penalty 1.1 \
      --seed 42
    ```

## Sampling Presets

Use these presets as starting points for common use cases:

=== "Node.js"

    ```javascript
    // Deterministic: best for code generation, factual Q&A
    const deterministic = { temperature: 0.0 };

    // Balanced: good for general conversation
    const balanced = { temperature: 0.7, topK: 40, topP: 0.9 };

    // Creative: for stories, brainstorming, poetry
    const creative = { temperature: 1.2, topK: 80, topP: 0.95 };

    const response = await context.generate(prompt, 200, balanced);
    ```

=== "Python"

    ```python
    # Deterministic: best for code generation, factual Q&A
    deterministic = SamplerParams(temperature=0.0)

    # Balanced: good for general conversation
    balanced = SamplerParams(temperature=0.7, top_k=40, top_p=0.9)

    # Creative: for stories, brainstorming, poetry
    creative = SamplerParams(temperature=1.2, top_k=80, top_p=0.95)

    response = context.generate(prompt, max_tokens=200, params=balanced)
    ```

=== "Rust"

    ```rust
    // Deterministic: best for code generation, factual Q&A
    let deterministic = SamplerParams { temperature: 0.0, ..Default::default() };

    // Balanced: good for general conversation
    let balanced = SamplerParams {
        temperature: 0.7, top_k: 40, top_p: 0.9, ..Default::default()
    };

    // Creative: for stories, brainstorming, poetry
    let creative = SamplerParams {
        temperature: 1.2, top_k: 80, top_p: 0.95, ..Default::default()
    };
    ```

=== "CLI"

    ```bash
    # Deterministic
    mullama run llama3.2:1b "What is 2+2?" --temperature 0

    # Balanced
    mullama run llama3.2:1b "Tell me about Rust" --temperature 0.7 --top-k 40

    # Creative
    mullama run llama3.2:1b "Write a haiku" --temperature 1.2 --top-k 80
    ```

For a comprehensive guide to all sampling strategies, see [Sampling Strategies](sampling.md).

## Token-by-Token Generation

For fine-grained control, generate one token at a time. This enables custom stop logic, progress tracking, and dynamic parameter adjustment:

=== "Node.js"

    ```javascript
    import { Model, Context, SamplerChain } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    // Tokenize the prompt
    const tokens = model.tokenize("The meaning of life is");

    // Feed prompt into context
    await context.decode(tokens);

    // Build a sampler chain
    const sampler = new SamplerChain()
      .addTemperature(0.7)
      .addTopK(40)
      .addTopP(0.9)
      .addDist(42);

    // Generate tokens one at a time
    let output = '';
    for (let i = 0; i < 200; i++) {
      const token = sampler.sample(context);

      if (model.isEndOfGeneration(token)) break;

      const text = model.tokenToString(token);
      output += text;
      process.stdout.write(text);

      await context.evalToken(token);
    }
    console.log('\n\nGenerated:', output);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, SamplerChain

    model = Model.load("./model.gguf")
    context = Context(model)

    # Tokenize the prompt
    tokens = model.tokenize("The meaning of life is")

    # Feed prompt into context
    context.decode(tokens)

    # Build a sampler chain
    sampler = (SamplerChain()
        .add_temperature(0.7)
        .add_top_k(40)
        .add_top_p(0.9)
        .add_dist(42))

    # Generate tokens one at a time
    output = ""
    for _ in range(200):
        token = sampler.sample(context)

        if model.is_end_of_generation(token):
            break

        text = model.token_to_string(token)
        output += text
        print(text, end="", flush=True)

        context.eval_token(token)

    print(f"\n\nGenerated: {output}")
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use mullama::sampling::{Sampler, SamplerChain};
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);
    let mut context = Context::new(model.clone(), ContextParams::default())?;

    // Tokenize the prompt
    let tokens = model.tokenize("The meaning of life is", true, false)?;
    context.decode(&tokens)?;

    // Build a sampler chain
    let mut chain = SamplerChain::new()
        .add(Sampler::temperature(0.7)?)
        .add(Sampler::top_k(40)?)
        .add(Sampler::top_p(0.9, 1)?)
        .add(Sampler::dist(42)?);

    // Generate tokens one at a time
    let mut output = String::new();
    for _ in 0..200 {
        let token = chain.sample(&context)?;

        if model.token_is_eog(token) {
            break;
        }

        let text = model.token_to_str(token, 0, false)?;
        output.push_str(&text);
        print!("{}", text);

        context.eval_token(token)?;
    }
    println!("\n\nGenerated: {}", output);
    ```

=== "CLI"

    ```bash
    # CLI handles token-by-token generation internally
    # Use --verbose to see per-token timing
    mullama run llama3.2:1b "The meaning of life is" --verbose
    ```

## Chat Templates

Format messages for chat/instruction-tuned models using their built-in templates:

=== "Node.js"

    ```javascript
    import { Model, Context, ChatMessage } from 'mullama';

    const model = await Model.load('./model.gguf');
    const context = new Context(model);

    const messages = [
      ChatMessage.system("You are a helpful coding assistant."),
      ChatMessage.user("Write a function to compute fibonacci numbers."),
    ];

    // Apply the model's built-in chat template
    const prompt = model.applyChatTemplate(messages);
    const response = await context.generate(prompt, 500);
    console.log(response);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ChatMessage

    model = Model.load("./model.gguf")
    context = Context(model)

    messages = [
        ChatMessage.system("You are a helpful coding assistant."),
        ChatMessage.user("Write a function to compute fibonacci numbers."),
    ]

    # Apply the model's built-in chat template
    prompt = model.apply_chat_template(messages)
    response = context.generate(prompt, max_tokens=500)
    print(response)
    ```

=== "Rust"

    ```rust
    use mullama::ChatMessage;

    let messages = vec![
        ChatMessage::system("You are a helpful coding assistant."),
        ChatMessage::user("Write a function to compute fibonacci numbers."),
    ];

    let prompt = model.apply_chat_template(&messages)?;
    let response = context.generate(&prompt, 500)?;
    println!("{}", response);
    ```

=== "CLI"

    ```bash
    # CLI automatically applies chat template
    mullama run llama3.2:1b "Write a function to compute fibonacci numbers." \
      --system "You are a helpful coding assistant."
    ```

### Chat Template Formats

Different models use different chat formats. Mullama automatically detects the correct template from model metadata. For models without embedded templates, you can format manually:

=== "ChatML (Qwen, etc.)"

    ```
    <|im_start|>system
    You are helpful.<|im_end|>
    <|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    ```

=== "Llama 3"

    ```
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are helpful.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    Hello!<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    ```

=== "Mistral"

    ```
    [INST] Hello! [/INST]
    ```

## Multi-Turn Conversations

Build multi-turn conversations by accumulating messages:

=== "Node.js"

    ```javascript
    import { Model, Context, ChatMessage } from 'mullama';
    import * as readline from 'readline';

    const model = await Model.load('./model.gguf');
    const context = new Context(model, { nCtx: 4096 });

    const messages = [
      ChatMessage.system("You are a helpful assistant."),
    ];

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    function ask(question) {
      return new Promise(resolve => rl.question(question, resolve));
    }

    while (true) {
      const userInput = await ask("You: ");
      if (userInput === 'quit') break;

      messages.push(ChatMessage.user(userInput));

      const prompt = model.applyChatTemplate(messages);
      const response = await context.generate(prompt, 500);

      console.log(`Assistant: ${response}`);
      messages.push(ChatMessage.assistant(response));
    }
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams, ChatMessage

    model = Model.load("./model.gguf")
    context = Context(model, ContextParams(n_ctx=4096))

    messages = [
        ChatMessage.system("You are a helpful assistant."),
    ]

    while True:
        user_input = input("You: ")
        if user_input == "quit":
            break

        messages.append(ChatMessage.user(user_input))

        prompt = model.apply_chat_template(messages)
        response = context.generate(prompt, max_tokens=500)

        print(f"Assistant: {response}")
        messages.append(ChatMessage.assistant(response))
    ```

=== "Rust"

    ```rust
    use mullama::ChatMessage;

    let mut messages = vec![
        ChatMessage::system("You are a helpful assistant."),
    ];

    loop {
        let user_input = get_user_input();  // Your input function
        messages.push(ChatMessage::user(&user_input));

        let prompt = model.apply_chat_template(&messages)?;
        let response = context.generate(&prompt, 500)?;

        println!("Assistant: {}", response);
        messages.push(ChatMessage::assistant(&response));
    }
    ```

=== "CLI"

    ```bash
    # Interactive chat mode
    mullama run llama3.2:1b --interactive \
      --system "You are a helpful assistant."
    ```

## Stop Conditions

Control when generation terminates using stop sequences:

=== "Node.js"

    ```javascript
    const response = await context.generate("Generate a list:", 500, {
      stopSequences: ["\n\n", "User:", "```"],
    });
    ```

=== "Python"

    ```python
    response = context.generate("Generate a list:", max_tokens=500, params=SamplerParams(
        stop_sequences=["\n\n", "User:", "```"],
    ))
    ```

=== "Rust"

    ```rust
    let params = SamplerParams {
        stop_sequences: vec![
            "\n\n".to_string(),
            "User:".to_string(),
            "```".to_string(),
        ],
        ..Default::default()
    };

    let response = context.generate_with_params(prompt, 500, params)?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Generate a list:" --stop "\n\n" --stop "User:"
    ```

## KV Cache Management

The KV (Key-Value) cache stores attention state. Managing it is important for long conversations:

=== "Node.js"

    ```javascript
    // Check context usage
    const nPast = context.tokenCount();
    const nCtx = context.contextSize();
    console.log(`Context usage: ${nPast}/${nCtx} tokens`);

    // Clear context when nearly full
    if (nPast > nCtx - 200) {
      context.clear();
      // Re-inject system prompt after clearing
      const systemTokens = model.tokenize("System: You are helpful.\n");
      await context.decode(systemTokens);
    }
    ```

=== "Python"

    ```python
    # Check context usage
    n_past = context.token_count()
    n_ctx = context.context_size()
    print(f"Context usage: {n_past}/{n_ctx} tokens")

    # Clear context when nearly full
    if n_past > n_ctx - 200:
        context.clear()
        # Re-inject system prompt after clearing
        system_tokens = model.tokenize("System: You are helpful.\n")
        context.decode(system_tokens)
    ```

=== "Rust"

    ```rust
    // Check context usage
    let n_past = context.n_past();
    let n_ctx = context.n_ctx();
    println!("Context usage: {}/{} tokens", n_past, n_ctx);

    // Clear context when nearly full
    if n_past > n_ctx - 200 {
        context.clear()?;
        let sys_tokens = model.tokenize("System: You are helpful.\n", true, true)?;
        context.decode(&sys_tokens)?;
    }
    ```

=== "CLI"

    ```bash
    # CLI manages context automatically in interactive mode
    mullama run llama3.2:1b --interactive --ctx-size 4096
    ```

### KV Cache Quantization

Reduce memory usage by quantizing the KV cache. Lower precision uses less memory at the cost of minor quality loss:

| Type | Bits | Memory (4096 ctx, 7B) | Quality Impact |
|------|------|-----------------------|----------------|
| F32 | 32 | ~4 GB | None |
| F16 | 16 | ~2 GB | Negligible |
| BF16 | 16 | ~2 GB | Negligible |
| Q8_0 | 8 | ~1 GB | Very minor |
| Q4_0 | 4 | ~0.5 GB | Minor |

!!! tip "Recommended"
    Use **F16** for most workloads. Switch to **Q8_0** or **Q4_0** when running with very large context windows (32K+) or on memory-constrained systems.

## Batch Processing

Process multiple prompts efficiently using the Batch API:

=== "Node.js"

    ```javascript
    import { Batch } from 'mullama';

    const prompts = [
      "Translate to French: Hello",
      "Translate to French: Goodbye",
      "Translate to French: Thank you",
    ];

    const batch = new Batch(512, prompts.length);

    for (let i = 0; i < prompts.length; i++) {
      const tokens = model.tokenize(prompts[i]);
      batch.addSequence(i, tokens);
    }

    await context.decodeBatch(batch);
    ```

=== "Python"

    ```python
    from mullama import Batch

    prompts = [
        "Translate to French: Hello",
        "Translate to French: Goodbye",
        "Translate to French: Thank you",
    ]

    batch = Batch(max_tokens=512, n_sequences=len(prompts))

    for i, prompt in enumerate(prompts):
        tokens = model.tokenize(prompt)
        batch.add_sequence(i, tokens)

    context.decode_batch(batch)
    ```

=== "Rust"

    ```rust
    use mullama::Batch;

    let prompts = vec![
        "Translate to French: Hello",
        "Translate to French: Goodbye",
        "Translate to French: Thank you",
    ];

    let mut batch = Batch::new(512, prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let tokens = model.tokenize(prompt, true, false)?;
        batch.add_sequence(i, &tokens)?;
    }

    context.decode_batch(&batch)?;
    ```

=== "CLI"

    ```bash
    # Batch via daemon REST API
    curl -X POST http://localhost:8080/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3.2:1b",
        "prompt": ["Translate to French: Hello", "Translate to French: Goodbye"]
      }'
    ```

## Performance Tips

1. **Use appropriate batch sizes** -- Larger `n_batch` values speed up prompt processing for long inputs
2. **Enable flash attention** -- Reduces memory usage and improves speed for long contexts
3. **Tune thread count** -- Set `n_threads` to the number of physical cores (not hyperthreads)
4. **Reuse contexts** -- Creating a new context is expensive; clear and reuse instead
5. **Use streaming** -- Better user experience and enables early stopping to save compute
6. **Match n_ctx to need** -- Smaller context windows use less memory and are faster
7. **Quantize KV cache** -- Use F16 or Q8_0 for large context windows to reduce memory pressure
8. **Offload to GPU** -- Even partial GPU offloading significantly accelerates generation

!!! warning "Thread Count"
    Setting `n_threads` higher than your physical core count can actually reduce performance due to hyperthreading overhead. Use `nproc --all` on Linux or check your CPU specifications.

## See Also

- [Streaming](streaming.md) -- Real-time token streaming for responsive UIs
- [Sampling Strategies](sampling.md) -- Deep dive into all sampling methods
- [Async Support](async.md) -- Non-blocking generation for concurrent workloads
- [API Reference: Context](../api/context.md) -- Complete Context API documentation
