# Async Support

Run inference without blocking your application's main thread. Async support enables concurrent generation, model pools for high-throughput servers, and seamless integration with async runtimes.

!!! abstract "Feature Gate"
    In Rust, enable the `async` feature flag:

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["async"] }
    ```

    Node.js and Python include async support by default.

## AsyncModel and AsyncContext

The async API mirrors the synchronous API but returns promises (Node.js), awaitables (Python), or futures (Rust):

=== "Node.js"

    ```javascript
    import { AsyncModel, AsyncContext } from 'mullama';

    // All operations are async by default in Node.js
    const model = await AsyncModel.load('./model.gguf');
    const context = new AsyncContext(model, { nCtx: 4096 });

    const response = await context.generate("Hello!", 100);
    console.log(response);
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, AsyncContext, ContextParams

    async def main():
        model = await AsyncModel.load("./model.gguf")
        context = AsyncContext(model, ContextParams(n_ctx=4096))

        response = await context.generate("Hello!", max_tokens=100)
        print(response)

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, AsyncContext, ContextParams};

    #[tokio::main]
    async fn main() -> Result<(), mullama::MullamaError> {
        let model = AsyncModel::load("model.gguf").await?;
        let mut context = AsyncContext::new(model, ContextParams::default()).await?;

        let response = context.generate("Hello!", 100).await?;
        println!("{}", response);
        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # The daemon handles async internally
    mullama run llama3.2:1b "Hello!"
    ```

## Promise-Based API (Node.js)

In Node.js, all model operations return Promises that can be awaited:

=== "Node.js"

    ```javascript
    import { AsyncModel, AsyncContext } from 'mullama';

    async function main() {
      const model = await AsyncModel.load('./model.gguf');
      const context = new AsyncContext(model);

      // Generate multiple responses concurrently
      const [response1, response2] = await Promise.all([
        context.generate("What is Rust?", 100),
        context.generate("What is Python?", 100),
      ]);

      console.log("Rust:", response1);
      console.log("Python:", response2);

      // Promise.race for timeout
      const result = await Promise.race([
        context.generate("Write an essay:", 1000),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 30000)
        ),
      ]);
    }

    main().catch(console.error);
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, AsyncContext

    async def main():
        model = await AsyncModel.load("./model.gguf")
        context = AsyncContext(model)

        # Generate multiple responses concurrently
        response1, response2 = await asyncio.gather(
            context.generate("What is Rust?", max_tokens=100),
            context.generate("What is Python?", max_tokens=100),
        )

        print(f"Rust: {response1}")
        print(f"Python: {response2}")

        # Timeout with asyncio
        try:
            result = await asyncio.wait_for(
                context.generate("Write an essay:", max_tokens=1000),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("Generation timed out")

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, AsyncContext, ContextParams};
    use tokio::time::{timeout, Duration};

    #[tokio::main]
    async fn main() -> Result<(), mullama::MullamaError> {
        let model = AsyncModel::load("model.gguf").await?;
        let mut ctx1 = AsyncContext::new(model.clone(), ContextParams::default()).await?;
        let mut ctx2 = AsyncContext::new(model.clone(), ContextParams::default()).await?;

        // Concurrent generation with join
        let (r1, r2) = tokio::join!(
            ctx1.generate("What is Rust?", 100),
            ctx2.generate("What is Python?", 100),
        );
        println!("Rust: {}", r1?);
        println!("Python: {}", r2?);

        // Timeout
        match timeout(Duration::from_secs(30),
            ctx1.generate("Write an essay:", 1000)
        ).await {
            Ok(Ok(result)) => println!("{}", result),
            Ok(Err(e)) => eprintln!("Generation error: {}", e),
            Err(_) => eprintln!("Generation timed out"),
        }

        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # CLI handles concurrency internally
    mullama run llama3.2:1b "What is Rust?"
    ```

## Tokio Runtime Management

In Rust, Mullama integrates with the Tokio async runtime. You can use an existing runtime or let Mullama create one:

=== "Node.js"

    ```javascript
    // Node.js uses its built-in event loop; no runtime management needed
    import { AsyncModel } from 'mullama';

    const model = await AsyncModel.load('./model.gguf');
    // All async operations run on the Node.js event loop
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel

    # Python uses asyncio event loop
    async def main():
        model = await AsyncModel.load("./model.gguf")
        # All async operations run on the asyncio event loop

    # Use uvloop for better performance (optional)
    # import uvloop
    # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, MullamaRuntime};

    // Option 1: Use #[tokio::main] (recommended)
    #[tokio::main]
    async fn main() {
        let model = AsyncModel::load("model.gguf").await.unwrap();
    }

    // Option 2: Create runtime explicitly
    fn main() {
        let runtime = MullamaRuntime::new()
            .worker_threads(4)
            .build()
            .unwrap();

        runtime.block_on(async {
            let model = AsyncModel::load("model.gguf").await.unwrap();
        });
    }

    // Option 3: Use existing runtime
    fn with_existing_runtime(rt: &tokio::runtime::Runtime) {
        rt.block_on(async {
            let model = AsyncModel::load("model.gguf").await.unwrap();
        });
    }
    ```

=== "CLI"

    ```bash
    # Daemon manages its own async runtime
    mullama serve --model llama3.2:1b --threads 4
    ```

## Cancellation Support

Cancel in-progress generation to free resources:

=== "Node.js"

    ```javascript
    import { AsyncModel, AsyncContext } from 'mullama';

    const model = await AsyncModel.load('./model.gguf');
    const context = new AsyncContext(model);

    // Use AbortController for cancellation
    const controller = new AbortController();

    // Cancel after 5 seconds
    setTimeout(() => controller.abort(), 5000);

    try {
      const response = await context.generate("Write a long essay:", 5000, {
        signal: controller.signal,
      });
      console.log(response);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Generation was cancelled');
      }
    }
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, AsyncContext

    async def main():
        model = await AsyncModel.load("./model.gguf")
        context = AsyncContext(model)

        # Cancel with asyncio task cancellation
        task = asyncio.create_task(
            context.generate("Write a long essay:", max_tokens=5000)
        )

        # Cancel after 5 seconds
        await asyncio.sleep(5)
        task.cancel()

        try:
            result = await task
        except asyncio.CancelledError:
            print("Generation was cancelled")

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, AsyncContext, ContextParams};
    use tokio::select;
    use tokio::time::{sleep, Duration};

    #[tokio::main]
    async fn main() -> Result<(), mullama::MullamaError> {
        let model = AsyncModel::load("model.gguf").await?;
        let mut context = AsyncContext::new(model, ContextParams::default()).await?;

        // Cancel with tokio::select!
        select! {
            result = context.generate("Write a long essay:", 5000) => {
                println!("{}", result?);
            }
            _ = sleep(Duration::from_secs(5)) => {
                println!("Generation cancelled after timeout");
            }
        }

        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # Ctrl+C to cancel generation
    mullama run llama3.2:1b "Write a long essay:" --max-tokens 5000
    ```

## Concurrent Generation

Generate multiple responses simultaneously using separate contexts:

=== "Node.js"

    ```javascript
    import { AsyncModel, AsyncContext } from 'mullama';

    const model = await AsyncModel.load('./model.gguf');

    const prompts = [
      "Explain quantum computing",
      "Explain machine learning",
      "Explain blockchain",
      "Explain cloud computing",
    ];

    // Create separate contexts for each prompt
    const results = await Promise.all(
      prompts.map(async (prompt) => {
        const context = new AsyncContext(model);
        return context.generate(prompt, 200);
      })
    );

    results.forEach((response, i) => {
      console.log(`\n--- ${prompts[i]} ---`);
      console.log(response);
    });
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, AsyncContext

    async def generate_one(model, prompt):
        context = AsyncContext(model)
        return await context.generate(prompt, max_tokens=200)

    async def main():
        model = await AsyncModel.load("./model.gguf")

        prompts = [
            "Explain quantum computing",
            "Explain machine learning",
            "Explain blockchain",
            "Explain cloud computing",
        ]

        # Generate all responses concurrently
        results = await asyncio.gather(
            *[generate_one(model, p) for p in prompts]
        )

        for prompt, response in zip(prompts, results):
            print(f"\n--- {prompt} ---")
            print(response)

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, AsyncContext, ContextParams};
    use std::sync::Arc;

    #[tokio::main]
    async fn main() -> Result<(), mullama::MullamaError> {
        let model = Arc::new(AsyncModel::load("model.gguf").await?);

        let prompts = vec![
            "Explain quantum computing",
            "Explain machine learning",
            "Explain blockchain",
            "Explain cloud computing",
        ];

        let handles: Vec<_> = prompts.iter().map(|prompt| {
            let model = Arc::clone(&model);
            let prompt = prompt.to_string();
            tokio::spawn(async move {
                let mut ctx = AsyncContext::new(
                    model, ContextParams::default()
                ).await?;
                ctx.generate(&prompt, 200).await
            })
        }).collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let response = handle.await.unwrap()?;
            println!("\n--- {} ---", prompts[i]);
            println!("{}", response);
        }

        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # Daemon supports parallel requests
    mullama serve --model llama3.2:1b --parallel 4

    # Multiple concurrent requests via REST API
    for prompt in "quantum computing" "machine learning" "blockchain"; do
      curl -s http://localhost:8080/v1/completions \
        -d "{\"prompt\": \"Explain $prompt\", \"max_tokens\": 200}" &
    done
    wait
    ```

!!! warning "Context Per Task"
    Each concurrent generation requires its own context. Contexts are not thread-safe for concurrent writes. Share the model (which is read-only) and create separate contexts for each concurrent task.

## Model Pools

For high-throughput servers, use a model pool to manage multiple pre-created contexts:

=== "Node.js"

    ```javascript
    import { AsyncModel, AsyncContext } from 'mullama';

    class ModelPool {
      constructor(model, poolSize = 4) {
        this.available = [];
        this.waiting = [];
        for (let i = 0; i < poolSize; i++) {
          this.available.push(new AsyncContext(model, { nCtx: 4096 }));
        }
      }

      async acquire() {
        if (this.available.length > 0) return this.available.pop();
        return new Promise(resolve => this.waiting.push(resolve));
      }

      release(context) {
        context.clear();
        if (this.waiting.length > 0) {
          this.waiting.shift()(context);
        } else {
          this.available.push(context);
        }
      }

      async generate(prompt, maxTokens) {
        const context = await this.acquire();
        try {
          return await context.generate(prompt, maxTokens);
        } finally {
          this.release(context);
        }
      }
    }

    const model = await AsyncModel.load('./model.gguf');
    const pool = new ModelPool(model, 4);

    // Handle concurrent requests
    const response = await pool.generate("Hello!", 100);
    ```

=== "Python"

    ```python
    import asyncio
    from mullama import AsyncModel, AsyncContext, ContextParams

    class ModelPool:
        def __init__(self, model, pool_size=4):
            self.semaphore = asyncio.Semaphore(pool_size)
            self.contexts = asyncio.Queue()
            for _ in range(pool_size):
                ctx = AsyncContext(model, ContextParams(n_ctx=4096))
                self.contexts.put_nowait(ctx)

        async def generate(self, prompt: str, max_tokens: int) -> str:
            async with self.semaphore:
                context = await self.contexts.get()
                try:
                    return await context.generate(prompt, max_tokens=max_tokens)
                finally:
                    context.clear()
                    await self.contexts.put(context)

    async def main():
        model = await AsyncModel.load("./model.gguf")
        pool = ModelPool(model, pool_size=4)

        # Handle concurrent requests
        results = await asyncio.gather(
            pool.generate("Hello!", 100),
            pool.generate("World!", 100),
        )
        for r in results:
            print(r)

    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{AsyncModel, AsyncContext, ContextParams};
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    struct ModelPool {
        model: Arc<AsyncModel>,
        semaphore: Arc<Semaphore>,
    }

    impl ModelPool {
        fn new(model: Arc<AsyncModel>, pool_size: usize) -> Self {
            Self {
                model,
                semaphore: Arc::new(Semaphore::new(pool_size)),
            }
        }

        async fn generate(&self, prompt: &str, max_tokens: usize)
            -> Result<String, mullama::MullamaError>
        {
            let _permit = self.semaphore.acquire().await.unwrap();
            let mut ctx = AsyncContext::new(
                self.model.clone(), ContextParams::default()
            ).await?;
            ctx.generate(prompt, max_tokens).await
        }
    }
    ```

=== "CLI"

    ```bash
    # Daemon provides built-in connection pooling
    mullama serve --model llama3.2:1b --parallel 4 --ctx-size 4096
    ```

!!! tip "Pool Sizing"
    Set pool size based on available memory. Each context consumes memory for its KV cache. For a 7B model with 4096 context and F16 KV cache, each context uses approximately 2 GB of RAM.

## See Also

- [Streaming](streaming.md) -- Combining async with real-time token streaming
- [Text Generation](generation.md) -- Core generation parameters
- [Memory Management](memory.md) -- Memory implications of concurrent contexts
- [API Reference: Async](../api/async.md) -- Complete Async API documentation
