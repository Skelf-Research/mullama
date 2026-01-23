---
title: "Tutorial: Batch Processing"
description: Process multiple prompts efficiently with parallel execution, progress reporting, and throughput optimization using Mullama.
---

# Batch Processing

Process multiple prompts or documents efficiently. This tutorial covers sequential batch generation, parallel processing with Rayon, batch embeddings, progress reporting, and error handling for individual items.

---

## What You'll Build

A batch processing system that:

- Generates responses for multiple prompts in sequence
- Processes embeddings in batches for high throughput
- Reports progress during long-running operations
- Handles errors per-item without aborting the batch
- Uses Rayon for parallel processing (Rust)
- Optimizes throughput with configuration tuning

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- A GGUF model file
- For Rust parallel processing: `parallel` feature enabled

```bash
mullama pull llama3.2:1b
```

---

## Batch Generation

Process multiple prompts sequentially, collecting all results.

=== "Node.js"
    ```javascript
    const { JsModel, JsContext } = require('mullama');

    const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });
    const ctx = new JsContext(model, { nCtx: 2048, nBatch: 512 });

    const prompts = [
        'Explain photosynthesis in one sentence:',
        'What is the speed of light?',
        'Define machine learning:',
        'What causes earthquakes?',
        'Explain gravity briefly:',
    ];

    function batchGenerate(prompts, maxTokens = 100, params = {}) {
        const results = [];
        const startTime = Date.now();

        for (let i = 0; i < prompts.length; i++) {
            const prompt = prompts[i];
            console.log(`[${i + 1}/${prompts.length}] Processing: "${prompt.slice(0, 40)}..."`);

            try {
                const text = ctx.generate(prompt, maxTokens, params);
                results.push({ prompt, response: text.trim(), status: 'success' });
            } catch (error) {
                results.push({ prompt, response: null, status: 'error', error: error.message });
                console.error(`  Error: ${error.message}`);
            }

            ctx.clearCache();

            // Progress
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = (i + 1) / elapsed;
            console.log(`  Done. (${rate.toFixed(1)} prompts/min)`);
        }

        const totalTime = (Date.now() - startTime) / 1000;
        console.log(`\nBatch complete: ${results.length} prompts in ${totalTime.toFixed(1)}s`);
        return results;
    }

    const results = batchGenerate(prompts, 80, { temperature: 0.3 });
    ```

=== "Python"
    ```python
    import time
    from mullama import Model, Context, SamplerParams

    model = Model.load("./model.gguf", n_gpu_layers=-1)
    ctx = Context(model, n_ctx=2048, n_batch=512)

    prompts = [
        "Explain photosynthesis in one sentence:",
        "What is the speed of light?",
        "Define machine learning:",
        "What causes earthquakes?",
        "Explain gravity briefly:",
    ]

    def batch_generate(prompts, max_tokens=100, params=None):
        results = []
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            print(f'[{i+1}/{len(prompts)}] Processing: "{prompt[:40]}..."')

            try:
                text = ctx.generate(prompt, max_tokens=max_tokens, params=params)
                results.append({"prompt": prompt, "response": text.strip(), "status": "success"})
            except Exception as e:
                results.append({"prompt": prompt, "response": None, "status": "error", "error": str(e)})
                print(f"  Error: {e}")

            ctx.clear_cache()

            # Progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            print(f"  Done. ({rate:.1f} prompts/min)")

        total_time = time.time() - start_time
        print(f"\nBatch complete: {len(results)} prompts in {total_time:.1f}s")
        return results

    results = batch_generate(prompts, max_tokens=80, params=SamplerParams(temperature=0.3))
    ```

---

## Batch Embeddings

Generate embeddings for many texts at once using the batch API.

=== "Node.js"
    ```javascript
    const { JsModel, JsEmbeddingGenerator, cosineSimilarity } = require('mullama');

    const model = JsModel.load('./model.gguf');
    const embedGen = new JsEmbeddingGenerator(model, 512, true);

    const texts = [
        'The cat sat on the mat.',
        'A feline rested on the rug.',
        'Dogs love to play fetch in the park.',
        'Machine learning transforms data into predictions.',
        'Neural networks are inspired by the brain.',
        'The weather today is sunny and warm.',
        'Quantum computing uses qubits instead of bits.',
        'Python is a popular programming language.',
    ];

    function batchEmbed(texts, batchSize = 32) {
        const allEmbeddings = [];
        const startTime = Date.now();

        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            console.log(`Embedding batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)}`);

            const embeddings = embedGen.embedBatch(batch);
            allEmbeddings.push(...embeddings);
        }

        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`Embedded ${texts.length} texts in ${elapsed.toFixed(2)}s`);
        console.log(`Throughput: ${(texts.length / elapsed).toFixed(1)} texts/sec`);
        console.log(`Dimension: ${allEmbeddings[0].length}`);

        return allEmbeddings;
    }

    const embeddings = batchEmbed(texts);

    // Show similarity matrix
    console.log('\nSimilarity matrix (first 4 texts):');
    for (let i = 0; i < 4; i++) {
        const row = [];
        for (let j = 0; j < 4; j++) {
            row.push(cosineSimilarity(embeddings[i], embeddings[j]).toFixed(3));
        }
        console.log(`  ${texts[i].slice(0, 30).padEnd(30)} | ${row.join(' | ')}`);
    }
    ```

=== "Python"
    ```python
    import time
    import numpy as np
    from mullama import Model, EmbeddingGenerator, cosine_similarity

    model = Model.load("./model.gguf")
    embed_gen = EmbeddingGenerator(model, n_ctx=512, normalize=True)

    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Dogs love to play fetch in the park.",
        "Machine learning transforms data into predictions.",
        "Neural networks are inspired by the brain.",
        "The weather today is sunny and warm.",
        "Quantum computing uses qubits instead of bits.",
        "Python is a popular programming language.",
    ]

    def batch_embed(texts, batch_size=32):
        all_embeddings = []
        start_time = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            embeddings = embed_gen.embed_batch(batch)
            all_embeddings.extend(embeddings)

        elapsed = time.time() - start_time
        print(f"Embedded {len(texts)} texts in {elapsed:.2f}s")
        print(f"Throughput: {len(texts) / elapsed:.1f} texts/sec")
        print(f"Dimension: {all_embeddings[0].shape}")

        return all_embeddings

    embeddings = batch_embed(texts)

    # Show similarity matrix
    print("\nSimilarity matrix (first 4 texts):")
    for i in range(4):
        row = [f"{cosine_similarity(embeddings[i], embeddings[j]):.3f}" for j in range(4)]
        print(f"  {texts[i][:30]:30s} | {' | '.join(row)}")
    ```

---

## Progress Reporting

Track and display progress for long-running batch operations.

=== "Node.js"
    ```javascript
    class BatchProgress {
        constructor(total) {
            this.total = total;
            this.completed = 0;
            this.errors = 0;
            this.startTime = Date.now();
        }

        tick(success = true) {
            this.completed++;
            if (!success) this.errors++;
            this.display();
        }

        display() {
            const elapsed = (Date.now() - this.startTime) / 1000;
            const rate = this.completed / elapsed;
            const eta = (this.total - this.completed) / rate;
            const pct = (this.completed / this.total * 100).toFixed(1);

            process.stdout.write(
                `\r[${this.completed}/${this.total}] ${pct}% | ` +
                `${rate.toFixed(1)}/s | ETA: ${eta.toFixed(0)}s | ` +
                `Errors: ${this.errors}`
            );
        }

        finish() {
            const elapsed = (Date.now() - this.startTime) / 1000;
            console.log(`\nCompleted: ${this.completed}/${this.total} ` +
                        `in ${elapsed.toFixed(1)}s (${this.errors} errors)`);
        }
    }
    ```

=== "Python"
    ```python
    import sys

    class BatchProgress:
        def __init__(self, total):
            self.total = total
            self.completed = 0
            self.errors = 0
            self.start_time = time.time()

        def tick(self, success=True):
            self.completed += 1
            if not success:
                self.errors += 1
            self.display()

        def display(self):
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            pct = self.completed / self.total * 100

            sys.stdout.write(
                f"\r[{self.completed}/{self.total}] {pct:.1f}% | "
                f"{rate:.1f}/s | ETA: {eta:.0f}s | "
                f"Errors: {self.errors}"
            )
            sys.stdout.flush()

        def finish(self):
            elapsed = time.time() - self.start_time
            print(f"\nCompleted: {self.completed}/{self.total} "
                  f"in {elapsed:.1f}s ({self.errors} errors)")
    ```

---

## Error Handling Per Item

Handle failures gracefully without aborting the entire batch.

=== "Node.js"
    ```javascript
    function resilientBatch(prompts, ctx, maxTokens = 100, maxRetries = 2) {
        const results = [];

        for (const prompt of prompts) {
            let attempts = 0;
            let success = false;

            while (attempts <= maxRetries && !success) {
                try {
                    const text = ctx.generate(prompt, maxTokens, { temperature: 0.3 });
                    results.push({ prompt, response: text.trim(), status: 'success', attempts });
                    success = true;
                } catch (error) {
                    attempts++;
                    if (attempts > maxRetries) {
                        results.push({
                            prompt, response: null,
                            status: 'failed', error: error.message, attempts
                        });
                    } else {
                        console.warn(`Retry ${attempts}/${maxRetries} for: "${prompt.slice(0, 30)}..."`);
                        ctx.clearCache();
                    }
                }
            }
            ctx.clearCache();
        }

        const successful = results.filter(r => r.status === 'success').length;
        const failed = results.filter(r => r.status === 'failed').length;
        console.log(`Results: ${successful} success, ${failed} failed`);
        return results;
    }
    ```

=== "Python"
    ```python
    def resilient_batch(prompts, ctx, max_tokens=100, max_retries=2):
        results = []

        for prompt in prompts:
            attempts = 0
            success = False

            while attempts <= max_retries and not success:
                try:
                    text = ctx.generate(prompt, max_tokens=max_tokens,
                                        params=SamplerParams(temperature=0.3))
                    results.append({"prompt": prompt, "response": text.strip(),
                                    "status": "success", "attempts": attempts})
                    success = True
                except Exception as e:
                    attempts += 1
                    if attempts > max_retries:
                        results.append({"prompt": prompt, "response": None,
                                        "status": "failed", "error": str(e),
                                        "attempts": attempts})
                    else:
                        print(f'Retry {attempts}/{max_retries} for: "{prompt[:30]}..."')
                        ctx.clear_cache()
            ctx.clear_cache()

        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        print(f"Results: {successful} success, {failed} failed")
        return results
    ```

---

## Parallel Processing (Rust)

For maximum throughput on multi-core systems, use Rayon for parallel batch processing.

```rust
use mullama::{Model, Context, ContextParams, SamplerParams, MullamaError};
use rayon::prelude::*;
use std::sync::Arc;

fn parallel_batch_generate(
    model: Arc<Model>,
    prompts: &[&str],
    max_tokens: usize,
) -> Vec<Result<String, MullamaError>> {
    prompts.par_iter()
        .map(|prompt| {
            // Each thread gets its own context
            let mut ctx_params = ContextParams::default();
            ctx_params.n_ctx = 2048;
            ctx_params.n_batch = 512;
            let mut context = Context::new(model.clone(), ctx_params)?;

            let mut sampler_params = SamplerParams::default();
            sampler_params.temperature = 0.3;

            let result = context.generate(prompt, max_tokens)?;
            Ok(result.trim().to_string())
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load("model.gguf")?);

    let prompts = vec![
        "Explain photosynthesis:",
        "What is gravity?",
        "Define entropy:",
        "What is DNA?",
        "Explain relativity:",
    ];

    println!("Processing {} prompts in parallel...", prompts.len());
    let start = std::time::Instant::now();

    let results = parallel_batch_generate(model, &prompts, 100);

    let elapsed = start.elapsed();
    let successful = results.iter().filter(|r| r.is_ok()).count();
    println!("Done: {}/{} successful in {:.2}s",
        successful, prompts.len(), elapsed.as_secs_f64());

    for (prompt, result) in prompts.iter().zip(results.iter()) {
        match result {
            Ok(text) => println!("Q: {}\nA: {}\n", prompt, text),
            Err(e) => println!("Q: {}\nError: {}\n", prompt, e),
        }
    }

    Ok(())
}
```

!!! note "Parallel vs Sequential in Bindings"
    The Node.js and Python bindings run inference sequentially because the underlying model context is not thread-safe. For true parallel processing, use the Rust API with Rayon, which creates separate contexts per thread.

---

## Throughput Optimization

| Technique | Impact | Description |
|-----------|--------|-------------|
| GPU offloading | 5-20x | Set `nGpuLayers: -1` for full GPU |
| Larger batch size | 1.5-3x | Increase `nBatch` (512-2048) |
| Lower max_tokens | Proportional | Generate only what you need |
| Greedy sampling | 1.1-1.3x | Temperature 0.0 avoids sampling overhead |
| Context reuse | 1.2x | Clear cache instead of recreating context |
| Quantized models | 2-4x | Use Q4_K_M instead of FP16 |

---

## What's Next

- [RAG Pipeline](rag.md) -- Use batch embeddings for document retrieval
- [Semantic Search](semantic-search.md) -- Build search from batch embeddings
- [API Server](api-server.md) -- Serve batch endpoints over HTTP
- [Advanced: Parallel Processing](../advanced/parallel.md) -- Deep dive into Rayon integration
