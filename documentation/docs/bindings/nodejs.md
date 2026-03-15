---
title: Node.js Bindings
description: High-performance Node.js bindings for Mullama LLM inference, built with napi-rs for native speed from JavaScript and TypeScript applications.
---

# Node.js Bindings

High-performance Node.js bindings for the Mullama LLM library, built with [napi-rs](https://napi.rs/) for native-speed inference from JavaScript and TypeScript. The bindings provide a synchronous API that runs inference directly in the Node.js process with zero serialization overhead.

## Installation

```bash
npm install mullama
```

Pre-built native modules are provided for the following platforms:

| Platform | Architecture | Package |
|----------|:------------:|---------|
| Linux | x64 | `mullama-linux-x64-gnu` |
| Linux | arm64 | `mullama-linux-arm64-gnu` |
| macOS | x64 (Intel) | `mullama-darwin-x64` |
| macOS | arm64 (Apple Silicon) | `mullama-darwin-arm64` |
| Windows | x64 | `mullama-win32-x64-msvc` |

The correct platform package is installed automatically when you run `npm install mullama`.

### Building from Source

If a pre-built binary is not available for your platform, you can build from source:

```bash
# Install the napi-rs CLI
npm install -g @napi-rs/cli

# Clone and build
cd bindings/node
npm install
npm run build
```

!!! info "Build Requirements"
    - Node.js >= 16
    - Rust toolchain (1.75+)
    - System dependencies (see [Platform Setup](../getting-started/platform-setup.md))
    - For GPU support, set the appropriate environment variable (`LLAMA_CUDA=1`, `LLAMA_METAL=1`, etc.) before building

### TypeScript Support

TypeScript type definitions are included in the package. No additional `@types/` installation is needed:

```typescript
import { JsModel, JsContext, JsSamplerParams } from 'mullama';
```

The package exports complete type definitions for all classes, interfaces, and utility functions.

## Quick Start

```javascript
const { JsModel, JsContext, samplerParamsGreedy } = require('mullama');

// Load a model with GPU acceleration
const model = JsModel.load('./llama-3.2-1b.Q4_K_M.gguf', { nGpuLayers: -1 });

// Create an inference context
const ctx = new JsContext(model, { nCtx: 2048 });

// Generate text
const text = ctx.generate('Once upon a time', 100);
console.log(text);
```

---

## API Reference

### JsModel

The `JsModel` class handles model loading and provides access to model information, tokenization, and chat template formatting.

#### `static load(path, params?)`

Load a model from a GGUF file.

```typescript
static load(path: string, params?: JsModelParams): JsModel
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `string` | (required) | Path to the GGUF model file |
| `params.nGpuLayers` | `number` | `0` | Layers to offload to GPU (0 = CPU only, -1 = all) |
| `params.useMmap` | `boolean` | `true` | Use memory mapping for model loading |
| `params.useMlock` | `boolean` | `false` | Lock model in memory (prevents swapping) |
| `params.vocabOnly` | `boolean` | `false` | Only load vocabulary (for tokenization only) |

**Returns:** `JsModel` instance

**Throws:** Error if the model file cannot be loaded or is invalid

```javascript
// CPU-only loading
const model = JsModel.load('./model.gguf');

// GPU-accelerated loading (all layers on GPU)
const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });

// Partial GPU offload (first 20 layers on GPU)
const model = JsModel.load('./model.gguf', { nGpuLayers: 20 });

// Load only vocabulary for tokenization tasks
const model = JsModel.load('./model.gguf', { vocabOnly: true });
```

---

#### `tokenize(text, addBos?, special?)`

Convert text to token IDs.

```typescript
tokenize(text: string, addBos?: boolean, special?: boolean): number[]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `string` | (required) | Text to tokenize |
| `addBos` | `boolean` | `true` | Add beginning-of-sequence token |
| `special` | `boolean` | `false` | Parse special tokens in the text |

```javascript
const tokens = model.tokenize('Hello, world!');
console.log(tokens); // [1, 10994, 29892, 3186, 29991]
console.log('Token count:', tokens.length);
```

---

#### `detokenize(tokens, removeSpecial?, unparseSpecial?)`

Convert token IDs back to text.

```typescript
detokenize(tokens: number[], removeSpecial?: boolean, unparseSpecial?: boolean): string
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `number[]` | (required) | Array of token IDs |
| `removeSpecial` | `boolean` | `false` | Remove special tokens from output |
| `unparseSpecial` | `boolean` | `false` | Include special token text in output |

```javascript
const text = model.detokenize([1, 10994, 29892, 3186, 29991]);
console.log(text); // "Hello, world!"
```

---

#### `applyChatTemplate(messages, addGenerationPrompt?)`

Format chat messages using the model's built-in chat template (e.g., ChatML, Llama-3 format).

```typescript
applyChatTemplate(messages: Array<[string, string]>, addGenerationPrompt?: boolean): string
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `messages` | `[string, string][]` | (required) | Array of `[role, content]` tuples |
| `addGenerationPrompt` | `boolean` | `true` | Add generation prompt at the end |

```javascript
const messages = [
    ['system', 'You are a helpful coding assistant.'],
    ['user', 'What is a closure in JavaScript?'],
];
const prompt = model.applyChatTemplate(messages);
const response = ctx.generate(prompt, 300);
```

---

#### `metadata()`

Get all model metadata as a key-value object.

```typescript
metadata(): Record<string, string>
```

```javascript
const meta = model.metadata();
for (const [key, value] of Object.entries(meta)) {
    console.log(`${key}: ${value}`);
}
```

---

#### `tokenIsEog(token)`

Check if a token is an end-of-generation token.

```typescript
tokenIsEog(token: number): boolean
```

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nCtxTrain` | `number` | Model's training context size |
| `nEmbd` | `number` | Embedding dimension |
| `nVocab` | `number` | Vocabulary size |
| `nLayer` | `number` | Number of layers |
| `nHead` | `number` | Number of attention heads |
| `tokenBos` | `number` | BOS (beginning-of-sequence) token ID |
| `tokenEos` | `number` | EOS (end-of-sequence) token ID |
| `size` | `number` | Model size in bytes |
| `nParams` | `number` | Number of parameters |
| `description` | `string` | Model description string |
| `architecture` | `string \| null` | Model architecture (e.g., "llama") |
| `name` | `string \| null` | Model name from metadata |

```javascript
const model = JsModel.load('model.gguf');

console.log('Architecture:', model.architecture);
console.log('Parameters:', model.nParams.toLocaleString());
console.log('Embedding dim:', model.nEmbd);
console.log('Vocabulary size:', model.nVocab);
console.log('Context size:', model.nCtxTrain);
console.log('Model size:', (model.size / 1e9).toFixed(2), 'GB');
```

---

### JsContext

The `JsContext` class provides the inference context for text generation.

#### `constructor(model, params?)`

Create a new inference context.

```typescript
constructor(model: JsModel, params?: JsContextParams)
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `JsModel` | (required) | Model to use for inference |
| `params.nCtx` | `number` | `0` | Context size (0 = model default) |
| `params.nBatch` | `number` | `2048` | Batch size for prompt processing |
| `params.nThreads` | `number` | `0` | Number of threads (0 = auto) |
| `params.embeddings` | `boolean` | `false` | Enable embeddings mode |

```javascript
// Default context (model's training context size)
const ctx = new JsContext(model);

// Custom context with 4096 token window
const ctx = new JsContext(model, { nCtx: 4096, nBatch: 512 });

// Multi-threaded with explicit thread count
const ctx = new JsContext(model, { nCtx: 2048, nThreads: 8 });
```

---

#### `generate(prompt, maxTokens?, params?)`

Generate text from a string prompt.

```typescript
generate(prompt: string, maxTokens?: number, params?: JsSamplerParams): string
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | `string` | (required) | Text prompt |
| `maxTokens` | `number` | `100` | Maximum tokens to generate |
| `params` | `JsSamplerParams` | defaults | Sampling parameters |

```javascript
// Basic generation with defaults
const text = ctx.generate('The meaning of life is', 50);

// With custom sampling parameters
const text = ctx.generate('Write a poem about the ocean:', 200, {
    temperature: 0.9,
    topP: 0.95,
    topK: 50,
    penaltyRepeat: 1.1,
});
```

---

#### `generateFromTokens(tokens, maxTokens?, params?)`

Generate text from pre-tokenized input. Useful for advanced control over the input or when generating from the same prompt repeatedly.

```typescript
generateFromTokens(tokens: number[], maxTokens?: number, params?: JsSamplerParams): string
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tokens` | `number[]` | (required) | Array of token IDs |
| `maxTokens` | `number` | `100` | Maximum tokens to generate |
| `params` | `JsSamplerParams` | defaults | Sampling parameters |

```javascript
// Pre-tokenize for repeated use
const tokens = model.tokenize('Hello, AI!');
const response = ctx.generateFromTokens(tokens, 100);
```

---

#### `generateStream(prompt, maxTokens?, params?)`

Generate text and return all token pieces as an array. This allows processing tokens individually for streaming-like output.

```typescript
generateStream(prompt: string, maxTokens?: number, params?: JsSamplerParams): string[]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | `string` | (required) | Text prompt |
| `maxTokens` | `number` | `100` | Maximum tokens to generate |
| `params` | `JsSamplerParams` | defaults | Sampling parameters |

**Returns:** Array of token strings (each element is one decoded token)

```javascript
const pieces = ctx.generateStream('Once upon a time', 200);
for (const piece of pieces) {
    process.stdout.write(piece);
}
console.log(); // newline at end
```

---

#### `clearCache()`

Clear the KV cache. Call this when starting a new conversation or switching between unrelated prompts.

```typescript
clearCache(): void
```

---

#### `getEmbeddings()`

Get embeddings for the last decoded tokens (requires the context to be created with `embeddings: true`).

```typescript
getEmbeddings(): number[] | null
```

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nCtx` | `number` | Current context size |
| `nBatch` | `number` | Current batch size |

---

### JsEmbeddingGenerator

The `JsEmbeddingGenerator` class generates text embeddings for semantic similarity, search, and RAG applications.

#### `constructor(model, nCtx?, normalize?)`

Create a new embedding generator.

```typescript
constructor(model: JsModel, nCtx?: number, normalize?: boolean)
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `JsModel` | (required) | Model to use (must support embeddings) |
| `nCtx` | `number` | `512` | Context size for embedding computation |
| `normalize` | `boolean` | `true` | Normalize output embeddings to unit length |

```javascript
const gen = new JsEmbeddingGenerator(model);
// or with custom settings
const gen = new JsEmbeddingGenerator(model, 1024, true);
```

---

#### `embed(text)`

Generate an embedding vector for a single text.

```typescript
embed(text: string): number[]
```

```javascript
const embedding = gen.embed('Hello, world!');
console.log('Dimensions:', embedding.length); // e.g., 4096
```

---

#### `embedBatch(texts)`

Generate embedding vectors for multiple texts efficiently. More efficient than calling `embed()` in a loop.

```typescript
embedBatch(texts: string[]): number[][]
```

```javascript
const texts = ['Hello', 'World', 'Mullama is fast'];
const embeddings = gen.embedBatch(texts);
console.log('Count:', embeddings.length); // 3
console.log('Dims:', embeddings[0].length); // e.g., 4096
```

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nEmbd` | `number` | Embedding dimension |

---

### JsSamplerParams

Interface for configuring text generation sampling behavior.

```typescript
interface JsSamplerParams {
    temperature?: number;     // Default: 0.8 (0.0 = deterministic)
    topK?: number;            // Default: 40 (0 = disabled)
    topP?: number;            // Default: 0.95 (1.0 = disabled)
    minP?: number;            // Default: 0.05 (0.0 = disabled)
    typicalP?: number;        // Default: 1.0 (1.0 = disabled)
    penaltyRepeat?: number;   // Default: 1.1 (1.0 = disabled)
    penaltyFreq?: number;     // Default: 0.0 (0.0 = disabled)
    penaltyPresent?: number;  // Default: 0.0 (0.0 = disabled)
    penaltyLastN?: number;    // Default: 64
    seed?: number;            // Default: 0 (0 = random)
}
```

### Options Interfaces

```typescript
interface JsModelParams {
    nGpuLayers?: number;   // GPU layers (0 = CPU, -1 = all)
    useMmap?: boolean;     // Memory mapping (default: true)
    useMlock?: boolean;    // Lock in memory (default: false)
    vocabOnly?: boolean;   // Vocabulary only (default: false)
}

interface JsContextParams {
    nCtx?: number;         // Context size (0 = model default)
    nBatch?: number;       // Batch size (default: 2048)
    nThreads?: number;     // Thread count (0 = auto)
    embeddings?: boolean;  // Embeddings mode (default: false)
}
```

---

### Utility Functions

#### `samplerParamsGreedy()`

Returns sampler parameters for deterministic (greedy) generation.

```typescript
function samplerParamsGreedy(): JsSamplerParams
// { temperature: 0.0, topK: 1, topP: 1.0, minP: 0.0, penaltyRepeat: 1.0 }
```

#### `samplerParamsCreative()`

Returns sampler parameters for creative, high-randomness generation.

```typescript
function samplerParamsCreative(): JsSamplerParams
// { temperature: 1.2, topK: 100, topP: 0.95, minP: 0.0, penaltyRepeat: 1.0 }
```

#### `samplerParamsPrecise()`

Returns sampler parameters for focused, low-randomness generation.

```typescript
function samplerParamsPrecise(): JsSamplerParams
// { temperature: 0.3, topK: 20, topP: 0.8, minP: 0.05, penaltyRepeat: 1.1 }
```

#### `cosineSimilarity(a, b)`

Compute the cosine similarity between two embedding vectors.

```typescript
function cosineSimilarity(a: number[], b: number[]): number
```

**Throws:** Error if vectors have different lengths.

#### `backendInit()`

Initialize the mullama backend. Called automatically on first model load.

```typescript
function backendInit(): void
```

#### `backendFree()`

Free mullama backend resources. Call before process exit for clean shutdown.

```typescript
function backendFree(): void
```

#### `supportsGpuOffload()`

Check if GPU offloading is available on the current system.

```typescript
function supportsGpuOffload(): boolean
```

#### `systemInfo()`

Get system information about the backend (CPU features, GPU support, etc.).

```typescript
function systemInfo(): string
```

#### `maxDevices()`

Get the maximum number of compute devices supported.

```typescript
function maxDevices(): number
```

#### `version()`

Get the library version string.

```typescript
function version(): string
```

---

## Examples

### Streaming Generation

```javascript
const { JsModel, JsContext, samplerParamsCreative } = require('mullama');

const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });
const ctx = new JsContext(model, { nCtx: 4096 });

// Generate with streaming output
const pieces = ctx.generateStream(
    'Write a short story about a robot learning to paint:\n\n',
    500,
    samplerParamsCreative()
);

for (const piece of pieces) {
    process.stdout.write(piece);
}
console.log('\n--- Done ---');
```

### Embeddings and RAG

```javascript
const { JsModel, JsEmbeddingGenerator, cosineSimilarity } = require('mullama');

// Load an embedding model
const model = JsModel.load('./nomic-embed-text-v1.5.Q4_K_M.gguf');
const gen = new JsEmbeddingGenerator(model);

// Document corpus
const documents = [
    'Node.js is a JavaScript runtime built on V8.',
    'Express.js is a minimal web framework for Node.js.',
    'React is a library for building user interfaces.',
    'PostgreSQL is a powerful relational database system.',
    'Docker containers package applications with their dependencies.',
    'Kubernetes orchestrates containerized applications at scale.',
];

// Generate embeddings for all documents
const docEmbeddings = gen.embedBatch(documents);

// Search function
function search(query, topK = 3) {
    const queryEmb = gen.embed(query);

    const results = docEmbeddings.map((emb, i) => ({
        text: documents[i],
        score: cosineSimilarity(queryEmb, emb),
    }));

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
}

// Query
const results = search('How do I deploy web applications?');
console.log('Search results:');
for (const { text, score } of results) {
    console.log(`  [${score.toFixed(4)}] ${text}`);
}
```

### Chat with Templates

```javascript
const { JsModel, JsContext, samplerParamsPrecise } = require('mullama');

const model = JsModel.load('./llama-3.2-1b-instruct.Q4_K_M.gguf', {
    nGpuLayers: -1,
});
const ctx = new JsContext(model, { nCtx: 4096 });

// Multi-turn conversation
const messages = [
    ['system', 'You are a helpful coding assistant. Be concise and provide examples.'],
    ['user', 'What is a closure in JavaScript?'],
];

const prompt = model.applyChatTemplate(messages);
const response = ctx.generate(prompt, 300, samplerParamsPrecise());
console.log('Assistant:', response);

// Continue the conversation
messages.push(['assistant', response]);
messages.push(['user', 'Can you show me a practical example?']);

ctx.clearCache();
const prompt2 = model.applyChatTemplate(messages);
const response2 = ctx.generate(prompt2, 400, samplerParamsPrecise());
console.log('Assistant:', response2);
```

### Express.js Server with Streaming SSE

```javascript
const express = require('express');
const { JsModel, JsContext, samplerParamsPrecise } = require('mullama');

const app = express();
app.use(express.json());

// Load model at startup
const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });

app.post('/api/generate', (req, res) => {
    const { prompt, maxTokens = 200, temperature = 0.7 } = req.body;

    const ctx = new JsContext(model, { nCtx: 2048 });
    const text = ctx.generate(prompt, maxTokens, { temperature });
    res.json({ text });
});

app.post('/api/generate/stream', (req, res) => {
    const { prompt, maxTokens = 200, temperature = 0.7 } = req.body;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const ctx = new JsContext(model, { nCtx: 2048 });
    const pieces = ctx.generateStream(prompt, maxTokens, { temperature });

    for (const piece of pieces) {
        res.write(`data: ${JSON.stringify({ token: piece })}\n\n`);
    }
    res.write('data: [DONE]\n\n');
    res.end();
});

app.post('/api/embed', (req, res) => {
    const { texts } = req.body;
    const { JsEmbeddingGenerator } = require('mullama');
    const gen = new JsEmbeddingGenerator(model);
    const embeddings = gen.embedBatch(texts);
    res.json({ embeddings });
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

### Fastify Server

```javascript
const fastify = require('fastify')({ logger: true });
const { JsModel, JsContext } = require('mullama');

const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });

fastify.post('/generate', {
    schema: {
        body: {
            type: 'object',
            required: ['prompt'],
            properties: {
                prompt: { type: 'string' },
                maxTokens: { type: 'number', default: 200 },
                temperature: { type: 'number', default: 0.8 },
            },
        },
    },
}, async (request, reply) => {
    const { prompt, maxTokens, temperature } = request.body;
    const ctx = new JsContext(model, { nCtx: 2048 });
    const text = ctx.generate(prompt, maxTokens, { temperature });
    return { text };
});

fastify.listen({ port: 3000 });
```

### Next.js API Route

```typescript
// app/api/generate/route.ts
import { JsModel, JsContext, JsSamplerParams } from 'mullama';
import { NextResponse } from 'next/server';

// Load model once at module level
let model: JsModel | null = null;

function getModel(): JsModel {
    if (!model) {
        model = JsModel.load(
            process.env.MODEL_PATH || './models/llama-3.2-1b.Q4_K_M.gguf',
            { nGpuLayers: -1 }
        );
    }
    return model;
}

export async function POST(request: Request) {
    const { prompt, maxTokens = 200, temperature = 0.8 } = await request.json();

    const m = getModel();
    const ctx = new JsContext(m, { nCtx: 2048 });
    const params: JsSamplerParams = { temperature };
    const text = ctx.generate(prompt, maxTokens, params);

    return NextResponse.json({ text });
}
```

### Worker Threads for Non-Blocking Generation

Since generation is synchronous and CPU-intensive, use worker threads to avoid blocking the event loop:

```javascript
// worker.js
const { parentPort, workerData } = require('worker_threads');
const { JsModel, JsContext } = require('mullama');

const { modelPath, prompt, maxTokens, params } = workerData;

const model = JsModel.load(modelPath, { nGpuLayers: -1 });
const ctx = new JsContext(model, { nCtx: 2048 });
const result = ctx.generate(prompt, maxTokens, params);

parentPort.postMessage({ result });
```

```javascript
// main.js
const { Worker } = require('worker_threads');

function generateAsync(modelPath, prompt, maxTokens = 200, params = {}) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./worker.js', {
            workerData: { modelPath, prompt, maxTokens, params },
        });
        worker.on('message', (msg) => resolve(msg.result));
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) reject(new Error(`Worker exited with code ${code}`));
        });
    });
}

// Usage
async function main() {
    const result = await generateAsync('./model.gguf', 'Hello, AI!', 100);
    console.log(result);
}
main();
```

### Complete Production Chatbot

```javascript
const { JsModel, JsContext, samplerParamsPrecise, supportsGpuOffload } = require('mullama');
const readline = require('readline');

class Chatbot {
    constructor(modelPath, systemPrompt = 'You are a helpful assistant.') {
        const gpuLayers = supportsGpuOffload() ? -1 : 0;
        this.model = JsModel.load(modelPath, { nGpuLayers: gpuLayers });
        this.ctx = new JsContext(this.model, { nCtx: 4096 });
        this.messages = [['system', systemPrompt]];
        this.params = samplerParamsPrecise();

        console.log(`Model: ${this.model.name || this.model.description}`);
        console.log(`Architecture: ${this.model.architecture}`);
        console.log(`GPU: ${supportsGpuOffload() ? 'enabled' : 'CPU only'}`);
        console.log(`Context: ${this.ctx.nCtx} tokens`);
        console.log('---');
    }

    chat(userMessage) {
        this.messages.push(['user', userMessage]);

        const prompt = this.model.applyChatTemplate(this.messages);

        // Check if we are approaching context limit
        const promptTokens = this.model.tokenize(prompt, false);
        if (promptTokens.length > this.ctx.nCtx - 500) {
            // Trim older messages but keep system prompt
            this.messages = [this.messages[0], ...this.messages.slice(-4)];
            console.log('[Context trimmed to fit window]');
        }

        this.ctx.clearCache();
        const response = this.ctx.generate(prompt, 500, this.params);

        this.messages.push(['assistant', response]);
        return response;
    }

    reset() {
        this.messages = [this.messages[0]];
        this.ctx.clearCache();
    }
}

// Interactive CLI
async function main() {
    const bot = new Chatbot('./llama-3.2-1b-instruct.Q4_K_M.gguf');

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const ask = () => {
        rl.question('\nYou: ', (input) => {
            if (!input || input === '/quit') {
                rl.close();
                return;
            }
            if (input === '/reset') {
                bot.reset();
                console.log('[Conversation reset]');
                ask();
                return;
            }

            const response = bot.chat(input);
            console.log(`\nAssistant: ${response}`);
            ask();
        });
    };

    ask();
}

main();
```

### Complete RAG Example

```javascript
const { JsModel, JsEmbeddingGenerator, JsContext, cosineSimilarity, samplerParamsPrecise } = require('mullama');

class RAGPipeline {
    constructor(embeddingModelPath, generationModelPath) {
        // Separate models for embedding and generation
        this.embModel = JsModel.load(embeddingModelPath);
        this.genModel = JsModel.load(generationModelPath, { nGpuLayers: -1 });
        this.embedGen = new JsEmbeddingGenerator(this.embModel, 512, true);
        this.documents = [];
        this.embeddings = [];
    }

    // Index documents
    addDocuments(docs) {
        const newEmbeddings = this.embedGen.embedBatch(docs);
        this.documents.push(...docs);
        this.embeddings.push(...newEmbeddings);
        console.log(`Indexed ${docs.length} documents (total: ${this.documents.length})`);
    }

    // Retrieve relevant documents
    retrieve(query, topK = 3) {
        const queryEmb = this.embedGen.embed(query);

        const scored = this.embeddings.map((emb, i) => ({
            index: i,
            text: this.documents[i],
            score: cosineSimilarity(queryEmb, emb),
        }));

        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK);
    }

    // Generate answer using retrieved context
    answer(question, topK = 3) {
        const relevant = this.retrieve(question, topK);

        const context = relevant
            .map((r, i) => `[${i + 1}] ${r.text}`)
            .join('\n');

        const prompt = this.genModel.applyChatTemplate([
            ['system', 'Answer the question based only on the provided context. If the context does not contain the answer, say so.'],
            ['user', `Context:\n${context}\n\nQuestion: ${question}`],
        ]);

        const ctx = new JsContext(this.genModel, { nCtx: 2048 });
        const answer = ctx.generate(prompt, 300, samplerParamsPrecise());

        return {
            answer,
            sources: relevant.map(r => ({ text: r.text, score: r.score })),
        };
    }
}

// Usage
const rag = new RAGPipeline(
    './nomic-embed-text-v1.5.Q4_K_M.gguf',
    './llama-3.2-1b-instruct.Q4_K_M.gguf'
);

rag.addDocuments([
    'Mullama is a Rust library for running LLMs locally with native performance.',
    'Mullama supports GGUF model format and can offload layers to GPU.',
    'The Node.js bindings use napi-rs for zero-overhead native integration.',
    'Embeddings can be generated using the EmbeddingGenerator class.',
    'Streaming generation returns tokens one by one as they are produced.',
    'Models can be loaded with partial GPU offloading for limited VRAM.',
]);

const result = rag.answer('How does Mullama integrate with Node.js?');
console.log('Answer:', result.answer);
console.log('\nSources:');
for (const src of result.sources) {
    console.log(`  [${src.score.toFixed(4)}] ${src.text}`);
}
```

---

## Error Handling

All API functions throw JavaScript `Error` objects with descriptive messages on failure:

```javascript
const { JsModel, JsContext } = require('mullama');

// Model loading errors
try {
    const model = JsModel.load('./nonexistent.gguf');
} catch (error) {
    console.error('Model loading failed:', error.message);
    // "Failed to load model: file not found: ./nonexistent.gguf"
}

// Generation errors
try {
    const model = JsModel.load('./model.gguf');
    const ctx = new JsContext(model);
    const result = ctx.generateFromTokens([], 100);
} catch (error) {
    console.error('Generation failed:', error.message);
}

// Embedding errors
try {
    const model = JsModel.load('./model.gguf');
    const { JsEmbeddingGenerator } = require('mullama');
    const gen = new JsEmbeddingGenerator(model);
    const emb = gen.embed(''); // empty string
} catch (error) {
    console.error('Embedding failed:', error.message);
}
```

Common error scenarios:

| Error | Cause | Solution |
|-------|-------|----------|
| File not found | Invalid model path | Check the path exists and is readable |
| Invalid GGUF | Corrupted or wrong format | Verify the file is a valid GGUF model |
| Out of memory | Model too large for RAM | Use a smaller quantization or enable GPU offloading |
| Context too large | `nCtx` exceeds model limit | Reduce `nCtx` to within `model.nCtxTrain` |
| Vector length mismatch | `cosineSimilarity` with different-sized vectors | Ensure both vectors come from the same model |

---

## Memory Management

### Model Lifecycle

Models are the most memory-intensive objects. Best practices:

1. **Load once, reuse everywhere** -- model loading is expensive (seconds for large models). Load at application startup and reuse.
2. **Multiple contexts from one model** -- creating contexts is cheap. Create a new context per request if needed.
3. **Garbage collection** -- models are freed when garbage collected, but you can call `backendFree()` for deterministic cleanup at shutdown.

```javascript
const { JsModel, JsContext, backendFree } = require('mullama');

// Load once at startup
const model = JsModel.load('./model.gguf', { nGpuLayers: -1 });

// Create contexts as needed (cheap)
function handleRequest(prompt) {
    const ctx = new JsContext(model, { nCtx: 2048 });
    return ctx.generate(prompt, 200);
    // ctx is garbage collected after this function returns
}

// Clean shutdown
process.on('exit', () => {
    backendFree();
});
```

---

## Performance Tips

1. **Use GPU offloading** when available -- set `nGpuLayers: -1` to offload all layers for maximum throughput.

2. **Reuse model instances** -- model loading takes seconds. Load once at startup and share across requests.

3. **Tune batch size** -- larger `nBatch` values speed up prompt processing but use more memory. The default of 2048 is good for most cases.

4. **Pre-tokenize** when generating from the same prompt repeatedly -- use `model.tokenize()` once and `ctx.generateFromTokens()` for each generation.

5. **Use appropriate context size** -- set `nCtx` to the minimum needed for your use case. A 2048-token context uses much less memory than 8192.

6. **Memory mapping** -- keep `useMmap: true` (default) for faster model loading and lower peak memory usage.

7. **Batch embeddings** -- use `gen.embedBatch()` rather than calling `gen.embed()` in a loop for significantly better throughput.

8. **Worker threads** -- for web servers, consider running generation in worker threads to keep the event loop responsive.

9. **Clear cache between conversations** -- call `ctx.clearCache()` when switching between unrelated prompts to avoid context pollution.

---

## TypeScript Full Example

```typescript
import {
    JsModel,
    JsContext,
    JsEmbeddingGenerator,
    JsModelParams,
    JsContextParams,
    JsSamplerParams,
    samplerParamsGreedy,
    samplerParamsCreative,
    samplerParamsPrecise,
    cosineSimilarity,
    supportsGpuOffload,
    version,
    backendFree,
} from 'mullama';

// Type-safe model loading
const modelParams: JsModelParams = {
    nGpuLayers: supportsGpuOffload() ? -1 : 0,
    useMmap: true,
};

const model: JsModel = JsModel.load('./model.gguf', modelParams);

// Type-safe context creation
const ctxParams: JsContextParams = {
    nCtx: 2048,
    nBatch: 512,
    nThreads: 4,
};

const ctx: JsContext = new JsContext(model, ctxParams);

// Custom sampling parameters
const params: JsSamplerParams = {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    penaltyRepeat: 1.1,
    seed: 42,
};

const result: string = ctx.generate('Hello, TypeScript!', 100, params);
console.log(`[mullama v${version()}] ${result}`);

// Type-safe embeddings
const gen: JsEmbeddingGenerator = new JsEmbeddingGenerator(model);
const embedding: number[] = gen.embed('Semantic search query');
console.log(`Embedding dimensions: ${embedding.length}`);

// Cleanup
backendFree();
```
