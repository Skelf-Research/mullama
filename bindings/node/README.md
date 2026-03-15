# Mullama Node.js Bindings

High-performance Node.js bindings for the Mullama LLM library, enabling fast local inference with GGUF models.

## Installation

### From npm (when published)

```bash
npm install mullama
```

### From Source

Requires Rust and the napi-rs CLI:

```bash
# Install napi CLI
npm install -g @napi-rs/cli

# Build and install
cd bindings/node
npm install
npm run build
```

## Quick Start

```javascript
const { Model, Context, samplerParamsGreedy } = require('mullama');

// Load a model
const model = Model.load('./model.gguf', { nGpuLayers: 32 });

// Create a context
const ctx = new Context(model, { nCtx: 2048 });

// Generate text
const text = ctx.generate('Once upon a time', 100);
console.log(text);

// With custom sampling parameters
const params = { temperature: 0.7, topP: 0.9 };
const text2 = ctx.generate('Hello, AI!', 50, params);
```

## Features

### Text Generation

```javascript
const { Model, Context, samplerParamsGreedy, samplerParamsCreative } = require('mullama');

const model = Model.load('model.gguf');
const ctx = new Context(model);

// Basic generation
const text = ctx.generate('Hello', 100);

// With custom parameters
const text2 = ctx.generate('Hello', 100, { temperature: 0.8, topK: 40 });

// Greedy (deterministic) generation
const text3 = ctx.generate('Hello', 100, samplerParamsGreedy());

// Creative generation
const text4 = ctx.generate('Hello', 100, samplerParamsCreative());
```

### Streaming Generation

```javascript
// Get tokens as they're generated
const pieces = ctx.generateStream('Once upon a time', 100);
for (const piece of pieces) {
    process.stdout.write(piece);
}
```

### Tokenization

```javascript
// Tokenize text
const tokens = model.tokenize('Hello, world!');
console.log('Tokens:', tokens);

// Detokenize back to text
const text = model.detokenize(tokens);
console.log('Text:', text);
```

### Embeddings

```javascript
const { Model, EmbeddingGenerator, cosineSimilarity } = require('mullama');

const model = Model.load('model.gguf');
const gen = new EmbeddingGenerator(model);

// Generate embeddings
const emb1 = gen.embed('Hello, world!');
const emb2 = gen.embed('Hi there!');

// Compute similarity
const similarity = cosineSimilarity(emb1, emb2);
console.log('Similarity:', similarity);

// Batch embedding
const texts = ['Hello', 'World', 'Test'];
const embeddings = gen.embedBatch(texts);
```

### Chat Templates

```javascript
// Format chat messages
const messages = [
    ['system', 'You are a helpful assistant.'],
    ['user', 'What is JavaScript?'],
];
const prompt = model.applyChatTemplate(messages);
const text = ctx.generate(prompt, 200);
```

### Model Information

```javascript
const model = Model.load('model.gguf');

console.log('Architecture:', model.architecture);
console.log('Parameters:', model.nParams);
console.log('Embedding dim:', model.nEmbd);
console.log('Vocabulary size:', model.nVocab);
console.log('Context size:', model.nCtxTrain);
console.log('Model size:', (model.size / 1e9).toFixed(2), 'GB');

// Get all metadata
const metadata = model.metadata();
for (const [key, value] of Object.entries(metadata)) {
    console.log(`${key}: ${value}`);
}
```

## API Reference

### Model

```typescript
class Model {
    static load(
        path: string,
        params?: {
            nGpuLayers?: number;  // GPU layers (0=CPU, -1=all)
            useMmap?: boolean;
            useMlock?: boolean;
            vocabOnly?: boolean;
        }
    ): Model;

    tokenize(text: string, addBos?: boolean, special?: boolean): number[];
    detokenize(tokens: number[], removeSpecial?: boolean): string;
    applyChatTemplate(messages: [string, string][], addGenerationPrompt?: boolean): string;
    metadata(): Record<string, string>;

    // Properties
    nCtxTrain: number;
    nEmbd: number;
    nVocab: number;
    nLayer: number;
    nHead: number;
    tokenBos: number;
    tokenEos: number;
    size: number;
    nParams: number;
    description: string;
    architecture: string | null;
    name: string | null;
}
```

### Context

```typescript
class Context {
    constructor(
        model: Model,
        params?: {
            nCtx?: number;      // 0 = model default
            nBatch?: number;
            nThreads?: number;  // 0 = auto
            embeddings?: boolean;
        }
    );

    generate(
        prompt: string,
        maxTokens?: number,
        params?: SamplerParams
    ): string;

    generateFromTokens(
        tokens: number[],
        maxTokens?: number,
        params?: SamplerParams
    ): string;

    generateStream(
        prompt: string,
        maxTokens?: number,
        params?: SamplerParams
    ): string[];

    clearCache(): void;

    // Properties
    nCtx: number;
    nBatch: number;
}
```

### SamplerParams

```typescript
interface SamplerParams {
    temperature?: number;    // Default: 0.8
    topK?: number;          // Default: 40
    topP?: number;          // Default: 0.95
    minP?: number;          // Default: 0.05
    typicalP?: number;      // Default: 1.0
    penaltyRepeat?: number; // Default: 1.1
    penaltyFreq?: number;   // Default: 0.0
    penaltyPresent?: number;// Default: 0.0
    penaltyLastN?: number;  // Default: 64
    seed?: number;          // Default: 0 (random)
}

function samplerParamsGreedy(): SamplerParams;
function samplerParamsCreative(): SamplerParams;
function samplerParamsPrecise(): SamplerParams;
```

### EmbeddingGenerator

```typescript
class EmbeddingGenerator {
    constructor(
        model: Model,
        nCtx?: number,      // Default: 512
        normalize?: boolean // Default: true
    );

    embed(text: string): number[];
    embedBatch(texts: string[]): number[][];

    // Properties
    nEmbd: number;
}
```

### Utility Functions

```typescript
function cosineSimilarity(a: number[], b: number[]): number;
function backendInit(): void;
function backendFree(): void;
function supportsGpuOffload(): boolean;
function systemInfo(): string;
function maxDevices(): number;
function version(): string;
```

## Development

### Building

```bash
# Install dependencies
npm install

# Build in debug mode
npm run build:debug

# Build in release mode
npm run build
```

### Testing

```bash
# Run tests (without model)
npm test

# Run tests with a model
MULLAMA_TEST_MODEL=/path/to/model.gguf npm test
```

## License

MIT OR Apache-2.0
