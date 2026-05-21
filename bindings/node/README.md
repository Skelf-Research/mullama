# Mullama for Node.js — local LLM inference with GGUF models, native llama.cpp performance

[![npm](https://img.shields.io/npm/v/mullama)](https://www.npmjs.com/package/mullama)
[![npm downloads](https://img.shields.io/npm/dm/mullama)](https://www.npmjs.com/package/mullama)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/cognisoc/mullama/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/cognisoc/mullama/ci.yml?branch=main&label=CI)](https://github.com/cognisoc/mullama/actions)

## What is Mullama for Node.js?

`mullama` is an npm package that runs local LLMs from GGUF files — Llama 3.2, Qwen 2.5, DeepSeek R1, Mistral, Phi 3, Gemma 2, LLaVA, and anything else in GGUF format — directly inside your Node.js, Bun, Deno, or Electron process. It's a `napi-rs` native module that wraps `llama.cpp` through a Rust core, giving you the same throughput as `llama.cpp` itself with zero HTTP overhead, no Ollama daemon, and no Python subprocess to manage.

If you want to embed a local LLM in a Node backend, an Electron desktop app, a CLI tool, or a serverless function — and you'd rather not shell out to `ollama` over `fetch` — `mullama` is a single `npm install` that covers it.

## Why use `mullama` over `node-llama-cpp` or `ollama` npm?

| | `mullama` | `node-llama-cpp` | `ollama` (npm) |
|---|:-:|:-:|:-:|
| In-process inference (no daemon) | ✓ | ✓ | ✗ (HTTP to `ollama serve`) |
| TypeScript types out of the box | ✓ | ✓ | ✓ |
| Pre-built binaries (linux x64/arm64, macOS x64/arm64, Windows) | ✓ | ✓ | n/a |
| GGUF models | ✓ | ✓ | ✓ |
| Streaming generation | ✓ | ✓ | ✓ |
| Built-in embeddings | ✓ | ✓ | ✓ (via HTTP) |
| Vision models (LLaVA, Moondream) | ✓ | ✓ | ✓ |
| Memory-safe core (no `node-gyp` build at install) | ✓ (`napi-rs` prebuilds) | requires native build | n/a |
| Ships an OpenAI-compatible HTTP server too | ✓ (`npx mullama serve`) | ✗ | requires Ollama |
| Electron + Bun support | ✓ | ✓ | ✓ |

## Install

```bash
npm install mullama
# or
pnpm add mullama
# or
yarn add mullama
# or
bun add mullama
```

Pre-built native modules ship for Node 18+ on linux-x64-gnu, linux-arm64-gnu, darwin-x64, darwin-arm64 (Metal), and win32-x64-msvc.

## Quick start

```typescript
import { Model, Context } from "mullama";

const model = Model.load("./llama3.2-1b.gguf", { nGpuLayers: 32 });
const ctx = new Context(model, { nCtx: 2048 });

console.log(ctx.generate("What is the capital of France?", 100));
```

CommonJS works too:

```javascript
const { Model, Context } = require("mullama");
```

Don't have a GGUF file yet? Grab one from Hugging Face:

```bash
npx huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## Streaming

```typescript
for (const piece of ctx.generateStream("Once upon a time", 100)) {
  process.stdout.write(piece);
}
```

## Embeddings (RAG-ready)

```typescript
import { Model, EmbeddingGenerator, cosineSimilarity } from "mullama";

const model = Model.load("./nomic-embed-text-v1.5.Q4_K_M.gguf");
const gen = new EmbeddingGenerator(model);

const a = gen.embed("How do I cook pasta?");
const b = gen.embed("What's the best way to boil noodles?");
console.log(cosineSimilarity(a, b));   // ≈ 0.9

const vectors = gen.embedBatch(["doc 1", "doc 2", "doc 3"]);
```

Works with any GGUF embedding model — `nomic-embed`, `bge:small`, `bge:large`, and others.

## Chat templates

```typescript
const messages: [string, string][] = [
  ["system", "You are a concise TypeScript tutor."],
  ["user", "Explain discriminated unions."],
];
const prompt = model.applyChatTemplate(messages);
console.log(ctx.generate(prompt, 300));
```

The model's built-in chat template (Llama 3, Qwen, Mistral, Phi, Gemma all supported) is read from the GGUF metadata automatically.

## Tokenization

```typescript
const tokens = model.tokenize("Hello, world!");
console.log(tokens);                    // [128000, 9906, 11, 1917, 0]
console.log(model.detokenize(tokens));  // "Hello, world!"
```

## Sampler presets

```typescript
import { samplerParamsGreedy, samplerParamsCreative, samplerParamsPrecise } from "mullama";

ctx.generate("Tell me a fact about France.", 100, samplerParamsGreedy());    // deterministic
ctx.generate("Write a poem about the moon.", 100, samplerParamsCreative());  // high randomness
ctx.generate("Summarize this article.", 100, samplerParamsPrecise());        // low randomness

// Or pass any combination of fields directly
ctx.generate("...", 100, { temperature: 0.7, topP: 0.9, topK: 40, penaltyRepeat: 1.1 });
```

## Model information

```typescript
console.log({
  architecture: model.architecture,
  parameters:   model.nParams,
  embeddingDim: model.nEmbd,
  vocabulary:   model.nVocab,
  contextSize:  model.nCtxTrain,
  sizeGB:       (model.size / 1e9).toFixed(2),
});
```

## Supported models

Any GGUF file works. Common picks:

- **Llama 3.2** — 1B, 3B Instruct (Q4_K_M is the sweet spot)
- **Llama 3.1** — 8B, 70B
- **Qwen 2.5** — 0.5B – 72B, plus Coder 7B/14B/32B
- **DeepSeek R1** — distilled 1.5B / 7B / 14B / 32B (reasoning)
- **Mistral / Mixtral / Codestral** — 7B, 8x7B MoE, 22B
- **Phi 3 / 3.5** — mini, medium
- **Gemma 2** — 2B, 9B, 27B
- **Vision** — LLaVA 1.5 7B/13B, LLaVA-Phi3, Moondream 2B
- **Embeddings** — `nomic-embed-text-v1.5`, `bge-small-en-v1.5`, `bge-large-en-v1.5`

## GPU acceleration

The pre-built npm package is CPU + Metal (Apple Silicon). For NVIDIA CUDA or AMD ROCm, install from source with the relevant env var:

```bash
LLAMA_CUDA=1 npm install mullama --build-from-source
```

| Backend | Env var | Hardware |
|---|---|---|
| CUDA | `LLAMA_CUDA=1` | NVIDIA |
| Metal | `LLAMA_METAL=1` | Apple Silicon (default for darwin-arm64 wheel) |
| ROCm | `LLAMA_HIPBLAS=1` | AMD |
| Vulkan | `LLAMA_VULKAN=1` | cross-platform |
| SYCL | `LLAMA_SYCL=1` | Intel Arc |

Pass `nGpuLayers: -1` to offload all layers, or a positive integer for partial offload.

## API reference

### `Model`

```typescript
class Model {
  static load(
    path: string,
    params?: {
      nGpuLayers?: number;   // 0=CPU, -1=all
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
  readonly nCtxTrain: number;
  readonly nEmbd: number;
  readonly nVocab: number;
  readonly nLayer: number;
  readonly nHead: number;
  readonly tokenBos: number;
  readonly tokenEos: number;
  readonly size: number;
  readonly nParams: number;
  readonly description: string;
  readonly architecture: string | null;
  readonly name: string | null;
}
```

### `Context`

```typescript
class Context {
  constructor(
    model: Model,
    params?: {
      nCtx?: number;        // 0 = model default
      nBatch?: number;
      nThreads?: number;    // 0 = auto
      embeddings?: boolean;
    }
  );

  generate(prompt: string, maxTokens?: number, params?: SamplerParams): string;
  generateFromTokens(tokens: number[], maxTokens?: number, params?: SamplerParams): string;
  generateStream(prompt: string, maxTokens?: number, params?: SamplerParams): Iterable<string>;
  clearCache(): void;

  readonly nCtx: number;
  readonly nBatch: number;
}
```

### `SamplerParams`

```typescript
interface SamplerParams {
  temperature?: number;     // Default: 0.8
  topK?: number;            // Default: 40
  topP?: number;            // Default: 0.95
  minP?: number;            // Default: 0.05
  typicalP?: number;        // Default: 1.0
  penaltyRepeat?: number;   // Default: 1.1
  penaltyFreq?: number;     // Default: 0.0
  penaltyPresent?: number;  // Default: 0.0
  penaltyLastN?: number;    // Default: 64
  seed?: number;            // Default: 0 (random)
}

function samplerParamsGreedy(): SamplerParams;
function samplerParamsCreative(): SamplerParams;
function samplerParamsPrecise(): SamplerParams;
```

### `EmbeddingGenerator`

```typescript
class EmbeddingGenerator {
  constructor(model: Model, nCtx?: number, normalize?: boolean);
  embed(text: string): number[];
  embedBatch(texts: string[]): number[][];
  readonly nEmbd: number;
}
```

### Utilities

```typescript
function cosineSimilarity(a: number[], b: number[]): number;
function backendInit(): void;
function backendFree(): void;
function supportsGpuOffload(): boolean;
function systemInfo(): string;
function maxDevices(): number;
function version(): string;
```

## FAQ

### Is Mullama production ready?

Yes. The 0.3.x line is used in production. The native module is built by `napi-rs`, which produces stable ABI-tagged prebuilds — your app won't break across Node minor versions.

### How does `mullama` compare to `node-llama-cpp`?

Same `llama.cpp` engine underneath. `mullama` ships an Ollama-compatible CLI + daemon, an OpenAI/Anthropic-compatible HTTP server, vision and embedding models, and is part of a polyglot library so the same models work identically from Python / Go / PHP / Rust. `node-llama-cpp` is more focused on the Node embedding use case with its own grammar and function-calling utilities.

### How does `mullama` compare to the `ollama` npm package?

The `ollama` npm package is an HTTP client — it talks to a running `ollama` daemon over `localhost:11434`. `mullama` runs inference in-process (no daemon, no HTTP, no Ollama install). If you already run `ollama serve`, `mullama` can also act as an HTTP client to it via the OpenAI-compatible API.

### Does Mullama work in Electron?

Yes. The native module is `napi-rs`-built, which is compatible with Electron's Node ABI. Use `electron-rebuild` if you hit a Node version mismatch. See the [Electron recipe](https://docs.cognisoc.com/mullama/recipes/electron/).

### Does Mullama work with Bun and Deno?

Bun: yes, `napi-rs` modules load via Bun's Node-API compatibility shim. Deno: yes via `npm:mullama` specifier. Workerd / Cloudflare Workers: not currently — requires a Node-compatible runtime.

### Are TypeScript types included?

Yes. The package ships `index.d.ts` generated by `napi-rs` from the Rust source — types stay in sync with the implementation automatically. No `@types/mullama` install needed.

### Does it support function calling / tool use?

Yes, via the underlying `llama.cpp` grammar-constrained sampling. The high-level `tools` parameter in the OpenAI-compatible HTTP API is also supported when running `npx mullama serve`. See [the tool-calling guide](https://docs.cognisoc.com/mullama/api/tool-calling/).

### Where do I report bugs?

[GitHub Issues](https://github.com/cognisoc/mullama/issues). For usage questions, [Discussions](https://github.com/cognisoc/mullama/discussions).

## Other language bindings

- [Rust core (`mullama` on crates.io)](https://crates.io/crates/mullama)
- [Python (`mullama` on PyPI)](https://pypi.org/project/mullama/) · [README](../python/README.md)
- [Go (`github.com/cognisoc/mullama` on pkg.go.dev)](https://pkg.go.dev/github.com/cognisoc/mullama) · [README](../go/README.md)
- [PHP (`mullama/mullama` on Packagist)](https://packagist.org/packages/mullama/mullama) · [README](../php/README.md)
- [C/C++ (FFI library)](../ffi/README.md)

## Documentation

Full Node.js docs at **[docs.cognisoc.com/mullama/bindings/nodejs](https://docs.cognisoc.com/mullama/bindings/nodejs/)**.

## Development

```bash
npm install
npm run build         # release
npm run build:debug   # debug
npm test
MULLAMA_TEST_MODEL=/path/to/model.gguf npm test
```

## License

MIT OR Apache-2.0
