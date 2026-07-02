# Mullama for Go — local LLM inference with GGUF models, native llama.cpp performance

[![Go Reference](https://pkg.go.dev/badge/github.com/cognisoc/mullama.svg)](https://pkg.go.dev/github.com/cognisoc/mullama)
[![Go Report Card](https://goreportcard.com/badge/github.com/cognisoc/mullama)](https://goreportcard.com/report/github.com/cognisoc/mullama)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/cognisoc/mullama/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/cognisoc/mullama/ci.yml?branch=main&label=CI)](https://github.com/cognisoc/mullama/actions)

## What is Mullama for Go?

`github.com/cognisoc/mullama` is a Go module that runs local LLMs from GGUF files — Llama 3.2, Qwen 2.5, DeepSeek R1, Mistral, Phi 3, Gemma 2, LLaVA, and any other GGUF model — directly inside your Go process via CGO. It binds to the Mullama FFI library (a Rust core wrapping `llama.cpp`), giving you native llama.cpp throughput with idiomatic Go ergonomics (struct fields, errors, defers) — no Ollama daemon, no HTTP round-trip, no separate Python or Node sidecar.

Use it as the Go-native alternative to shelling out to `ollama` from your service, or when you want to embed an LLM directly in a Go CLI, microservice, or worker without orchestrating a separate inference server.

## Why use `mullama` over `ollama-go` or shelling out to Ollama?

| | `mullama` | `ollama-go` (HTTP client) | Shell out to `llama.cpp` |
|---|:-:|:-:|:-:|
| In-process inference (no daemon) | ✓ | ✗ (HTTP to `ollama serve`) | ✓ (via `exec`) |
| Idiomatic Go API (structs + errors) | ✓ | partial | ✗ (raw stdin/stdout) |
| Streaming via callback | ✓ | ✓ | requires pipe parsing |
| Built-in embeddings | ✓ | ✓ (via HTTP) | requires separate binary |
| Goroutine-safe model handle | ✓ | n/a | n/a |
| Vision / multimodal | ✓ | ✓ | ✓ |
| Pre-built FFI library for linux x64/arm64, macOS x64/arm64, Windows | ✓ | n/a | depends on build |

## Install

```bash
go get github.com/cognisoc/mullama
```

The Go package uses CGO to load `libmullama_ffi.so` / `.dylib` / `.dll`. Pre-built libraries are attached to every GitHub release as `mullama-ffi-<version>-<target-triple>.tar.gz` — download the one for your platform and either install it at a standard location or point `CGO_LDFLAGS` at it. The release `tar.gz` also contains the C header (`mullama.h`) the cgo bindings expect.

## Quick start

```go
package main

import (
	"fmt"
	"log"

	"github.com/cognisoc/mullama"
)

func main() {
	mullama.BackendInit()
	defer mullama.BackendFree()

	model, err := mullama.LoadModel("./llama3.2-1b.gguf", &mullama.ModelParams{NGPULayers: 32})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	ctx, err := mullama.NewContext(model, &mullama.ContextParams{NCtx: 2048})
	if err != nil {
		log.Fatal(err)
	}
	defer ctx.Free()

	text, err := ctx.Generate("What is the capital of France?", 100, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
}
```

## Streaming

```go
err := ctx.GenerateStream("Once upon a time", 100, nil, func(token string) bool {
	fmt.Print(token)
	return true   // return false to stop early
})
```

## Embeddings (RAG-ready)

```go
gen, err := mullama.NewEmbeddingGenerator(model, 512, true)
if err != nil { log.Fatal(err) }
defer gen.Free()

a, _ := gen.Embed("How do I cook pasta?")
b, _ := gen.Embed("What's the best way to boil noodles?")
sim, _ := mullama.CosineSimilarity(a, b)
fmt.Printf("similarity: %f\n", sim)

vectors, _ := gen.EmbedBatch([]string{"doc 1", "doc 2", "doc 3"})
```

## Sampler presets

```go
greedy := mullama.GreedySamplerParams()       // deterministic
creative := mullama.CreativeSamplerParams()   // high randomness
precise := mullama.PreciseSamplerParams()     // low randomness

text, _ := ctx.Generate("Tell me a fact about France.", 100, &greedy)
```

## Tokenization

```go
tokens, _ := model.Tokenize("Hello, world!", true, false)
fmt.Println(tokens)                  // [128000 9906 11 1917 0]
text, _ := model.Detokenize(tokens)
fmt.Println(text)                    // "Hello, world!"
```

## Supported models

Any GGUF file works. Common picks:

- **Llama 3.2** — 1B, 3B Instruct
- **Llama 3.1** — 8B, 70B
- **Qwen 2.5** — 0.5B – 72B, Coder variants
- **DeepSeek R1** — distilled 1.5B / 7B / 14B / 32B
- **Mistral / Mixtral / Codestral** — 7B, 8x7B MoE, 22B
- **Phi 3 / 3.5** — mini, medium
- **Gemma 2** — 2B, 9B, 27B
- **Vision** — LLaVA 1.5 7B/13B, Moondream 2B
- **Embeddings** — `nomic-embed`, `bge:small`, `bge:large`

## GPU acceleration

The pre-built FFI library is CPU + Metal (Apple Silicon). For NVIDIA CUDA, AMD ROCm, Vulkan, or Intel SYCL, download the CUDA variant from a release (`mullama-ffi-<version>-x86_64-unknown-linux-gnu-cuda.tar.gz`) or build from source with the relevant env var (`LLAMA_CUDA=1`, `LLAMA_HIPBLAS=1`, etc.).

Pass `NGPULayers: -1` to offload all layers, or a positive integer for partial offload.

## API reference

### Backend lifecycle

```go
func BackendInit()
func BackendFree()
func SupportsGPUOffload() bool
func SystemInfo() string
func MaxDevices() int
func Version() string
```

### `Model`

```go
type ModelParams struct {
	NGPULayers int32  // 0=CPU, -1=all
	UseMmap    bool
	UseMlock   bool
	VocabOnly  bool
}

func LoadModel(path string, params *ModelParams) (*Model, error)

func (m *Model) Free()
func (m *Model) Tokenize(text string, addBos, special bool) ([]int32, error)
func (m *Model) Detokenize(tokens []int32) (string, error)

// Properties
func (m *Model) NCtxTrain() int32
func (m *Model) NEmbd() int32
func (m *Model) NVocab() int32
func (m *Model) NLayer() int32
func (m *Model) NHead() int32
func (m *Model) TokenBOS() int32
func (m *Model) TokenEOS() int32
func (m *Model) Size() uint64
func (m *Model) NParams() uint64
func (m *Model) Description() string
func (m *Model) TokenIsEOG(token int32) bool
```

### `Context`

```go
type ContextParams struct {
	NCtx       uint32   // 0 = model default
	NBatch     uint32
	NThreads   int32    // 0 = auto
	Embeddings bool
}

func NewContext(model *Model, params *ContextParams) (*Context, error)
func (c *Context) Free()
func (c *Context) Generate(prompt string, maxTokens int, params *SamplerParams) (string, error)
func (c *Context) GenerateFromTokens(tokens []int32, maxTokens int, params *SamplerParams) (string, error)
func (c *Context) GenerateStream(prompt string, maxTokens int, params *SamplerParams, callback StreamCallback) error
func (c *Context) ClearCache()
func (c *Context) NCtx() uint32
func (c *Context) NBatch() uint32

type StreamCallback func(token string) bool   // return false to stop
```

### `SamplerParams`

```go
type SamplerParams struct {
	Temperature    float32   // Default: 0.8
	TopK           int32     // Default: 40
	TopP           float32   // Default: 0.95
	MinP           float32   // Default: 0.05
	TypicalP       float32   // Default: 1.0
	PenaltyRepeat  float32   // Default: 1.1
	PenaltyFreq    float32   // Default: 0.0
	PenaltyPresent float32   // Default: 0.0
	PenaltyLastN   int32     // Default: 64
	PenalizeNL     bool      // Default: true
	IgnoreEOS      bool      // Default: false
	Seed           uint32    // Default: 0 (random)
}

func DefaultSamplerParams() SamplerParams
func GreedySamplerParams() SamplerParams
func CreativeSamplerParams() SamplerParams
func PreciseSamplerParams() SamplerParams
```

### `EmbeddingGenerator`

```go
func NewEmbeddingGenerator(model *Model, nCtx uint32, normalize bool) (*EmbeddingGenerator, error)
func (g *EmbeddingGenerator) Free()
func (g *EmbeddingGenerator) Embed(text string) ([]float32, error)
func (g *EmbeddingGenerator) EmbedBatch(texts []string) ([][]float32, error)
func (g *EmbeddingGenerator) NEmbd() int32

func CosineSimilarity(a, b []float32) (float32, error)
```

## FAQ

### Is Mullama production ready?

Yes. The 0.3.x line is used in production. `Model` handles are goroutine-safe (the Rust core uses Arc reference counting internally). Each goroutine should hold its own `*Context` — contexts are not safe for concurrent generation, but you can fan out multiple contexts off one model cheaply.

### Does it require CGO?

Yes. The Go bindings load `libmullama_ffi` via CGO. Set `CGO_ENABLED=1` when building. Pre-built FFI libraries are attached to every GitHub release; alternatively, build from source with `cargo build --release -p mullama-ffi`.

### How does Mullama compare to `ollama-go`?

`ollama-go` is an HTTP client to a running `ollama serve` daemon. Mullama runs inference in-process via CGO — no daemon, no HTTP, no Ollama install. If you already run `ollama serve`, Mullama can also act as a client to it via the OpenAI-compatible API on the same port.

### How does Mullama compare to `langchaingo`?

Different layers. `langchaingo` is an orchestration framework (chains, agents, retrievers). Mullama is the inference engine that backs those chains. You can plug Mullama into `langchaingo` as the LLM and embedding provider — see [the LangChain Go recipe](https://docs.cognisoc.com/mullama/integrations/langchaingo/).

### How do I deploy a Mullama Go service in a Docker container?

Use the official CPU base image (`ghcr.io/cognisoc/mullama:<version>`) as a multi-stage builder for the FFI library, then copy `libmullama_ffi.so` into your Go runtime stage. See the [Docker recipe](https://docs.cognisoc.com/mullama/deployment/docker-go/).

### Can I cross-compile a Mullama Go binary?

Cross-compilation with CGO is awkward. The supported pattern is: build on the target platform (or use `xgo` / `goreleaser` with `cgo: true`). For static linking, the FFI build supports `.a` output — pass `LDFLAGS="-l:libmullama_ffi.a"` and link against the static library.

### Where do I report bugs?

[GitHub Issues](https://github.com/cognisoc/mullama/issues). For usage questions, [Discussions](https://github.com/cognisoc/mullama/discussions).

## Other language bindings

- [Rust core (`mullama` on crates.io)](https://crates.io/crates/mullama)
- [Python (`mullama` on PyPI)](https://pypi.org/project/mullama/) · [README](../python/README.md)
- [Node.js / TypeScript (`mullama` on npm)](https://www.npmjs.com/package/mullama) · [README](../node/README.md)
- [PHP (`mullama/mullama` on Packagist)](https://packagist.org/packages/mullama/mullama) · [README](../php/README.md)
- [C/C++ (FFI library)](../ffi/README.md)

## Documentation

Full Go docs at **[docs.cognisoc.com/mullama/bindings/go](https://docs.cognisoc.com/mullama/bindings/go/)**.

## Development

```bash
# Build FFI library
cargo build --release -p mullama-ffi

# Build + test Go bindings
cd bindings/go
go build ./...
go test -v
MULLAMA_TEST_MODEL=/path/to/model.gguf go test -v
```

## Releasing

```bash
git tag v0.3.1
git push origin v0.3.1
GOPROXY=https://proxy.golang.org go list -m github.com/cognisoc/mullama@v0.3.1
```

## License

MIT OR Apache-2.0

---

## Part of the Cognisoc stack

**[Cognisoc](https://www.cognisoc.com)** builds open-source LLM inference for every language and every device — *LLM inference, everywhere.* This project is one of six:

| Project | Language | What it does |
|---|---|---|
| mullama **(this project)** | Python · Node · Go · PHP · Rust · C | Local LLM runtime & server, drop-in Ollama alternative |
| [unillm](https://github.com/cognisoc/unillm) | Rust | Modular inference runtime, 47 architectures |
| [llamafu](https://github.com/cognisoc/llamafu) | Dart / Flutter | On-device inference for mobile apps |
| [llmdot](https://github.com/cognisoc/llmdot) | C# / .NET | Local GGUF inference for the .NET ecosystem |
| [cllm](https://github.com/cognisoc/cllm) | C | Bare-metal unikernel — boots straight into inference |
| [zigllm](https://github.com/cognisoc/zigllm) | Zig | Learn LLMs by building one, from tensors to text |

🌐 [cognisoc.com](https://www.cognisoc.com) · 📚 [docs.cognisoc.com](https://docs.cognisoc.com) · 🐙 [github.com/cognisoc](https://github.com/cognisoc)
