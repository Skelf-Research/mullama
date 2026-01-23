---
title: Go Bindings
description: Idiomatic Go bindings for Mullama LLM inference via cgo, with goroutine-safe model sharing and native error handling patterns.
---

# Go Bindings

Idiomatic Go bindings for the Mullama LLM library, built with [cgo](https://pkg.go.dev/cmd/cgo) for direct integration with the C FFI layer. The bindings provide goroutine-safe model sharing, `defer`-based resource cleanup, and standard Go error patterns.

## Installation

```bash
go get github.com/neul-labs/mullama-go
```

### Prerequisites

The Go bindings require the pre-built `libmullama_ffi` shared library and header file:

```bash
# Build the FFI library from the mullama source
cargo build --release -p mullama-ffi

# The library is output to: target/release/libmullama_ffi.so (Linux)
#                            target/release/libmullama_ffi.dylib (macOS)
```

Set the appropriate environment variables so cgo can find the library:

```bash
export CGO_CFLAGS="-I/path/to/mullama/bindings/ffi/include"
export CGO_LDFLAGS="-L/path/to/mullama/target/release -lmullama_ffi"
export LD_LIBRARY_PATH="/path/to/mullama/target/release:$LD_LIBRARY_PATH"
```

!!! info "Requirements"
    - Go >= 1.21
    - cgo enabled (`CGO_ENABLED=1`, the default)
    - `libmullama_ffi` shared library
    - `mullama.h` header file

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/neul-labs/mullama-go"
)

func main() {
    // Load a model with GPU offloading
    model, err := mullama.LoadModel("./model.gguf", &mullama.ModelParams{
        NGPULayers: 32,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    // Create an inference context
    ctx, err := mullama.NewContext(model, &mullama.ContextParams{
        NCtx: 2048,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer ctx.Free()

    // Generate text
    text, err := ctx.Generate("Once upon a time", 100, nil)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(text)
}
```

---

## API Reference

### Backend Functions

#### `BackendInit()`

Initialize the mullama backend. Called automatically on first model load. Safe to call multiple times (uses `sync.Once` internally).

```go
func BackendInit()
```

#### `BackendFree()`

Free backend resources. Call before program exit for clean shutdown.

```go
func BackendFree()
```

#### `SupportsGPUOffload()`

Check if GPU offloading is available.

```go
func SupportsGPUOffload() bool
```

#### `SystemInfo()`

Get system information about the backend.

```go
func SystemInfo() string
```

#### `MaxDevices()`

Get the maximum number of compute devices.

```go
func MaxDevices() int
```

#### `Version()`

Get the library version.

```go
func Version() string
```

---

### ModelParams

Configuration for model loading.

```go
type ModelParams struct {
    // NGPULayers is the number of layers to offload to GPU
    // 0 = CPU only, -1 = all layers
    NGPULayers int32

    // UseMmap enables memory mapping for model loading
    UseMmap bool

    // UseMlock locks the model in memory (prevents swapping)
    UseMlock bool

    // VocabOnly loads only the vocabulary (for tokenization)
    VocabOnly bool
}
```

#### `DefaultModelParams()`

Returns sensible defaults for model loading.

```go
func DefaultModelParams() ModelParams
// NGPULayers: 0, UseMmap: true, UseMlock: false, VocabOnly: false
```

---

### Model

The `Model` type represents a loaded LLM model.

#### `LoadModel(path, params)`

Load a model from a GGUF file.

```go
func LoadModel(path string, params *ModelParams) (*Model, error)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `string` | Path to the GGUF model file |
| `params` | `*ModelParams` | Loading parameters (nil for defaults) |

**Returns:** `(*Model, error)`

```go
model, err := mullama.LoadModel("./model.gguf", &mullama.ModelParams{
    NGPULayers: -1,  // offload all layers
    UseMmap:    true,
})
if err != nil {
    log.Fatalf("Failed to load model: %v", err)
}
defer model.Free()
```

!!! warning "Memory Management"
    Always call `model.Free()` when done, or use `defer model.Free()`. The Go garbage collector will also call `Free()` via a finalizer, but explicit cleanup is recommended for deterministic resource release.

---

#### `model.Free()`

Release model resources. Safe to call multiple times.

```go
func (m *Model) Free()
```

---

#### `model.Tokenize(text, addBos, special)`

Convert text to token IDs.

```go
func (m *Model) Tokenize(text string, addBos bool, special bool) ([]int32, error)
```

```go
tokens, err := model.Tokenize("Hello, world!", true, false)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Tokens: %v\n", tokens)
```

---

#### `model.Detokenize(tokens, removeSpecial, unparseSpecial)`

Convert token IDs back to text.

```go
func (m *Model) Detokenize(tokens []int32, removeSpecial bool, unparseSpecial bool) (string, error)
```

```go
text, err := model.Detokenize(tokens, false, false)
if err != nil {
    log.Fatal(err)
}
fmt.Println(text)
```

---

#### Model Properties

```go
func (m *Model) NCtxTrain() int32    // Training context size
func (m *Model) NEmbd() int32        // Embedding dimension
func (m *Model) NVocab() int32       // Vocabulary size
func (m *Model) NLayer() int32       // Number of layers
func (m *Model) NHead() int32        // Number of attention heads
func (m *Model) TokenBOS() int32     // BOS token ID
func (m *Model) TokenEOS() int32     // EOS token ID
func (m *Model) Size() uint64        // Model size in bytes
func (m *Model) NParams() uint64     // Number of parameters
func (m *Model) Description() string // Model description
func (m *Model) TokenIsEOG(token int32) bool // Check if token is EOG
```

```go
fmt.Printf("Model: %s\n", model.Description())
fmt.Printf("Parameters: %d\n", model.NParams())
fmt.Printf("Layers: %d\n", model.NLayer())
fmt.Printf("Embedding dim: %d\n", model.NEmbd())
fmt.Printf("Context size: %d\n", model.NCtxTrain())
```

---

### ContextParams

Configuration for context creation.

```go
type ContextParams struct {
    // NCtx is the context size (0 = model default)
    NCtx uint32

    // NBatch is the batch size for prompt processing
    NBatch uint32

    // NThreads is the number of threads (0 = auto)
    NThreads int32

    // Embeddings enables embeddings mode
    Embeddings bool
}
```

#### `DefaultContextParams()`

Returns sensible defaults for context creation.

```go
func DefaultContextParams() ContextParams
// NCtx: 0, NBatch: 2048, NThreads: runtime.NumCPU(), Embeddings: false
```

---

### Context

The `Context` type represents an inference context.

#### `NewContext(model, params)`

Create a new inference context from a model.

```go
func NewContext(model *Model, params *ContextParams) (*Context, error)
```

```go
ctx, err := mullama.NewContext(model, &mullama.ContextParams{
    NCtx:     4096,
    NBatch:   512,
    NThreads: 8,
})
if err != nil {
    log.Fatal(err)
}
defer ctx.Free()
```

---

#### `ctx.Free()`

Release context resources.

```go
func (c *Context) Free()
```

---

#### `ctx.Generate(prompt, maxTokens, params)`

Generate text from a string prompt.

```go
func (c *Context) Generate(prompt string, maxTokens int, params *SamplerParams) (string, error)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `string` | Text prompt |
| `maxTokens` | `int` | Maximum tokens to generate |
| `params` | `*SamplerParams` | Sampling parameters (nil for defaults) |

```go
text, err := ctx.Generate("Hello, AI!", 100, nil)
if err != nil {
    log.Fatal(err)
}
fmt.Println(text)
```

---

#### `ctx.GenerateFromTokens(tokens, maxTokens, params)`

Generate text from pre-tokenized input.

```go
func (c *Context) GenerateFromTokens(tokens []int32, maxTokens int, params *SamplerParams) (string, error)
```

```go
tokens, _ := model.Tokenize("Hello!", true, false)
text, err := ctx.GenerateFromTokens(tokens, 100, nil)
```

---

#### `ctx.GenerateStream(prompt, maxTokens, params, callback)`

Generate text with a streaming callback.

```go
func (c *Context) GenerateStream(prompt string, maxTokens int, params *SamplerParams, callback StreamCallback) error
```

The `StreamCallback` type:

```go
type StreamCallback func(token string) bool
```

Return `true` from the callback to continue generation, `false` to stop.

```go
err := ctx.GenerateStream("Once upon a time", 200, nil, func(token string) bool {
    fmt.Print(token)
    return true // continue
})
if err != nil {
    log.Fatal(err)
}
fmt.Println()
```

---

#### `ctx.ClearCache()`

Clear the KV cache.

```go
func (c *Context) ClearCache()
```

---

#### Context Properties

```go
func (c *Context) NCtx() uint32   // Context size
func (c *Context) NBatch() uint32 // Batch size
```

---

### SamplerParams

Configuration for text generation sampling.

```go
type SamplerParams struct {
    Temperature    float32
    TopK           int32
    TopP           float32
    MinP           float32
    TypicalP       float32
    PenaltyRepeat  float32
    PenaltyFreq    float32
    PenaltyPresent float32
    PenaltyLastN   int32
    Seed           uint32
}
```

#### Preset Functions

```go
func DefaultSamplerParams() SamplerParams   // temperature=0.8, topK=40
func GreedySamplerParams() SamplerParams    // temperature=0.0, topK=1
func CreativeSamplerParams() SamplerParams  // temperature=1.2, topK=100
func PreciseSamplerParams() SamplerParams   // temperature=0.3, topK=20
```

```go
// Deterministic generation
text, _ := ctx.Generate("2+2=", 10, &mullama.GreedySamplerParams())

// Creative generation
params := mullama.CreativeSamplerParams()
text, _ := ctx.Generate("Write a story:", 300, &params)

// Custom parameters
params := &mullama.SamplerParams{
    Temperature:   0.7,
    TopK:          50,
    TopP:          0.9,
    PenaltyRepeat: 1.1,
}
text, _ := ctx.Generate("Hello", 100, params)
```

---

### EmbeddingGenerator

The `EmbeddingGenerator` type generates text embeddings.

#### `NewEmbeddingGenerator(model, nCtx, normalize)`

Create a new embedding generator.

```go
func NewEmbeddingGenerator(model *Model, nCtx uint32, normalize bool) (*EmbeddingGenerator, error)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `*Model` | Model for embeddings |
| `nCtx` | `uint32` | Context size (0 = 512 default) |
| `normalize` | `bool` | Normalize embeddings to unit length |

```go
eg, err := mullama.NewEmbeddingGenerator(model, 512, true)
if err != nil {
    log.Fatal(err)
}
defer eg.Free()
```

---

#### `eg.Free()`

Release embedding generator resources.

```go
func (eg *EmbeddingGenerator) Free()
```

---

#### `eg.Embed(text)`

Generate an embedding vector for text.

```go
func (eg *EmbeddingGenerator) Embed(text string) ([]float32, error)
```

```go
embedding, err := eg.Embed("Hello, world!")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Dimensions: %d\n", len(embedding))
```

---

#### `eg.EmbedBatch(texts)`

Generate embeddings for multiple texts.

```go
func (eg *EmbeddingGenerator) EmbedBatch(texts []string) ([][]float32, error)
```

```go
texts := []string{"Hello", "World", "Test"}
embeddings, err := eg.EmbedBatch(texts)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Count: %d\n", len(embeddings))
```

---

#### `eg.NEmbd()`

Get the embedding dimension.

```go
func (eg *EmbeddingGenerator) NEmbd() int32
```

---

### Utility Functions

#### `CosineSimilarity(a, b)`

Compute cosine similarity between two vectors.

```go
func CosineSimilarity(a, b []float32) (float32, error)
```

**Returns:** Similarity value between -1 and 1, or error if vectors have different lengths.

```go
sim, err := mullama.CosineSimilarity(emb1, emb2)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Similarity: %.4f\n", sim)
```

---

### Error Variables

The package defines sentinel errors for common failure cases:

```go
var (
    ErrNullPointer  = errors.New("null pointer")
    ErrModelLoad    = errors.New("failed to load model")
    ErrContext      = errors.New("failed to create context")
    ErrTokenization = errors.New("tokenization failed")
    ErrGeneration   = errors.New("generation failed")
    ErrEmbedding    = errors.New("embedding generation failed")
    ErrInvalidInput = errors.New("invalid input")
)
```

Use `errors.Is()` to check for specific error types:

```go
_, err := mullama.LoadModel("bad_path.gguf", nil)
if err != nil {
    fmt.Printf("Error: %v\n", err) // detailed message from FFI layer
}
```

---

## Examples

### Basic Generation

```go
package main

import (
    "fmt"
    "log"

    "github.com/neul-labs/mullama-go"
)

func main() {
    defer mullama.BackendFree()

    model, err := mullama.LoadModel("./model.gguf", &mullama.ModelParams{
        NGPULayers: -1,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    ctx, err := mullama.NewContext(model, &mullama.ContextParams{
        NCtx: 2048,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer ctx.Free()

    params := mullama.PreciseSamplerParams()
    text, err := ctx.Generate("Explain Go concurrency in one paragraph:", 200, &params)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(text)
}
```

### Streaming Generation

```go
package main

import (
    "fmt"
    "log"

    "github.com/neul-labs/mullama-go"
)

func main() {
    model, err := mullama.LoadModel("./model.gguf", nil)
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    ctx, err := mullama.NewContext(model, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer ctx.Free()

    fmt.Print("Response: ")
    err = ctx.GenerateStream("Tell me a joke:", 150, nil, func(token string) bool {
        fmt.Print(token)
        return true
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println()
}
```

### Embeddings and Similarity

```go
package main

import (
    "fmt"
    "log"
    "sort"

    "github.com/neul-labs/mullama-go"
)

type SearchResult struct {
    Text  string
    Score float32
}

func main() {
    model, err := mullama.LoadModel("./embedding-model.gguf", nil)
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    eg, err := mullama.NewEmbeddingGenerator(model, 512, true)
    if err != nil {
        log.Fatal(err)
    }
    defer eg.Free()

    // Index documents
    documents := []string{
        "Go is a statically typed language",
        "Python is dynamically typed",
        "Rust emphasizes memory safety",
        "The weather is nice today",
    }

    docEmbeddings, err := eg.EmbedBatch(documents)
    if err != nil {
        log.Fatal(err)
    }

    // Query
    queryEmb, err := eg.Embed("Which language is memory safe?")
    if err != nil {
        log.Fatal(err)
    }

    // Rank results
    var results []SearchResult
    for i, docEmb := range docEmbeddings {
        score, _ := mullama.CosineSimilarity(queryEmb, docEmb)
        results = append(results, SearchResult{
            Text:  documents[i],
            Score: score,
        })
    }

    sort.Slice(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })

    fmt.Println("Search results:")
    for _, r := range results {
        fmt.Printf("  [%.4f] %s\n", r.Score, r.Text)
    }
}
```

### Concurrency with Goroutines

```go
package main

import (
    "fmt"
    "log"
    "sync"

    "github.com/neul-labs/mullama-go"
)

func main() {
    model, err := mullama.LoadModel("./model.gguf", &mullama.ModelParams{
        NGPULayers: -1,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    prompts := []string{
        "What is Go?",
        "What is Rust?",
        "What is Python?",
    }

    var wg sync.WaitGroup
    results := make([]string, len(prompts))

    for i, prompt := range prompts {
        wg.Add(1)
        go func(idx int, p string) {
            defer wg.Done()

            // Each goroutine gets its own context
            ctx, err := mullama.NewContext(model, &mullama.ContextParams{
                NCtx: 1024,
            })
            if err != nil {
                log.Printf("Context error: %v", err)
                return
            }
            defer ctx.Free()

            text, err := ctx.Generate(p, 100, nil)
            if err != nil {
                log.Printf("Generation error: %v", err)
                return
            }
            results[idx] = text
        }(i, prompt)
    }

    wg.Wait()

    for i, result := range results {
        fmt.Printf("\n--- %s ---\n%s\n", prompts[i], result)
    }
}
```

!!! warning "Thread Safety"
    A `Model` can be shared across goroutines, but each goroutine must create its own `Context`. Contexts are not thread-safe and must not be shared between goroutines.

---

## Memory Management

Go bindings use finalizers for automatic cleanup, but explicit `Free()` calls are strongly recommended:

```go
model, err := mullama.LoadModel("./model.gguf", nil)
if err != nil {
    log.Fatal(err)
}
defer model.Free()  // Always defer Free() immediately after creation

ctx, err := mullama.NewContext(model, nil)
if err != nil {
    log.Fatal(err)
}
defer ctx.Free()

eg, err := mullama.NewEmbeddingGenerator(model, 512, true)
if err != nil {
    log.Fatal(err)
}
defer eg.Free()
```

**Key rules:**

1. Always `defer obj.Free()` immediately after successful creation.
2. `Free()` is safe to call multiple times (idempotent).
3. Do not use objects after calling `Free()`.
4. The GC finalizer will call `Free()` if you forget, but timing is non-deterministic.

---

## Error Handling

Go bindings follow idiomatic Go error handling patterns:

```go
model, err := mullama.LoadModel("./model.gguf", nil)
if err != nil {
    // err contains the detailed error message from the FFI layer
    log.Fatalf("Model load failed: %v", err)
}

ctx, err := mullama.NewContext(model, nil)
if err != nil {
    log.Fatalf("Context creation failed: %v", err)
}

text, err := ctx.Generate("Hello", 100, nil)
if err != nil {
    log.Fatalf("Generation failed: %v", err)
}
```

All errors include descriptive messages from the underlying C FFI layer, making debugging straightforward.

---

## HTTP Server Example

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "github.com/neul-labs/mullama-go"
)

var model *mullama.Model

type GenerateRequest struct {
    Prompt      string  `json:"prompt"`
    MaxTokens   int     `json:"max_tokens"`
    Temperature float32 `json:"temperature"`
}

type GenerateResponse struct {
    Text string `json:"text"`
}

func generateHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req GenerateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    if req.MaxTokens == 0 {
        req.MaxTokens = 200
    }
    if req.Temperature == 0 {
        req.Temperature = 0.8
    }

    ctx, err := mullama.NewContext(model, &mullama.ContextParams{NCtx: 2048})
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer ctx.Free()

    params := &mullama.SamplerParams{
        Temperature: req.Temperature,
        TopK:        40,
        TopP:        0.95,
    }

    text, err := ctx.Generate(req.Prompt, req.MaxTokens, params)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(GenerateResponse{Text: text})
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req GenerateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    if req.MaxTokens == 0 {
        req.MaxTokens = 200
    }

    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "Streaming not supported", http.StatusInternalServerError)
        return
    }

    ctx, err := mullama.NewContext(model, &mullama.ContextParams{NCtx: 2048})
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer ctx.Free()

    err = ctx.GenerateStream(req.Prompt, req.MaxTokens, nil, func(token string) bool {
        data, _ := json.Marshal(map[string]string{"token": token})
        fmt.Fprintf(w, "data: %s\n\n", data)
        flusher.Flush()
        return true
    })
    if err != nil {
        log.Printf("Stream error: %v", err)
    }

    fmt.Fprintf(w, "data: [DONE]\n\n")
    flusher.Flush()
}

func main() {
    defer mullama.BackendFree()

    var err error
    model, err = mullama.LoadModel("./model.gguf", &mullama.ModelParams{
        NGPULayers: -1,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    http.HandleFunc("/generate", generateHandler)
    http.HandleFunc("/generate/stream", streamHandler)

    log.Println("Server listening on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

---

## Performance Tips

1. **GPU offloading** -- set `NGPULayers` to `-1` for maximum throughput.

2. **Reuse models** -- model loading is expensive. Load once at startup and share across goroutines.

3. **Per-goroutine contexts** -- contexts are not thread-safe. Create a new context per goroutine or per request.

4. **Defer Free()** -- always `defer obj.Free()` immediately after creation to prevent resource leaks.

5. **Batch embeddings** -- use `EmbedBatch()` for multiple texts rather than calling `Embed()` in a loop.

6. **Context pooling** -- for high-throughput servers, consider a sync.Pool of pre-created contexts (remember to call `ClearCache()` between uses).
