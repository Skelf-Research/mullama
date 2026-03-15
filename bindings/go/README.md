# Mullama Go Bindings

High-performance Go bindings for the Mullama LLM library, enabling fast local inference with GGUF models.

## Installation

```bash
go get github.com/neul-labs/mullama
```

**Note**: This package requires the Mullama FFI library to be built and available. See [Building from Source](#building-from-source) below.

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/neul-labs/mullama"
)

func main() {
    // Initialize backend
    mullama.BackendInit()
    defer mullama.BackendFree()

    // Load a model
    model, err := mullama.LoadModel("./model.gguf", &mullama.ModelParams{
        NGPULayers: 32,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Free()

    // Create a context
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

## Features

### Text Generation

```go
// Basic generation
text, err := ctx.Generate("Hello", 100, nil)

// With custom parameters
params := &mullama.SamplerParams{
    Temperature: 0.7,
    TopP:        0.9,
}
text, err = ctx.Generate("Hello", 100, params)

// Greedy (deterministic) generation
params = mullama.GreedySamplerParams()
text, err = ctx.Generate("Hello", 100, &params)

// Creative generation
params = mullama.CreativeSamplerParams()
text, err = ctx.Generate("Hello", 100, &params)
```

### Streaming Generation

```go
// Stream with callback
err := ctx.GenerateStream("Once upon a time", 100, nil, func(token string) bool {
    fmt.Print(token)
    return true // return false to stop generation
})
```

### Tokenization

```go
// Tokenize text
tokens, err := model.Tokenize("Hello, world!", true, false)
fmt.Printf("Tokens: %v\n", tokens)

// Detokenize back to text
text, err := model.Detokenize(tokens, false, false)
fmt.Printf("Text: %s\n", text)
```

### Embeddings

```go
// Create embedding generator
gen, err := mullama.NewEmbeddingGenerator(model, 512, true)
if err != nil {
    log.Fatal(err)
}
defer gen.Free()

// Generate embeddings
emb1, err := gen.Embed("Hello, world!")
emb2, err := gen.Embed("Hi there!")

// Compute similarity
similarity, err := mullama.CosineSimilarity(emb1, emb2)
fmt.Printf("Similarity: %f\n", similarity)

// Batch embedding
texts := []string{"Hello", "World", "Test"}
embeddings, err := gen.EmbedBatch(texts)
```

### Model Information

```go
model, _ := mullama.LoadModel("model.gguf", nil)

fmt.Printf("Embedding dim: %d\n", model.NEmbd())
fmt.Printf("Vocabulary size: %d\n", model.NVocab())
fmt.Printf("Context size: %d\n", model.NCtxTrain())
fmt.Printf("Model size: %.2f GB\n", float64(model.Size())/1e9)
fmt.Printf("Parameters: %d\n", model.NParams())
fmt.Printf("Description: %s\n", model.Description())
```

## API Reference

### Model

```go
type ModelParams struct {
    NGPULayers int32  // GPU layers (0=CPU, -1=all)
    UseMmap    bool   // Memory mapping
    UseMlock   bool   // Lock in memory
    VocabOnly  bool   // Only load vocabulary
}

func LoadModel(path string, params *ModelParams) (*Model, error)
func (m *Model) Free()
func (m *Model) Tokenize(text string, addBos, special bool) ([]int32, error)
func (m *Model) Detokenize(tokens []int32, removeSpecial, unparseSpecial bool) (string, error)

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

### Context

```go
type ContextParams struct {
    NCtx       uint32 // Context size (0 = model default)
    NBatch     uint32 // Batch size
    NThreads   int32  // Threads (0 = auto)
    Embeddings bool   // Enable embeddings
}

func NewContext(model *Model, params *ContextParams) (*Context, error)
func (c *Context) Free()
func (c *Context) Generate(prompt string, maxTokens int, params *SamplerParams) (string, error)
func (c *Context) GenerateFromTokens(tokens []int32, maxTokens int, params *SamplerParams) (string, error)
func (c *Context) GenerateStream(prompt string, maxTokens int, params *SamplerParams, callback StreamCallback) error
func (c *Context) ClearCache()

// Properties
func (c *Context) NCtx() uint32
func (c *Context) NBatch() uint32
```

### SamplerParams

```go
type SamplerParams struct {
    Temperature    float32 // Default: 0.8
    TopK           int32   // Default: 40
    TopP           float32 // Default: 0.95
    MinP           float32 // Default: 0.05
    TypicalP       float32 // Default: 1.0
    PenaltyRepeat  float32 // Default: 1.1
    PenaltyFreq    float32 // Default: 0.0
    PenaltyPresent float32 // Default: 0.0
    PenaltyLastN   int32   // Default: 64
    Seed           uint32  // Default: 0 (random)
}

func DefaultSamplerParams() SamplerParams
func GreedySamplerParams() SamplerParams
func CreativeSamplerParams() SamplerParams
func PreciseSamplerParams() SamplerParams
```

### EmbeddingGenerator

```go
func NewEmbeddingGenerator(model *Model, nCtx uint32, normalize bool) (*EmbeddingGenerator, error)
func (eg *EmbeddingGenerator) Free()
func (eg *EmbeddingGenerator) Embed(text string) ([]float32, error)
func (eg *EmbeddingGenerator) EmbedBatch(texts []string) ([][]float32, error)
func (eg *EmbeddingGenerator) NEmbd() int32
```

### Utility Functions

```go
func CosineSimilarity(a, b []float32) (float32, error)
func BackendInit()
func BackendFree()
func SupportsGPUOffload() bool
func SystemInfo() string
func MaxDevices() int
func Version() string
```

## Building from Source

### Prerequisites

1. Build the Mullama FFI library:

```bash
cd /path/to/mullama
cargo build --release -p mullama-ffi
```

2. The CGO flags in `mullama.go` expect the library at `../../target/release`. Adjust paths as needed.

### Running Tests

```bash
# Without model
go test -v

# With model
MULLAMA_TEST_MODEL=/path/to/model.gguf go test -v
```

## License

MIT OR Apache-2.0
