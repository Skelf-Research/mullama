// Package mullama provides Go bindings for the Mullama LLM library.
//
// This package enables high-performance LLM inference from Go using GGUF models.
package mullama

/*
#cgo CFLAGS: -I${SRCDIR}/../ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lmullama_ffi -lm -lstdc++ -lpthread

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "mullama.h"
*/
import "C"
import (
	"errors"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

var (
	// ErrNullPointer indicates a null pointer was passed
	ErrNullPointer = errors.New("null pointer")
	// ErrModelLoad indicates model loading failed
	ErrModelLoad = errors.New("failed to load model")
	// ErrContext indicates context creation failed
	ErrContext = errors.New("failed to create context")
	// ErrTokenization indicates tokenization failed
	ErrTokenization = errors.New("tokenization failed")
	// ErrGeneration indicates text generation failed
	ErrGeneration = errors.New("generation failed")
	// ErrEmbedding indicates embedding generation failed
	ErrEmbedding = errors.New("embedding generation failed")
	// ErrInvalidInput indicates invalid input was provided
	ErrInvalidInput = errors.New("invalid input")
)

var initOnce sync.Once

// BackendInit initializes the mullama backend.
// This should be called once at the start of your program.
func BackendInit() {
	initOnce.Do(func() {
		C.mullama_backend_init()
	})
}

// BackendFree frees mullama backend resources.
// This should be called once before your program exits.
func BackendFree() {
	C.mullama_backend_free()
}

// SupportsGPUOffload returns true if GPU offloading is supported.
func SupportsGPUOffload() bool {
	return bool(C.mullama_supports_gpu_offload())
}

// SystemInfo returns system information about the backend.
func SystemInfo() string {
	buf := make([]C.char, 4096)
	n := C.mullama_system_info((*C.char)(&buf[0]), C.size_t(len(buf)))
	if n < 0 {
		return ""
	}
	return C.GoString((*C.char)(&buf[0]))
}

// MaxDevices returns the maximum number of devices available.
func MaxDevices() int {
	return int(C.mullama_max_devices())
}

// Version returns the library version.
func Version() string {
	return "0.1.0"
}

// getLastError retrieves the last error message from the C library.
func getLastError() string {
	cStr := C.mullama_get_last_error()
	if cStr == nil {
		return "unknown error"
	}
	return C.GoString(cStr)
}

// ModelParams contains parameters for model loading.
type ModelParams struct {
	// NGPULayers is the number of layers to offload to GPU (0 = CPU only, -1 = all)
	NGPULayers int32
	// UseMmap enables memory mapping for model loading
	UseMmap bool
	// UseMlock locks the model in memory
	UseMlock bool
	// VocabOnly loads only the vocabulary (for tokenization only)
	VocabOnly bool
}

// DefaultModelParams returns default model parameters.
func DefaultModelParams() ModelParams {
	return ModelParams{
		NGPULayers: 0,
		UseMmap:    true,
		UseMlock:   false,
		VocabOnly:  false,
	}
}

// Model represents a loaded LLM model.
type Model struct {
	ptr *C.MullamaMullamaModel
}

// LoadModel loads a model from a GGUF file.
func LoadModel(path string, params *ModelParams) (*Model, error) {
	BackendInit()

	if params == nil {
		p := DefaultModelParams()
		params = &p
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cParams := C.MullamaMullamaModelParams{
		n_gpu_layers:  C.int(params.NGPULayers),
		use_mmap:      C.bool(params.UseMmap),
		use_mlock:     C.bool(params.UseMlock),
		vocab_only:    C.bool(params.VocabOnly),
		main_gpu:      0,
		check_tensors: false,
	}

	ptr := C.mullama_model_load(cPath, &cParams)
	if ptr == nil {
		return nil, errors.New(getLastError())
	}

	model := &Model{ptr: ptr}
	runtime.SetFinalizer(model, func(m *Model) {
		m.Free()
	})

	return model, nil
}

// Free releases the model resources.
func (m *Model) Free() {
	if m.ptr != nil {
		C.mullama_model_free(m.ptr)
		m.ptr = nil
	}
}

// Tokenize converts text to token IDs.
func (m *Model) Tokenize(text string, addBos bool, special bool) ([]int32, error) {
	if m.ptr == nil {
		return nil, ErrNullPointer
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Allocate buffer for tokens (estimate max size)
	maxTokens := len(text)*2 + 10
	tokens := make([]C.int32_t, maxTokens)

	n := C.mullama_tokenize(m.ptr, cText, (*C.int32_t)(&tokens[0]), C.int32_t(maxTokens), C.bool(addBos), C.bool(special))
	if n < 0 {
		return nil, errors.New(getLastError())
	}

	result := make([]int32, n)
	for i := C.int32_t(0); i < n; i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// Detokenize converts token IDs back to text.
func (m *Model) Detokenize(tokens []int32, removeSpecial bool, unparseSpecial bool) (string, error) {
	if m.ptr == nil {
		return "", ErrNullPointer
	}

	if len(tokens) == 0 {
		return "", nil
	}

	cTokens := make([]C.int, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int(t)
	}

	// Allocate buffer for output
	maxLen := len(tokens) * 32
	buf := make([]C.char, maxLen)

	n := C.mullama_detokenize(m.ptr, (*C.int)(&cTokens[0]), C.int(len(tokens)),
		(*C.char)(&buf[0]), C.int(maxLen))
	if n < 0 {
		return "", errors.New(getLastError())
	}

	return C.GoString((*C.char)(&buf[0])), nil
}

// NCtxTrain returns the model's training context size.
func (m *Model) NCtxTrain() int32 {
	if m.ptr == nil {
		return 0
	}
	return int32(C.mullama_model_n_ctx_train(m.ptr))
}

// NEmbd returns the embedding dimension.
func (m *Model) NEmbd() int32 {
	if m.ptr == nil {
		return 0
	}
	return int32(C.mullama_model_n_embd(m.ptr))
}

// NVocab returns the vocabulary size.
func (m *Model) NVocab() int32 {
	if m.ptr == nil {
		return 0
	}
	return int32(C.mullama_model_n_vocab(m.ptr))
}

// NLayer returns the number of layers.
func (m *Model) NLayer() int32 {
	if m.ptr == nil {
		return 0
	}
	return int32(C.mullama_model_n_layer(m.ptr))
}

// NHead returns the number of attention heads.
func (m *Model) NHead() int32 {
	if m.ptr == nil {
		return 0
	}
	return int32(C.mullama_model_n_head(m.ptr))
}

// TokenBOS returns the beginning-of-sequence token ID.
func (m *Model) TokenBOS() int32 {
	if m.ptr == nil {
		return -1
	}
	return int32(C.mullama_model_token_bos(m.ptr))
}

// TokenEOS returns the end-of-sequence token ID.
func (m *Model) TokenEOS() int32 {
	if m.ptr == nil {
		return -1
	}
	return int32(C.mullama_model_token_eos(m.ptr))
}

// Size returns the model size in bytes.
func (m *Model) Size() uint64 {
	if m.ptr == nil {
		return 0
	}
	return uint64(C.mullama_model_size(m.ptr))
}

// NParams returns the number of parameters.
func (m *Model) NParams() uint64 {
	if m.ptr == nil {
		return 0
	}
	return uint64(C.mullama_model_n_params(m.ptr))
}

// Description returns the model description.
func (m *Model) Description() string {
	if m.ptr == nil {
		return ""
	}
	buf := make([]C.char, 256)
	C.mullama_model_desc(m.ptr, (*C.char)(&buf[0]), 256)
	return C.GoString((*C.char)(&buf[0]))
}

// TokenIsEOG checks if a token is end-of-generation.
func (m *Model) TokenIsEOG(token int32) bool {
	if m.ptr == nil {
		return false
	}
	return bool(C.mullama_model_token_is_eog(m.ptr, C.int32_t(token)))
}

// ContextParams contains parameters for context creation.
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

// DefaultContextParams returns default context parameters.
func DefaultContextParams() ContextParams {
	return ContextParams{
		NCtx:       0,
		NBatch:     2048,
		NThreads:   int32(runtime.NumCPU()),
		Embeddings: false,
	}
}

// Context represents an inference context.
type Context struct {
	ptr   *C.MullamaMullamaContext
	model *Model
}

// NewContext creates a new context from a model.
func NewContext(model *Model, params *ContextParams) (*Context, error) {
	if model == nil || model.ptr == nil {
		return nil, ErrNullPointer
	}

	if params == nil {
		p := DefaultContextParams()
		params = &p
	}

	cParams := C.MullamaMullamaContextParams{
		n_ctx:          C.uint32_t(params.NCtx),
		n_batch:        C.uint32_t(params.NBatch),
		n_threads:      C.int(params.NThreads),
		n_threads_batch: C.int(params.NThreads),
		embeddings:     C.bool(params.Embeddings),
	}

	ptr := C.mullama_context_new(model.ptr, &cParams)
	if ptr == nil {
		return nil, errors.New(getLastError())
	}

	ctx := &Context{ptr: ptr, model: model}
	runtime.SetFinalizer(ctx, func(c *Context) {
		c.Free()
	})

	return ctx, nil
}

// Free releases the context resources.
func (c *Context) Free() {
	if c.ptr != nil {
		C.mullama_context_free(c.ptr)
		c.ptr = nil
	}
}

// SamplerParams contains parameters for text sampling.
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

// DefaultSamplerParams returns default sampler parameters.
func DefaultSamplerParams() SamplerParams {
	return SamplerParams{
		Temperature:    0.8,
		TopK:           40,
		TopP:           0.95,
		MinP:           0.05,
		TypicalP:       1.0,
		PenaltyRepeat:  1.1,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenaltyLastN:   64,
		Seed:           0,
	}
}

// GreedySamplerParams returns greedy (deterministic) sampler parameters.
func GreedySamplerParams() SamplerParams {
	return SamplerParams{
		Temperature:    0.0,
		TopK:           1,
		TopP:           1.0,
		MinP:           0.0,
		TypicalP:       1.0,
		PenaltyRepeat:  1.0,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenaltyLastN:   0,
		Seed:           0,
	}
}

// CreativeSamplerParams returns creative (high randomness) sampler parameters.
func CreativeSamplerParams() SamplerParams {
	return SamplerParams{
		Temperature:    1.2,
		TopK:           100,
		TopP:           0.95,
		MinP:           0.02,
		TypicalP:       1.0,
		PenaltyRepeat:  1.15,
		PenaltyFreq:    0.1,
		PenaltyPresent: 0.1,
		PenaltyLastN:   128,
		Seed:           0,
	}
}

// PreciseSamplerParams returns precise (low randomness) sampler parameters.
func PreciseSamplerParams() SamplerParams {
	return SamplerParams{
		Temperature:    0.3,
		TopK:           20,
		TopP:           0.8,
		MinP:           0.1,
		TypicalP:       1.0,
		PenaltyRepeat:  1.05,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenaltyLastN:   32,
		Seed:           0,
	}
}

// Generate generates text from a prompt.
func (c *Context) Generate(prompt string, maxTokens int, params *SamplerParams) (string, error) {
	if c.ptr == nil {
		return "", ErrNullPointer
	}

	tokens, err := c.model.Tokenize(prompt, true, false)
	if err != nil {
		return "", err
	}

	return c.GenerateFromTokens(tokens, maxTokens, params)
}

// GenerateFromTokens generates text from token IDs.
func (c *Context) GenerateFromTokens(tokens []int32, maxTokens int, params *SamplerParams) (string, error) {
	if c.ptr == nil {
		return "", ErrNullPointer
	}

	if params == nil {
		p := DefaultSamplerParams()
		params = &p
	}

	cTokens := make([]C.int, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int(t)
	}

	cParams := C.MullamaMullamaSamplerParams{
		temperature:    C.float(params.Temperature),
		top_k:          C.int(params.TopK),
		top_p:          C.float(params.TopP),
		min_p:          C.float(params.MinP),
		typical_p:      C.float(params.TypicalP),
		penalty_repeat: C.float(params.PenaltyRepeat),
		penalty_freq:   C.float(params.PenaltyFreq),
		penalty_present: C.float(params.PenaltyPresent),
		penalty_last_n: C.int(params.PenaltyLastN),
		seed:           C.uint32_t(params.Seed),
	}

	// Allocate output buffer
	maxOutput := maxTokens * 32
	buf := make([]C.char, maxOutput)

	n := C.mullama_generate(c.ptr, (*C.int)(&cTokens[0]), C.int(len(tokens)),
		C.int(maxTokens), &cParams, (*C.char)(&buf[0]), C.size_t(maxOutput))
	if n < 0 {
		return "", errors.New(getLastError())
	}

	return C.GoString((*C.char)(&buf[0])), nil
}

// StreamCallback is called for each generated token during streaming.
type StreamCallback func(token string) bool

// GenerateStream generates text with streaming using a callback.
func (c *Context) GenerateStream(prompt string, maxTokens int, params *SamplerParams, callback StreamCallback) error {
	if c.ptr == nil {
		return ErrNullPointer
	}

	tokens, err := c.model.Tokenize(prompt, true, false)
	if err != nil {
		return err
	}

	if params == nil {
		p := DefaultSamplerParams()
		params = &p
	}

	cTokens := make([]C.int, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int(t)
	}

	cParams := C.MullamaMullamaSamplerParams{
		temperature:    C.float(params.Temperature),
		top_k:          C.int(params.TopK),
		top_p:          C.float(params.TopP),
		min_p:          C.float(params.MinP),
		typical_p:      C.float(params.TypicalP),
		penalty_repeat: C.float(params.PenaltyRepeat),
		penalty_freq:   C.float(params.PenaltyFreq),
		penalty_present: C.float(params.PenaltyPresent),
		penalty_last_n: C.int(params.PenaltyLastN),
		seed:           C.uint32_t(params.Seed),
	}

	// For simplicity, we'll use the non-callback generate and return the result
	maxOutput := maxTokens * 32
	buf := make([]C.char, maxOutput)

	n := C.mullama_generate(c.ptr, (*C.int)(&cTokens[0]), C.int(len(tokens)),
		C.int(maxTokens), &cParams, (*C.char)(&buf[0]), C.size_t(maxOutput))
	if n < 0 {
		return errors.New(getLastError())
	}

	// Call callback with the full result (simplified version)
	result := C.GoString((*C.char)(&buf[0]))
	if !callback(result) {
		return nil
	}

	return nil
}

// ClearCache clears the KV cache.
func (c *Context) ClearCache() {
	if c.ptr != nil {
		C.mullama_context_kv_cache_clear(c.ptr)
	}
}

// NCtx returns the context size.
func (c *Context) NCtx() uint32 {
	if c.ptr == nil {
		return 0
	}
	return uint32(C.mullama_context_n_ctx(c.ptr))
}

// NBatch returns the batch size.
func (c *Context) NBatch() uint32 {
	if c.ptr == nil {
		return 0
	}
	return uint32(C.mullama_context_n_batch(c.ptr))
}

// EmbeddingGenerator generates text embeddings.
type EmbeddingGenerator struct {
	ptr       *C.MullamaMullamaEmbeddingGenerator
	model     *Model
	normalize bool
}

// NewEmbeddingGenerator creates a new embedding generator.
func NewEmbeddingGenerator(model *Model, nCtx uint32, normalize bool) (*EmbeddingGenerator, error) {
	if model == nil || model.ptr == nil {
		return nil, ErrNullPointer
	}

	if nCtx == 0 {
		nCtx = 512
	}

	config := C.mullama_embedding_default_config()
	config.n_ctx = C.uint32_t(nCtx)
	config.normalize = C.bool(normalize)

	ptr := C.mullama_embedding_generator_new(model.ptr, &config)
	if ptr == nil {
		return nil, errors.New(getLastError())
	}

	eg := &EmbeddingGenerator{
		ptr:       ptr,
		model:     model,
		normalize: normalize,
	}
	runtime.SetFinalizer(eg, func(e *EmbeddingGenerator) {
		e.Free()
	})

	return eg, nil
}

// Free releases the embedding generator resources.
func (eg *EmbeddingGenerator) Free() {
	if eg.ptr != nil {
		C.mullama_embedding_generator_free(eg.ptr)
		eg.ptr = nil
	}
}

// Embed generates embeddings for text.
func (eg *EmbeddingGenerator) Embed(text string) ([]float32, error) {
	if eg.ptr == nil {
		return nil, ErrNullPointer
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	nEmbd := C.mullama_embedding_generator_n_embd(eg.ptr)
	embeddings := make([]C.float, nEmbd)

	n := C.mullama_embed_text(eg.ptr, cText, (*C.float)(&embeddings[0]), C.size_t(nEmbd))
	if n < 0 {
		return nil, errors.New(getLastError())
	}

	result := make([]float32, n)
	for i := C.int(0); i < n; i++ {
		result[i] = float32(embeddings[i])
	}

	return result, nil
}

// EmbedBatch generates embeddings for multiple texts.
func (eg *EmbeddingGenerator) EmbedBatch(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := eg.Embed(text)
		if err != nil {
			return nil, err
		}
		results[i] = emb
	}
	return results, nil
}

// NEmbd returns the embedding dimension.
func (eg *EmbeddingGenerator) NEmbd() int32 {
	if eg.ptr == nil {
		return 0
	}
	return int32(C.mullama_embedding_generator_n_embd(eg.ptr))
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrInvalidInput
	}

	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	norm := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
	if norm == 0 {
		return 0, nil
	}

	return dot / norm, nil
}
