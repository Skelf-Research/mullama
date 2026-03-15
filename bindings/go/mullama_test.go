package mullama

import (
	"math"
	"os"
	"testing"
)

var testModelPath = os.Getenv("MULLAMA_TEST_MODEL")
var modelAvailable = testModelPath != "" && fileExists(testModelPath)

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Error("Version should not be empty")
	}
	if v != "0.1.0" {
		t.Errorf("Expected version 0.1.0, got %s", v)
	}
}

func TestSystemInfo(t *testing.T) {
	info := SystemInfo()
	if info == "" {
		t.Error("SystemInfo should not be empty")
	}
}

func TestSupportsGPUOffload(t *testing.T) {
	// Just ensure it doesn't panic
	_ = SupportsGPUOffload()
}

func TestMaxDevices(t *testing.T) {
	devices := MaxDevices()
	if devices < 1 {
		t.Errorf("MaxDevices should be at least 1, got %d", devices)
	}
}

func TestDefaultModelParams(t *testing.T) {
	params := DefaultModelParams()
	if params.NGPULayers != 0 {
		t.Errorf("Default NGPULayers should be 0, got %d", params.NGPULayers)
	}
	if !params.UseMmap {
		t.Error("Default UseMmap should be true")
	}
	if params.UseMlock {
		t.Error("Default UseMlock should be false")
	}
	if params.VocabOnly {
		t.Error("Default VocabOnly should be false")
	}
}

func TestDefaultContextParams(t *testing.T) {
	params := DefaultContextParams()
	if params.NCtx != 0 {
		t.Errorf("Default NCtx should be 0, got %d", params.NCtx)
	}
	if params.NBatch != 2048 {
		t.Errorf("Default NBatch should be 2048, got %d", params.NBatch)
	}
	if params.NThreads <= 0 {
		t.Error("Default NThreads should be positive")
	}
	if params.Embeddings {
		t.Error("Default Embeddings should be false")
	}
}

func TestDefaultSamplerParams(t *testing.T) {
	params := DefaultSamplerParams()
	if math.Abs(float64(params.Temperature-0.8)) > 0.01 {
		t.Errorf("Default Temperature should be 0.8, got %f", params.Temperature)
	}
	if params.TopK != 40 {
		t.Errorf("Default TopK should be 40, got %d", params.TopK)
	}
	if math.Abs(float64(params.TopP-0.95)) > 0.01 {
		t.Errorf("Default TopP should be 0.95, got %f", params.TopP)
	}
}

func TestGreedySamplerParams(t *testing.T) {
	params := GreedySamplerParams()
	if params.Temperature != 0.0 {
		t.Errorf("Greedy Temperature should be 0.0, got %f", params.Temperature)
	}
	if params.TopK != 1 {
		t.Errorf("Greedy TopK should be 1, got %d", params.TopK)
	}
}

func TestCreativeSamplerParams(t *testing.T) {
	params := CreativeSamplerParams()
	if params.Temperature <= 1.0 {
		t.Errorf("Creative Temperature should be > 1.0, got %f", params.Temperature)
	}
	if params.TopK <= 40 {
		t.Errorf("Creative TopK should be > 40, got %d", params.TopK)
	}
}

func TestPreciseSamplerParams(t *testing.T) {
	params := PreciseSamplerParams()
	if params.Temperature >= 0.5 {
		t.Errorf("Precise Temperature should be < 0.5, got %f", params.Temperature)
	}
	if params.TopK >= 40 {
		t.Errorf("Precise TopK should be < 40, got %d", params.TopK)
	}
}

func TestCosineSimilarity(t *testing.T) {
	t.Run("identical vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{1.0, 0.0, 0.0}
		sim, err := CosineSimilarity(a, b)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(float64(sim-1.0)) > 0.001 {
			t.Errorf("Expected similarity ~1.0, got %f", sim)
		}
	})

	t.Run("orthogonal vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{0.0, 1.0, 0.0}
		sim, err := CosineSimilarity(a, b)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(float64(sim)) > 0.001 {
			t.Errorf("Expected similarity ~0.0, got %f", sim)
		}
	})

	t.Run("opposite vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{-1.0, 0.0, 0.0}
		sim, err := CosineSimilarity(a, b)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if math.Abs(float64(sim+1.0)) > 0.001 {
			t.Errorf("Expected similarity ~-1.0, got %f", sim)
		}
	})

	t.Run("mismatched lengths", func(t *testing.T) {
		a := []float32{1.0, 0.0}
		b := []float32{1.0, 0.0, 0.0}
		_, err := CosineSimilarity(a, b)
		if err == nil {
			t.Error("Expected error for mismatched lengths")
		}
	})
}

func TestLoadModelNonexistent(t *testing.T) {
	_, err := LoadModel("/nonexistent/path/model.gguf", nil)
	if err == nil {
		t.Error("Expected error loading nonexistent model")
	}
}

// Tests that require a model file

func TestLoadModel(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	if model.NVocab() <= 0 {
		t.Error("NVocab should be positive")
	}
	if model.NEmbd() <= 0 {
		t.Error("NEmbd should be positive")
	}
}

func TestModelProperties(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	if model.NCtxTrain() <= 0 {
		t.Error("NCtxTrain should be positive")
	}
	if model.NLayer() <= 0 {
		t.Error("NLayer should be positive")
	}
	if model.NHead() <= 0 {
		t.Error("NHead should be positive")
	}
	if model.Size() <= 0 {
		t.Error("Size should be positive")
	}
	if model.NParams() <= 0 {
		t.Error("NParams should be positive")
	}
	if model.TokenBOS() < 0 {
		t.Error("TokenBOS should be non-negative")
	}
	if model.TokenEOS() < 0 {
		t.Error("TokenEOS should be non-negative")
	}
}

func TestTokenization(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	text := "Hello, world!"
	tokens, err := model.Tokenize(text, true, false)
	if err != nil {
		t.Fatalf("Tokenization failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Tokenization should produce tokens")
	}

	decoded, err := model.Detokenize(tokens, false, false)
	if err != nil {
		t.Fatalf("Detokenization failed: %v", err)
	}

	if decoded == "" {
		t.Error("Detokenization should produce text")
	}
}

func TestCreateContext(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	ctx, err := NewContext(model, nil)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Free()

	if ctx.NCtx() <= 0 {
		t.Error("NCtx should be positive")
	}
	if ctx.NBatch() <= 0 {
		t.Error("NBatch should be positive")
	}
}

func TestContextWithParams(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	params := ContextParams{
		NCtx:   512,
		NBatch: 256,
	}

	ctx, err := NewContext(model, &params)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Free()

	if ctx.NCtx() != 512 {
		t.Errorf("Expected NCtx 512, got %d", ctx.NCtx())
	}
	if ctx.NBatch() != 256 {
		t.Errorf("Expected NBatch 256, got %d", ctx.NBatch())
	}
}

func TestGenerateText(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	ctx, err := NewContext(model, &ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Free()

	params := GreedySamplerParams()
	text, err := ctx.Generate("Hello", 10, &params)
	if err != nil {
		t.Fatalf("Generation failed: %v", err)
	}

	if text == "" {
		t.Error("Generated text should not be empty")
	}
}

func TestClearCache(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	ctx, err := NewContext(model, nil)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Free()

	// Should not panic
	ctx.ClearCache()
}

func TestEmbeddingGenerator(t *testing.T) {
	if !modelAvailable {
		t.Skip("No test model available")
	}

	model, err := LoadModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	gen, err := NewEmbeddingGenerator(model, 512, true)
	if err != nil {
		t.Fatalf("Failed to create embedding generator: %v", err)
	}
	defer gen.Free()

	if gen.NEmbd() <= 0 {
		t.Error("NEmbd should be positive")
	}
}
