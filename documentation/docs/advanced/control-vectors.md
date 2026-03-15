# Control Vectors

Steer model behavior at inference time using control vectors -- modifying generation style, safety, personality, and other characteristics without fine-tuning the model weights.

!!! info "Feature Gate"
    This feature requires the `control-vectors` feature flag.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["control-vectors"] }
    ```

## Overview

Control vectors provide:

- **ControlVector** loading from pre-computed vector files
- **ControlVectorManager** for organizing and switching between vectors
- **Scale adjustment** to control the strength of steering
- **Runtime switching** without reloading the model
- **Composition** of multiple vectors simultaneously
- **Layer-specific application** for fine-grained control

---

## What Are Control Vectors?

Control vectors are directions in a model's activation space that correspond to specific behaviors. By adding or subtracting these vectors during inference, you can steer the model's output without modifying its weights.

```
Without control vector:
  Input:  "Write a poem about the ocean"
  Output: "The ocean vast and wide, stretching to the sky..."

With "creative_writing" vector (scale=1.5):
  Input:  "Write a poem about the ocean"
  Output: "In cerulean depths where starlight drowns,
           the ancient tides compose their endless hymns..."

With "concise" vector (scale=1.0):
  Input:  "Write a poem about the ocean"
  Output: "Salt, waves, infinity."
```

Unlike fine-tuning, control vectors:

- Require **no training data** for application
- Can be **enabled/disabled at runtime**
- Support **variable strength** via scale parameter
- Are **composable** (combine multiple vectors)
- Use **minimal memory** (one vector per controlled dimension)

---

## Loading Control Vectors

=== "Node.js"

    ```javascript
    const { ControlVector, loadModel } = require('mullama');

    const model = await loadModel('model.gguf');

    // Load a control vector from file
    const creative = await ControlVector.load('creative_writing.gguf');
    console.log(`Loaded vector: ${creative.name}, layers: ${creative.layers}`);

    // Apply to model with scale
    model.applyControlVector(creative, { scale: 1.5 });

    // Generate with the control vector active
    const output = await model.generate('Write a poem about the ocean:', {
      maxTokens: 200
    });
    console.log(output);
    ```

=== "Python"

    ```python
    from mullama import ControlVector, load_model

    model = await load_model("model.gguf")

    # Load a control vector from file
    creative = ControlVector.load("creative_writing.gguf")
    print(f"Loaded vector: {creative.name}, layers: {creative.layers}")

    # Apply to model with scale
    model.apply_control_vector(creative, scale=1.5)

    # Generate with the control vector active
    output = await model.generate("Write a poem about the ocean:", max_tokens=200)
    print(output)
    ```

=== "Rust"

    ```rust
    use mullama::{ControlVector, Model, Context, ContextParams};
    use std::sync::Arc;

    let model = Arc::new(Model::load("model.gguf")?);

    // Load a control vector from file
    let creative = ControlVector::load("creative_writing.gguf")?;
    println!("Loaded vector: {}, layers: {}", creative.name(), creative.num_layers());

    // Apply to context with scale
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;
    ctx.apply_control_vector(&creative, 1.5)?;

    // Generate with the control vector active
    let output = ctx.generate("Write a poem about the ocean:", 200)?;
    println!("{}", output);
    ```

=== "CLI"

    ```bash
    # Apply a control vector during generation
    mullama run model.gguf "Write a poem about the ocean:" \
      --control-vector creative_writing.gguf \
      --control-scale 1.5

    # Apply multiple vectors
    mullama run model.gguf "Explain quantum physics:" \
      --control-vector creative_writing.gguf:1.5 \
      --control-vector concise.gguf:0.8
    ```

---

## ControlVectorManager

Manage multiple control vectors and switch between them at runtime.

=== "Node.js"

    ```javascript
    const { ControlVectorManager } = require('mullama');

    const manager = new ControlVectorManager();

    // Register multiple vectors
    await manager.register('creative', 'creative_writing.gguf');
    await manager.register('formal', 'formal_tone.gguf');
    await manager.register('concise', 'concise_output.gguf');
    await manager.register('safe', 'safety.gguf');

    // List available vectors
    const vectors = manager.list();
    vectors.forEach(v => console.log(`  ${v.name}: ${v.layers} layers`));

    // Apply a single vector
    manager.activate('creative', { scale: 1.2 });

    // Apply multiple vectors simultaneously
    manager.activateMultiple([
      { name: 'creative', scale: 1.0 },
      { name: 'concise', scale: 0.5 }
    ]);

    // Deactivate all
    manager.deactivateAll();
    ```

=== "Python"

    ```python
    from mullama import ControlVectorManager

    manager = ControlVectorManager()

    # Register multiple vectors
    await manager.register("creative", "creative_writing.gguf")
    await manager.register("formal", "formal_tone.gguf")
    await manager.register("concise", "concise_output.gguf")
    await manager.register("safe", "safety.gguf")

    # List available vectors
    for v in manager.list():
        print(f"  {v.name}: {v.layers} layers")

    # Apply a single vector
    manager.activate("creative", scale=1.2)

    # Apply multiple vectors simultaneously
    manager.activate_multiple([
        {"name": "creative", "scale": 1.0},
        {"name": "concise", "scale": 0.5}
    ])

    # Deactivate all
    manager.deactivate_all()
    ```

=== "Rust"

    ```rust
    use mullama::ControlVectorManager;

    let mut manager = ControlVectorManager::new();

    // Register multiple vectors
    manager.register("creative", "creative_writing.gguf")?;
    manager.register("formal", "formal_tone.gguf")?;
    manager.register("concise", "concise_output.gguf")?;
    manager.register("safe", "safety.gguf")?;

    // List available vectors
    for info in manager.list() {
        println!("  {}: {} layers", info.name, info.num_layers);
    }

    // Apply a single vector
    manager.activate("creative", 1.2)?;

    // Apply multiple vectors
    manager.activate_multiple(&[
        ("creative", 1.0),
        ("concise", 0.5),
    ])?;

    // Deactivate all
    manager.deactivate_all();
    ```

### Manager Methods

| Method | Description |
|--------|-------------|
| `register(name, path)` | Load and register a control vector |
| `list()` | List all registered vectors |
| `activate(name, scale)` | Apply a vector to generation |
| `activate_multiple(specs)` | Apply multiple vectors |
| `deactivate(name)` | Remove a specific vector |
| `deactivate_all()` | Remove all active vectors |
| `is_active(name)` | Check if a vector is currently active |

---

## Scale Adjustment

The `scale` parameter controls how strongly the vector influences generation:

| Scale | Effect |
|-------|--------|
| 0.0 | No effect (vector disabled) |
| 0.5 | Subtle steering |
| 1.0 | Standard effect |
| 1.5 | Strong steering |
| 2.0+ | Very strong (may reduce coherence) |
| -1.0 | Inverted effect (opposite behavior) |

=== "Node.js"

    ```javascript
    // Subtle creative boost
    model.applyControlVector(creative, { scale: 0.5 });

    // Strong creative influence
    model.applyControlVector(creative, { scale: 2.0 });

    // Invert the vector (make output LESS creative)
    model.applyControlVector(creative, { scale: -1.0 });
    ```

=== "Python"

    ```python
    # Subtle creative boost
    model.apply_control_vector(creative, scale=0.5)

    # Strong creative influence
    model.apply_control_vector(creative, scale=2.0)

    # Invert the vector (make output LESS creative)
    model.apply_control_vector(creative, scale=-1.0)
    ```

=== "Rust"

    ```rust
    // Subtle creative boost
    ctx.apply_control_vector(&creative, 0.5)?;

    // Strong creative influence
    ctx.apply_control_vector(&creative, 2.0)?;

    // Invert the vector (make output LESS creative)
    ctx.apply_control_vector(&creative, -1.0)?;
    ```

!!! warning "Scale Too High"
    Scales above 2.0 often degrade output quality. Start with 1.0 and adjust in increments of 0.25 until you achieve the desired effect.

---

## Use Cases

### Style Control

Steer writing style without changing the prompt:

```rust
// Academic writing
ctx.apply_control_vector(&academic_vector, 1.2)?;
let output = ctx.generate("Explain photosynthesis:", 200)?;

// Casual/conversational
ctx.remove_control_vector()?;
ctx.apply_control_vector(&casual_vector, 1.0)?;
let output = ctx.generate("Explain photosynthesis:", 200)?;
```

### Safety

Apply safety vectors to reduce harmful outputs:

```rust
// Always-on safety vector
ctx.apply_control_vector(&safety_vector, 1.5)?;
```

### Personality

Give the model consistent character traits:

```rust
manager.activate_multiple(&[
    ("helpful", 1.0),
    ("concise", 0.8),
    ("technical", 0.6),
])?;
```

### Language Formality

Switch between formal and informal registers:

```rust
// Formal mode for business contexts
manager.activate("formal", 1.2)?;

// Switch to casual for chat
manager.deactivate("formal");
manager.activate("casual", 1.0)?;
```

---

## Finding and Creating Control Vectors

### Pre-trained Vectors

Control vectors are typically available from:

- Model publishers alongside their GGUF models
- Community repositories (e.g., HuggingFace)
- Purpose-built training pipelines

### Creating Custom Vectors

!!! note "Advanced Topic"
    Creating control vectors requires generating contrast pairs and computing the principal direction of difference in the model's activation space.

The general process:

1. **Collect contrast pairs** - Positive and negative examples of the desired behavior
2. **Extract activations** - Run both sets through the model
3. **Compute difference** - Find the direction that separates the two sets (PCA on differences)
4. **Validate** - Test the vector on held-out examples

```rust
// Creating vectors is done offline, then loaded at runtime:
// 1. Train: python train_control_vector.py --model model.gguf \
//           --positive positive_examples.txt \
//           --negative negative_examples.txt \
//           --output my_vector.gguf
//
// 2. Use in production:
let my_vector = ControlVector::load("my_vector.gguf")?;
ctx.apply_control_vector(&my_vector, 1.0)?;
```

---

## Combining with Other Generation Parameters

Control vectors work alongside all other generation parameters:

=== "Node.js"

    ```javascript
    // Control vector + sampling parameters
    model.applyControlVector(creative, { scale: 1.2 });

    const output = await model.generate('Tell me a story:', {
      maxTokens: 500,
      temperature: 0.8,   // Works with temperature
      topK: 50,           // Works with top-k
      topP: 0.9,          // Works with top-p
      repetitionPenalty: 1.1
    });
    ```

=== "Python"

    ```python
    # Control vector + sampling parameters
    model.apply_control_vector(creative, scale=1.2)

    output = await model.generate("Tell me a story:",
        max_tokens=500,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )
    ```

=== "Rust"

    ```rust
    // Control vector + sampling parameters
    ctx.apply_control_vector(&creative, 1.2)?;

    let params = SamplingParams::new()
        .temperature(0.8)
        .top_k(50)
        .top_p(0.9)
        .repetition_penalty(1.1);

    let output = ctx.generate_with_params("Tell me a story:", 500, &params)?;
    ```

!!! tip "Interaction with Temperature"
    Control vectors modify the model's internal representations, while temperature affects the final sampling distribution. Both can be used together -- the vector shifts what the model "wants to say" and temperature controls how deterministic the selection is.

---

## Removing and Switching Vectors at Runtime

Control vectors can be dynamically changed between generations.

=== "Node.js"

    ```javascript
    // Start with creative mode
    model.applyControlVector(creative, { scale: 1.0 });
    const poem = await model.generate('Write a haiku:', { maxTokens: 50 });

    // Switch to formal mode for next request
    model.removeControlVector();
    model.applyControlVector(formal, { scale: 1.0 });
    const report = await model.generate('Summarize Q3 earnings:', { maxTokens: 200 });

    // Remove all vectors
    model.removeControlVector();
    ```

=== "Python"

    ```python
    # Start with creative mode
    model.apply_control_vector(creative, scale=1.0)
    poem = await model.generate("Write a haiku:", max_tokens=50)

    # Switch to formal mode for next request
    model.remove_control_vector()
    model.apply_control_vector(formal, scale=1.0)
    report = await model.generate("Summarize Q3 earnings:", max_tokens=200)

    # Remove all vectors
    model.remove_control_vector()
    ```

=== "Rust"

    ```rust
    // Start with creative mode
    ctx.apply_control_vector(&creative, 1.0)?;
    let poem = ctx.generate("Write a haiku:", 50)?;

    // Switch to formal mode for next request
    ctx.remove_control_vector()?;
    ctx.apply_control_vector(&formal, 1.0)?;
    let report = ctx.generate("Summarize Q3 earnings:", 200)?;

    // Remove all vectors
    ctx.remove_control_vector()?;
    ```

---

## ControlVector Reference

```rust
pub struct ControlVector {
    name: String,
    data: Vec<Vec<f32>>,      // Per-layer vectors
    num_layers: usize,
    dimension: usize,
}

impl ControlVector {
    /// Load from a GGUF file
    pub fn load(path: &str) -> Result<Self, MullamaError>;

    /// Get the vector name
    pub fn name(&self) -> &str;

    /// Number of model layers covered
    pub fn num_layers(&self) -> usize;

    /// Embedding dimension
    pub fn dimension(&self) -> usize;

    /// Scale the vector by a factor (returns new vector)
    pub fn scaled(&self, scale: f32) -> Self;

    /// Add two control vectors together
    pub fn add(&self, other: &Self) -> Result<Self, MullamaError>;
}
```

---

## Memory and Performance Impact

Control vectors have minimal overhead:

| Metric | Impact |
|--------|--------|
| Memory | ~4-8 MB per vector (depends on model size) |
| Latency per token | < 0.1ms additional |
| Load time | < 100ms |
| Switching cost | Negligible |

!!! success "Production Ready"
    Control vectors add negligible latency (< 0.1ms per token) and minimal memory overhead. They are suitable for production deployments where per-request behavior customization is needed.

---

## See Also

- [Generation Guide](../guide/generation.md) - Text generation fundamentals
- [Sampling Parameters](../guide/sampling.md) - Sampling configuration
- [API: Sampling](../api/sampling.md) - Sampling API reference
