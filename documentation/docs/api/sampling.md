# Sampling API

Control how tokens are selected during generation.

## SamplingParams

```rust
pub struct SamplingParams {
    // Temperature
    pub temperature: f32,

    // Top-K sampling
    pub top_k: i32,

    // Top-P (nucleus) sampling
    pub top_p: f32,

    // Min-P sampling
    pub min_p: f32,

    // Repetition penalty
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,

    // Presence/frequency penalties
    pub presence_penalty: f32,
    pub frequency_penalty: f32,

    // Stop sequences
    pub stop_sequences: Vec<String>,

    // Seed for reproducibility
    pub seed: Option<u64>,
}
```

## Default Values

```rust
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
            seed: None,
        }
    }
}
```

---

## Temperature

Controls randomness in token selection.

| Value | Effect |
|-------|--------|
| 0.0 | Deterministic (greedy) |
| 0.1-0.5 | Focused, consistent |
| 0.6-0.9 | Balanced creativity |
| 1.0+ | More random/creative |

```rust
// Deterministic
let params = SamplingParams { temperature: 0.0, ..Default::default() };

// Creative
let params = SamplingParams { temperature: 1.2, ..Default::default() };
```

---

## Top-K Sampling

Limit selection to the K most likely tokens.

```rust
let params = SamplingParams {
    top_k: 40,  // Consider only top 40 tokens
    ..Default::default()
};
```

| Value | Effect |
|-------|--------|
| 1 | Greedy (only best token) |
| 10-40 | Focused |
| 50-100 | More variety |
| 0 or -1 | Disabled |

---

## Top-P (Nucleus) Sampling

Select tokens until cumulative probability reaches P.

```rust
let params = SamplingParams {
    top_p: 0.9,  // Top 90% probability mass
    ..Default::default()
};
```

| Value | Effect |
|-------|--------|
| 0.1-0.5 | Very focused |
| 0.9 | Standard |
| 0.95+ | More variety |
| 1.0 | Disabled |

---

## Min-P Sampling

Filter tokens below a minimum probability relative to the top token.

```rust
let params = SamplingParams {
    min_p: 0.05,  // Must be at least 5% of top probability
    ..Default::default()
};
```

---

## Repetition Control

### Repeat Penalty

Penalize tokens that have appeared recently.

```rust
let params = SamplingParams {
    repeat_penalty: 1.1,  // Multiplicative penalty
    repeat_last_n: 64,    // Look back 64 tokens
    ..Default::default()
};
```

| Value | Effect |
|-------|--------|
| 1.0 | No penalty |
| 1.05-1.15 | Light penalty |
| 1.2+ | Strong penalty |

### Presence Penalty

Penalize any token that has appeared at all.

```rust
let params = SamplingParams {
    presence_penalty: 0.5,  // Additive penalty
    ..Default::default()
};
```

### Frequency Penalty

Penalize based on how often a token appeared.

```rust
let params = SamplingParams {
    frequency_penalty: 0.5,  // Per-occurrence penalty
    ..Default::default()
};
```

---

## Stop Sequences

Stop generation when specific strings are produced.

```rust
let params = SamplingParams {
    stop_sequences: vec![
        "\n\n".to_string(),
        "User:".to_string(),
        "</s>".to_string(),
    ],
    ..Default::default()
};
```

---

## Reproducibility

Set a seed for deterministic generation.

```rust
let params = SamplingParams {
    seed: Some(42),
    temperature: 0.0,  // Also set temperature to 0 for full determinism
    ..Default::default()
};
```

---

## Preset Configurations

### Deterministic (Code/Facts)

```rust
let params = SamplingParams {
    temperature: 0.0,
    ..Default::default()
};
```

### Balanced (General Use)

```rust
let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    ..Default::default()
};
```

### Creative (Stories/Brainstorming)

```rust
let params = SamplingParams {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 100,
    repeat_penalty: 1.2,
    ..Default::default()
};
```

### Strict (Following Instructions)

```rust
let params = SamplingParams {
    temperature: 0.3,
    top_p: 0.8,
    top_k: 20,
    repeat_penalty: 1.15,
    ..Default::default()
};
```

---

## Usage Example

```rust
use mullama::{Context, SamplingParams};

let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
    stop_sequences: vec!["\n\n".to_string()],
    ..Default::default()
};

let response = context.generate_with_params(
    "Write a poem about rust programming:",
    200,
    params
)?;
```

---

## Best Practices

1. **Start with defaults** - They work well for most cases
2. **Adjust temperature first** - It has the biggest impact
3. **Use repetition penalty** - Prevents loops
4. **Add stop sequences** - For cleaner output
5. **Test with your model** - Different models respond differently
