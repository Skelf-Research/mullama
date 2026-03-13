# Advanced Sampling Guide

Sampling is the process of selecting the next token during text generation. Mullama provides comprehensive support for all modern sampling strategies used in llama.cpp, allowing you to fine-tune generation quality for your specific use case.

## Quick Start

The simplest way to set up sampling:

```rust
use mullama::{SamplerParams, Model};

let sampler_params = SamplerParams::default()
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9);

let sampler = sampler_params.build_chain(&model);
```

## Sampling Strategies

### Temperature Sampling

Controls randomness in token selection. Lower values make output more deterministic.

```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.1);  // Very focused, deterministic
    // .with_temperature(0.7);  // Balanced (recommended)
    // .with_temperature(1.2);  // Creative, more random
```

**Guidelines:**
- `0.1-0.3`: Highly focused, good for factual content
- `0.6-0.8`: Balanced, good for general conversation
- `0.9-1.2`: Creative, good for storytelling
- `1.5+`: Very random, experimental

### Top-K Sampling

Limits consideration to the K most likely tokens.

```rust
let sampler_params = SamplerParams::default()
    .with_top_k(20);   // Conservative, focused
    // .with_top_k(40);   // Balanced (recommended)
    // .with_top_k(80);   // More variety
```

**Guidelines:**
- `10-30`: Conservative, predictable output
- `40-60`: Good balance of quality and variety
- `80-100`: More creative and diverse

### Top-P (Nucleus) Sampling

Selects from the smallest set of tokens whose cumulative probability exceeds P.

```rust
let sampler_params = SamplerParams::default()
    .with_top_p(0.9);    // Standard setting
    // .with_top_p(0.95);   // More variety
    // .with_top_p(0.8);    // More focused
```

**Guidelines:**
- `0.8-0.85`: Focused, consistent output
- `0.9-0.95`: Balanced variety and quality
- `0.95-0.99`: Maximum variety while maintaining quality

### Min-P Sampling

Sets a minimum probability threshold relative to the most likely token.

```rust
let sampler_params = SamplerParams::default()
    .with_min_p(0.05);   // 5% of the top token's probability
```

**Benefits:**
- More adaptive than fixed thresholds
- Maintains quality across different contexts
- Good for preventing very unlikely tokens

### Typical Sampling

Selects tokens based on their "typicalness" rather than just probability.

```rust
let sampler_params = SamplerParams::default()
    .with_typical_p(0.95);  // Standard setting
```

**Use Cases:**
- Better for maintaining narrative consistency
- Reduces repetition naturally
- Good for creative writing

## Repetition Control

### Repetition Penalty

Penalizes tokens that have appeared recently.

```rust
let sampler_params = SamplerParams::default()
    .with_repeat_penalty(1.1)      // Penalty multiplier
    .with_repeat_last_n(64);       // Look back N tokens
```

**Guidelines:**
- `1.0`: No penalty (default)
- `1.05-1.1`: Light penalty, maintains flow
- `1.1-1.2`: Standard penalty for most use cases
- `1.2+`: Strong penalty, may affect quality

### Frequency and Presence Penalties

More sophisticated repetition control.

```rust
let sampler_params = SamplerParams::default()
    .with_frequency_penalty(0.1)   // Penalize frequent tokens
    .with_presence_penalty(0.1);   // Penalize any repetition
```

## Advanced Sampling

### Mirostat Sampling

Maintains consistent perplexity (text difficulty).

```rust
use mullama::MirostatType;

let sampler_params = SamplerParams::default()
    .with_mirostat_type(MirostatType::V2)
    .with_mirostat_tau(5.0)       // Target entropy
    .with_mirostat_eta(0.1);      // Learning rate
```

**Benefits:**
- Maintains consistent text quality
- Adapts to content complexity
- Good for long-form generation

### TFS (Tail-Free Sampling)

Removes unlikely tokens based on derivative analysis.

```rust
let sampler_params = SamplerParams::default()
    .with_tfs_z(1.0);  // Standard setting
```

## Custom Sampling Chains

You can build custom sampling chains for specific use cases:

```rust
use mullama::sampling::*;

// Creative writing chain
let creative_sampler = SamplerChain::new()
    .add_repetition_penalty(1.15, 128)  // Strong anti-repetition
    .add_top_k(60)                      // Good variety
    .add_top_p(0.95)                    // High nucleus threshold
    .add_typical_p(0.95)                // Maintain typicalness
    .add_temperature(0.9);              // Creative temperature

// Factual/precise chain
let precise_sampler = SamplerChain::new()
    .add_repetition_penalty(1.05, 64)  // Light anti-repetition
    .add_min_p(0.1)                     // Higher minimum threshold
    .add_top_k(30)                      // Conservative variety
    .add_temperature(0.3);              // Low temperature

// Balanced conversation chain
let balanced_sampler = SamplerChain::new()
    .add_repetition_penalty(1.1, 64)
    .add_top_k(40)
    .add_top_p(0.9)
    .add_temperature(0.7);
```

## Use Case Examples

### Chatbot Assistant

```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9)
    .with_repeat_penalty(1.1)
    .with_repeat_last_n(64);
```

### Creative Writing

```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.9)
    .with_top_k(60)
    .with_top_p(0.95)
    .with_typical_p(0.95)
    .with_repeat_penalty(1.15)
    .with_repeat_last_n(128);
```

### Code Generation

```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.2)
    .with_top_k(20)
    .with_top_p(0.85)
    .with_repeat_penalty(1.05)
    .with_min_p(0.1);
```

### Factual Q&A

```rust
let sampler_params = SamplerParams::default()
    .with_temperature(0.3)
    .with_top_k(30)
    .with_top_p(0.8)
    .with_min_p(0.1)
    .with_repeat_penalty(1.05);
```

### Long-form Content

```rust
let sampler_params = SamplerParams::default()
    .with_mirostat_type(MirostatType::V2)
    .with_mirostat_tau(5.0)
    .with_mirostat_eta(0.1)
    .with_repeat_penalty(1.1)
    .with_repeat_last_n(256);  // Longer lookback
```

## Dynamic Sampling

You can adjust sampling parameters during generation:

```rust
use mullama::{SamplerChain, SamplerParams};

let mut sampler = SamplerParams::default()
    .with_temperature(0.7)
    .build_chain(&model);

// Generate some tokens...
for i in 0..50 {
    let token = sampler.sample(&context)?;
    context.eval_token(token)?;

    // Increase creativity as we go
    if i % 10 == 0 {
        let new_temp = 0.7 + (i as f32 * 0.01);
        sampler = SamplerParams::default()
            .with_temperature(new_temp)
            .build_chain(&model);
    }
}
```

## Guidance and Constraints

### Grammar-based Sampling

```rust
use mullama::Grammar;

// Define a simple JSON grammar
let json_grammar = Grammar::from_string(r#"
root ::= object
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? ws "}"
string ::= "\"" [^"]* "\""
value ::= string | number | object | array | "true" | "false" | "null"
array ::= "[" ws (value ("," ws value)*)? ws "]"
number ::= [0-9]+
ws ::= [ \t\n]*
"#)?;

let sampler_params = SamplerParams::default()
    .with_grammar(json_grammar)
    .with_temperature(0.7);
```

### Logit Bias

Bias specific tokens during sampling:

```rust
use std::collections::HashMap;

let mut logit_bias = HashMap::new();
logit_bias.insert(13, -1.0);  // Reduce probability of token 13
logit_bias.insert(42, 0.5);   // Increase probability of token 42

let sampler_params = SamplerParams::default()
    .with_logit_bias(logit_bias)
    .with_temperature(0.7);
```

## Performance Considerations

### Sampling Speed

Different sampling methods have different performance characteristics:

1. **Fastest**: Temperature only
2. **Fast**: Top-K, Top-P
3. **Medium**: Min-P, Typical-P
4. **Slower**: Mirostat, TFS
5. **Slowest**: Grammar-based

### Memory Usage

- Larger Top-K values use more memory
- Grammar-based sampling requires additional memory
- Longer repetition lookback (`repeat_last_n`) uses more memory

### Batch Processing

For maximum throughput with multiple sequences:

```rust
use mullama::Batch;

let batch = Batch::new()
    .add_sequence(tokens1, 0)
    .add_sequence(tokens2, 1)
    .add_sequence(tokens3, 2);

context.eval_batch(&batch)?;

// Sample for all sequences
for seq_id in 0..3 {
    let token = sampler.sample_sequence(&context, seq_id)?;
    // Process token...
}
```

## Troubleshooting

### Poor Quality Output

1. **Too repetitive**: Increase `repeat_penalty` or `repeat_last_n`
2. **Too random**: Lower `temperature` or `top_p`
3. **Too predictable**: Increase `temperature` or `top_k`
4. **Inconsistent**: Try Mirostat or Typical-P sampling

### Performance Issues

1. **Slow sampling**: Use simpler sampling methods (Temperature + Top-K)
2. **Memory issues**: Reduce `top_k` or `repeat_last_n`
3. **GPU utilization**: Ensure model layers are on GPU

### Common Mistakes

1. **Over-tuning**: Start with defaults and adjust gradually
2. **Conflicting settings**: Don't use very low temperature with high top_p
3. **Wrong use case**: Match sampling strategy to your specific needs

## Best Practices

1. **Start Simple**: Begin with temperature + top_p, then add complexity
2. **Test Iteratively**: Make small adjustments and evaluate results
3. **Consider Context**: Different content types need different settings
4. **Monitor Quality**: Use evaluation metrics for objective assessment
5. **Cache Settings**: Save good configurations for reuse

## Sampling Presets

Here are some proven presets for common scenarios:

```rust
// Preset configurations
impl SamplerParams {
    pub fn chatbot() -> Self {
        Self::default()
            .with_temperature(0.7)
            .with_top_k(40)
            .with_top_p(0.9)
            .with_repeat_penalty(1.1)
    }

    pub fn creative_writing() -> Self {
        Self::default()
            .with_temperature(0.9)
            .with_top_p(0.95)
            .with_typical_p(0.95)
            .with_repeat_penalty(1.15)
    }

    pub fn precise_factual() -> Self {
        Self::default()
            .with_temperature(0.3)
            .with_top_k(30)
            .with_min_p(0.1)
            .with_repeat_penalty(1.05)
    }
}
```

This comprehensive sampling system gives you complete control over text generation quality, allowing you to optimize for any use case while maintaining the highest performance.