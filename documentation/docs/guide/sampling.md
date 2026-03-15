# Sampling Strategies

Sampling is the process of selecting the next token from the model's probability distribution during text generation. Mullama provides comprehensive support for all modern sampling strategies, allowing fine-grained control over output quality, creativity, and consistency.

## Overview

After the model computes logits (raw scores) for each token in the vocabulary, a sampler converts these scores into a probability distribution and selects the next token. Different sampling strategies produce different characteristics in the output.

## Temperature

Temperature scales the logits before converting them to probabilities. It is the most fundamental sampling parameter:

- **0.0** -- Deterministic (always picks the highest probability token)
- **0.1-0.5** -- Conservative, focused output
- **0.7-0.9** -- Balanced creativity and coherence
- **1.0** -- Raw model probabilities (no scaling)
- **1.2+** -- More creative, less predictable

=== "Node.js"

    ```javascript
    // Deterministic output
    const factual = await context.generate("What is 2+2?", 50, {
      temperature: 0.0,
    });

    // Creative output
    const creative = await context.generate("Write a poem:", 200, {
      temperature: 1.2,
    });
    ```

=== "Python"

    ```python
    from mullama import SamplerParams

    # Deterministic output
    factual = context.generate("What is 2+2?", max_tokens=50,
        params=SamplerParams(temperature=0.0))

    # Creative output
    creative = context.generate("Write a poem:", max_tokens=200,
        params=SamplerParams(temperature=1.2))
    ```

=== "Rust"

    ```rust
    use mullama::SamplerParams;

    // Deterministic output
    let factual = context.generate_with_params("What is 2+2?", 50,
        SamplerParams { temperature: 0.0, ..Default::default() })?;

    // Creative output
    let creative = context.generate_with_params("Write a poem:", 200,
        SamplerParams { temperature: 1.2, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    # Deterministic
    mullama run llama3.2:1b "What is 2+2?" --temperature 0

    # Creative
    mullama run llama3.2:1b "Write a poem:" --temperature 1.2
    ```

## Top-K Sampling

Top-K limits the candidate pool to the K highest-probability tokens. This prevents the model from selecting extremely unlikely tokens while preserving variety.

- **K=1** -- Equivalent to greedy/deterministic
- **K=10-40** -- Conservative, coherent output
- **K=40-100** -- Balanced variety
- **K=0** -- Disabled (consider all tokens)

=== "Node.js"

    ```javascript
    const response = await context.generate("Tell me a story:", 200, {
      temperature: 0.8,
      topK: 40,  // Consider only top 40 tokens at each step
    });
    ```

=== "Python"

    ```python
    response = context.generate("Tell me a story:", max_tokens=200,
        params=SamplerParams(temperature=0.8, top_k=40))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Tell me a story:", 200,
        SamplerParams { temperature: 0.8, top_k: 40, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Tell me a story:" --temperature 0.8 --top-k 40
    ```

## Top-P (Nucleus) Sampling

Top-P selects from the smallest set of tokens whose cumulative probability exceeds the threshold P. Unlike Top-K, it adapts to the distribution -- using fewer tokens when the model is confident and more when uncertain.

- **P=0.1** -- Very conservative
- **P=0.9** -- Standard, good for most tasks
- **P=0.95** -- Slightly more diverse
- **P=1.0** -- Disabled (consider all tokens)

=== "Node.js"

    ```javascript
    const response = await context.generate("Explain quantum physics:", 200, {
      temperature: 0.7,
      topP: 0.9,  // Include tokens until 90% cumulative probability
    });
    ```

=== "Python"

    ```python
    response = context.generate("Explain quantum physics:", max_tokens=200,
        params=SamplerParams(temperature=0.7, top_p=0.9))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Explain quantum physics:", 200,
        SamplerParams { temperature: 0.7, top_p: 0.9, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Explain quantum physics:" --temperature 0.7 --top-p 0.9
    ```

## Min-P Sampling

Min-P filters out tokens whose probability is below a fraction of the top token's probability. This provides adaptive filtering that scales with model confidence.

- **P=0.0** -- Disabled
- **P=0.05** -- Standard (remove tokens less than 5% of top probability)
- **P=0.1** -- Conservative
- **P=0.5** -- Very conservative

=== "Node.js"

    ```javascript
    const response = await context.generate("Write code:", 200, {
      temperature: 0.7,
      minP: 0.05,  // Remove tokens < 5% of top token probability
    });
    ```

=== "Python"

    ```python
    response = context.generate("Write code:", max_tokens=200,
        params=SamplerParams(temperature=0.7, min_p=0.05))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Write code:", 200,
        SamplerParams { temperature: 0.7, min_p: 0.05, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write code:" --temperature 0.7 --min-p 0.05
    ```

!!! tip "Min-P vs Top-P"
    Min-P is generally preferred over Top-P for most applications. It adapts better to varying confidence levels and produces more consistent quality across different prompts.

## Typical Sampling

Typical sampling selects tokens whose information content (surprisal) is close to the expected information content of the model. This filters out both overly predictable and overly surprising tokens.

- **P=1.0** -- Disabled (default)
- **P=0.95** -- Mild filtering
- **P=0.8** -- Moderate filtering
- **P=0.5** -- Aggressive filtering

=== "Node.js"

    ```javascript
    const response = await context.generate("Write prose:", 200, {
      typicalP: 0.95,
    });
    ```

=== "Python"

    ```python
    response = context.generate("Write prose:", max_tokens=200,
        params=SamplerParams(typical_p=0.95))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Write prose:", 200,
        SamplerParams { typical_p: 0.95, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write prose:" --typical-p 0.95
    ```

## Tail-Free Sampling

Tail-free sampling removes tokens in the "tail" of the probability distribution by analyzing the second derivative of the sorted probabilities. This effectively cuts off the long tail of improbable tokens.

- **Z=1.0** -- Disabled
- **Z=0.95** -- Mild tail removal
- **Z=0.5** -- Aggressive tail removal

=== "Node.js"

    ```javascript
    const response = await context.generate("Generate text:", 200, {
      tfsZ: 0.95,
    });
    ```

=== "Python"

    ```python
    response = context.generate("Generate text:", max_tokens=200,
        params=SamplerParams(tfs_z=0.95))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Generate text:", 200,
        SamplerParams { tfs_z: 0.95, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Generate text:" --tfs-z 0.95
    ```

## Mirostat (Adaptive Perplexity)

Mirostat dynamically adjusts the sampling parameters to maintain a target perplexity level. Instead of fixed parameters, it adapts on-the-fly to keep output quality consistent throughout generation.

### Mirostat v1

Controls perplexity using a learning rate and target surprise value:

=== "Node.js"

    ```javascript
    const response = await context.generate("Write an essay:", 500, {
      mirostat: 1,      // Enable Mirostat v1
      mirostatTau: 5.0, // Target perplexity (lower = more focused)
      mirostatEta: 0.1, // Learning rate
    });
    ```

=== "Python"

    ```python
    response = context.generate("Write an essay:", max_tokens=500,
        params=SamplerParams(mirostat=1, mirostat_tau=5.0, mirostat_eta=0.1))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Write an essay:", 500,
        SamplerParams {
            mirostat: 1, mirostat_tau: 5.0, mirostat_eta: 0.1,
            ..Default::default()
        })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write an essay:" \
      --mirostat 1 --mirostat-tau 5.0 --mirostat-eta 0.1
    ```

### Mirostat v2

An improved version with better convergence:

=== "Node.js"

    ```javascript
    const response = await context.generate("Write an essay:", 500, {
      mirostat: 2,      // Enable Mirostat v2
      mirostatTau: 5.0, // Target perplexity
      mirostatEta: 0.1, // Learning rate
    });
    ```

=== "Python"

    ```python
    response = context.generate("Write an essay:", max_tokens=500,
        params=SamplerParams(mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Write an essay:", 500,
        SamplerParams {
            mirostat: 2, mirostat_tau: 5.0, mirostat_eta: 0.1,
            ..Default::default()
        })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write an essay:" \
      --mirostat 2 --mirostat-tau 5.0 --mirostat-eta 0.1
    ```

!!! info "When to Use Mirostat"
    Mirostat is particularly useful for long-form generation where fixed parameters may cause quality drift. It maintains consistent output quality regardless of generation length.

## Repetition Penalties

Prevent the model from repeating itself by penalizing tokens that have already appeared:

### Repeat Penalty

Applies a multiplicative penalty to tokens that have appeared in the last N tokens:

- **1.0** -- No penalty (disabled)
- **1.1** -- Mild penalty (recommended for most use cases)
- **1.5** -- Strong penalty

### Frequency Penalty

Penalizes based on how many times a token has appeared (higher count = higher penalty):

### Presence Penalty

Penalizes any token that has appeared at all (binary: appeared or not):

=== "Node.js"

    ```javascript
    const response = await context.generate("Write a story:", 500, {
      penaltyRepeat: 1.1,     // Multiplicative repeat penalty
      penaltyFreq: 0.1,       // Frequency-based penalty
      penaltyPresent: 0.1,    // Presence-based penalty
      repeatLastN: 64,        // Lookback window (last 64 tokens)
    });
    ```

=== "Python"

    ```python
    response = context.generate("Write a story:", max_tokens=500,
        params=SamplerParams(
            penalty_repeat=1.1,
            penalty_freq=0.1,
            penalty_present=0.1,
            repeat_last_n=64,
        ))
    ```

=== "Rust"

    ```rust
    let response = context.generate_with_params("Write a story:", 500,
        SamplerParams {
            penalty_repeat: 1.1,
            penalty_freq: 0.1,
            penalty_present: 0.1,
            repeat_last_n: 64,
            ..Default::default()
        })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Write a story:" \
      --repeat-penalty 1.1 \
      --frequency-penalty 0.1 \
      --presence-penalty 0.1
    ```

## Logit Bias

Directly modify the probability of specific tokens. Positive values increase likelihood, negative values decrease it:

=== "Node.js"

    ```javascript
    // Bias specific token IDs
    const response = await context.generate("Colors:", 100, {
      logitBias: {
        1234: 5.0,    // Strongly prefer token 1234
        5678: -100.0, // Effectively ban token 5678
      },
    });
    ```

=== "Python"

    ```python
    response = context.generate("Colors:", max_tokens=100,
        params=SamplerParams(
            logit_bias={1234: 5.0, 5678: -100.0}
        ))
    ```

=== "Rust"

    ```rust
    use std::collections::HashMap;

    let mut bias = HashMap::new();
    bias.insert(1234, 5.0);    // Prefer
    bias.insert(5678, -100.0); // Ban

    let response = context.generate_with_params("Colors:", 100,
        SamplerParams { logit_bias: bias, ..Default::default() })?;
    ```

=== "CLI"

    ```bash
    mullama run llama3.2:1b "Colors:" --logit-bias "1234:5.0,5678:-100.0"
    ```

## Sampler Chains

Combine multiple sampling strategies into a chain. Samplers are applied in order, each filtering the candidates before passing to the next:

=== "Node.js"

    ```javascript
    import { SamplerChain } from 'mullama';

    const sampler = new SamplerChain()
      .addRepetitionPenalty(1.1, 64)  // First: penalize repetitions
      .addTemperature(0.8)            // Then: scale logits
      .addTopK(40)                    // Then: keep top 40
      .addTopP(0.9)                   // Then: nucleus filter
      .addMinP(0.05)                  // Then: min-p filter
      .addDist(42);                   // Finally: sample with seed

    // Use the chain for token-by-token generation
    const token = sampler.sample(context);
    ```

=== "Python"

    ```python
    from mullama import SamplerChain

    sampler = (SamplerChain()
        .add_repetition_penalty(1.1, 64)  # First: penalize repetitions
        .add_temperature(0.8)              # Then: scale logits
        .add_top_k(40)                     # Then: keep top 40
        .add_top_p(0.9)                    # Then: nucleus filter
        .add_min_p(0.05)                   # Then: min-p filter
        .add_dist(42))                     # Finally: sample with seed

    # Use the chain for token-by-token generation
    token = sampler.sample(context)
    ```

=== "Rust"

    ```rust
    use mullama::sampling::{Sampler, SamplerChain};

    let chain = SamplerChain::new()
        .add(Sampler::repetition_penalty(1.1, 64)?)  // Penalize repetitions
        .add(Sampler::temperature(0.8)?)              // Scale logits
        .add(Sampler::top_k(40)?)                     // Keep top 40
        .add(Sampler::top_p(0.9, 1)?)                 // Nucleus filter
        .add(Sampler::min_p(0.05, 1)?)                // Min-p filter
        .add(Sampler::dist(42)?);                     // Sample with seed

    let token = chain.sample(&context)?;
    ```

=== "CLI"

    ```bash
    # CLI applies samplers in the standard order automatically
    mullama run llama3.2:1b "Hello:" \
      --repeat-penalty 1.1 \
      --temperature 0.8 \
      --top-k 40 \
      --top-p 0.9 \
      --min-p 0.05 \
      --seed 42
    ```

!!! warning "Sampler Order Matters"
    The order of samplers in the chain affects output. The recommended order is: penalties, temperature, top-k, top-p/min-p, distribution sampling. Applying temperature after top-k produces different results than the reverse.

## Recommended Presets

| Use Case | Temperature | Top-K | Top-P | Min-P | Repeat Penalty |
|----------|-------------|-------|-------|-------|----------------|
| Code generation | 0.0 | -- | -- | -- | 1.0 |
| Factual Q&A | 0.1 | 10 | -- | 0.1 | 1.0 |
| General chat | 0.7 | 40 | 0.9 | 0.05 | 1.1 |
| Creative writing | 1.0 | 80 | 0.95 | 0.02 | 1.1 |
| Brainstorming | 1.2 | 100 | 0.99 | 0.01 | 1.2 |
| Long-form essays | Mirostat v2 | -- | -- | -- | 1.1 |

## SIMD Acceleration

Mullama uses SIMD (Single Instruction, Multiple Data) instructions to accelerate sampling operations on supported hardware. This is automatic and requires no configuration:

- **x86_64**: AVX2/AVX-512 for probability calculations
- **ARM**: NEON for vectorized operations
- **Apple Silicon**: Accelerate framework integration

The performance benefit is most noticeable with large vocabularies (32K+ tokens) and complex sampler chains.

## See Also

- [Text Generation](generation.md) -- Using sampling with generation
- [Structured Output](structured-output.md) -- Grammar-constrained sampling
- [Grammar Constraints](grammar.md) -- GBNF grammar syntax
- [API Reference: Sampling](../api/sampling.md) -- Complete Sampling API documentation
