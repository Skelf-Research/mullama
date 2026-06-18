use crate::{Model, MullamaError, SamplerChain, SamplerParams};
use std::sync::Arc;

/// Builder for creating samplers with fluent API
#[derive(Debug, Clone)]
pub struct SamplerBuilder {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    penalty_last_n: i32,
    seed: u32,
}

impl SamplerBuilder {
    /// Create a new sampler builder
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            seed: 0,
        }
    }

    /// Set temperature for sampling
    ///
    /// # Arguments
    ///
    /// * `temp` - Temperature value (0.0 = deterministic, higher = more random)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .temperature(0.7);
    /// ```
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k sampling
    ///
    /// # Arguments
    ///
    /// * `k` - Number of top tokens to consider (0 = disabled)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .top_k(50);
    /// ```
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) sampling
    ///
    /// # Arguments
    ///
    /// * `p` - Cumulative probability threshold (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .nucleus(0.9); // Top-p sampling with p=0.9
    /// ```
    pub fn nucleus(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set minimum probability threshold
    ///
    /// # Arguments
    ///
    /// * `min_p` - Minimum probability for token consideration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .min_probability(0.02);
    /// ```
    pub fn min_probability(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Configure penalties using a closure
    ///
    /// # Arguments
    ///
    /// * `config` - Closure that configures penalty settings
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .penalties(|p| p
    ///         .repetition(1.15)
    ///         .frequency(0.1)
    ///         .presence(0.1)
    ///         .lookback(128)
    ///     );
    /// ```
    pub fn penalties<F>(mut self, config: F) -> Self
    where
        F: FnOnce(PenaltyBuilder) -> PenaltyBuilder,
    {
        let penalty_builder = PenaltyBuilder {
            repeat: self.penalty_repeat,
            freq: self.penalty_freq,
            present: self.penalty_present,
            last_n: self.penalty_last_n,
        };

        let configured = config(penalty_builder);
        self.penalty_repeat = configured.repeat;
        self.penalty_freq = configured.freq;
        self.penalty_present = configured.present;
        self.penalty_last_n = configured.last_n;
        self
    }

    /// Set random seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed (0 = random)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .seed(12345);
    /// ```
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Apply a preset configuration
    ///
    /// # Arguments
    ///
    /// * `preset` - Preset configuration function
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{SamplerBuilder, presets};
    ///
    /// let builder = SamplerBuilder::new()
    ///     .preset(presets::creative_sampling);
    /// ```
    pub fn preset<F>(self, preset: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        preset(self)
    }

    /// Build the sampler
    ///
    /// # Arguments
    ///
    /// * `model` - The model to create the sampler for
    ///
    /// # Returns
    ///
    /// A `SamplerChain` ready for use
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, SamplerBuilder};
    ///
    /// # fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let sampler = SamplerBuilder::new()
    ///     .temperature(0.8)
    ///     .top_k(50)
    ///     .build(model)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self, model: Arc<Model>) -> Result<SamplerChain, MullamaError> {
        let params = SamplerParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            penalty_repeat: self.penalty_repeat,
            penalty_freq: self.penalty_freq,
            penalty_present: self.penalty_present,
            penalty_last_n: self.penalty_last_n,
            seed: self.seed,
            ..Default::default()
        };

        params.build_chain(model)
    }
}

impl Default for SamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring penalties
#[derive(Debug, Clone)]
pub struct PenaltyBuilder {
    repeat: f32,
    freq: f32,
    present: f32,
    last_n: i32,
}

impl PenaltyBuilder {
    /// Set repetition penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)
    pub fn repetition(mut self, penalty: f32) -> Self {
        self.repeat = penalty;
        self
    }

    /// Set frequency penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Frequency penalty
    pub fn frequency(mut self, penalty: f32) -> Self {
        self.freq = penalty;
        self
    }

    /// Set presence penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Presence penalty
    pub fn presence(mut self, penalty: f32) -> Self {
        self.present = penalty;
        self
    }

    /// Set lookback window for penalties
    ///
    /// # Arguments
    ///
    /// * `tokens` - Number of tokens to look back for penalty calculation
    pub fn lookback(mut self, tokens: i32) -> Self {
        self.last_n = tokens;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::presets;

    #[test]
    fn test_sampler_builder() {
        let builder = SamplerBuilder::new()
            .temperature(0.8)
            .top_k(50)
            .nucleus(0.95);

        assert_eq!(builder.temperature, 0.8);
        assert_eq!(builder.top_k, 50);
        assert_eq!(builder.top_p, 0.95);
    }

    #[test]
    fn test_penalty_builder() {
        let builder = SamplerBuilder::new()
            .penalties(|p| p.repetition(1.2).frequency(0.1).presence(0.1).lookback(128));

        assert_eq!(builder.penalty_repeat, 1.2);
        assert_eq!(builder.penalty_freq, 0.1);
        assert_eq!(builder.penalty_present, 0.1);
        assert_eq!(builder.penalty_last_n, 128);
    }

    #[test]
    fn test_presets() {
        let creative = SamplerBuilder::new().preset(presets::creative_sampling);

        assert!(creative.temperature > 0.8);
        assert!(creative.top_k > 50);

        let precise = SamplerBuilder::new().preset(presets::precise_sampling);

        assert!(precise.temperature < 0.3);
        assert!(precise.top_k < 20);
    }
}
