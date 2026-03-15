//! Sampler FFI bindings
//!
//! This module provides C ABI structures and functions for configuring
//! sampling parameters used during text generation.

use libc::c_int;
use mullama::sys::LLAMA_DEFAULT_SEED;

/// Sampler parameters for text generation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MullamaSamplerParams {
    /// Temperature for randomness (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: c_int,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-p sampling (0.0 = disabled)
    pub min_p: f32,
    /// Typical sampling (1.0 = disabled)
    pub typical_p: f32,
    /// Repeat penalty (1.0 = disabled)
    pub penalty_repeat: f32,
    /// Frequency penalty (0.0 = disabled)
    pub penalty_freq: f32,
    /// Presence penalty (0.0 = disabled)
    pub penalty_present: f32,
    /// Number of last tokens to consider for penalty
    pub penalty_last_n: c_int,
    /// Whether to penalize newline tokens
    pub penalize_nl: bool,
    /// Whether to ignore EOS tokens
    pub ignore_eos: bool,
    /// Random seed (0 = random)
    pub seed: u32,
}

impl Default for MullamaSamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            penalize_nl: true,
            ignore_eos: false,
            seed: LLAMA_DEFAULT_SEED,
        }
    }
}

/// Get default sampler parameters
#[no_mangle]
pub extern "C" fn mullama_sampler_default_params() -> MullamaSamplerParams {
    MullamaSamplerParams::default()
}

/// Create greedy sampler params (deterministic, always picks top token)
#[no_mangle]
pub extern "C" fn mullama_sampler_greedy_params() -> MullamaSamplerParams {
    MullamaSamplerParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        typical_p: 1.0,
        penalty_repeat: 1.0,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        penalty_last_n: 0,
        penalize_nl: false,
        ignore_eos: false,
        seed: 0,
    }
}

/// Create creative sampler params (higher randomness)
#[no_mangle]
pub extern "C" fn mullama_sampler_creative_params() -> MullamaSamplerParams {
    MullamaSamplerParams {
        temperature: 1.2,
        top_k: 100,
        top_p: 0.95,
        min_p: 0.02,
        typical_p: 1.0,
        penalty_repeat: 1.15,
        penalty_freq: 0.1,
        penalty_present: 0.1,
        penalty_last_n: 128,
        penalize_nl: true,
        ignore_eos: false,
        seed: LLAMA_DEFAULT_SEED,
    }
}

/// Create precise sampler params (lower randomness, more focused)
#[no_mangle]
pub extern "C" fn mullama_sampler_precise_params() -> MullamaSamplerParams {
    MullamaSamplerParams {
        temperature: 0.3,
        top_k: 20,
        top_p: 0.8,
        min_p: 0.1,
        typical_p: 1.0,
        penalty_repeat: 1.05,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        penalty_last_n: 32,
        penalize_nl: true,
        ignore_eos: false,
        seed: LLAMA_DEFAULT_SEED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = mullama_sampler_default_params();
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.top_p, 0.95);
    }

    #[test]
    fn test_greedy_params() {
        let params = mullama_sampler_greedy_params();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_k, 1);
    }

    #[test]
    fn test_creative_params() {
        let params = mullama_sampler_creative_params();
        assert!(params.temperature > 1.0);
    }
}
