//! Complete sampling system with 100% llama.cpp API coverage
//!
//! This module provides the full sampling functionality including:
//! - All sampling strategies (greedy, top-k, top-p, temperature, etc.)
//! - Sampler chains for combining multiple strategies
//! - Advanced samplers (mirostat, typical, tail-free, etc.)
//! - Grammar-constrained sampling
//! - Penalty systems for repetition control
//! - Logit bias for token preference control

use crate::{sys, error::MullamaError, model::Model, context::Context, token::TokenId};
use std::{ffi::CString, ptr, sync::Arc};
use std::os::raw::c_void;

/// High-level sampler wrapper providing safe access to llama.cpp sampling
pub struct Sampler {
    sampler_ptr: *mut sys::llama_sampler,
    _model: Option<Arc<Model>>, // Keep model alive if needed
}

impl Sampler {
    /// Create a greedy sampler (always picks highest probability token)
    pub fn greedy() -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_greedy() };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a distribution sampler with random seed
    pub fn dist(seed: u32) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_dist(seed) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a top-k sampler
    pub fn top_k(k: i32) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_top_k(k) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a top-p (nucleus) sampler
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_top_p(p, min_keep) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a min-p sampler
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_min_p(p, min_keep) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a tail-free sampling (TFS) sampler
    pub fn tail_free(z: f32, min_keep: usize) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_tail_free(z, min_keep) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a typical sampling sampler
    pub fn typical(p: f32, min_keep: usize) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_typical(p, min_keep) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a temperature sampler
    pub fn temperature(temperature: f32) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_temp(temperature) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create an extended temperature sampler with additional parameters
    pub fn temperature_ext(temperature: f32, delta: f32, exponent: f32) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_temp_ext(temperature, delta, exponent) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a Mirostat sampler (version 1)
    pub fn mirostat(model: Arc<Model>, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
        let sampler_ptr = unsafe {
            sys::llama_sampler_init_mirostat(vocab_ptr, seed, tau, eta, m)
        };
        Self {
            sampler_ptr,
            _model: Some(model),
        }
    }

    /// Create a Mirostat v2 sampler
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        let sampler_ptr = unsafe { sys::llama_sampler_init_mirostat_v2(seed, tau, eta) };
        Self {
            sampler_ptr,
            _model: None,
        }
    }

    /// Create a grammar-constrained sampler
    pub fn grammar(model: Arc<Model>, grammar_str: &str, grammar_root: &str) -> Result<Self, MullamaError> {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };

        let c_grammar_str = CString::new(grammar_str)
            .map_err(|_| MullamaError::SamplingError("Invalid grammar string".to_string()))?;
        let c_grammar_root = CString::new(grammar_root)
            .map_err(|_| MullamaError::SamplingError("Invalid grammar root".to_string()))?;

        let sampler_ptr = unsafe {
            sys::llama_sampler_init_grammar(
                vocab_ptr,
                c_grammar_str.as_ptr(),
                c_grammar_root.as_ptr(),
            )
        };

        if sampler_ptr.is_null() {
            return Err(MullamaError::SamplingError("Failed to create grammar sampler".to_string()));
        }

        Ok(Self {
            sampler_ptr,
            _model: Some(model),
        })
    }

    /// Create a penalties sampler for repetition control
    pub fn penalties(
        model: Arc<Model>,
        special_eos_id: TokenId,
        linefeed_id: TokenId,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
        penalize_nl: bool,
        ignore_eos: bool,
    ) -> Self {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
        let sampler_ptr = unsafe {
            sys::llama_sampler_init_penalties(
                vocab_ptr,
                special_eos_id as sys::llama_token,
                linefeed_id as sys::llama_token,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
                penalize_nl as sys::c_bool,
                ignore_eos as sys::c_bool,
            )
        };

        Self {
            sampler_ptr,
            _model: Some(model),
        }
    }

    /// Create a logit bias sampler for token preference control
    pub fn logit_bias(model: Arc<Model>, logit_biases: &[LogitBias]) -> Self {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };

        let sys_biases: Vec<sys::llama_logit_bias> = logit_biases
            .iter()
            .map(|bias| sys::llama_logit_bias {
                token: bias.token as sys::llama_token,
                bias: bias.bias,
            })
            .collect();

        let sampler_ptr = unsafe {
            sys::llama_sampler_init_logit_bias(
                vocab_ptr,
                sys_biases.len() as i32,
                sys_biases.as_ptr(),
            )
        };

        Self {
            sampler_ptr,
            _model: Some(model),
        }
    }

    /// Sample a token from the given context at the specified position
    pub fn sample(&mut self, context: &mut Context, idx: i32) -> TokenId {
        let token = unsafe {
            sys::llama_sampler_sample(self.sampler_ptr, context.as_ptr(), idx)
        };
        token as TokenId
    }

    /// Accept a token (for stateful samplers like Mirostat)
    pub fn accept(&mut self, token: TokenId) {
        unsafe {
            sys::llama_sampler_accept(self.sampler_ptr, token as sys::llama_token);
        }
    }

    /// Apply this sampler to a token data array
    pub fn apply(&mut self, candidates: &mut TokenDataArray) {
        unsafe {
            sys::llama_sampler_apply(self.sampler_ptr, &mut candidates.inner);
        }
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        unsafe {
            sys::llama_sampler_reset(self.sampler_ptr);
        }
    }

    /// Clone this sampler
    pub fn try_clone(&self) -> Result<Self, MullamaError> {
        let cloned_ptr = unsafe { sys::llama_sampler_clone(self.sampler_ptr) };
        if cloned_ptr.is_null() {
            return Err(MullamaError::SamplingError("Failed to clone sampler".to_string()));
        }

        Ok(Self {
            sampler_ptr: cloned_ptr,
            _model: self._model.clone(),
        })
    }

    /// Get the name of this sampler
    pub fn name(&self) -> String {
        let name_ptr = unsafe { sys::llama_sampler_name(self.sampler_ptr) };
        if name_ptr.is_null() {
            return "unknown".to_string();
        }

        unsafe {
            std::ffi::CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .to_string()
        }
    }

    /// Get performance data for this sampler
    pub fn perf_data(&self) -> SamplerPerfData {
        let data = unsafe { sys::llama_perf_sampler(self.sampler_ptr) };
        SamplerPerfData {
            t_sample_ms: data.t_sample_ms,
            n_sample: data.n_sample,
        }
    }

    /// Print performance information
    pub fn perf_print(&self) {
        unsafe {
            sys::llama_perf_sampler_print(self.sampler_ptr);
        }
    }

    /// Reset performance counters
    pub fn perf_reset(&mut self) {
        unsafe {
            sys::llama_perf_sampler_reset(self.sampler_ptr);
        }
    }

    /// Get the internal sampler pointer (for advanced use)
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_sampler {
        self.sampler_ptr
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if !self.sampler_ptr.is_null() {
            unsafe {
                sys::llama_sampler_free(self.sampler_ptr);
            }
        }
    }
}

/// Sampler chain for combining multiple sampling strategies
pub struct SamplerChain {
    chain_ptr: *mut sys::llama_sampler,
}

impl SamplerChain {
    /// Create a new sampler chain
    pub fn new(params: SamplerChainParams) -> Self {
        let sys_params = sys::llama_sampler_chain_params {
            no_perf: params.no_perf as sys::c_bool,
        };

        let chain_ptr = unsafe { sys::llama_sampler_chain_init(sys_params) };
        Self { chain_ptr }
    }

    /// Create a chain with default parameters
    pub fn default() -> Self {
        Self::new(SamplerChainParams::default())
    }

    /// Add a sampler to the chain (takes ownership)
    pub fn add(&mut self, sampler: Sampler) {
        unsafe {
            sys::llama_sampler_chain_add(self.chain_ptr, sampler.sampler_ptr);
        }
        // Prevent the sampler from being dropped since the chain now owns it
        std::mem::forget(sampler);
    }

    /// Get a sampler from the chain by index
    pub fn get(&self, index: i32) -> Option<*mut sys::llama_sampler> {
        let sampler_ptr = unsafe { sys::llama_sampler_chain_get(self.chain_ptr, index) };
        if sampler_ptr.is_null() {
            None
        } else {
            Some(sampler_ptr)
        }
    }

    /// Get the number of samplers in the chain
    pub fn len(&self) -> i32 {
        unsafe { sys::llama_sampler_chain_n(self.chain_ptr) }
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove a sampler from the chain by index
    pub fn remove(&mut self, index: i32) -> Option<Sampler> {
        let removed_ptr = unsafe { sys::llama_sampler_chain_remove(self.chain_ptr, index) };
        if removed_ptr.is_null() {
            None
        } else {
            Some(Sampler {
                sampler_ptr: removed_ptr,
                _model: None, // Can't preserve model reference
            })
        }
    }

    /// Sample using the entire chain
    pub fn sample(&mut self, context: &mut Context, idx: i32) -> TokenId {
        let token = unsafe {
            sys::llama_sampler_sample(self.chain_ptr, context.as_ptr(), idx)
        };
        token as TokenId
    }

    /// Accept a token for all samplers in the chain
    pub fn accept(&mut self, token: TokenId) {
        unsafe {
            sys::llama_sampler_accept(self.chain_ptr, token as sys::llama_token);
        }
    }

    /// Reset all samplers in the chain
    pub fn reset(&mut self) {
        unsafe {
            sys::llama_sampler_reset(self.chain_ptr);
        }
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        if !self.chain_ptr.is_null() {
            unsafe {
                sys::llama_sampler_free(self.chain_ptr);
            }
        }
    }
}

/// Parameters for creating a sampler chain
#[derive(Debug, Clone)]
pub struct SamplerChainParams {
    pub no_perf: bool,
}

impl Default for SamplerChainParams {
    fn default() -> Self {
        Self { no_perf: false }
    }
}

/// Logit bias for controlling token preferences
#[derive(Debug, Clone)]
pub struct LogitBias {
    pub token: TokenId,
    pub bias: f32,
}

/// Token data with probability information
#[derive(Debug, Clone)]
pub struct TokenData {
    pub id: TokenId,
    pub logit: f32,
    pub p: f32,
}

/// Array of token candidates for sampling
pub struct TokenDataArray {
    inner: sys::llama_token_data_array,
    _data: Vec<sys::llama_token_data>, // Keep data alive
}

impl TokenDataArray {
    /// Create a new token data array from candidates
    pub fn new(mut candidates: Vec<TokenData>) -> Self {
        let mut data: Vec<sys::llama_token_data> = candidates
            .iter()
            .map(|candidate| sys::llama_token_data {
                id: candidate.id as sys::llama_token,
                logit: candidate.logit,
                p: candidate.p,
            })
            .collect();

        let inner = sys::llama_token_data_array {
            data: data.as_mut_ptr(),
            size: data.len(),
            selected: -1,
            sorted: false as sys::c_bool,
        };

        Self {
            inner,
            _data: data,
        }
    }

    /// Get the number of candidates
    pub fn len(&self) -> usize {
        self.inner.size
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the selected token index
    pub fn selected(&self) -> Option<usize> {
        if self.inner.selected >= 0 {
            Some(self.inner.selected as usize)
        } else {
            None
        }
    }

    /// Check if the array is sorted
    pub fn is_sorted(&self) -> bool {
        self.inner.sorted as bool
    }

    /// Get candidates as a slice
    pub fn candidates(&self) -> &[TokenData] {
        unsafe {
            let data_slice = std::slice::from_raw_parts(
                self.inner.data as *const sys::llama_token_data,
                self.inner.size,
            );
            std::mem::transmute(data_slice)
        }
    }
}

/// Performance data for a sampler
#[derive(Debug, Clone)]
pub struct SamplerPerfData {
    pub t_sample_ms: f64,
    pub n_sample: i32,
}

/// Default sampler parameters for common use cases
#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalty_last_n: i32,
    pub penalize_nl: bool,
    pub ignore_eos: bool,
    pub seed: u32,
}

impl Default for SamplerParams {
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
            seed: sys::LLAMA_DEFAULT_SEED,
        }
    }
}

impl SamplerParams {
    /// Create a typical sampling chain from these parameters
    pub fn build_chain(&self, model: Arc<Model>) -> SamplerChain {
        let mut chain = SamplerChain::default();

        // Add penalties first
        if self.penalty_repeat != 1.0 || self.penalty_freq != 0.0 || self.penalty_present != 0.0 {
            let penalties = Sampler::penalties(
                model.clone(),
                model.token_eos(),
                model.token_nl(),
                self.penalty_last_n,
                self.penalty_repeat,
                self.penalty_freq,
                self.penalty_present,
                self.penalize_nl,
                self.ignore_eos,
            );
            chain.add(penalties);
        }

        // Add top-k filtering
        if self.top_k > 0 {
            chain.add(Sampler::top_k(self.top_k));
        }

        // Add tail-free sampling if enabled
        if self.typical_p < 1.0 && self.typical_p > 0.0 {
            chain.add(Sampler::typical(self.typical_p, 1));
        }

        // Add top-p filtering
        if self.top_p < 1.0 {
            chain.add(Sampler::top_p(self.top_p, 1));
        }

        // Add min-p filtering
        if self.min_p > 0.0 {
            chain.add(Sampler::min_p(self.min_p, 1));
        }

        // Add temperature scaling
        if self.temperature > 0.0 {
            chain.add(Sampler::temperature(self.temperature));
        }

        // Add final distribution sampler
        chain.add(Sampler::dist(self.seed));

        chain
    }
}