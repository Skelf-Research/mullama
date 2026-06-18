use mullama::SamplerParams;
use napi_derive::napi;

/// Sampler parameters for text generation.
#[napi(object)]
#[derive(Clone)]
pub struct JsSamplerParams {
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub typical_p: Option<f64>,
    pub penalty_repeat: Option<f64>,
    pub penalty_freq: Option<f64>,
    pub penalty_present: Option<f64>,
    pub penalty_last_n: Option<i32>,
    pub seed: Option<u32>,
}

impl Default for JsSamplerParams {
    fn default() -> Self {
        JsSamplerParams {
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: Some(0.05),
            typical_p: Some(1.0),
            penalty_repeat: Some(1.1),
            penalty_freq: Some(0.0),
            penalty_present: Some(0.0),
            penalty_last_n: Some(64),
            seed: Some(0),
        }
    }
}

impl From<&JsSamplerParams> for SamplerParams {
    fn from(p: &JsSamplerParams) -> Self {
        SamplerParams {
            temperature: p.temperature.unwrap_or(0.8) as f32,
            top_k: p.top_k.unwrap_or(40),
            top_p: p.top_p.unwrap_or(0.95) as f32,
            min_p: p.min_p.unwrap_or(0.05) as f32,
            typical_p: p.typical_p.unwrap_or(1.0) as f32,
            penalty_repeat: p.penalty_repeat.unwrap_or(1.1) as f32,
            penalty_freq: p.penalty_freq.unwrap_or(0.0) as f32,
            penalty_present: p.penalty_present.unwrap_or(0.0) as f32,
            penalty_last_n: p.penalty_last_n.unwrap_or(64),
            seed: p.seed.unwrap_or(0),
            ..Default::default()
        }
    }
}

#[napi]
pub fn sampler_params_greedy() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        min_p: Some(0.0),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.0),
        penalty_freq: Some(0.0),
        penalty_present: Some(0.0),
        penalty_last_n: Some(0),
        seed: Some(0),
    }
}

#[napi]
pub fn sampler_params_creative() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(1.2),
        top_k: Some(100),
        top_p: Some(0.95),
        min_p: Some(0.02),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.15),
        penalty_freq: Some(0.1),
        penalty_present: Some(0.1),
        penalty_last_n: Some(128),
        seed: Some(0),
    }
}

#[napi]
pub fn sampler_params_precise() -> JsSamplerParams {
    JsSamplerParams {
        temperature: Some(0.3),
        top_k: Some(20),
        top_p: Some(0.8),
        min_p: Some(0.1),
        typical_p: Some(1.0),
        penalty_repeat: Some(1.05),
        penalty_freq: Some(0.0),
        penalty_present: Some(0.0),
        penalty_last_n: Some(32),
        seed: Some(0),
    }
}
