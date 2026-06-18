use mullama::{Context, ContextParams};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

use crate::model::JsModel;
use crate::napi_error;

/// Embedding generator for creating text embeddings
#[napi]
pub struct JsEmbeddingGenerator {
    context: Context,
    model: Arc<mullama::Model>,
    normalize: bool,
}

#[napi]
impl JsEmbeddingGenerator {
    #[napi(constructor)]
    pub fn new(model: &JsModel, n_ctx: Option<u32>, normalize: Option<bool>) -> Result<Self> {
        let params = ContextParams {
            n_ctx: n_ctx.unwrap_or(512),
            embeddings: true,
            pooling_type: mullama::sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            ..Default::default()
        };

        let model_arc = model.get_inner();
        let context = Context::new(Arc::new((*model_arc).clone()), params)
            .map_err(|e| napi_error("Failed to create context", e))?;

        Ok(Self {
            context,
            model: model_arc,
            normalize: normalize.unwrap_or(true),
        })
    }

    #[napi]
    pub fn embed(&mut self, text: String) -> Result<Vec<f64>> {
        let tokens = self
            .model
            .tokenize(&text, true, false)
            .map_err(|e| napi_error("Tokenization failed", e))?;

        self.context.kv_cache_clear();
        self.context
            .decode(&tokens)
            .map_err(|e| napi_error("Decode failed", e))?;

        match self.context.get_embeddings() {
            Some(embeddings) => {
                let mut values: Vec<f64> = embeddings.iter().map(|&value| value as f64).collect();

                if self.normalize {
                    let norm: f64 = values.iter().map(|value| value * value).sum::<f64>().sqrt();
                    if norm > 0.0 {
                        for value in &mut values {
                            *value /= norm;
                        }
                    }
                }

                Ok(values)
            }
            None => Err(Error::from_reason("No embeddings available")),
        }
    }

    #[napi]
    pub fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            embeddings.push(self.embed(text)?);
        }

        Ok(embeddings)
    }

    #[napi(getter)]
    pub fn n_embd(&self) -> i32 {
        self.model.n_embd()
    }
}
