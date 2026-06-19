use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::Mutex as TokioMutex;

use super::config::{detect_quantization_from_path, ModelLoadConfig};
use super::loaded::LoadedModel;
use crate::daemon::protocol::ModelInfo;
use crate::{Context, ContextParams, Model, ModelParams, MullamaError};

#[cfg(feature = "multimodal")]
use crate::{MtmdContext, MtmdParams};

struct LoadedCore {
    model: Arc<Model>,
    context: Context,
    ctx_params: ContextParams,
}

/// Multi-model manager with lock-free concurrent access
pub struct ModelManager {
    models: DashMap<String, Arc<LoadedModel>>,
    default_model: RwLock<Option<String>>,
    total_tokens: AtomicU64,
    mutation_lock: TokioMutex<()>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
            default_model: RwLock::new(None),
            total_tokens: AtomicU64::new(0),
            mutation_lock: TokioMutex::new(()),
        }
    }

    pub async fn load(&self, config: ModelLoadConfig) -> Result<ModelInfo, MullamaError> {
        let _mutation_guard = self.mutation_lock.lock().await;

        if self.models.contains_key(&config.alias) {
            return Err(MullamaError::OperationFailed(format!(
                "Model with alias '{}' already loaded",
                config.alias
            )));
        }

        let mut model_params = ModelParams {
            n_gpu_layers: config.gpu_layers,
            ..ModelParams::default()
        };
        if let Some(mmap) = config.use_mmap {
            model_params.use_mmap = mmap;
        }
        model_params.use_mlock = config.use_mlock;
        if let Some(ref mode) = config.split_mode {
            model_params.split_mode = match mode.to_lowercase().as_str() {
                "layer" => crate::sys::llama_split_mode::LLAMA_SPLIT_MODE_LAYER,
                "row" => crate::sys::llama_split_mode::LLAMA_SPLIT_MODE_ROW,
                _ => crate::sys::llama_split_mode::LLAMA_SPLIT_MODE_NONE,
            };
        }

        let path = config.path.clone();
        let info_path = config.path.clone();
        let context_size = config.context_size;
        let threads = config.threads;
        let gpu_layers = config.gpu_layers;
        let model_config = config.model_config.clone().unwrap_or_default();
        let context_pool_size = config.context_pool_size;
        let quantization = detect_quantization_from_path(&path);
        let flash_attn = config.flash_attn;
        let cache_type_k = config.cache_type_k.clone();
        let cache_type_v = config.cache_type_v.clone();
        let rope_freq_base = config.rope_freq_base;
        let rope_freq_scale = config.rope_freq_scale;
        let n_batch = config.n_batch;
        let defrag_thold = config.defrag_thold;

        let load_result = tokio::task::spawn_blocking(move || -> Result<LoadedCore, MullamaError> {
            let model = Arc::new(Model::load_with_params(&path, model_params)?);

            let mut ctx_params = ContextParams {
                n_ctx: context_size,
                n_threads: threads,
                n_threads_batch: threads,
                ..ContextParams::default()
            };
            ctx_params.flash_attn_type = if flash_attn {
                crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
            } else {
                crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED
            };
            if let Some(ref k) = cache_type_k {
                if let Some(kt) = crate::context::KvCacheType::from_str(k) {
                    ctx_params.type_k = kt;
                }
            }
            if let Some(ref v) = cache_type_v {
                if let Some(vt) = crate::context::KvCacheType::from_str(v) {
                    ctx_params.type_v = vt;
                }
            }
            if let Some(base) = rope_freq_base {
                ctx_params.rope_freq_base = base;
            }
            if let Some(scale) = rope_freq_scale {
                ctx_params.rope_freq_scale = scale;
            }
            if let Some(batch) = n_batch {
                ctx_params.n_batch = batch;
            }
            if let Some(thold) = defrag_thold {
                ctx_params.defrag_thold = thold;
            }

            let context = Context::new(model.clone(), ctx_params.clone())?;

            Ok(LoadedCore {
                model,
                context,
                ctx_params,
            })
        }).await.map_err(|e| MullamaError::OperationFailed(format!("Model loading task failed: {}", e)))??;

        let LoadedCore { model, context, ctx_params } = load_result;

        #[cfg(feature = "multimodal")]
        let mtmd_context = if let Some(ref mmproj_path) = config.mmproj_path {
            let mut mtmd_params = MtmdParams::default();
            mtmd_params.n_threads = config.threads;
            match MtmdContext::new(mmproj_path, &model, mtmd_params) {
                Ok(ctx) => {
                    eprintln!(
                        "  Multimodal: vision={}, audio={}",
                        ctx.supports_vision(),
                        ctx.supports_audio()
                    );
                    Some(ctx)
                }
                Err(e) => {
                    eprintln!("  Warning: Failed to load mmproj: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let info = ModelInfo {
            path: info_path.clone(),
            parameters: model.n_params(),
            context_size,
            vocab_size: model.n_vocab() as u32,
            gpu_layers,
            quantization,
        };

        #[cfg(feature = "multimodal")]
        let loaded = Arc::new(LoadedModel::new(
            config.alias.clone(),
            model,
            context,
            info.clone(),
            mtmd_context,
            ctx_params,
            model_config,
            context_pool_size,
        )?);

        #[cfg(not(feature = "multimodal"))]
        let loaded = Arc::new(LoadedModel::new(
            config.alias.clone(),
            model,
            context,
            info.clone(),
            ctx_params,
            model_config,
            context_pool_size,
        )?);

        self.models.insert(config.alias.clone(), loaded);

        {
            let mut default = self.default_model.write();
            if default.is_none() {
                *default = Some(config.alias);
            }
        }

        Ok(info)
    }

    pub async fn unload(&self, alias: &str) -> Result<(), MullamaError> {
        let _mutation_guard = self.mutation_lock.lock().await;

        if let Some(model_ref) = self.models.get(alias) {
            if model_ref.active_count() > 0 {
                return Err(MullamaError::OperationFailed(format!(
                    "Model '{}' has {} active requests",
                    alias,
                    model_ref.active_count()
                )));
            }
        }

        if self.models.remove(alias).is_none() {
            return Err(MullamaError::OperationFailed(format!(
                "Model '{}' not found",
                alias
            )));
        }

        {
            let mut default = self.default_model.write();
            if default.as_deref() == Some(alias) {
                *default = self.models.iter().next().map(|r| r.key().clone());
            }
        }

        Ok(())
    }

    pub async fn get(&self, alias: Option<&str>) -> Result<Arc<LoadedModel>, MullamaError> {
        let key = match alias {
            Some(a) => a.to_string(),
            None => {
                let default = self.default_model.read();
                default.clone().ok_or_else(|| {
                    MullamaError::OperationFailed("No default model set".to_string())
                })?
            }
        };

        self.models
            .get(&key)
            .map(|r| r.value().clone())
            .ok_or_else(|| MullamaError::OperationFailed(format!("Model '{}' not found", key)))
    }

    pub async fn set_default(&self, alias: &str) -> Result<(), MullamaError> {
        let _mutation_guard = self.mutation_lock.lock().await;

        if !self.models.contains_key(alias) {
            return Err(MullamaError::OperationFailed(format!(
                "Model '{}' not found",
                alias
            )));
        }

        let mut default = self.default_model.write();
        *default = Some(alias.to_string());
        Ok(())
    }

    pub fn default_alias(&self) -> Option<String> {
        self.default_model.read().clone()
    }

    pub fn list(&self) -> Vec<(String, ModelInfo, bool, u32)> {
        let default = self.default_model.read();

        self.models
            .iter()
            .map(|entry| {
                let alias = entry.key().clone();
                let model = entry.value();
                (
                    alias.clone(),
                    model.info.clone(),
                    default.as_deref() == Some(alias.as_str()),
                    model.active_count(),
                )
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.models.len()
    }

    pub fn add_tokens(&self, count: u64) {
        self.total_tokens.fetch_add(count, Ordering::Relaxed);
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    pub fn has_models(&self) -> bool {
        !self.models.is_empty()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}
