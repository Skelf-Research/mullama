use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::config::ModelConfig;
use super::pool::ContextPool;
use super::stats::ModelStats;
use crate::daemon::protocol::ModelInfo;
use crate::{Context, ContextParams, Model, MullamaError};

#[cfg(feature = "multimodal")]
use crate::MtmdContext;

/// A loaded model instance with its context pool
pub struct LoadedModel {
    pub alias: String,
    pub model: Arc<Model>,
    pool: ContextPool,
    pub info: ModelInfo,
    pub active_requests: AtomicU32,
    pub config: ModelConfig,
    pub stats: ModelStats,
    #[cfg(feature = "multimodal")]
    pub mtmd_context: Option<tokio::sync::RwLock<MtmdContext>>,
}

impl LoadedModel {
    #[cfg(feature = "multimodal")]
    pub fn new(
        alias: String,
        model: Arc<Model>,
        context: Context,
        info: ModelInfo,
        mtmd_context: Option<MtmdContext>,
        ctx_params: ContextParams,
        config: ModelConfig,
        context_pool_size: usize,
    ) -> Result<Self, MullamaError> {
        let pool = ContextPool::new(model.clone(), context, ctx_params, context_pool_size)?;

        Ok(Self {
            alias,
            model,
            pool,
            info,
            active_requests: AtomicU32::new(0),
            config,
            stats: ModelStats::new(),
            mtmd_context: mtmd_context.map(tokio::sync::RwLock::new),
        })
    }

    #[cfg(not(feature = "multimodal"))]
    pub fn new(
        alias: String,
        model: Arc<Model>,
        context: Context,
        info: ModelInfo,
        ctx_params: ContextParams,
        config: ModelConfig,
        context_pool_size: usize,
    ) -> Result<Self, MullamaError> {
        let pool = ContextPool::new(model.clone(), context, ctx_params, context_pool_size)?;

        Ok(Self {
            alias,
            model,
            pool,
            info,
            active_requests: AtomicU32::new(0),
            config,
            stats: ModelStats::new(),
        })
    }

    pub async fn acquire_context(&self) -> tokio::sync::RwLockWriteGuard<'_, Context> {
        self.pool.acquire().await
    }

    pub async fn get_context(&self) -> tokio::sync::RwLockReadGuard<'_, Context> {
        self.pool.read().await
    }

    pub fn pool_size(&self) -> usize {
        self.pool.size()
    }

    #[cfg(feature = "multimodal")]
    pub fn has_multimodal(&self) -> bool {
        self.mtmd_context.is_some()
    }

    #[cfg(not(feature = "multimodal"))]
    pub fn has_multimodal(&self) -> bool {
        false
    }

    pub fn acquire(&self) {
        self.active_requests.fetch_add(1, Ordering::SeqCst);
    }

    pub fn release(&self) {
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn active_count(&self) -> u32 {
        self.active_requests.load(Ordering::SeqCst)
    }
}

/// Guard for tracking active requests
pub struct RequestGuard {
    model: Arc<LoadedModel>,
}

impl RequestGuard {
    pub fn new(model: Arc<LoadedModel>) -> Self {
        model.acquire();
        Self { model }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.model.release();
    }
}
