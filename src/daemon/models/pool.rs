use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{RwLock as TokioRwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{Context, ContextParams, Model, MullamaError};

/// Round-robin context pool for a loaded model.
pub struct ContextPool {
    contexts: Vec<TokioRwLock<Context>>,
    next_context: AtomicUsize,
}

impl ContextPool {
    pub fn new(
        model: Arc<Model>,
        context: Context,
        ctx_params: ContextParams,
        context_pool_size: usize,
    ) -> Result<Self, MullamaError> {
        let context_pool_size = context_pool_size.max(1);
        let mut contexts = Vec::with_capacity(context_pool_size);
        contexts.push(TokioRwLock::new(context));

        for _ in 1..context_pool_size {
            let ctx = Context::new(model.clone(), ctx_params.clone())?;
            contexts.push(TokioRwLock::new(ctx));
        }

        Ok(Self {
            contexts,
            next_context: AtomicUsize::new(0),
        })
    }

    pub async fn acquire(&self) -> RwLockWriteGuard<'_, Context> {
        let idx = self.next_context.fetch_add(1, Ordering::Relaxed) % self.contexts.len();
        self.contexts[idx].write().await
    }

    /// Acquire a specific pool slot by index. Used for cross-turn KV reuse,
    /// where a session is pinned to one slot so its KV cache persists across
    /// requests (a round-robin acquire would land on a different slot each turn
    /// and wipe the benefit). The index is validated against the pool size.
    pub async fn acquire_at(&self, idx: usize) -> RwLockWriteGuard<'_, Context> {
        let idx = idx.min(self.contexts.len().saturating_sub(1));
        self.contexts[idx].write().await
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, Context> {
        let idx = self.next_context.load(Ordering::Relaxed) % self.contexts.len();
        self.contexts[idx].read().await
    }

    pub fn size(&self) -> usize {
        self.contexts.len()
    }
}
