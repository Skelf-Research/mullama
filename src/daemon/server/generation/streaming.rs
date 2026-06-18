use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;

use super::super::super::models::ModelManager;
use super::super::super::protocol::{generate_completion_id, StreamChunk};
use super::super::Daemon;
use super::common::GenerationResult;
use crate::MullamaError;

/// Pre-computed state for a streaming generation task.
pub(super) struct StreamingSetup {
    pub tx: mpsc::Sender<StreamChunk>,
    pub request_id_arc: Arc<str>,
    pub cancel_flag: Arc<AtomicBool>,
    models_ref: Arc<ModelManager>,
    cancellations: Arc<DashMap<String, Arc<AtomicBool>>>,
    active_requests: Arc<AtomicU32>,
    request_id_for_cleanup: String,
    pub stop_sequences: Vec<String>,
    pub max_stop_len: usize,
}

impl StreamingSetup {
    pub fn finish(self, result: &Result<GenerationResult, MullamaError>) {
        if let Ok(r) = result {
            self.models_ref.add_tokens(r.completion_tokens as u64);
        }
        self.cancellations.remove(&self.request_id_for_cleanup);
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Daemon {
    pub(super) fn prepare_streaming(
        &self,
        stop_sequences: Vec<String>,
    ) -> (StreamingSetup, mpsc::Receiver<StreamChunk>, String) {
        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let setup = StreamingSetup {
            tx,
            request_id_arc,
            cancel_flag,
            models_ref: self.models.clone(),
            cancellations: Arc::clone(&self.cancellations),
            active_requests: Arc::clone(&self.active_requests),
            request_id_for_cleanup: request_id.clone(),
            stop_sequences,
            max_stop_len,
        };

        (setup, rx, request_id)
    }
}
