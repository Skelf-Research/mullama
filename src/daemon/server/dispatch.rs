use std::sync::atomic::Ordering;

use super::super::protocol::{ErrorCode, Request, Response};
use super::Daemon;

impl Daemon {
    /// Map transport-layer requests to daemon service methods.
    pub async fn handle_request(&self, request: Request) -> Response {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match request {
            Request::Ping => Response::Pong {
                uptime_secs: self.start_time.elapsed().as_secs(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Request::Status => self.handle_status().await,
            Request::ListModels => self.handle_list_models().await,

            Request::LoadModel(params) => self.handle_load_model(params).await,

            Request::UnloadModel { alias } => self.handle_unload_model(&alias).await,
            Request::SetDefaultModel { alias } => self.handle_set_default(&alias).await,

            Request::ChatCompletion(params) => self.handle_chat_completion(params).await,

            Request::Completion(params) => self.handle_completion(params).await,

            Request::Embeddings { model, input } => self.handle_embeddings(model, input).await,

            Request::Tokenize { model, text } => self.handle_tokenize(model, &text).await,

            Request::Cancel { request_id } => {
                if self.cancel_request(&request_id) {
                    Response::Cancelled { request_id }
                } else {
                    Response::error(
                        ErrorCode::InvalidRequest,
                        format!("No active request found with id '{}'", request_id),
                    )
                }
            }

            Request::Shutdown => {
                self.store.flush();
                self.shutdown.store(true, Ordering::SeqCst);
                Response::ShuttingDown
            }
        }
    }
}
