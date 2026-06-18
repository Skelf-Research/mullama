use super::super::Daemon;
use crate::daemon::protocol::{
    EmbeddingData, EmbeddingInput, EmbeddingsResponse, ErrorCode, Response, Usage,
};
use crate::embedding::{EmbeddingConfig, EmbeddingGenerator};

impl Daemon {
    pub async fn handle_embeddings(
        &self,
        model: Option<String>,
        input: EmbeddingInput,
    ) -> Response {
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let config = EmbeddingConfig::default();
        let mut generator = match EmbeddingGenerator::new(loaded.model.clone(), config) {
            Ok(g) => g,
            Err(e) => {
                return Response::error(
                    ErrorCode::Internal,
                    format!("Failed to create embedding generator: {}", e),
                )
            }
        };

        let texts: Vec<String> = match &input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Multiple(texts) => texts.clone(),
        };

        let model_clone = loaded.model.clone();
        let texts_clone = texts.clone();
        let embed_result = tokio::task::block_in_place(|| {
            let mut total_tokens = 0usize;
            for text in &texts_clone {
                if let Ok(tokens) = model_clone.tokenize(text, true, false) {
                    total_tokens += tokens.len();
                }
            }

            let text_refs: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
            let embeddings = generator.embed_batch(&text_refs)?;

            Ok::<_, crate::MullamaError>((embeddings, total_tokens))
        });

        match embed_result {
            Ok((embeddings, total_tokens)) => {
                let data: Vec<EmbeddingData> = embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(i, embedding)| EmbeddingData {
                        object: "embedding".to_string(),
                        embedding,
                        index: i as u32,
                    })
                    .collect();

                Response::Embeddings(EmbeddingsResponse {
                    object: "list".to_string(),
                    data,
                    model: loaded.alias.clone(),
                    usage: Usage {
                        prompt_tokens: total_tokens as u32,
                        completion_tokens: 0,
                        total_tokens: total_tokens as u32,
                    },
                })
            }
            Err(e) => Response::error(
                ErrorCode::GenerationFailed,
                format!("Failed to generate embeddings: {}", e),
            ),
        }
    }

    pub(crate) async fn handle_tokenize(&self, model: Option<String>, text: &str) -> Response {
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let model_clone = loaded.model.clone();
        let text_owned = text.to_string();
        tokio::task::block_in_place(move || {
            match model_clone.tokenize(&text_owned, false, false) {
                Ok(tokens) => {
                    let count = tokens.len();
                    Response::Tokens { tokens, count }
                }
                Err(e) => Response::error(ErrorCode::Internal, e.to_string()),
            }
        })
    }
}
