use thiserror::Error;

#[derive(Error, Debug)]
pub enum MullamaError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    
    #[error("Failed to create context: {0}")]
    ContextError(String),
    
    #[error("Tokenization failed: {0}")]
    TokenizationError(String),
    
    #[error("Generation failed: {0}")]
    GenerationError(String),
    
    #[error("Session error: {0}")]
    SessionError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("FFI error: {0}")]
    FfiError(String),
}