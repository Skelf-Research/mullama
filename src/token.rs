/// Token identifier - matches the C type from llama.cpp
pub type TokenId = i32;

/// Represents a token with its metadata
#[derive(Debug, Clone)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
}