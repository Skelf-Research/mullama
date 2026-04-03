use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use super::types::{ErrorDetail, ErrorResponse};
use crate::daemon::protocol::ErrorCode;

pub struct ApiError {
    pub(super) message: String,
    pub(super) status: StatusCode,
}

impl ApiError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(super) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: StatusCode::BAD_REQUEST,
        }
    }

    pub(super) fn from_protocol_error(code: ErrorCode, message: impl Into<String>) -> Self {
        let message = message.into();
        let status = match code {
            ErrorCode::ModelNotFound => StatusCode::NOT_FOUND,
            ErrorCode::InvalidRequest => StatusCode::BAD_REQUEST,
            ErrorCode::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            ErrorCode::Timeout => StatusCode::GATEWAY_TIMEOUT,
            ErrorCode::Cancelled => StatusCode::CONFLICT,
            ErrorCode::ModelLoadFailed
            | ErrorCode::NoDefaultModel
            | ErrorCode::GenerationFailed
            | ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self { message, status }
    }

    pub(super) fn not_found(model: &str) -> Self {
        Self {
            message: format!("Model '{}' not found", model),
            status: StatusCode::NOT_FOUND,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message: self.message,
                error_type: "api_error".to_string(),
                code: None,
            },
        });
        (self.status, body).into_response()
    }
}
