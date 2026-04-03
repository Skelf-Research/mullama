use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{
    extract::State,
    http::{header::AUTHORIZATION, HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

use super::types::{ErrorDetail, ErrorResponse};

#[derive(Clone)]
pub(super) struct HttpAuthState {
    pub(super) api_key: Arc<str>,
}

#[derive(Clone)]
pub(super) struct HttpRateLimitState {
    pub(super) limit: u64,
    pub(super) second: Arc<AtomicU64>,
    pub(super) count: Arc<AtomicU64>,
}

fn header_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(token) = value.strip_prefix("Bearer ") {
            return Some(token.trim());
        }
    }

    headers.get("x-api-key").and_then(|v| v.to_str().ok())
}

pub(super) async fn require_api_key(
    State(auth): State<HttpAuthState>,
    headers: HeaderMap,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if let Some(key) = header_api_key(&headers) {
        if key == auth.api_key.as_ref() {
            return next.run(request).await;
        }
    }

    let body = Json(ErrorResponse {
        error: ErrorDetail {
            message: "Missing or invalid API key".to_string(),
            error_type: "authentication_error".to_string(),
            code: Some("invalid_api_key".to_string()),
        },
    });
    (StatusCode::UNAUTHORIZED, body).into_response()
}

pub(super) async fn enforce_rate_limit(
    State(rate): State<HttpRateLimitState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let now = super::types::unix_timestamp_secs();
    let seen_second = rate.second.load(Ordering::Relaxed);
    if seen_second != now
        && rate
            .second
            .compare_exchange(seen_second, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        rate.count.store(0, Ordering::Relaxed);
    }

    let count = rate.count.fetch_add(1, Ordering::Relaxed) + 1;
    if count > rate.limit {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Rate limit exceeded".to_string(),
                    error_type: "rate_limit_error".to_string(),
                    code: Some("rate_limited".to_string()),
                },
            }),
        )
            .into_response();
    }

    next.run(request).await
}
