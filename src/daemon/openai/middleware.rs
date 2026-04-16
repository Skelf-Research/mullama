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
pub struct HttpAuthState {
    pub api_key: Arc<str>,
}

#[derive(Clone)]
pub struct HttpRateLimitState {
    pub limit: u64,
    pub second: Arc<AtomicU64>,
    pub count: Arc<AtomicU64>,
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        let mut diff = 0u8;
        diff |= a.len().wrapping_sub(b.len()) as u8;
        // still walk the shorter to avoid length-based timing
        for i in 0..a.len().min(b.len()) {
            diff |= a[i] ^ b[i];
        }
        diff == 0
    } else {
        let mut diff = 0u8;
        for i in 0..a.len() {
            diff |= a[i] ^ b[i];
        }
        diff == 0
    }
}

fn header_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(token) = value.strip_prefix("Bearer ") {
            return Some(token.trim());
        }
    }

    headers.get("x-api-key").and_then(|v| v.to_str().ok())
}

pub async fn require_api_key(
    State(auth): State<HttpAuthState>,
    headers: HeaderMap,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if let Some(key) = header_api_key(&headers) {
        if constant_time_eq(key.as_bytes(), auth.api_key.as_bytes()) {
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

pub async fn enforce_rate_limit(
    State(rate): State<HttpRateLimitState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let now = super::types::unix_timestamp_secs();
    let seen_second = rate.second.load(Ordering::Relaxed);

    if seen_second != now {
        if rate
            .second
            .compare_exchange(seen_second, now, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            rate.count.store(1, Ordering::Relaxed);
        } else {
            rate.count.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        rate.count.fetch_add(1, Ordering::Relaxed);
    }

    let count = rate.count.load(Ordering::Relaxed);
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
