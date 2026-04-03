use axum::{
    http::{header, Uri, StatusCode},
    response::IntoResponse,
};

/// Redirect /ui to /ui/
pub(super) async fn ui_redirect() -> impl IntoResponse {
    (
        StatusCode::TEMPORARY_REDIRECT,
        [(header::LOCATION, "/ui/")],
        "",
    )
}

/// Serve embedded UI assets
pub(super) async fn serve_ui_handler(uri: Uri) -> impl IntoResponse {
    crate::daemon::ui::serve_ui(uri).await
}
