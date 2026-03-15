//! Embedded Web UI
//!
//! This module embeds and serves the Vue.js web UI for Mullama.
//! The UI is built separately and embedded at compile time when the `embedded-ui` feature is enabled.

use axum::{
    body::Body,
    http::{header, Response, StatusCode, Uri},
    response::IntoResponse,
};

#[cfg(feature = "embedded-ui")]
use include_dir::{include_dir, Dir};

/// Embedded UI assets directory (only available with embedded-ui feature)
#[cfg(feature = "embedded-ui")]
static UI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui/dist");

/// Serve embedded UI assets
pub async fn serve_ui(uri: Uri) -> impl IntoResponse {
    #[cfg(feature = "embedded-ui")]
    {
        let path = uri.path().trim_start_matches("/ui/");
        let path = if path.is_empty() { "index.html" } else { path };

        // Try to find the file in the embedded directory
        match UI_DIR.get_file(path) {
            Some(file) => {
                let mime_type = mime_guess::from_path(path)
                    .first_or_octet_stream()
                    .to_string();

                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, mime_type)
                    .header(header::CACHE_CONTROL, "public, max-age=31536000")
                    .body(Body::from(file.contents().to_vec()))
                    .unwrap()
            }
            None => {
                // For SPA routing, serve index.html for non-asset paths
                if !path.contains('.') {
                    if let Some(index) = UI_DIR.get_file("index.html") {
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
                            .body(Body::from(index.contents().to_vec()))
                            .unwrap();
                    }
                }

                // Return 404 for missing files
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header(header::CONTENT_TYPE, "text/plain")
                    .body(Body::from("Not Found"))
                    .unwrap()
            }
        }
    }

    #[cfg(not(feature = "embedded-ui"))]
    {
        let _ = uri; // Suppress unused warning
        // Return a helpful message when UI is not embedded
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(Body::from(
                r#"<!DOCTYPE html>
<html>
<head>
    <title>Mullama - Web UI Not Available</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
        pre { background: #f0f0f0; padding: 15px; border-radius: 8px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Web UI Not Available</h1>
    <p>The Mullama web UI was not embedded in this build.</p>
    <p>To build with the embedded web UI:</p>
    <pre>cd ui && npm install && npm run build
cargo build --release --features daemon,embedded-ui</pre>
    <p>You can still use the CLI and API:</p>
    <ul>
        <li><code>mullama run llama3.2:1b "Hello"</code> - Run a model</li>
        <li><code>mullama chat</code> - Interactive TUI chat</li>
        <li><code>curl http://localhost:8080/v1/models</code> - List models via API</li>
    </ul>
</body>
</html>"#,
            ))
            .unwrap()
    }
}

/// Check if embedded UI is available
pub fn ui_available() -> bool {
    #[cfg(feature = "embedded-ui")]
    {
        UI_DIR.get_file("index.html").is_some()
    }
    #[cfg(not(feature = "embedded-ui"))]
    {
        false
    }
}

/// Get UI index HTML (for serving at root if desired)
#[allow(dead_code)]
pub fn get_index_html() -> Option<&'static [u8]> {
    #[cfg(feature = "embedded-ui")]
    {
        UI_DIR.get_file("index.html").map(|f| f.contents())
    }
    #[cfg(not(feature = "embedded-ui"))]
    {
        None
    }
}
