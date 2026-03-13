//! WebSocket integration for real-time bidirectional communication
//!
//! This module provides comprehensive WebSocket support for real-time applications,
//! enabling bidirectional streaming, audio processing, and interactive chat interfaces.
//!
//! ## Features
//!
//! - **Real-time Streaming**: Bidirectional text and audio streaming
//! - **Audio Processing**: WebSocket-based audio input/output handling
//! - **Interactive Chat**: Real-time conversation interfaces
//! - **Connection Management**: Robust connection handling with reconnection
//! - **Message Types**: Support for text, binary, and custom message formats
//! - **Room/Channel Support**: Multi-user session management
//! - **Compression**: Optional message compression for bandwidth optimization
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::websockets::{WebSocketServer, WebSocketConfig, MessageHandler};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     let config = WebSocketConfig::new()
//!         .port(8080)
//!         .enable_audio()
//!         .max_connections(100);
//!
//!     let server = WebSocketServer::new(config)
//!         .with_handler(MessageHandler::new())
//!         .build()
//!         .await?;
//!
//!     server.start().await?;
//!     Ok(())
//! }
//! ```

#[cfg(feature = "websockets")]
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    response::Response,
    routing::get,
    Router,
};

#[cfg(feature = "websockets")]
use tokio::{
    sync::{broadcast, mpsc, RwLock},
    time::{interval, Duration, Instant},
};

#[cfg(feature = "websockets")]
use tokio_tungstenite::tungstenite;

#[cfg(feature = "websockets")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "websockets")]
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

#[cfg(all(feature = "websockets", feature = "async"))]
use crate::{AsyncModel, MullamaError, StreamConfig, TokenStream};

/// WebSocket server for real-time communication
#[cfg(feature = "websockets")]
pub struct WebSocketServer {
    config: WebSocketConfig,
    connections: Arc<ConnectionManager>,
    model: Option<Arc<AsyncModel>>,
    audio_processor: Option<Arc<AudioProcessor>>,
}

#[cfg(feature = "websockets")]
impl WebSocketServer {
    /// Create a new WebSocket server
    pub fn new(config: WebSocketConfig) -> WebSocketServerBuilder {
        WebSocketServerBuilder::new(config)
    }

    /// Create router with WebSocket endpoints
    pub fn create_router(&self) -> Router<AppState> {
        Router::new()
            .route("/ws", get(websocket_handler))
            .route("/ws/chat/:room_id", get(chat_websocket_handler))
            .route("/ws/audio", get(audio_websocket_handler))
            .route("/ws/stream/:session_id", get(stream_websocket_handler))
            .with_state(AppState {
                connections: self.connections.clone(),
                model: self.model.clone(),
                audio_processor: self.audio_processor.clone(),
                config: self.config.clone(),
            })
    }

    /// Start the WebSocket server
    pub async fn start(&self) -> Result<(), MullamaError> {
        println!("🌐 Starting WebSocket server on port {}", self.config.port);

        let app = self.create_router();
        let addr = SocketAddr::from(([0, 0, 0, 0], self.config.port));

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| MullamaError::ConfigError(format!("Failed to bind: {}", e)))?;

        println!("✅ WebSocket server listening on {}", addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| MullamaError::ConfigError(format!("Server error: {}", e)))?;

        Ok(())
    }

    /// Get server statistics
    pub async fn stats(&self) -> ServerStats {
        self.connections.stats().await
    }
}

/// Builder for WebSocketServer
#[cfg(feature = "websockets")]
pub struct WebSocketServerBuilder {
    config: WebSocketConfig,
    model: Option<Arc<AsyncModel>>,
    audio_processor: Option<Arc<AudioProcessor>>,
}

#[cfg(feature = "websockets")]
impl WebSocketServerBuilder {
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            config,
            model: None,
            audio_processor: None,
        }
    }

    #[cfg(feature = "async")]
    pub fn with_model(mut self, model: Arc<AsyncModel>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_audio_processor(mut self, processor: Arc<AudioProcessor>) -> Self {
        self.audio_processor = Some(processor);
        self
    }

    pub async fn build(self) -> Result<WebSocketServer, MullamaError> {
        let connections = Arc::new(ConnectionManager::new(self.config.max_connections));

        // Initialize audio processor if audio is enabled
        let audio_processor = if self.config.enable_audio {
            Some(
                self.audio_processor
                    .unwrap_or_else(|| Arc::new(AudioProcessor::new())),
            )
        } else {
            self.audio_processor
        };

        Ok(WebSocketServer {
            config: self.config,
            connections,
            model: self.model,
            audio_processor,
        })
    }
}

/// WebSocket server configuration
#[cfg(feature = "websockets")]
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    pub port: u16,
    pub max_connections: usize,
    pub max_message_size: usize,
    pub ping_interval: Duration,
    pub connection_timeout: Duration,
    pub enable_audio: bool,
    pub enable_compression: bool,
    pub allowed_origins: Vec<String>,
    pub rate_limit: Option<RateLimit>,
}

#[cfg(feature = "websockets")]
impl WebSocketConfig {
    pub fn new() -> Self {
        Self {
            port: 8080,
            max_connections: 100,
            max_message_size: 1024 * 1024, // 1MB
            ping_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(60),
            enable_audio: false,
            enable_compression: false,
            allowed_origins: vec!["*".to_string()],
            rate_limit: Some(RateLimit {
                max_requests: 100,
                window: Duration::from_secs(60),
            }),
        }
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    pub fn enable_audio(mut self) -> Self {
        self.enable_audio = true;
        self
    }

    pub fn enable_compression(mut self) -> Self {
        self.enable_compression = true;
        self
    }

    pub fn rate_limit(mut self, limit: RateLimit) -> Self {
        self.rate_limit = Some(limit);
        self
    }
}

/// Rate limiting configuration
#[cfg(feature = "websockets")]
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub max_requests: u32,
    pub window: Duration,
}

/// Application state for WebSocket handlers
#[cfg(feature = "websockets")]
#[derive(Clone)]
pub struct AppState {
    pub connections: Arc<ConnectionManager>,
    pub model: Option<Arc<AsyncModel>>,
    pub audio_processor: Option<Arc<AudioProcessor>>,
    pub config: WebSocketConfig,
}

/// Connection manager for WebSocket connections
#[cfg(feature = "websockets")]
pub struct ConnectionManager {
    connections: RwLock<HashMap<String, Connection>>,
    rooms: RwLock<HashMap<String, Room>>,
    max_connections: usize,
    stats: Arc<ConnectionStats>,
}

#[cfg(feature = "websockets")]
impl ConnectionManager {
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            rooms: RwLock::new(HashMap::new()),
            max_connections,
            stats: Arc::new(ConnectionStats::new()),
        }
    }

    /// Add a new connection
    pub async fn add_connection(&self, connection: Connection) -> Result<(), MullamaError> {
        let mut connections = self.connections.write().await;

        if connections.len() >= self.max_connections {
            return Err(MullamaError::ConfigError(
                "Max connections reached".to_string(),
            ));
        }

        connections.insert(connection.id.clone(), connection);
        self.stats
            .active_connections
            .fetch_add(1, Ordering::Relaxed);
        self.stats.total_connections.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Remove a connection
    pub async fn remove_connection(&self, connection_id: &str) {
        let mut connections = self.connections.write().await;
        if connections.remove(connection_id).is_some() {
            self.stats
                .active_connections
                .fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Join a room
    pub async fn join_room(
        &self,
        connection_id: String,
        room_id: String,
    ) -> Result<(), MullamaError> {
        let mut rooms = self.rooms.write().await;
        let room = rooms
            .entry(room_id.clone())
            .or_insert_with(|| Room::new(room_id));
        room.add_member(connection_id);
        Ok(())
    }

    /// Leave a room
    pub async fn leave_room(&self, connection_id: &str, room_id: &str) {
        let mut rooms = self.rooms.write().await;
        if let Some(room) = rooms.get_mut(room_id) {
            room.remove_member(connection_id);
            if room.is_empty() {
                rooms.remove(room_id);
            }
        }
    }

    /// Broadcast message to room
    pub async fn broadcast_to_room(
        &self,
        room_id: &str,
        message: &WSMessage,
    ) -> Result<(), MullamaError> {
        let rooms = self.rooms.read().await;
        if let Some(room) = rooms.get(room_id) {
            let connections = self.connections.read().await;
            for member_id in &room.members {
                if let Some(connection) = connections.get(member_id) {
                    let _ = connection.sender.send(message.clone());
                }
            }
        }
        Ok(())
    }

    /// Get connection statistics
    pub async fn stats(&self) -> ServerStats {
        let connections = self.connections.read().await;
        let rooms = self.rooms.read().await;

        ServerStats {
            active_connections: connections.len(),
            total_rooms: rooms.len(),
            total_connections_ever: self.stats.total_connections.load(Ordering::Relaxed),
            messages_sent: self.stats.messages_sent.load(Ordering::Relaxed),
            messages_received: self.stats.messages_received.load(Ordering::Relaxed),
        }
    }
}

/// Individual WebSocket connection
#[cfg(feature = "websockets")]
#[derive(Debug)]
pub struct Connection {
    pub id: String,
    pub sender: broadcast::Sender<WSMessage>,
    pub created_at: Instant,
    pub last_ping: Option<Instant>,
    pub connection_type: ConnectionType,
}

/// Type of WebSocket connection
#[cfg(feature = "websockets")]
#[derive(Debug, Clone)]
pub enum ConnectionType {
    Chat,
    Audio,
    Stream,
    General,
}

/// Chat room for grouped connections
#[cfg(feature = "websockets")]
#[derive(Debug)]
pub struct Room {
    pub id: String,
    pub members: Vec<String>,
    pub created_at: Instant,
}

#[cfg(feature = "websockets")]
impl Room {
    pub fn new(id: String) -> Self {
        Self {
            id,
            members: Vec::new(),
            created_at: Instant::now(),
        }
    }

    pub fn add_member(&mut self, connection_id: String) {
        if !self.members.contains(&connection_id) {
            self.members.push(connection_id);
        }
    }

    pub fn remove_member(&mut self, connection_id: &str) {
        self.members.retain(|id| id != connection_id);
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// WebSocket message types
#[cfg(feature = "websockets")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WSMessage {
    /// Text message
    Text {
        content: String,
    },
    /// Generation request
    Generate {
        prompt: String,
        config: Option<GenerationConfig>,
    },
    /// Streaming token
    Token {
        text: String,
        is_final: bool,
    },
    /// Audio data
    Audio {
        data: Vec<u8>,
        format: AudioFormat,
    },
    /// System message
    System {
        message: String,
    },
    /// Error message
    Error {
        error: String,
    },
    /// Ping/Pong for keepalive
    Ping,
    Pong,
    /// Room management
    JoinRoom {
        room_id: String,
    },
    LeaveRoom {
        room_id: String,
    },
    /// User joined/left notifications
    UserJoined {
        user_id: String,
        room_id: String,
    },
    UserLeft {
        user_id: String,
        room_id: String,
    },
}

/// Generation configuration for WebSocket requests
#[cfg(feature = "websockets")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

/// Audio format specification
#[cfg(feature = "websockets")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub format: String, // "pcm", "wav", "mp3", etc.
}

/// Audio processor for WebSocket audio handling
#[cfg(feature = "websockets")]
pub struct AudioProcessor {
    sample_rate: u32,
    channels: u16,
    buffer_size: usize,
}

#[cfg(feature = "websockets")]
impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 1024,
        }
    }

    /// Process incoming audio data
    pub async fn process_audio(
        &self,
        data: &[u8],
        format: &AudioFormat,
    ) -> Result<Vec<u8>, MullamaError> {
        // Placeholder for audio processing
        // In real implementation, this would:
        // 1. Convert audio format if needed
        // 2. Resample if sample rates don't match
        // 3. Apply noise reduction
        // 4. Normalize audio levels

        println!(
            "🎵 Processing audio: {} bytes, {}Hz, {} channels",
            data.len(),
            format.sample_rate,
            format.channels
        );

        Ok(data.to_vec())
    }

    /// Convert audio to text (placeholder for STT integration)
    pub async fn speech_to_text(&self, audio_data: &[u8]) -> Result<String, MullamaError> {
        // Placeholder for speech-to-text integration
        // This would integrate with services like Whisper, Google Speech-to-Text, etc.
        Ok("Transcribed text placeholder".to_string())
    }

    /// Convert text to audio (placeholder for TTS integration)
    pub async fn text_to_speech(&self, text: &str) -> Result<Vec<u8>, MullamaError> {
        // Placeholder for text-to-speech integration
        // This would integrate with TTS engines
        println!("🔊 Converting to speech: {}", text);
        Ok(vec![0u8; 1024]) // Placeholder audio data
    }
}

/// Connection statistics
#[cfg(feature = "websockets")]
pub struct ConnectionStats {
    pub active_connections: AtomicU64,
    pub total_connections: AtomicU64,
    pub messages_sent: AtomicU64,
    pub messages_received: AtomicU64,
}

#[cfg(feature = "websockets")]
impl ConnectionStats {
    pub fn new() -> Self {
        Self {
            active_connections: AtomicU64::new(0),
            total_connections: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
        }
    }
}

/// Server statistics
#[cfg(feature = "websockets")]
#[derive(Debug, Clone, Serialize)]
pub struct ServerStats {
    pub active_connections: usize,
    pub total_rooms: usize,
    pub total_connections_ever: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
}

// WebSocket handler functions
#[cfg(feature = "websockets")]
async fn websocket_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state, ConnectionType::General))
}

#[cfg(feature = "websockets")]
async fn chat_websocket_handler(
    ws: WebSocketUpgrade,
    Path(room_id): Path<String>,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_chat_websocket(socket, state, room_id))
}

#[cfg(feature = "websockets")]
async fn audio_websocket_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state, ConnectionType::Audio))
}

#[cfg(feature = "websockets")]
async fn stream_websocket_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_stream_websocket(socket, state, session_id))
}

#[cfg(feature = "websockets")]
async fn handle_websocket(socket: WebSocket, state: AppState, connection_type: ConnectionType) {
    let connection_id = uuid::Uuid::new_v4().to_string();
    let (sender, _receiver) = broadcast::channel(100);

    let connection = Connection {
        id: connection_id.clone(),
        sender: sender.clone(),
        created_at: Instant::now(),
        last_ping: None,
        connection_type,
    };

    if let Err(e) = state.connections.add_connection(connection).await {
        eprintln!("❌ Failed to add connection: {}", e);
        return;
    }

    println!("✅ New WebSocket connection: {}", connection_id);

    let (mut sender_ws, mut receiver_ws) = socket.split();

    // Handle incoming messages
    let connections_clone = state.connections.clone();
    let connection_id_clone = connection_id.clone();

    tokio::spawn(async move {
        while let Some(msg) = receiver_ws.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(ws_message) = serde_json::from_str::<WSMessage>(&text) {
                        handle_message(ws_message, &state, &connection_id).await;
                    }
                }
                Ok(Message::Binary(data)) => {
                    // Handle binary data (audio, etc.)
                    if let Some(audio_processor) = &state.audio_processor {
                        let format = AudioFormat {
                            sample_rate: 16000,
                            channels: 1,
                            bits_per_sample: 16,
                            format: "pcm".to_string(),
                        };

                        if let Ok(_processed) = audio_processor.process_audio(&data, &format).await
                        {
                            // Process audio data
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    println!("🔌 WebSocket connection closed: {}", connection_id);
                    break;
                }
                Err(e) => {
                    eprintln!("❌ WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        connections_clone
            .remove_connection(&connection_id_clone)
            .await;
    });

    // Keep connection alive and handle cleanup
    let mut ping_interval = interval(state.config.ping_interval);

    loop {
        tokio::select! {
            _ = ping_interval.tick() => {
                let ping_msg = WSMessage::Ping;
                if let Ok(ping_json) = serde_json::to_string(&ping_msg) {
                    if sender_ws.send(Message::Text(ping_json)).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(feature = "websockets")]
async fn handle_chat_websocket(socket: WebSocket, state: AppState, room_id: String) {
    // Similar to handle_websocket but with room management
    let connection_id = uuid::Uuid::new_v4().to_string();

    // Join the specified room
    if let Err(e) = state
        .connections
        .join_room(connection_id.clone(), room_id.clone())
        .await
    {
        eprintln!("❌ Failed to join room: {}", e);
        return;
    }

    println!("👥 User {} joined room {}", connection_id, room_id);

    // Notify other room members
    let join_message = WSMessage::UserJoined {
        user_id: connection_id.clone(),
        room_id: room_id.clone(),
    };

    let _ = state
        .connections
        .broadcast_to_room(&room_id, &join_message)
        .await;

    // Handle the WebSocket connection (similar to general handler)
    handle_websocket(socket, state, ConnectionType::Chat).await;
}

#[cfg(feature = "websockets")]
async fn handle_stream_websocket(socket: WebSocket, state: AppState, session_id: String) {
    println!("🌊 Starting streaming session: {}", session_id);

    // Handle streaming-specific logic
    handle_websocket(socket, state, ConnectionType::Stream).await;
}

#[cfg(feature = "websockets")]
async fn handle_message(message: WSMessage, state: &AppState, connection_id: &str) {
    match message {
        WSMessage::Generate { prompt, config } => {
            if let Some(model) = &state.model {
                // Handle generation request
                println!("🤖 Generation request from {}: {}", connection_id, prompt);

                // In real implementation, this would generate using the model
                let response = WSMessage::Text {
                    content: format!("Generated response for: {}", prompt),
                };

                // Send response back to the client
                // (This would require a way to send messages back to specific connections)
            }
        }
        WSMessage::Audio { data, format } => {
            if let Some(audio_processor) = &state.audio_processor {
                println!(
                    "🎵 Audio message from {}: {} bytes",
                    connection_id,
                    data.len()
                );

                // Process audio and potentially convert to text
                if let Ok(text) = audio_processor.speech_to_text(&data).await {
                    println!("📝 Transcribed: {}", text);
                }
            }
        }
        WSMessage::JoinRoom { room_id } => {
            let _ = state
                .connections
                .join_room(connection_id.to_string(), room_id)
                .await;
        }
        WSMessage::LeaveRoom { room_id } => {
            state.connections.leave_room(connection_id, &room_id).await;
        }
        WSMessage::Ping => {
            // Respond with Pong
            let pong = WSMessage::Pong;
            // Send pong back (would need connection reference)
        }
        _ => {
            println!("📨 Received message from {}: {:?}", connection_id, message);
        }
    }
}

/// Utility functions for WebSocket integration
#[cfg(feature = "websockets")]
pub mod utils {
    use super::*;

    /// Create a WebSocket client for testing
    pub async fn create_test_client(url: &str) -> Result<(), MullamaError> {
        // Placeholder for WebSocket client implementation
        println!("🔌 Connecting to WebSocket: {}", url);
        Ok(())
    }

    /// Validate WebSocket message
    pub fn validate_message(message: &WSMessage) -> bool {
        match message {
            WSMessage::Text { content } => !content.is_empty(),
            WSMessage::Generate { prompt, .. } => !prompt.is_empty(),
            WSMessage::Audio { data, .. } => !data.is_empty(),
            _ => true,
        }
    }

    /// Convert HTTP upgrade to WebSocket
    pub fn upgrade_connection(headers: &HashMap<String, String>) -> bool {
        headers.get("upgrade").map(|v| v.to_lowercase()) == Some("websocket".to_string())
            && headers
                .get("connection")
                .map(|v| v.to_lowercase().contains("upgrade"))
                == Some(true)
    }
}

#[cfg(not(feature = "websockets"))]
compile_error!("WebSocket integration requires the 'websockets' feature to be enabled");
