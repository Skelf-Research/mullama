# WebSockets

Build real-time, bidirectional communication channels for streaming LLM inference, audio processing, and interactive chat applications.

!!! info "Feature Gate"
    This feature requires the `websockets` feature flag, which transitively enables `async`.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["websockets"] }
    ```

## Overview

Mullama's WebSocket integration provides:

- **WebSocketServer** for managing client connections
- **WebSocketConfig** with comprehensive tuning options
- **ConnectionManager** for client lifecycle and multi-user support
- **Typed message protocol** (Text, Binary, Audio, Custom)
- **Room/channel support** for organized communication
- **Message compression** for bandwidth optimization
- **Automatic reconnection** handling
- **Server statistics** and monitoring

---

## WebSocketServer

The `WebSocketServer` is the primary entry point for WebSocket-based communication.

=== "Node.js"

    ```javascript
    const { WebSocketServer } = require('mullama');

    const server = new WebSocketServer({
      port: 8080,
      maxConnections: 100,
      enableAudio: true,
      enableCompression: true
    });

    server.onConnect((conn) => {
      console.log(`Client connected: ${conn.id}`);
    });

    server.onMessage(async (msg, conn) => {
      if (msg.type === 'Generate') {
        const stream = await server.generateStream(msg.data.prompt);
        for await (const token of stream) {
          await conn.send({ type: 'Token', data: { text: token, isFinal: false } });
        }
        await conn.send({ type: 'Token', data: { text: '', isFinal: true } });
      }
    });

    server.onDisconnect((conn) => {
      console.log(`Client disconnected: ${conn.id}`);
    });

    await server.start();
    ```

=== "Python"

    ```python
    from mullama import WebSocketServer, WebSocketConfig

    config = WebSocketConfig(
        port=8080,
        max_connections=100,
        enable_audio=True,
        enable_compression=True
    )

    server = WebSocketServer(config)

    @server.on_connect
    async def handle_connect(conn):
        print(f"Client connected: {conn.id}")

    @server.on_message
    async def handle_message(msg, conn):
        if msg.type == "Generate":
            async for token in server.generate_stream(msg.data.prompt):
                await conn.send({"type": "Token", "data": {"text": token, "is_final": False}})
            await conn.send({"type": "Token", "data": {"text": "", "is_final": True}})

    @server.on_disconnect
    async def handle_disconnect(conn):
        print(f"Client disconnected: {conn.id}")

    await server.start()
    ```

=== "Rust"

    ```rust
    use mullama::{WebSocketServer, WebSocketConfig, WSMessage};

    let server = WebSocketServer::new(
        WebSocketConfig::new()
            .port(8080)
            .max_connections(100)
            .enable_audio()
            .enable_compression()
    )
    .on_connect(|conn| async move {
        println!("Client connected: {}", conn.id());
        Ok(())
    })
    .on_message(|msg, conn| async move {
        match msg {
            WSMessage::Generate { prompt, config } => {
                conn.send(WSMessage::Token {
                    text: "Response...".into(),
                    is_final: true,
                }).await?;
            }
            _ => {}
        }
        Ok(())
    })
    .on_disconnect(|conn| async move {
        println!("Client disconnected: {}", conn.id());
        Ok(())
    })
    .build()
    .await?;

    server.start().await?;
    ```

=== "CLI"

    ```bash
    # Start WebSocket server via daemon
    mullama serve --model model.gguf --ws-port 8080

    # Test with websocat
    echo '{"type":"Generate","data":{"prompt":"Hello"}}' | \
      websocat ws://localhost:8080
    ```

### Server Methods

| Method | Description |
|--------|-------------|
| `start()` | Start accepting connections |
| `stop()` | Gracefully stop the server |
| `broadcast(msg)` | Send message to all connected clients |
| `send_to(id, msg)` | Send message to a specific client |
| `connection_count()` | Get current number of connections |
| `stats()` | Get server statistics |

---

## WebSocketConfig

Configure server behavior with the builder pattern.

```rust
use mullama::WebSocketConfig;
use std::time::Duration;

let config = WebSocketConfig::new()
    .port(8080)
    .max_connections(200)
    .enable_audio()
    .enable_compression()
    .ping_interval(Duration::from_secs(30))
    .message_size_limit(16 * 1024 * 1024)  // 16 MB
    .connection_timeout(Duration::from_secs(300));
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `port` | Server listening port | 8080 |
| `max_connections` | Maximum concurrent clients | 100 |
| `enable_audio` | Enable audio message handling | false |
| `enable_compression` | Enable per-message deflate | false |
| `ping_interval` | Keep-alive ping frequency | 30s |
| `message_size_limit` | Maximum message size in bytes | 1 MB |
| `connection_timeout` | Idle connection timeout | 300s |

---

## ConnectionManager

The `ConnectionManager` handles client lifecycle, tracking active connections and managing resources.

=== "Node.js"

    ```javascript
    const manager = server.connectionManager;

    // Get all active connections
    const connections = manager.activeConnections();

    // Check specific client
    const isConnected = manager.isConnected('client-123');

    // Disconnect a client
    await manager.disconnect('client-123');

    // Get connection info
    const info = manager.getInfo('client-123');
    console.log(`Connected at: ${info.connectedAt}`);
    ```

=== "Python"

    ```python
    manager = server.connection_manager

    # Get all active connections
    connections = manager.active_connections()

    # Check specific client
    is_connected = manager.is_connected("client-123")

    # Disconnect a client
    await manager.disconnect("client-123")
    ```

=== "Rust"

    ```rust
    let manager = server.connection_manager();

    // Get all active connection IDs
    let connections = manager.active_connections();

    // Check if a specific client is connected
    let is_connected = manager.is_connected("client-123");

    // Disconnect a specific client
    manager.disconnect("client-123").await?;
    ```

---

## WSMessage Types

The message protocol defines structured types for all WebSocket communication.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WSMessage {
    // Text content
    Text { content: String },

    // Binary data
    Binary { data: Vec<u8>, mime_type: Option<String> },

    // Generation request
    Generate { prompt: String, config: Option<GenerationConfig> },

    // Audio data
    Audio { data: Vec<u8>, format: AudioFormat },

    // Custom application message
    Custom { kind: String, payload: serde_json::Value },

    // Streaming token
    Token { text: String, is_final: bool },

    // Error reporting
    Error { message: String, code: u32 },

    // Keep-alive
    Ping,
    Pong,
}
```

### Message Flow

```
Client                          Server
  |                               |
  |--- Generate { prompt } ------>|
  |                               |-- Process prompt
  |<-- Token { text, false } -----|
  |<-- Token { text, false } -----|
  |<-- Token { text, true }  -----|  (is_final = true)
  |                               |
  |--- Audio { data } ----------->|
  |<-- Text { transcript } -------|
  |                               |
```

---

## Room/Channel Support

Organize connections into logical groups for targeted messaging.

=== "Node.js"

    ```javascript
    server.onConnect(async (conn) => {
      // Join a room based on user preference
      await conn.joinRoom('general');
    });

    server.onMessage(async (msg, conn) => {
      if (msg.type === 'Custom' && msg.data.kind === 'chat') {
        // Broadcast to room members only
        await server.broadcastToRoom('general', {
          type: 'Text',
          data: { content: `${conn.id}: ${msg.data.payload.text}` }
        });
      }
    });

    // Send to specific room
    await server.broadcastToRoom('announcements', {
      type: 'Text',
      data: { content: 'Server maintenance in 5 minutes' }
    });
    ```

=== "Python"

    ```python
    @server.on_connect
    async def handle_connect(conn):
        await conn.join_room("general")

    @server.on_message
    async def handle_message(msg, conn):
        if msg.type == "Custom" and msg.data["kind"] == "chat":
            await server.broadcast_to_room("general", {
                "type": "Text",
                "data": {"content": f"{conn.id}: {msg.data['payload']['text']}"}
            })

    # Send to specific room
    await server.broadcast_to_room("announcements", {
        "type": "Text",
        "data": {"content": "Server maintenance in 5 minutes"}
    })
    ```

=== "Rust"

    ```rust
    .on_connect(|conn| async move {
        conn.join_room("general").await?;
        Ok(())
    })
    .on_message(|msg, conn| async move {
        if let WSMessage::Custom { kind, payload } = msg {
            if kind == "chat" {
                server.broadcast_to_room("general", WSMessage::Text {
                    content: format!("{}: {}", conn.id(), payload["text"]),
                }).await?;
            }
        }
        Ok(())
    })
    ```

---

## Audio Streaming over WebSocket

Process real-time audio from clients for transcription or voice commands.

!!! note "Additional Feature Required"
    Audio streaming over WebSocket requires both `websockets` and `streaming-audio` features.

    ```toml
    mullama = { version = "0.1", features = ["websockets", "streaming-audio"] }
    ```

=== "Node.js"

    ```javascript
    // Client-side: capture and send audio
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    recorder.ondataavailable = (event) => {
      ws.send(JSON.stringify({
        type: 'Audio',
        data: {
          data: Array.from(new Uint8Array(event.data)),
          format: 'webm'
        }
      }));
    };

    recorder.start(100); // Send chunks every 100ms
    ```

=== "Python"

    ```python
    import sounddevice as sd
    import numpy as np

    # Capture and send audio chunks
    def audio_callback(indata, frames, time, status):
        audio_bytes = indata.tobytes()
        await ws.send({
            "type": "Audio",
            "data": {"data": list(audio_bytes), "format": "pcm"}
        })

    with sd.InputStream(callback=audio_callback, samplerate=16000, channels=1):
        await asyncio.sleep(30)  # Record for 30 seconds
    ```

=== "Rust"

    ```rust
    .on_message(|msg, conn| async move {
        match msg {
            WSMessage::Audio { data, format } => {
                let chunk = AudioChunk::from_bytes(&data, format)?;
                let processed = processor.process_chunk(&chunk).await?;

                if processed.voice_detected {
                    let transcript = transcribe_audio(&processed).await?;
                    conn.send(WSMessage::Text {
                        content: transcript,
                    }).await?;
                }
            }
            _ => {}
        }
        Ok(())
    })
    ```

---

## Message Compression

Enable per-message deflate compression to reduce bandwidth usage for text-heavy communication.

```rust
let config = WebSocketConfig::new()
    .enable_compression()
    .compression_threshold(128);  // Only compress messages > 128 bytes
```

!!! tip "When to Use Compression"
    - Enable for chat applications with large text messages
    - Enable for JSON-heavy protocols
    - Disable for binary/audio data (already compressed)
    - Disable for very small messages (overhead exceeds savings)

---

## Automatic Reconnection

Clients can implement automatic reconnection when the connection drops.

=== "Node.js"

    ```javascript
    class ReconnectingWebSocket {
      constructor(url, options = {}) {
        this.url = url;
        this.maxRetries = options.maxRetries || 5;
        this.retryDelay = options.retryDelay || 1000;
        this.retries = 0;
        this.connect();
      }

      connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.retries = 0;
          console.log('Connected');
        };

        this.ws.onclose = () => {
          if (this.retries < this.maxRetries) {
            this.retries++;
            const delay = this.retryDelay * Math.pow(2, this.retries - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.retries})`);
            setTimeout(() => this.connect(), delay);
          }
        };
      }
    }

    const ws = new ReconnectingWebSocket('ws://localhost:8080', {
      maxRetries: 5,
      retryDelay: 1000
    });
    ```

=== "Python"

    ```python
    import asyncio
    import websockets

    async def connect_with_retry(url, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                async with websockets.connect(url) as ws:
                    retries = 0  # Reset on successful connection
                    async for message in ws:
                        await handle_message(message)
            except websockets.ConnectionClosed:
                retries += 1
                delay = min(2 ** retries, 30)
                print(f"Reconnecting in {delay}s (attempt {retries})")
                await asyncio.sleep(delay)
    ```

---

## Server Statistics and Monitoring

Monitor server health and performance with `ServerStats`.

=== "Node.js"

    ```javascript
    const stats = server.stats();
    console.log(`Active connections: ${stats.activeConnections}`);
    console.log(`Messages received: ${stats.messagesReceived}`);
    console.log(`Messages sent: ${stats.messagesSent}`);
    console.log(`Bytes received: ${stats.bytesReceived}`);
    console.log(`Bytes sent: ${stats.bytesSent}`);
    console.log(`Uptime: ${stats.uptimeSeconds}s`);
    ```

=== "Python"

    ```python
    stats = server.stats()
    print(f"Active connections: {stats.active_connections}")
    print(f"Messages received: {stats.messages_received}")
    print(f"Messages sent: {stats.messages_sent}")
    print(f"Bytes received: {stats.bytes_received}")
    print(f"Bytes sent: {stats.bytes_sent}")
    print(f"Uptime: {stats.uptime}")
    ```

=== "Rust"

    ```rust
    let stats = server.stats();

    println!("Active connections: {}", stats.active_connections);
    println!("Total messages received: {}", stats.messages_received);
    println!("Total messages sent: {}", stats.messages_sent);
    println!("Bytes received: {}", stats.bytes_received);
    println!("Bytes sent: {}", stats.bytes_sent);
    println!("Uptime: {:?}", stats.uptime);
    ```

---

## Complete Example: Chat Server

A multi-user chat server with streaming responses and conversation history.

=== "Node.js"

    ```javascript
    const { WebSocketServer, loadModel } = require('mullama');

    async function main() {
      const model = await loadModel('model.gguf');
      const clients = new Map();

      const server = new WebSocketServer({ port: 8080, maxConnections: 50 });

      server.onConnect((conn) => {
        clients.set(conn.id, { history: [] });
        console.log(`Client ${conn.id} connected`);
      });

      server.onMessage(async (msg, conn) => {
        if (msg.type === 'Text') {
          const state = clients.get(conn.id);
          state.history.push(`User: ${msg.data.content}`);

          const prompt = state.history.join('\n') + '\nAssistant:';
          let response = '';

          for await (const token of model.generateStream(prompt, { maxTokens: 300 })) {
            response += token;
            await conn.send({ type: 'Token', data: { text: token, isFinal: false } });
          }

          await conn.send({ type: 'Token', data: { text: '', isFinal: true } });
          state.history.push(`Assistant: ${response}`);
        }
      });

      server.onDisconnect((conn) => {
        clients.delete(conn.id);
      });

      await server.start();
      console.log('Chat server running on ws://localhost:8080');
    }

    main();
    ```

=== "Python"

    ```python
    from mullama import WebSocketServer, WebSocketConfig, load_model

    async def main():
        model = await load_model("model.gguf")
        clients = {}

        server = WebSocketServer(WebSocketConfig(port=8080, max_connections=50))

        @server.on_connect
        async def on_connect(conn):
            clients[conn.id] = {"history": []}

        @server.on_message
        async def on_message(msg, conn):
            if msg.type == "Text":
                state = clients[conn.id]
                state["history"].append(f"User: {msg.data['content']}")

                prompt = "\n".join(state["history"]) + "\nAssistant:"
                response = ""

                async for token in model.generate_stream(prompt, max_tokens=300):
                    response += token
                    await conn.send({"type": "Token", "data": {"text": token, "is_final": False}})

                await conn.send({"type": "Token", "data": {"text": "", "is_final": True}})
                state["history"].append(f"Assistant: {response}")

        @server.on_disconnect
        async def on_disconnect(conn):
            del clients[conn.id]

        await server.start()

    import asyncio
    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{WebSocketServer, WebSocketConfig, WSMessage, Model, Context, ContextParams};
    use std::sync::Arc;
    use tokio::sync::{mpsc, RwLock};
    use std::collections::HashMap;

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let model = Arc::new(Model::load("model.gguf")?);
        let clients: Arc<RwLock<HashMap<String, Vec<String>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let server = WebSocketServer::new(
            WebSocketConfig::new()
                .port(8080)
                .max_connections(50)
        )
        .on_connect({
            let clients = clients.clone();
            move |conn| {
                let clients = clients.clone();
                async move {
                    clients.write().await.insert(conn.id().to_string(), Vec::new());
                    Ok(())
                }
            }
        })
        .on_message({
            let model = model.clone();
            let clients = clients.clone();
            move |msg, conn| {
                let model = model.clone();
                let clients = clients.clone();
                async move {
                    if let WSMessage::Text { content } = msg {
                        let prompt = {
                            let mut map = clients.write().await;
                            let history = map.entry(conn.id().to_string())
                                .or_insert_with(Vec::new);
                            history.push(format!("User: {}", content));
                            history.join("\n") + "\nAssistant:"
                        };

                        let (tx, mut rx) = mpsc::channel::<String>(64);
                        let model = model.clone();

                        tokio::task::spawn_blocking(move || {
                            let mut ctx = Context::new(model, ContextParams::default()).unwrap();
                            ctx.generate_streaming(&prompt, 300, |token| {
                                tx.blocking_send(token.to_string()).is_ok()
                            }).ok();
                        });

                        let mut full_response = String::new();
                        while let Some(token) = rx.recv().await {
                            full_response.push_str(&token);
                            conn.send(WSMessage::Token { text: token, is_final: false }).await?;
                        }
                        conn.send(WSMessage::Token { text: String::new(), is_final: true }).await?;

                        let mut map = clients.write().await;
                        if let Some(history) = map.get_mut(conn.id()) {
                            history.push(format!("Assistant: {}", full_response));
                        }
                    }
                    Ok(())
                }
            }
        })
        .build()
        .await?;

        println!("Chat server running on ws://localhost:8080");
        server.start().await?;
        Ok(())
    }
    ```

---

## Integration with Web Framework

Combine WebSocket support with the Axum web framework for a unified server.

```rust
use mullama::{create_router, AppState, WebSocketServer, WebSocketConfig};

let app_state = AppState::new(model.clone())
    .enable_streaming()
    .build();

// REST API on port 3000
let rest_app = create_router(app_state);

// WebSocket on port 8080
let ws_server = WebSocketServer::new(
    WebSocketConfig::new().port(8080)
).build().await?;

// Run both concurrently
tokio::select! {
    _ = axum::Server::bind(&"0.0.0.0:3000".parse()?).serve(rest_app.into_make_service()) => {},
    _ = ws_server.start() => {},
}
```

---

## Error Handling

!!! warning "Connection Cleanup"
    Always implement `on_disconnect` to clean up per-client resources (chat history, pending tasks, etc.) to prevent memory leaks.

The server handles these automatically:

1. Client drops connection mid-generation -- generation task cancelled
2. Message too large -- Error message sent, connection maintained
3. Invalid message format -- Error message sent
4. Ping timeout -- Connection closed, `on_disconnect` called

---

## See Also

- [Web Framework](web-framework.md) - REST API integration
- [Streaming Audio](streaming-audio.md) - Real-time audio processing
- [Streaming Guide](../guide/streaming.md) - Token streaming patterns
