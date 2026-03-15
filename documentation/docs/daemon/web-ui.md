---
title: Web UI
description: Browser-based management interface with chat, model management, playground, and monitoring
---

# Web UI

Mullama includes an embedded Web UI built with Vue 3, Vite, and Tailwind CSS that provides a browser-based interface for managing models, chatting, testing APIs, and monitoring the daemon.

**Access URL:** `http://localhost:8080/ui/`

## Overview

The Web UI provides five main views:

| View | Path | Description |
|------|------|-------------|
| Dashboard | `/ui/` | System status, loaded models, quick actions |
| Models | `/ui/models` | Browse, download, load, and manage models |
| Chat | `/ui/chat` | Multi-conversation chat with streaming |
| Playground | `/ui/playground` | API testing and curl generation |
| Settings | `/ui/settings` | Theme, defaults, and display options |

---

## Building the Web UI

The Web UI is a Vue.js single-page application located in the `ui/` directory. It must be built before being embedded into the daemon binary.

### Prerequisites

- Node.js (v18+)
- npm

### Build Steps

```bash
# Navigate to the UI directory
cd ui

# Install dependencies
npm install

# Build the production bundle
npm run build

# Return to project root
cd ..
```

### Build Daemon with Embedded UI

After building the UI, compile the daemon binary with the `embedded-ui` feature flag:

```bash
cargo build --release --features daemon,embedded-ui
```

!!! note "Build Order"
    The UI must be built **before** compiling the Rust binary. The `embedded-ui` feature uses Rust's `include_dir!` macro to embed the built UI assets at compile time.

### Development Mode

For UI development with hot-reload:

```bash
cd ui
npm run dev
```

This starts a Vite dev server (typically on port 5173) that proxies API requests to the Mullama daemon running on port 8080. Make sure the daemon is running separately:

```bash
# In another terminal
mullama serve --model llama3.2:1b
```

---

## Accessing the Web UI

Once the daemon is running with the embedded UI:

```bash
# Start the daemon
mullama serve --model llama3.2:1b

# Open in browser
open http://localhost:8080/ui/
```

The UI is served at `/ui/` and all sub-paths (`/ui/*`) route to the Vue.js SPA using client-side routing.

---

## Dashboard

The Dashboard provides an at-a-glance overview of the daemon state:

### System Status Card

- **Uptime** -- How long the daemon has been running
- **Version** -- Mullama daemon version
- **HTTP Endpoint** -- Address and port of the API server
- **IPC Socket** -- Path to the IPC socket

### Loaded Models Card

- List of all currently loaded models
- Each model shows: name, parameter count, GPU layers, context size
- Default model indicator
- Active request count per model
- Quick actions: unload, set as default

### Quick Actions

- **Load Default Model** -- One-click to download and load a recommended model
- **Open Chat** -- Navigate to the chat interface
- **View Metrics** -- Link to raw Prometheus metrics

### Statistics

- **Total Requests** -- Lifetime request count
- **Tokens Generated** -- Total tokens produced
- **Active Requests** -- Currently processing
- **GPU Available** -- Whether GPU acceleration is active

---

## Models Page

The Models view provides comprehensive model management.

### Browse Available Models

A card grid of pre-configured default models with:

- Model name and description
- Size indicator (1B, 7B, 14B, etc.)
- Capability tags (chat, reasoning, code, vision, embeddings)
- Download button with size estimate
- Status indicator (not downloaded, downloading, available, loaded)

### Download Progress

When pulling models, the UI displays:

- File name and total size
- Progress bar with percentage
- Download speed (MB/s)
- Estimated time remaining
- Cancel button

### Loaded Models Management

For each loaded model:

- **Unload** -- Free memory by removing from inference engine
- **Set Default** -- Make this the default model for API requests
- **Details** -- View parameters, context size, GPU layers, file path
- **Active Requests** -- Count of in-flight requests

### Custom Model Loading

Form to load a custom model:

- Model alias (text input)
- GGUF file path (file picker or text input)
- GPU layers (slider: 0-99)
- Context size (dropdown: 2048, 4096, 8192, 16384, 32768)
- Set as default (checkbox)

---

## Chat

The Chat view provides a rich conversational interface.

### Features

- **Real-time Streaming** -- Tokens appear as they are generated via Server-Sent Events
- **Markdown Rendering** -- Full markdown support including headings, lists, bold, italic, links
- **Code Highlighting** -- Syntax-highlighted code blocks for 50+ languages with copy button
- **Thinking Display** -- Collapsible reasoning blocks for models with thinking tokens
- **Model Selection** -- Dropdown to switch between loaded models mid-conversation
- **Conversation History** -- Sidebar with multiple conversations
- **System Prompts** -- Configure system prompts per conversation
- **Stop Generation** -- Button to halt streaming generation
- **Token Counter** -- Shows prompt and completion token counts
- **Speed Display** -- Tokens per second during generation

### Conversation Management

- **New Conversation** -- Start fresh with a clean context
- **Rename** -- Give conversations meaningful names
- **Delete** -- Remove conversations
- **Export** -- Download as markdown

### Thinking Display

For models configured with thinking tokens (e.g., DeepSeek-R1):

```
[Thinking] (click to expand)
  Let me work through this step by step...
  First, I need to consider the problem...
  The key insight is...

[Response]
The answer is 42.
```

The thinking section is collapsed by default and can be expanded with a click.

### Code Blocks

Code blocks in responses feature:

- Language detection and label
- Syntax highlighting (highlight.js)
- One-click copy to clipboard button
- Line numbers (optional, toggled in settings)

---

## Playground

The Playground provides direct API testing capabilities with a form-based interface.

### Request Builder

- **Endpoint Selector** -- Choose between:
    - Chat Completions (`/v1/chat/completions`)
    - Text Completions (`/v1/completions`)
    - Embeddings (`/v1/embeddings`)
    - Anthropic Messages (`/v1/messages`)
    - Raw Generate (`/api/generate`)

- **Model Selector** -- Dropdown of loaded models

- **Messages Editor** -- Add/remove/edit messages with role selection (system, user, assistant)

- **Prompt Input** -- For text completion endpoint

### Parameter Tuning

Adjustable parameters with sliders and inputs:

| Parameter | Control | Range |
|-----------|---------|-------|
| Temperature | Slider | 0.0 - 2.0 |
| Top P | Slider | 0.0 - 1.0 |
| Top K | Number input | 1 - 100 |
| Max Tokens | Number input | 1 - 32768 |
| Presence Penalty | Slider | -2.0 - 2.0 |
| Frequency Penalty | Slider | -2.0 - 2.0 |
| Stream | Toggle | on/off |

### Response Viewer

- Formatted JSON response with syntax highlighting
- Expandable/collapsible JSON tree
- Response metadata (status code, timing, token counts)
- Raw text view for generated content

### curl Generation

Every request configuration can be exported as a curl command:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'
```

Copy-to-clipboard button for immediate terminal use.

---

## Settings

The Settings view manages UI preferences and defaults.

### Theme

- **Auto** -- Follow system preference (light/dark)
- **Light** -- Light color scheme
- **Dark** -- Dark color scheme

### Generation Defaults

Default values for the Playground and Chat:

- Default temperature
- Default max tokens
- Default top_p
- Default model (when multiple are loaded)

### Display Options

- Code highlighting theme (vs-dark, github, monokai, etc.)
- Show line numbers in code blocks
- Markdown rendering (on/off)
- Show timestamps on messages
- Show token counts

### Connection

- API endpoint URL (default: auto-detect from page URL)
- API key (if configured on the server)

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Vue 3 (Composition API) | Reactive UI components |
| Build Tool | Vite | Fast builds, HMR in development |
| Styling | Tailwind CSS | Utility-first CSS framework |
| Icons | Heroicons | UI icons |
| Markdown | markdown-it | Markdown rendering |
| Code Highlighting | highlight.js | Syntax highlighting |
| HTTP Client | Fetch API | API communication |
| Streaming | EventSource / fetch + ReadableStream | SSE and NDJSON streaming |
| State Management | Vue reactivity | Application state |
| Routing | Vue Router | Client-side routing |

---

## API Communication

The Web UI communicates with the daemon through the same REST API documented in the [REST API](rest-api.md), [OpenAI API](openai-api.md), and [Anthropic API](anthropic-api.md) pages.

| UI Action | API Endpoint | Method |
|-----------|-------------|--------|
| Dashboard status | `/api/system/status` | GET |
| List models | `/api/models` | GET |
| Load model | `/api/models/load` | POST |
| Unload model | `/api/models/:name/unload` | POST |
| Pull model | `/api/models/pull` | POST |
| Default models | `/api/defaults` | GET |
| Use default model | `/api/defaults/:name/use` | POST |
| Chat (streaming) | `/v1/chat/completions` | POST |
| Embeddings | `/v1/embeddings` | POST |

---

## Building Without Embedded UI

If you do not need the Web UI, build the daemon without the `embedded-ui` feature:

```bash
cargo build --release --features daemon
```

In this case, accessing `/ui/` will return a message indicating the UI is not available:

```json
{
  "error": "Web UI not available. Build with --features embedded-ui"
}
```

---

## URL Routes

| Route | View | Description |
|-------|------|-------------|
| `/ui/` | Dashboard | System overview and quick actions |
| `/ui/chat` | Chat | Conversational interface |
| `/ui/chat/:id` | Chat | Specific conversation |
| `/ui/models` | Models | Model management |
| `/ui/playground` | Playground | API testing |
| `/ui/settings` | Settings | UI configuration |

All routes use client-side routing. Refreshing any page works correctly as the server returns the SPA for all `/ui/*` paths.

---

## Troubleshooting

### UI Not Loading

!!! warning "Common Issues"
    - Ensure the daemon was built with `--features daemon,embedded-ui`
    - Check that `npm run build` completed successfully in the `ui/` directory before compiling
    - Verify the daemon is running: `mullama daemon status`
    - Check the correct port: `curl http://localhost:8080/ui/`

### API Connection Errors

- The UI connects to the same host/port it is served from
- Ensure no firewall rules block the HTTP port (default: 8080)
- Check CORS is not being blocked (the daemon enables permissive CORS by default)
- If using a reverse proxy, ensure WebSocket/SSE passthrough is configured

### Stale UI Build

If the UI shows outdated content after code changes:

```bash
cd ui
rm -rf dist node_modules
npm install
npm run build
cd ..
cargo build --release --features daemon,embedded-ui
```

### Streaming Not Working

If chat responses appear all at once instead of streaming:

- Check that `proxy_buffering off` is set in your nginx config
- Ensure `chunked_transfer_encoding off` is not blocking SSE
- Verify the daemon's streaming endpoint works directly: `curl -N http://localhost:8080/v1/chat/completions ...`

### Dark Mode Issues

If the theme does not match your system preference:

- Use the Settings page to explicitly set light or dark mode
- Check that your browser supports the `prefers-color-scheme` media query
- Hard refresh the page (Ctrl+Shift+R) after changing system theme
