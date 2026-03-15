---
title: "Tutorial: Voice Assistant"
description: Build a real-time voice assistant with audio capture, voice activity detection, speech processing, and streaming LLM responses using Mullama.
---

# Voice Assistant

Build a real-time voice assistant that captures audio, detects speech, processes it through a local LLM, and streams responses -- all running privately with no cloud dependencies.

---

## What You'll Build

A voice-to-text-to-response pipeline that:

- Captures audio in real-time from the microphone
- Detects speech using Voice Activity Detection (VAD)
- Processes audio through the multimodal pipeline
- Generates streaming LLM responses
- Manages conversation context across turns
- Handles errors gracefully with automatic recovery

---

## Prerequisites

- A chat-capable GGUF model (instruct-tuned recommended)
- Audio system dependencies:

=== "Linux (Ubuntu/Debian)"
    ```bash
    sudo apt install -y libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev
    ```

=== "macOS"
    ```bash
    # CoreAudio is available by default
    ```

- Rust toolchain (this tutorial uses Rust primarily since audio capture requires native access)
- Features: `streaming-audio`, `multimodal`, `async`, `streaming`

```bash
mullama pull llama3.2:1b  # Pull a model for generation
```

---

## Architecture Overview

```
Microphone --> Audio Capture --> Ring Buffer --> VAD (Voice Activity Detection)
                                                      |
                                                      v (speech detected)
                                               Audio Processing --> Multimodal Processor
                                                                          |
                                                                          v
                                                                   Transcription
                                                                          |
                                                                          v
                                                                  LLM Generation --> Streaming Output
```

Key components:

- **`StreamingAudioProcessor`** -- Real-time audio capture with ring buffer
- **Voice Activity Detection (VAD)** -- Detects when the user is speaking
- **`MultimodalProcessor`** -- Converts audio to text (speech-to-text)
- **`TokenStream`** -- Generates and streams the LLM response

---

## Step 1: Audio Capture Configuration

Configure audio capture parameters for speech recognition.

```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig};

// Configure audio for speech recognition
let audio_config = AudioStreamConfig::new()
    .sample_rate(16000)              // 16kHz -- standard for speech
    .channels(1)                      // Mono audio
    .chunk_duration_ms(100)           // 100ms chunks for low latency
    .enable_voice_detection(true)     // Built-in VAD
    .vad_threshold(0.2)              // Sensitivity (0.0-1.0, lower = more sensitive)
    .enable_noise_reduction(true)     // Filter background noise
    .silence_timeout_ms(1500);        // End utterance after 1.5s silence

let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate in Hz (16kHz for speech) |
| `channels` | 1 | Number of audio channels (mono for speech) |
| `chunk_duration_ms` | 100 | Size of each audio chunk in milliseconds |
| `enable_voice_detection` | `true` | Enable Voice Activity Detection |
| `vad_threshold` | 0.2 | VAD sensitivity (lower = more sensitive) |
| `enable_noise_reduction` | `true` | Filter background noise |
| `silence_timeout_ms` | 1500 | Silence duration to end an utterance |

---

## Step 2: Voice Activity Detection

Listen for speech and capture complete utterances.

```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig, MullamaError};
use futures::StreamExt;

async fn listen_for_speech(
    audio_processor: &mut StreamingAudioProcessor,
) -> Result<Vec<f32>, MullamaError> {
    let mut audio_stream = audio_processor.start_capture().await?;
    let mut speech_buffer: Vec<f32> = Vec::new();
    let mut is_speaking = false;

    println!("Listening...");

    while let Some(chunk) = audio_stream.next().await {
        let processed = audio_processor.process_chunk(&chunk).await?;

        if processed.voice_detected && processed.signal_level > 0.1 {
            if !is_speaking {
                println!("[Speech detected]");
                is_speaking = true;
            }
            speech_buffer.extend_from_slice(&processed.audio_data);
        } else if is_speaking {
            // Silence after speech -- utterance complete
            println!("[End of utterance: {} samples]", speech_buffer.len());
            break;
        }
    }

    Ok(speech_buffer)
}
```

---

## Step 3: Speech-to-Text

Convert captured audio to text using the multimodal processor.

```rust
use mullama::{MultimodalProcessor, AudioInput};

async fn transcribe_audio(
    multimodal: &MultimodalProcessor,
    audio_data: &[f32],
    sample_rate: u32,
) -> Result<String, MullamaError> {
    // Create audio input from captured samples
    let audio_input = AudioInput::from_samples(audio_data, sample_rate, 1)?;

    // Process through the multimodal pipeline
    let result = multimodal.process_audio(&audio_input).await?;

    match result.transcript {
        Some(text) if !text.trim().is_empty() => {
            println!("Transcribed: \"{}\"", text);
            Ok(text)
        }
        _ => {
            println!("(no speech recognized)");
            Ok(String::new())
        }
    }
}
```

---

## Step 4: Generate Streaming Response

Generate a response using the transcribed text.

```rust
use mullama::{AsyncModel, StreamConfig, TokenStream};
use futures::StreamExt;
use std::io::Write;

async fn generate_response(
    model: &AsyncModel,
    transcript: &str,
    history: &str,
) -> Result<String, MullamaError> {
    let prompt = format!("{}User: {}\nAssistant:", history, transcript);

    let config = StreamConfig::default()
        .max_tokens(200)       // Keep responses concise for voice
        .temperature(0.7)
        .top_k(40);

    let mut stream = TokenStream::new(model.clone(), &prompt, config).await?;
    let mut response = String::new();

    print!("Assistant: ");
    while let Some(token) = stream.next().await {
        let token = token?;
        if response.contains("\nUser:") { break; }
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
        response.push_str(&token.text);
        if token.is_final { break; }
    }
    println!();

    Ok(response.trim().to_string())
}
```

---

## Step 5: Conversation Loop

Tie all components together in a continuous voice interaction loop.

```rust
use mullama::prelude::*;
use mullama::{
    AsyncModel, StreamingAudioProcessor, AudioStreamConfig,
    MultimodalProcessor, AudioInput, StreamConfig, TokenStream,
};
use futures::StreamExt;

async fn run_voice_assistant() -> Result<(), MullamaError> {
    // Initialize model
    let model = AsyncModel::load("path/to/model.gguf").await?;

    // Initialize audio
    let audio_config = AudioStreamConfig::new()
        .sample_rate(16000)
        .enable_voice_detection(true)
        .vad_threshold(0.2)
        .enable_noise_reduction(true)
        .silence_timeout_ms(1500);
    let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;

    // Initialize multimodal processor
    let multimodal = MultimodalProcessor::new()
        .enable_audio_processing()
        .build();

    // Conversation state
    let mut history = String::from(
        "System: You are a helpful voice assistant. Keep responses concise \
         and conversational since they will be spoken aloud.\n\n"
    );

    println!("Voice Assistant ready! Start speaking...");
    println!("(Press Ctrl+C to exit)\n");

    loop {
        // Listen for speech
        let audio_data = listen_for_speech(&mut audio_processor).await?;
        if audio_data.is_empty() { continue; }

        // Transcribe
        let transcript = transcribe_audio(&multimodal, &audio_data, 16000).await?;
        if transcript.is_empty() { continue; }

        println!("You: {}", transcript);

        // Generate response
        let response = generate_response(&model, &transcript, &history).await?;

        // Update history (keep bounded)
        history.push_str(&format!("User: {}\nAssistant: {}\n", transcript, response));
        if history.len() > 3000 {
            let trim_point = history.len() - 2000;
            if let Some(pos) = history[trim_point..].find('\n') {
                history = history[trim_point + pos..].to_string();
            }
        }

        println!();
    }
}
```

---

## Latency Optimization

For real-time voice interaction, minimize latency at each stage:

| Stage | Target Latency | Tips |
|-------|----------------|------|
| Audio capture | < 100ms | Use small chunk sizes (50-100ms) |
| VAD detection | < 50ms | Lower threshold for faster detection |
| Transcription | < 500ms | Use smaller/quantized speech models |
| LLM first token | < 200ms | Use small models, GPU acceleration |
| Total round-trip | < 1.5s | Overlap processing stages |

```rust
// Optimize for low latency
let audio_config = AudioStreamConfig::new()
    .chunk_duration_ms(50)           // Smaller chunks = faster detection
    .vad_threshold(0.15)             // More sensitive = faster trigger
    .silence_timeout_ms(1000);       // Shorter timeout = faster response

let stream_config = StreamConfig::default()
    .max_tokens(100)                 // Shorter responses for voice
    .temperature(0.6);               // Slightly more focused
```

---

## Error Handling

Handle audio and processing errors gracefully with retry logic.

```rust
async fn robust_voice_loop(
    audio_processor: &mut StreamingAudioProcessor,
    multimodal: &MultimodalProcessor,
    model: &AsyncModel,
) -> Result<(), MullamaError> {
    let mut consecutive_errors = 0;

    loop {
        match process_one_turn(audio_processor, multimodal, model).await {
            Ok(_) => consecutive_errors = 0,
            Err(MullamaError::AudioError(e)) => {
                eprintln!("Audio error: {}. Retrying...", e);
                consecutive_errors += 1;
                if consecutive_errors >= 5 {
                    eprintln!("Restarting audio system...");
                    audio_processor.restart().await?;
                    consecutive_errors = 0;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
            Err(MullamaError::ModelError(e)) => {
                eprintln!("Model error: {}. Skipping turn.", e);
                consecutive_errors += 1;
            }
            Err(e) => return Err(e),
        }

        if consecutive_errors >= 10 {
            return Err(MullamaError::SystemError("Too many errors".into()));
        }
    }
}
```

---

## Ring Buffer Architecture

The `StreamingAudioProcessor` uses a ring buffer for zero-allocation audio handling:

```
Audio Input (continuous stream from microphone)
    |
    v
+---+---+---+---+---+---+---+---+
| C | C | C | C | C | C | C | C |  Ring Buffer (pre-allocated chunks)
+---+---+---+---+---+---+---+---+
        ^               ^
        |               |
      Read            Write
      Pointer         Pointer
```

Benefits:

- **No allocations** during capture (pre-allocated buffer)
- **Constant latency** regardless of processing speed
- **Overflow protection** -- oldest data is overwritten if processing falls behind

---

## Complete Working Example

```rust
use mullama::prelude::*;
use mullama::{
    AsyncModel, StreamingAudioProcessor, AudioStreamConfig,
    MultimodalProcessor, AudioInput, StreamConfig, TokenStream,
};
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("Mullama Voice Assistant");
    println!("=======================\n");

    #[cfg(all(
        feature = "streaming-audio",
        feature = "multimodal",
        feature = "async",
        feature = "streaming"
    ))]
    {
        let model_path = std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "path/to/model.gguf".to_string());

        // Load model
        println!("Loading model...");
        let model = AsyncModel::load(&model_path).await?;
        println!("Model loaded!");

        // Configure audio
        let audio_config = AudioStreamConfig::new()
            .sample_rate(16000)
            .channels(1)
            .enable_voice_detection(true)
            .vad_threshold(0.2)
            .enable_noise_reduction(true)
            .silence_timeout_ms(1500);
        let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;

        // Configure multimodal processor
        let multimodal = MultimodalProcessor::new()
            .enable_audio_processing()
            .build();

        let mut history = String::from(
            "System: You are a helpful, concise voice assistant.\n\n"
        );

        println!("\n--- Ready. Speak into your microphone. Ctrl+C to exit. ---\n");

        loop {
            // Capture speech
            let mut audio_stream = audio_processor.start_capture().await?;
            let mut speech_buffer: Vec<f32> = Vec::new();
            let mut heard_speech = false;

            while let Some(chunk) = audio_stream.next().await {
                let processed = audio_processor.process_chunk(&chunk).await?;
                if processed.voice_detected && processed.signal_level > 0.1 {
                    if !heard_speech {
                        print!("[listening] ");
                        std::io::stdout().flush().unwrap();
                        heard_speech = true;
                    }
                    speech_buffer.extend_from_slice(&processed.audio_data);
                } else if heard_speech {
                    println!("done");
                    break;
                }
            }

            if speech_buffer.is_empty() { continue; }

            // Transcribe
            let audio_input = AudioInput::from_samples(&speech_buffer, 16000, 1)?;
            let result = multimodal.process_audio(&audio_input).await?;
            let transcript = match result.transcript {
                Some(text) if !text.trim().is_empty() => text,
                _ => { println!("(no speech recognized)"); continue; }
            };
            println!("You: {}", transcript);

            // Generate response
            let prompt = format!("{}User: {}\nAssistant:", history, transcript);
            let config = StreamConfig::default()
                .max_tokens(150)
                .temperature(0.7);

            let mut stream = TokenStream::new(model.clone(), &prompt, config).await?;
            let mut response = String::new();

            print!("Assistant: ");
            while let Some(token) = stream.next().await {
                let token = token?;
                if response.contains("\nUser:") { break; }
                print!("{}", token.text);
                std::io::stdout().flush().unwrap();
                response.push_str(&token.text);
                if token.is_final { break; }
            }
            println!("\n");

            // Update history
            history.push_str(&format!(
                "User: {}\nAssistant: {}\n", transcript, response.trim()
            ));
            if history.len() > 3000 {
                let trim = history.len() - 2000;
                if let Some(p) = history[trim..].find('\n') {
                    history = history[trim + p..].to_string();
                }
            }
        }
    }

    #[cfg(not(all(
        feature = "streaming-audio", feature = "multimodal",
        feature = "async", feature = "streaming"
    )))]
    {
        println!("This example requires features: streaming-audio, multimodal, async, streaming");
        println!("Run with:");
        println!("  cargo run --example voice_assistant --features \"streaming-audio,multimodal,async,streaming\"");
    }

    Ok(())
}
```

---

## Extension: Text-to-Speech Output

Add TTS output so the assistant speaks its responses aloud. This requires an external TTS engine since Mullama focuses on inference:

```rust
// Example using system TTS (platform-specific)
fn speak_response(text: &str) {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("say")
            .arg(text)
            .spawn()
            .expect("Failed to invoke macOS TTS");
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("espeak-ng")
            .arg(text)
            .spawn()
            .expect("Failed to invoke espeak-ng. Install with: sudo apt install espeak-ng");
    }
}
```

---

## What's Next

- [Multimodal Processing](multimodal.md) -- Understand the multimodal pipeline in depth
- [Streaming Generation](streaming.md) -- Deep dive into token streaming patterns
- [API Server](api-server.md) -- Expose voice capabilities over HTTP/WebSocket
- [Advanced: Streaming Audio](../advanced/streaming-audio.md) -- Low-level audio architecture
