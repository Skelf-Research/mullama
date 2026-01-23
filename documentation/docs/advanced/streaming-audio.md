# Streaming Audio

Capture and process real-time audio with voice activity detection, noise reduction, and low-latency ring buffer architecture for voice-enabled AI applications.

!!! info "Feature Gate"
    This feature requires the `streaming-audio` feature flag, which transitively enables `multimodal`.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["streaming-audio"] }
    ```

## Overview

The streaming audio subsystem provides:

- **StreamingAudioProcessor** for real-time microphone capture
- **AudioStreamConfig** with device, sample rate, channels, and chunk size
- **Voice Activity Detection (VAD)** with configurable threshold
- **Ring buffer architecture** for low-latency, lock-free processing
- **AudioChunk** processing with metadata
- **StreamingMetrics** for latency and buffer utilization monitoring
- **Device enumeration** and selection
- **Noise reduction** pipeline

---

## Platform Requirements

!!! warning "System Dependencies Required"
    Streaming audio requires platform-specific audio libraries to be installed.

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt install -y libasound2-dev libpulse-dev libflac-dev \
        libvorbis-dev libopus-dev
    ```

    Uses **ALSA** or **PulseAudio** as the audio backend.

=== "macOS"

    No additional dependencies required. Uses **CoreAudio** natively.

=== "Windows"

    No additional dependencies required. Uses **DirectSound/WASAPI** natively.

---

## StreamingAudioProcessor

The main entry point for real-time audio capture and processing.

=== "Node.js"

    ```javascript
    const { StreamingAudioProcessor } = require('mullama');

    const processor = new StreamingAudioProcessor({
      sampleRate: 16000,
      channels: 1,
      enableVad: true,
      enableNoiseReduction: true,
      vadThreshold: 0.3
    });

    await processor.initialize();
    const stream = await processor.startCapture();

    for await (const chunk of stream) {
      if (chunk.voiceDetected) {
        console.log(`Speech detected! Level: ${chunk.signalLevel.toFixed(2)}`);
        // Process with AI model...
      }
    }

    await processor.stopCapture();
    ```

=== "Python"

    ```python
    from mullama import StreamingAudioProcessor, AudioStreamConfig

    config = AudioStreamConfig(
        sample_rate=16000,
        channels=1,
        enable_vad=True,
        enable_noise_reduction=True,
        vad_threshold=0.3
    )

    processor = StreamingAudioProcessor(config)
    await processor.initialize()

    async for chunk in processor.start_capture():
        if chunk.voice_detected:
            print(f"Speech detected! Level: {chunk.signal_level:.2f}")
            # Process with AI model...

    await processor.stop_capture()
    ```

=== "Rust"

    ```rust
    use mullama::{StreamingAudioProcessor, AudioStreamConfig, AudioChunk};

    let config = AudioStreamConfig::new()
        .sample_rate(16000)
        .channels(1)
        .enable_voice_detection(true)
        .enable_noise_reduction(true)
        .vad_threshold(0.3);

    let mut processor = StreamingAudioProcessor::new(config)?;
    processor.initialize().await?;

    let mut audio_stream = processor.start_capture().await?;

    while let Some(chunk) = audio_stream.next().await {
        if chunk.voice_detected {
            println!("Speech detected! Signal level: {:.2}", chunk.signal_level);
            // Process with AI model...
        }
    }

    processor.stop_capture().await?;
    ```

### Processor Methods

| Method | Description |
|--------|-------------|
| `new(config)` | Create a new processor |
| `initialize()` | Initialize audio devices |
| `start_capture()` | Begin audio capture, returns `AudioStream` |
| `stop_capture()` | Stop audio capture |
| `process_chunk(chunk)` | Process a single audio chunk |
| `metrics()` | Get streaming performance metrics |
| `list_input_devices()` | Enumerate available input devices |

---

## AudioStreamConfig

Configure audio capture parameters using the builder pattern.

```rust
use mullama::{AudioStreamConfig, DevicePreference};

let config = AudioStreamConfig::new()
    .sample_rate(16000)          // Sample rate in Hz
    .channels(1)                 // Mono audio
    .chunk_size(4096)            // Samples per chunk
    .buffer_size(4096)           // Ring buffer size
    .enable_noise_reduction(true)
    .enable_voice_detection(true)
    .vad_threshold(0.25)         // VAD sensitivity
    .max_latency_ms(50)          // Maximum acceptable latency
    .device_preference(DevicePreference::Default);
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sample_rate` | Audio sample rate in Hz | 16000 |
| `channels` | Number of audio channels (1=mono, 2=stereo) | 1 (mono) |
| `chunk_size` | Samples delivered per chunk | 1024 |
| `buffer_size` | Ring buffer size in samples | 4096 |
| `enable_noise_reduction` | Enable noise reduction pipeline | false |
| `enable_voice_detection` | Enable VAD processing | false |
| `vad_threshold` | Voice activity detection threshold (0.0-1.0) | 0.3 |
| `max_latency_ms` | Maximum acceptable processing latency | 100 |
| `device_preference` | Audio input device selection | Default |

---

## Voice Activity Detection (VAD)

VAD automatically detects when speech is present in the audio stream, enabling efficient processing by only sending speech segments to the model.

### How It Works

The energy-based VAD computes the RMS (Root Mean Square) energy of each audio chunk and compares it against the configured threshold:

1. **Energy Computation** - Calculate RMS of the audio samples
2. **Threshold Comparison** - Compare against `vad_threshold`
3. **State Machine** - Track speech start/end transitions
4. **Hysteresis** - Prevent rapid on/off toggling

### Configuring VAD Sensitivity

=== "Node.js"

    ```javascript
    // High sensitivity - detects quiet speech (may include noise)
    const highSensitivity = new StreamingAudioProcessor({
      enableVad: true, vadThreshold: 0.1
    });

    // Medium sensitivity - balanced (recommended)
    const balanced = new StreamingAudioProcessor({
      enableVad: true, vadThreshold: 0.3
    });

    // Low sensitivity - only detects clear, loud speech
    const lowSensitivity = new StreamingAudioProcessor({
      enableVad: true, vadThreshold: 0.6
    });
    ```

=== "Python"

    ```python
    # High sensitivity - detects quiet speech (may include noise)
    config = AudioStreamConfig(enable_vad=True, vad_threshold=0.1)

    # Medium sensitivity - balanced (recommended)
    config = AudioStreamConfig(enable_vad=True, vad_threshold=0.3)

    # Low sensitivity - only detects clear, loud speech
    config = AudioStreamConfig(enable_vad=True, vad_threshold=0.6)
    ```

=== "Rust"

    ```rust
    // High sensitivity - detects quiet speech (may include noise)
    let config = AudioStreamConfig::new()
        .enable_voice_detection(true)
        .vad_threshold(0.1);

    // Medium sensitivity - balanced (recommended)
    let config = AudioStreamConfig::new()
        .enable_voice_detection(true)
        .vad_threshold(0.3);

    // Low sensitivity - only detects clear, loud speech
    let config = AudioStreamConfig::new()
        .enable_voice_detection(true)
        .vad_threshold(0.6);
    ```

### Using VAD Results

```rust
while let Some(chunk) = audio_stream.next().await {
    match (chunk.voice_detected, &previous_state) {
        (true, false) => {
            println!("Speech started");
            speech_buffer.clear();
            speech_buffer.extend(&chunk.samples);
        }
        (true, true) => {
            // Ongoing speech
            speech_buffer.extend(&chunk.samples);
        }
        (false, true) => {
            println!("Speech ended, processing {} samples", speech_buffer.len());
            let result = process_speech(&speech_buffer).await?;
            println!("Transcription: {}", result);
        }
        (false, false) => {
            // Silence, do nothing
        }
    }
    previous_state = chunk.voice_detected;
}
```

---

## Ring Buffer Architecture

The streaming audio system uses a ring buffer for low-latency, lock-free audio processing.

```
+----------------------------------------------+
|  Ring Buffer (buffer_size samples)           |
|                                              |
|  [write_pos]                                 |
|       v                                      |
|  +--+--+--+--+--+--+--+--+--+--+--+--+    |
|  |##|##|##|##|  |  |  |  |##|##|##|##|    |
|  +--+--+--+--+--+--+--+--+--+--+--+--+    |
|                          ^                    |
|                     [read_pos]                |
|                                              |
|  ## = data available for processing          |
+----------------------------------------------+
```

**Benefits:**

- **Zero-copy processing** - Chunks reference buffer memory directly
- **Constant memory** - Fixed allocation regardless of stream duration
- **Lock-free** - Uses atomic operations for thread-safe read/write
- **Overflow handling** - Oldest data is overwritten when buffer is full

### Buffer Size Guidelines

| Use Case | Recommended Buffer Size | Latency |
|----------|------------------------|---------|
| Voice commands | 2048 samples | ~128ms at 16kHz |
| Transcription | 4096 samples | ~256ms at 16kHz |
| Music analysis | 8192 samples | ~512ms at 16kHz |
| Low-latency | 1024 samples | ~64ms at 16kHz |

---

## StreamingMetrics

Monitor the performance of the audio streaming pipeline.

=== "Node.js"

    ```javascript
    const metrics = await processor.metrics();
    console.log(`Avg latency: ${metrics.avgLatency}ms`);
    console.log(`Max latency: ${metrics.maxLatency}ms`);
    console.log(`Dropped frames: ${metrics.droppedFrames}`);
    console.log(`Buffer overruns: ${metrics.bufferOverruns}`);
    console.log(`Chunks processed: ${metrics.chunksProcessed}`);
    console.log(`Voice activity: ${(metrics.voiceActivityRatio * 100).toFixed(1)}%`);
    ```

=== "Python"

    ```python
    metrics = await processor.metrics()
    print(f"Avg latency: {metrics.avg_latency}")
    print(f"Max latency: {metrics.max_latency}")
    print(f"Dropped frames: {metrics.dropped_frames}")
    print(f"Buffer overruns: {metrics.buffer_overruns}")
    print(f"Chunks processed: {metrics.chunks_processed}")
    print(f"Voice activity: {metrics.voice_activity_ratio * 100:.1f}%")
    ```

=== "Rust"

    ```rust
    let metrics = processor.metrics().await;

    println!("Processing latency: {:?}", metrics.avg_latency);
    println!("Max latency: {:?}", metrics.max_latency);
    println!("Dropped frames: {}", metrics.dropped_frames);
    println!("Buffer overruns: {}", metrics.buffer_overruns);
    println!("Total chunks processed: {}", metrics.chunks_processed);
    println!("Voice activity ratio: {:.1}%", metrics.voice_activity_ratio * 100.0);
    println!("Signal-to-noise ratio: {:.1} dB", metrics.snr_db);
    ```

!!! warning "Dropped Frames"
    If `dropped_frames` is non-zero, your processing pipeline is too slow. Consider:

    - Increasing `buffer_size`
    - Reducing `sample_rate`
    - Simplifying your processing callback
    - Using a dedicated processing thread

---

## Device Enumeration and Selection

List and select specific audio input devices.

=== "Node.js"

    ```javascript
    const devices = processor.listInputDevices();
    devices.forEach(d => console.log(`Device: ${d}`));

    // Use a specific device
    const processor2 = new StreamingAudioProcessor({
      sampleRate: 16000,
      device: devices[0]
    });
    ```

=== "Python"

    ```python
    devices = processor.list_input_devices()
    for d in devices:
        print(f"Device: {d}")

    # Use a specific device
    config = AudioStreamConfig(sample_rate=16000, device=devices[0])
    processor2 = StreamingAudioProcessor(config)
    ```

=== "Rust"

    ```rust
    let devices = processor.list_input_devices()?;
    for device in &devices {
        println!("Available device: {}", device);
    }

    // Use a specific device
    let config = AudioStreamConfig::new()
        .sample_rate(16000)
        .device_preference(DevicePreference::ByName(devices[0].clone()));

    let mut processor = StreamingAudioProcessor::new(config)?;
    ```

---

## Noise Reduction

When enabled, the noise reduction pipeline processes audio before delivery.

```rust
let config = AudioStreamConfig::new()
    .sample_rate(16000)
    .channels(1)
    .enable_noise_reduction(true);
```

The pipeline includes:

1. **Spectral subtraction** - Estimates and removes stationary noise
2. **High-pass filtering** - Removes low-frequency rumble
3. **Noise gate** - Silences audio below a threshold

!!! tip "Performance Impact"
    Noise reduction adds approximately 2-5ms of latency per chunk. Disable it for ultra-low-latency applications where audio quality is already high.

---

## Integration with Multimodal Processor

Feed captured audio directly into the multimodal processing pipeline.

```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig, MultimodalProcessor};

let audio_config = AudioStreamConfig::new()
    .sample_rate(16000)
    .channels(1)
    .enable_voice_detection(true);

let mut audio_proc = StreamingAudioProcessor::new(audio_config)?;
let multimodal = MultimodalProcessor::new(model.clone())?;

audio_proc.initialize().await?;
let mut stream = audio_proc.start_capture().await?;

while let Some(chunk) = stream.next().await {
    if chunk.voice_detected {
        let audio_input = chunk.to_audio_input();
        let response = multimodal.process_audio(audio_input).await?;
        println!("Response: {}", response);
    }
}
```

---

## Complete Example: Voice Input

=== "Node.js"

    ```javascript
    const { StreamingAudioProcessor, loadModel } = require('mullama');

    async function main() {
      const model = await loadModel('whisper-model.gguf');

      const processor = new StreamingAudioProcessor({
        sampleRate: 16000,
        channels: 1,
        enableVad: true,
        enableNoiseReduction: true,
        vadThreshold: 0.25,
        bufferSize: 4096
      });

      await processor.initialize();
      console.log('Listening for voice commands...');

      let speechBuffer = [];
      let inSpeech = false;
      let silenceCount = 0;

      for await (const chunk of processor.startCapture()) {
        if (chunk.voiceDetected) {
          if (!inSpeech) console.log('[Listening...]');
          inSpeech = true;
          silenceCount = 0;
          speechBuffer.push(...chunk.samples);
        } else if (inSpeech) {
          silenceCount++;
          if (silenceCount >= 10) {
            inSpeech = false;
            const transcript = await model.transcribe(speechBuffer);
            console.log(`Command: ${transcript}`);
            if (transcript.includes('stop')) break;
            speechBuffer = [];
          }
        }
      }

      await processor.stopCapture();
    }

    main();
    ```

=== "Python"

    ```python
    from mullama import StreamingAudioProcessor, AudioStreamConfig, load_model

    async def main():
        model = await load_model("whisper-model.gguf")

        config = AudioStreamConfig(
            sample_rate=16000,
            channels=1,
            enable_vad=True,
            enable_noise_reduction=True,
            vad_threshold=0.25,
            buffer_size=4096
        )

        processor = StreamingAudioProcessor(config)
        await processor.initialize()
        print("Listening for voice commands...")

        speech_buffer = []
        in_speech = False
        silence_count = 0

        async for chunk in processor.start_capture():
            if chunk.voice_detected:
                if not in_speech:
                    print("[Listening...]")
                in_speech = True
                silence_count = 0
                speech_buffer.extend(chunk.samples)
            elif in_speech:
                silence_count += 1
                if silence_count >= 10:
                    in_speech = False
                    transcript = await model.transcribe(speech_buffer)
                    print(f"Command: {transcript}")
                    if "stop" in transcript:
                        break
                    speech_buffer = []

        await processor.stop_capture()

    import asyncio
    asyncio.run(main())
    ```

=== "Rust"

    ```rust
    use mullama::{
        StreamingAudioProcessor, AudioStreamConfig,
        Model, Context, ContextParams,
    };
    use std::sync::Arc;

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let model = Arc::new(Model::load("whisper-model.gguf")?);

        let config = AudioStreamConfig::new()
            .sample_rate(16000)
            .channels(1)
            .enable_voice_detection(true)
            .enable_noise_reduction(true)
            .vad_threshold(0.25)
            .buffer_size(4096);

        let mut processor = StreamingAudioProcessor::new(config)?;
        processor.initialize().await?;
        println!("Listening for voice commands...");

        let mut audio_stream = processor.start_capture().await?;
        let mut speech_buffer: Vec<f32> = Vec::new();
        let mut in_speech = false;
        let mut silence_counter = 0;

        while let Some(chunk) = audio_stream.next().await {
            if chunk.voice_detected {
                if !in_speech { println!("[Listening...]"); }
                in_speech = true;
                silence_counter = 0;
                speech_buffer.extend_from_slice(&chunk.samples);
            } else if in_speech {
                silence_counter += 1;
                if silence_counter >= 10 {
                    in_speech = false;
                    let model = model.clone();
                    let buffer = speech_buffer.clone();
                    let transcript = tokio::task::spawn_blocking(move || {
                        let mut ctx = Context::new(model, ContextParams::default()).unwrap();
                        ctx.process_audio(&buffer)
                    }).await??;
                    println!("Command: {}", transcript.trim());
                    if transcript.contains("stop") { break; }
                    speech_buffer.clear();
                }
            }
        }

        processor.stop_capture().await?;
        Ok(())
    }
    ```

---

## AudioChunk Reference

Each chunk delivered by the audio stream contains processed audio data and metadata.

```rust
pub struct AudioChunk {
    pub samples: Vec<f32>,       // Audio samples (normalized -1.0 to 1.0)
    pub channels: u16,           // Number of channels
    pub sample_rate: u32,        // Sample rate in Hz
    pub timestamp: Instant,      // When this chunk was captured
    pub duration: Duration,      // Duration of this chunk
    pub voice_detected: bool,    // VAD result
    pub signal_level: f32,       // RMS signal level (0.0 to 1.0)
}
```

---

## See Also

- [Format Conversion](format-conversion.md) - Audio format conversion
- [Multimodal Guide](../guide/multimodal.md) - Multimodal processing
- [WebSockets](websockets.md) - Audio streaming over WebSocket
