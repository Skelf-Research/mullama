# Multi-stage build for Mullama
# Stage 1: Build
FROM rust:1.75-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev libpulse-dev libflac-dev libvorbis-dev libopus-dev \
    libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libclang-dev cmake pkg-config git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY . .

# Initialize submodules
RUN git submodule update --init --recursive || true

# Build release binary with daemon feature
RUN cargo build --release --features daemon

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libasound2 libpulse0 libflac12 libvorbis0a libopus0 \
    libpng16-16 libjpeg62-turbo libtiff6 libwebp7 \
    ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash mullama

# Copy binary from builder
COPY --from=builder /app/target/release/mullama /usr/local/bin/mullama

# Copy model configs
COPY --from=builder /app/configs /etc/mullama/configs

# Create model storage directory
RUN mkdir -p /models && chown mullama:mullama /models

USER mullama
WORKDIR /home/mullama

# Expose HTTP API port
EXPOSE 8080

# Health check against the OpenAI-compatible API
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/v1/models || exit 1

# Default: start the daemon server
ENTRYPOINT ["mullama"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
