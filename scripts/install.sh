#!/bin/sh
# Mullama installer script for Linux and macOS
# Usage: curl -fsSL https://skelfresearch.com/mullama/install.sh | sh

set -e

# Configuration
REPO="skelf-research/mullama"
BINARY_NAME="mullama"
INSTALL_DIR="${MULLAMA_INSTALL_DIR:-$HOME/.local/bin}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

success() {
    printf "${GREEN}[OK]${NC} %s\n" "$1"
}

warn() {
    printf "${YELLOW}[WARN]${NC} %s\n" "$1"
}

error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1" >&2
    exit 1
}

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            OS="linux"
            ;;
        Darwin)
            OS="darwin"
            ;;
        *)
            error "Unsupported operating system: $OS"
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)
            ARCH="x64"
            ;;
        aarch64|arm64)
            ARCH="arm64"
            ;;
        *)
            error "Unsupported architecture: $ARCH"
            ;;
    esac

    PLATFORM="${OS}-${ARCH}"
    info "Detected platform: $PLATFORM"
}

# Check for CUDA support on Linux
detect_gpu() {
    GPU_VARIANT=""

    if [ "$OS" = "linux" ] && [ "$ARCH" = "x64" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                info "NVIDIA GPU detected"
                printf "Would you like to install the CUDA-enabled version? [y/N] "
                read -r response
                case "$response" in
                    [yY][eE][sS]|[yY])
                        GPU_VARIANT="-cuda"
                        info "Installing CUDA version"
                        ;;
                    *)
                        info "Installing CPU version"
                        ;;
                esac
            fi
        fi
    fi
}

# Get the latest release version
get_latest_version() {
    if command -v curl >/dev/null 2>&1; then
        VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    elif command -v wget >/dev/null 2>&1; then
        VERSION=$(wget -qO- "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    else
        error "Neither curl nor wget found. Please install one of them."
    fi

    if [ -z "$VERSION" ]; then
        error "Failed to get latest version"
    fi

    info "Latest version: v$VERSION"
}

# Download and install
download_and_install() {
    FILENAME="${BINARY_NAME}-${PLATFORM}${GPU_VARIANT}.tar.gz"
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/v$VERSION/$FILENAME"

    info "Downloading $FILENAME..."

    # Create temp directory
    TMP_DIR=$(mktemp -d)
    trap "rm -rf $TMP_DIR" EXIT

    # Download
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$FILENAME" || error "Download failed"
    else
        wget -q "$DOWNLOAD_URL" -O "$TMP_DIR/$FILENAME" || error "Download failed"
    fi

    success "Downloaded successfully"

    # Extract
    info "Extracting..."
    tar -xzf "$TMP_DIR/$FILENAME" -C "$TMP_DIR"

    # Create install directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"

    # Install
    info "Installing to $INSTALL_DIR..."
    mv "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"

    success "Installed $BINARY_NAME to $INSTALL_DIR"
}

# Update PATH if necessary
update_path() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*)
            # Already in PATH
            ;;
        *)
            warn "$INSTALL_DIR is not in your PATH"

            # Detect shell and suggest appropriate config file
            SHELL_NAME=$(basename "$SHELL")
            case "$SHELL_NAME" in
                bash)
                    SHELL_CONFIG="$HOME/.bashrc"
                    ;;
                zsh)
                    SHELL_CONFIG="$HOME/.zshrc"
                    ;;
                fish)
                    SHELL_CONFIG="$HOME/.config/fish/config.fish"
                    ;;
                *)
                    SHELL_CONFIG="$HOME/.profile"
                    ;;
            esac

            echo ""
            echo "Add the following to $SHELL_CONFIG:"
            echo ""
            if [ "$SHELL_NAME" = "fish" ]; then
                echo "  set -gx PATH \$PATH $INSTALL_DIR"
            else
                echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
            fi
            echo ""
            echo "Then run: source $SHELL_CONFIG"
            echo ""
            ;;
    esac
}

# Verify installation
verify_installation() {
    if [ -x "$INSTALL_DIR/$BINARY_NAME" ]; then
        VERSION_OUTPUT=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null || echo "unknown")
        success "Installation verified: $VERSION_OUTPUT"
    else
        error "Installation verification failed"
    fi
}

# Print post-install message
print_success() {
    echo ""
    echo "============================================"
    printf "${GREEN}Mullama installed successfully!${NC}\n"
    echo "============================================"
    echo ""
    echo "Quick start:"
    echo "  mullama run llama3.2:1b    # Run a model"
    echo "  mullama serve              # Start the daemon"
    echo "  mullama chat               # Interactive chat"
    echo ""
    echo "For more information, visit:"
    echo "  https://github.com/$REPO"
    echo ""
}

# Main
main() {
    echo ""
    echo "================================"
    echo "   Mullama Installer"
    echo "================================"
    echo ""

    detect_platform
    detect_gpu
    get_latest_version
    download_and_install
    update_path
    verify_installation
    print_success
}

main "$@"
