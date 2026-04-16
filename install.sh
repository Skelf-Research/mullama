#!/bin/sh
# Mullama installer
# Usage: curl -fsSL https://mullama.cognisoc.com/install.sh | sh
#
# Drop-in Ollama replacement. All-in-one LLM toolkit.
# https://github.com/cognisoc/mullama

set -e

# Delegate to the full installer script
exec curl -fsSL https://raw.githubusercontent.com/cognisoc/mullama/main/scripts/install.sh | sh
