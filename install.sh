#!/bin/sh
# Mullama installer
# Usage: curl -fsSL https://skelfresearch.com/mullama/install.sh | sh
#
# Drop-in Ollama replacement. All-in-one LLM toolkit.
# https://github.com/skelf-research/mullama

set -e

# Delegate to the full installer script
exec curl -fsSL https://raw.githubusercontent.com/skelf-research/mullama/main/scripts/install.sh | sh
