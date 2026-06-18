#!/usr/bin/env bash
# Bench setup: pull each GGUF once with mullama, then register the SAME file in
# ollama so both engines run byte-identical weights (strict parity).
#
# Prereqs:
#   - `cargo build --features daemon` (provides the `mullama` binary)
#   - `ollama` on PATH and `ollama serve` running (or start it: `ollama serve &`)
#
# Usage:
#   bash bench/setup.sh                 # set up all models
#   bash bench/setup.sh qwen2.5-0.5b    # set up one model
set -euo pipefail

# alias|hf-repo  (filename omitted -> mullama auto-detects Q4_K_M, case-insensitively)
MODELS=(
  "qwen2.5-0.5b|bartowski/Qwen2.5-0.5B-Instruct-GGUF"
  "llama3.2-1b|unsloth/Llama-3.2-1B-Instruct-GGUF"
  "qwen2.5-1.5b|bartowski/Qwen2.5-1.5B-Instruct-GGUF"
  "llama3.2-3b|unsloth/Llama-3.2-3B-Instruct-GGUF"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${1:-}"

# Locate the mullama binary: prefer a built target, fall back to `cargo run`.
MULLAMA_BIN="$(command -v mullama || true)"
if [[ -z "$MULLAMA_BIN" ]]; then
  if [[ -x "target/debug/mullama" ]]; then
    MULLAMA_BIN="target/debug/mullama"
  elif [[ -x "target/release/mullama" ]]; then
    MULLAMA_BIN="target/release/mullama"
  else
    MULLAMA_BIN="cargo run --features daemon --bin mullama --"
  fi
fi
echo "using mullama: $MULLAMA_BIN"

require_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "error: ollama not found on PATH" >&2; exit 1
  fi
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "error: ollama server not reachable at http://127.0.0.1:11434" >&2
    echo "       start it with: ollama serve &" >&2; exit 1
  fi
}
require_ollama

pull_path() {
  # Pull (idempotent: re-uses cache) and parse the "  Path: <...>" line.
  local spec="hf:$2"
  local out
  out="$($MULLAMA_BIN pull "$spec" 2>&1)" || {
    echo "mullama pull failed for $spec:" >&2; echo "$out" >&2; exit 1;
  }
  echo "$out" | grep -E '^  Path:' | awk '{print $2}'
}

# Per-architecture ollama TEMPLATE directive.
#
# A bare `FROM <gguf>` Modelfile makes ollama fall back to the passthrough
# template `{{ .Prompt }}` — ollama does NOT read the GGUF's embedded
# tokenizer.chat_template. That breaks chat parity: mullama applies the
# embedded chatml/llama3 template (via llama.cpp's llama_chat_apply_template)
# while ollama sends the raw prompt with no role markers, so the two engines
# see different bytes and diverge from the first token. Writing the matching
# Go template here makes ollama apply the same formatting mullama does.
template_for() {
  case "$1" in
    qwen2.5-*)
      # Qwen2.5 chatml — matches llama.cpp's qwen jinja template for the
      # common (no-tools, single-turn) case.
      cat <<'TPL'
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
TPL
      ;;
    llama3.2-*)
      # Llama-3.2 — matches llama.cpp's llama3 jinja template (ollama adds
      # the BOS token itself, as does mullama via add_bos_token metadata).
      cat <<'TPL'
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
TPL
      ;;
    *)
      # Unknown arch: leave ollama's default (passthrough). Chat parity for
      # such models is expected to fail and will be logged by the bench.
      return 1
      ;;
  esac
}

setup_one() {
  local alias="$1" repo="$2"
  echo "==> $alias  ($repo)"

  local path
  path="$(pull_path "$alias" "$repo")"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "error: could not resolve local GGUF path for $alias" >&2
    echo "       pull output did not contain a valid '  Path:' line" >&2
    exit 1
  fi
  echo "    gguf: $path"

  # Generate an ollama Modelfile FROM the exact same file, plus the
  # architecture-matching TEMPLATE so chat parity is meaningful.
  local mf="$SCRIPT_DIR/Modelfile.$alias"
  {
    printf 'FROM %s\n' "$path"
    if template_for "$alias" >"$mf.tpl" 2>/dev/null; then
      cat "$mf.tpl"
      rm -f "$mf.tpl"
    fi
  } > "$mf"
  echo "    modelfile: $mf"

  # (Re)create the ollama model from that file.
  ollama create "$alias" -f "$mf" >/dev/null
  echo "    ollama:   created $alias"

  # Tell the user how to load it in mullama.
  echo "    to serve in mullama:"
  echo "      $MULLAMA_BIN serve --model $alias:$path"
  echo
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r alias repo <<< "$entry"
  if [[ -n "$ONLY" && "$alias" != "$ONLY" ]]; then
    continue
  fi
  setup_one "$alias" "$repo"
done

echo "setup complete."
echo "next:"
echo "  1) start mullama daemon with the models, e.g."
echo "       $MULLAMA_BIN serve --model qwen2.5-0.5b:<path> --model llama3.2-1b:<path> ..."
echo "  2) run the bench:"
echo "       cargo run --features daemon --bin mullama-bench -- --models qwen2.5-0.5b,llama3.2-1b,qwen2.5-1.5b,llama3.2-3b --mode both --report report.json"