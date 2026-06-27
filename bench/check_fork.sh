#!/usr/bin/env bash
# Assert the llama.cpp submodule still points at the cognisoc fork and matches
# the parent repo's pinned SHA. Run before a build, or wire into CI.
#
# Background: mullama pins `cognisoc/llama.cpp` (branch `mullama-parity`), which
# carries the "Align native core with Ollama 0.24.0" patch. A stray
# `git submodule update --remote` or a manual `git checkout` against upstream
# silently drops that patch. This guard catches both.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

expected_url="https://github.com/cognisoc/llama.cpp.git"
actual_url="$(git -C llama.cpp remote get-url origin 2>/dev/null || true)"
if [[ "$actual_url" != "$expected_url" ]]; then
  echo "error: llama.cpp submodule remote is '$actual_url'" >&2
  echo "       expected '$expected_url'" >&2
  echo "       run: git submodule sync llama.cpp" >&2
  exit 1
fi

pinned="$(git ls-tree HEAD llama.cpp | awk '{print $3}')"
actual="$(git -C llama.cpp rev-parse HEAD)"
if [[ "$pinned" != "$actual" ]]; then
  echo "error: llama.cpp submodule HEAD is $actual" >&2
  echo "       parent repo pins        $pinned" >&2
  echo "       run: git submodule update --init llama.cpp" >&2
  exit 1
fi

echo "ok: llama.cpp at cognisoc fork, pinned commit $actual"
