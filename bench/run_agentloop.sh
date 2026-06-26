#!/usr/bin/env bash
#
# Agent-loop benchmark: cross-turn KV reuse vs stateless baseline.
#
# Starts a mullama daemon, runs the agent-loop bench twice (with and without
# session KV reuse), and writes two report JSONs. Computes the per-turn prefill
# speedup so you can see the O(history) -> O(delta) collapse.
#
# Usage:
#   MODEL_BLOB=/path/to/blob bench/run_agentloop.sh
# or set MODEL to an alias the daemon can resolve.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Model: a GGUF blob path or "alias:path". Override via $MODEL.
MODEL="${MODEL:-qwen2.5-0.5b:${MODEL_BLOB:?set MODEL or MODEL_BLOB}}"
PORT="${PORT:-18110}"
MAXTOK="${MAXTOK:-64}"
# Optional: Ollama-matched backend for exact numerics.
BACKEND_ENV=()
if [[ -n "${GGML_BACKEND_PATH:-}" ]]; then
  BACKEND_ENV=(env "GGML_BACKEND_PATH=$GGML_BACKEND_PATH")
fi

pkill -f "mullama serve.*--http-port $PORT" 2>/dev/null || true
sleep 1

"${BACKEND_ENV[@]}" target/release/mullama serve --model "$MODEL" \
  --http-port "$PORT" -c 8192 --context-pool-size 2 -t 8 --batch-size 512 \
  >/tmp/agentloop_srv.log 2>&1 &
DPID=$!
trap 'kill "$DPID" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
  curl -fsS "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 && break
  sleep 1
done

echo "### WITH KV REUSE ###"
target/release/mullama-bench \
  --mullama-url "http://127.0.0.1:$PORT" --models qwen2.5-0.5b \
  --mode agent-loop --trace-file bench/trace.jsonl \
  --agent-max-tokens "$MAXTOK" --temperature 0 \
  --report /tmp/agentloop_reuse.json

echo "### BASELINE (--no-kv-reuse) ###"
target/release/mullama-bench \
  --mullama-url "http://127.0.0.1:$PORT" --models qwen2.5-0.5b \
  --mode agent-loop --trace-file bench/trace.jsonl \
  --agent-max-tokens "$MAXTOK" --temperature 0 --no-kv-reuse \
  --report /tmp/agentloop_noreuse.json

echo
echo "### PER-TURN PREFILL SPEEDUP (reuse vs baseline) ###"
python3 - <<'PY'
import json
def recs(d):
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and 'turn' in v[0]:
            return v
    return []
r = {(x['trace_id'], x['turn']): x for x in recs(json.load(open('/tmp/agentloop_reuse.json')))}
n = {(x['trace_id'], x['turn']): x for x in recs(json.load(open('/tmp/agentloop_noreuse.json')))}
print(f"{'trace':<12}{'turn':>5}{'p_toks':>8}{'reuse_ms':>10}{'base_ms':>10}{'speedup':>9}")
for k in sorted(r):
    a, b = r[k], n.get(k)
    if not b: continue
    rp = (a.get('prompt_eval_ns') or 0) / 1e6
    bp = (b.get('prompt_eval_ns') or 0) / 1e6
    sp = bp / rp if rp > 0 else 0
    print(f"{k[0]:<12}{k[1]:>5}{a['prompt_tokens']:>8}{rp:>10.1f}{bp:>10.1f}{sp:>8.1f}x")
PY
