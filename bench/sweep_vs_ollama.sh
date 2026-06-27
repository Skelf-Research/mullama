#!/usr/bin/env bash
# Full-sweep bench: mullama Phase C (default-on, current build) vs ollama.
#
# For each model in the sweep, starts a mullama daemon, runs scenarios
# {1,4,8 sessions} × {stateless, session-pinned}, kills the daemon, runs
# the same scenarios against ollama. Writes one JSON line per run to
# bench/sweep_results.jsonl plus a markdown summary to bench/sweep_summary.md.
#
# Models are taken from bench/models.toml conventions; their GGUF paths are
# auto-detected via mullama pull's cache layout.

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Where to drop results.
RESULTS="$REPO/bench/sweep_results.jsonl"
SUMMARY="$REPO/bench/sweep_summary.md"
: > "$RESULTS"

MODELS=(
  "qwen2.5-0.5b|bartowski--Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
  "llama3.2-1b|unsloth--Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  "qwen2.5-1.5b|bartowski--Qwen2.5-1.5B-Instruct-GGUF/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
  "llama3.2-3b|unsloth--Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
CACHE="$HOME/Library/Caches/mullama/models"

SESSIONS=(4 8)
MAX_TOKENS=48
TURNS=4

wait_ready() {
  local url=$1; local tries=180
  while ! curl -fsS "$url/api/version" >/dev/null 2>&1; do
    tries=$((tries-1)); [ $tries -le 0 ] && return 1
    sleep 1
  done
  return 0
}

run_scenario() {
  local engine=$1 url=$2 model=$3 sessions=$4 mode=$5
  local extra=""
  if [ "$mode" = "stateless" ]; then extra="--no-session"; fi
  local label="${engine}_${model}_${mode}_s${sessions}"
  local line
  line=$(python3 "$REPO/bench/concurrent_sessions.py" --url "$url" --model "$model" --sessions "$sessions" --turns "$TURNS" --max-tokens "$MAX_TOKENS" $extra --label "$label" 2>&1 | tail -1)
  # Stamp engine/model/mode into the JSON for easy parsing.
  echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); d['engine']='$engine'; d['model']='$model'; d['mode']='$mode'; print(json.dumps(d))" >> "$RESULTS"
  # Cosmetic console echo — the JSONL is the source of truth.
  echo "  [$engine $model $mode sess=$sessions] $line" \
    | python3 -c 'import sys,json,re;
line = sys.stdin.read().strip();
m = re.search(r"({.*})", line);
d = json.loads(m.group(1));
print(f"  [{d[\"engine\"]:<7} {d[\"model\"]:<15} {d[\"mode\"]:<9} sess={d[\"sessions\"]:<2}] wall={d[\"concurrent_total_wall_s\"]:6.2f} agg_tok/s={d[\"concurrent_agg_tok_s\"]:6.1f} scaling={d[\"throughput_scaling_vs_1session\"]:5.2f}x infl={d[\"median_session_latency_inflation\"]:5.2f}x")
' || true
}

bench_one_model() {
  local alias=$1 path=$2
  echo
  echo "### $alias ###"

  # Warm ollama with this model (load it into VRAM).
  curl -fsS http://127.0.0.1:11434/api/generate -d "{\"model\":\"$alias\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" >/dev/null 2>&1 || true

  # --- mullama side ---
  pkill -f "mullama serve.*--http-port 18110" 2>/dev/null || true; sleep 1
  "$REPO/target/release/mullama" serve --model "$alias:$path" \
    --http-port 18110 -c 8192 -t 8 --batch-size 2048 --hydration off \
    >/tmp/sweep_srv.log 2>&1 &
  local DPID=$!
  if ! wait_ready http://127.0.0.1:18110; then
    echo "  mullama failed to start; skipping"
    return
  fi
  # Warmup
  curl -fsS -X POST http://127.0.0.1:18110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$alias"'","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1
  echo "  mullama:"
  for s in "${SESSIONS[@]}"; do
    run_scenario mullama http://127.0.0.1:18110 "$alias" "$s" stateless
    run_scenario mullama http://127.0.0.1:18110 "$alias" "$s" session
  done
  kill "$DPID" 2>/dev/null || true
  pkill -f "mullama serve.*--http-port 18110" 2>/dev/null || true
  sleep 1

  # --- ollama side ---
  echo "  ollama:"
  for s in "${SESSIONS[@]}"; do
    run_scenario ollama http://127.0.0.1:11434 "$alias" "$s" stateless
    run_scenario ollama http://127.0.0.1:11434 "$alias" "$s" session
  done
}

main() {
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r alias subpath <<<"$entry"
    local path="$CACHE/$subpath"
    if [ ! -f "$path" ]; then
      echo "skipping $alias: $path not found (run bench/setup.sh $alias first)"
      continue
    fi
    bench_one_model "$alias" "$path"
  done

  echo
  echo "=== Done — $(wc -l < "$RESULTS" | tr -d ' ') runs written to $RESULTS ==="
}

main "$@"
