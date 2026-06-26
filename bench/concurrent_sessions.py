#!/usr/bin/env python3
"""Concurrent multi-session throughput benchmark.

Fires K independent sessions at a mullama daemon in parallel, each running the
same fixed agent loop (distinct session ids, so distinct pinned KV slots), and
measures aggregate throughput. Run against a daemon started with different
--context-pool-size values to see whether throughput scales with pool size —
i.e. whether sessions genuinely decode in parallel or serialize on a shared
resource.

The honest question this answers: does pool-size P > 1 actually serve P
sessions concurrently, or is it single-flight under the hood?

Usage:
  python3 bench/concurrent_sessions.py --url http://127.0.0.1:8110 \
      --model qwen2.5-0.5b --sessions 4 --turns 4 --max-tokens 48
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request


TURNS = [
    "List the files in a typical Rust project and what each does.",
    "Which of those files would hold the HTTP routing logic?",
    "How would you add a new endpoint there?",
    "What tests would you write for it?",
    "Summarize the change as a commit message.",
    "Now describe the rollback plan in two sentences.",
]


def post(url, body, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_session(url, model, sess_id, turns, max_tokens, use_session):
    """Run one multi-turn session; return (completion_tokens, prompt_eval_ns,
    eval_ns, wall_secs)."""
    history = []
    total_completion = 0
    total_prompt_eval_ns = 0
    total_eval_ns = 0
    t0 = time.perf_counter()
    for i in range(turns):
        history.append({"role": "user", "content": TURNS[i % len(TURNS)]})
        body = {
            "model": model,
            "messages": history,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if use_session:
            body["session"] = sess_id
        v = post(f"{url}/v1/chat/completions", body)
        msg = v["choices"][0]["message"]["content"]
        usage = v.get("usage", {})
        total_completion += usage.get("completion_tokens", 0)
        tim = v.get("timings", {}) or {}
        total_prompt_eval_ns += tim.get("prompt_eval_ns", 0) or 0
        total_eval_ns += tim.get("eval_ns", 0) or 0
        history.append({"role": "assistant", "content": msg})
    wall = time.perf_counter() - t0
    return (total_completion, total_prompt_eval_ns, total_eval_ns, wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8110")
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--no-session", action="store_true",
                    help="disable KV reuse (stateless)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    use_session = not args.no_session

    # Warm the model (first request pays load/JIT) with a throwaway call.
    try:
        post(f"{args.url}/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "temperature": 0, "stream": False,
        })
    except Exception as e:
        print(f"warm-up failed: {e}")

    # 1) Single-session baseline: one session alone, no contention.
    base = run_session(args.url, args.model, "baseline-0", args.turns,
                       args.max_tokens, use_session)
    base_wall = base[3]
    base_tok = base[0]
    base_tok_s = base_tok / base_wall if base_wall > 0 else 0

    # 2) S sessions concurrently.
    t0 = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as ex:
        futs = [
            ex.submit(run_session, args.url, args.model, f"concsess-{k}",
                      args.turns, args.max_tokens, use_session)
            for k in range(args.sessions)
        ]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total_wall = time.perf_counter() - t0

    total_completion = sum(r[0] for r in results)
    per_session_walls = sorted(r[3] for r in results)
    agg_tok_s = total_completion / total_wall if total_wall > 0 else 0

    # Throughput scaling vs a single session: how many sessions' worth of
    # tokens/sec we actually get. ~1.0 = fully serialized, ~S = perfectly
    # parallel. This is the clean metric (no lock-wait contamination).
    throughput_scaling = agg_tok_s / base_tok_s if base_tok_s > 0 else 0
    # Latency cost: how much slower a session is under contention vs alone.
    median_wall = per_session_walls[len(per_session_walls) // 2]
    latency_inflation = median_wall / base_wall if base_wall > 0 else 0

    label = args.label or ("session" if use_session else "stateless")
    print(json.dumps({
        "label": label,
        "sessions": args.sessions,
        "turns": args.turns,
        "max_tokens": args.max_tokens,
        "baseline_1session_tok_s": round(base_tok_s, 2),
        "baseline_1session_wall_s": round(base_wall, 3),
        "concurrent_total_wall_s": round(total_wall, 3),
        "concurrent_agg_tok_s": round(agg_tok_s, 2),
        "throughput_scaling_vs_1session": round(throughput_scaling, 2),
        "median_session_latency_inflation": round(latency_inflation, 2),
    }))


if __name__ == "__main__":
    main()
