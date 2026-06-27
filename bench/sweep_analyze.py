#!/usr/bin/env python3
"""Render bench/sweep_results.jsonl as a markdown summary table.

Per model we emit one block with (sessions × mode) rows; each row shows
mullama and ollama side-by-side and the wall/throughput ratios. Aimed at
quick scanning rather than statistical rigor — the JSONL has the full
numbers if you want to crunch them yourself.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_block(model, rows):
    out = [f"\n### {model}", ""]
    out.append(
        "| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |"
    )
    out.append("|---:|---|---|---:|---:|---:|---:|---:|")
    # Group by (sessions, mode) so mullama+ollama land together.
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["sessions"], r["mode"])].append(r)
    # Sort by sessions then mode (stateless first).
    for key in sorted(by_key.keys(), key=lambda k: (k[0], 0 if k[1] == "stateless" else 1)):
        sess, mode = key
        for engine in ("mullama", "ollama"):
            r = next((x for x in by_key[key] if x["engine"] == engine), None)
            if not r:
                continue
            out.append(
                f"| {sess} | {mode} | {engine} | "
                f"{r['concurrent_total_wall_s']:.2f} | "
                f"{r['concurrent_agg_tok_s']:.1f} | "
                f"{r['baseline_1session_tok_s']:.1f} | "
                f"{r['throughput_scaling_vs_1session']:.2f}× | "
                f"{r['median_session_latency_inflation']:.2f}× |"
            )
        # Headline ratio line per (sessions,mode): mullama vs ollama.
        m = next((x for x in by_key[key] if x["engine"] == "mullama"), None)
        o = next((x for x in by_key[key] if x["engine"] == "ollama"), None)
        if m and o and o["concurrent_total_wall_s"] > 0:
            wall_ratio = o["concurrent_total_wall_s"] / m["concurrent_total_wall_s"]
            tok_ratio = (
                m["concurrent_agg_tok_s"] / o["concurrent_agg_tok_s"]
                if o["concurrent_agg_tok_s"] > 0
                else float("inf")
            )
            out.append(
                f"| | | **mullama / ollama** | "
                f"**{wall_ratio:.2f}× faster** | "
                f"**{tok_ratio:.2f}× more** | | | |"
            )
    return "\n".join(out)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/sweep_results.jsonl")
    if not src.exists():
        sys.exit(f"no results file: {src}")
    rows = load(src)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print("# Mullama vs ollama — sweep results")
    print()
    print(f"Source: `{src}` ({len(rows)} runs across {len(by_model)} models).")
    print()
    print(f"Bench: `concurrent_sessions.py`, 4 turns × 48 max_tokens per turn,")
    print(f"temperature=0. Sessions vary by row.")
    print()
    # Stable model ordering by size hint in the alias.
    model_order = sorted(by_model.keys(), key=lambda m: (
        m,  # alphabetical fallback
    ))
    # Better: by typical size ordering.
    size_hint = {"0.5": 0, "1b": 1, "1.5": 2, "3b": 3, "7b": 4, "8b": 5}
    def size_key(name):
        for s, idx in size_hint.items():
            if s in name.lower():
                return idx
        return 99
    model_order = sorted(by_model.keys(), key=lambda m: (size_key(m), m))
    for m in model_order:
        print(fmt_block(m, by_model[m]))

    # Final headline: best concurrency advantage per model.
    print("\n## Headline (best concurrency advantage per model)")
    print()
    print("| model | sessions | mode | mullama wall | ollama wall | speedup |")
    print("|---|---:|---|---:|---:|---:|")
    for m in model_order:
        best = None
        for key in [(s, mo) for s in (4, 8) for mo in ("stateless", "session")]:
            ms = [r for r in by_model[m] if r["sessions"] == key[0] and r["mode"] == key[1] and r["engine"] == "mullama"]
            os_ = [r for r in by_model[m] if r["sessions"] == key[0] and r["mode"] == key[1] and r["engine"] == "ollama"]
            if not ms or not os_:
                continue
            wm = ms[0]["concurrent_total_wall_s"]
            wo = os_[0]["concurrent_total_wall_s"]
            if wm == 0:
                continue
            ratio = wo / wm
            if best is None or ratio > best[0]:
                best = (ratio, key[0], key[1], wm, wo)
        if best:
            ratio, sess, mode, wm, wo = best
            print(f"| {m} | {sess} | {mode} | {wm:.2f}s | {wo:.2f}s | **{ratio:.2f}×** |")


if __name__ == "__main__":
    main()
