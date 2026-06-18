#!/usr/bin/env python3
"""Quick engine-throughput A/B: mullama daemon vs ollama, same GGUF, greedy+raw.

Measures engine tok/s = completion_tokens / eval_ns (both engines' internal
timings), which is the apples-to-apples compute number (no HTTP/IPC overhead).
Run with the mullama daemon already up on --mullama-url.
"""
import json, sys, time, urllib.request

MULLAMA = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
OLLAMA  = "http://127.0.0.1:11434"
MODEL   = "qwen2.5-0.5b"
# A prompt that yields a long, steady generation (~256 tokens) for amortized perf.
PROMPT  = "Write a long, detailed Python tutorial covering functions, classes, and modules with many examples."
MAXTOK  = 256

def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def mullama_run():
    # /v1/completions (raw) — server applies NO chat template to raw prompts,
    # matching ollama /api/generate raw:true.
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": MAXTOK,
            "temperature": 0.0, "stream": False}
    r = post(f"{MULLAMA}/v1/completions", body)
    usage = r.get("usage", {})
    t = r.get("timings") or {}
    ct = usage.get("completion_tokens") or t.get("completion_tokens") or 0
    eval_ns = t.get("eval_ns")
    return ct, eval_ns, r

def ollama_run():
    body = {"model": MODEL, "prompt": PROMPT, "stream": False, "raw": True,
            "options": {"temperature": 0.0, "num_predict": MAXTOK}}
    r = post(f"{OLLAMA}/api/generate", body)
    ct = r.get("eval_count", 0)
    eval_ns = r.get("eval_duration")  # ns
    return ct, eval_ns, r

print("=== warmup (1 run each, discarded) ===")
try: mullama_run()
except Exception as e: print("mullama warmup err:", e)
try: ollama_run()
except Exception as e: print("ollama warmup err:", e)

print(f"=== measured (engine tok/s, {MAXTOK} max tokens, temp=0) ===")
mc, mn, mr = mullama_run()
oc, on, _   = ollama_run()
m_tps = (mc / mn * 1e9) if mn else 0.0
o_tps = (oc / on * 1e9) if on else 0.0
print(f"mullama: {mc} tok / {mn/1e6:.1f} ms = {m_tps:.1f} tok/s")
print(f"ollama:  {oc} tok / {on/1e6:.1f} ms = {o_tps:.1f} tok/s")
ratio = m_tps / o_tps if o_tps else 0
print(f"mullama/ollama = {ratio:.3f}x  (gap {1/ratio:.2f}x)" if ratio else "n/a")