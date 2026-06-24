//! Benchmark and parity testing vs ollama.
//!
//! Runs a fixed prompt suite through both a mullama daemon (OpenAI-compatible
//! HTTP API) and an ollama server on the *same* GGUF files, then reports:
//!
//! - **Parity**: at `temperature=0` (greedy) the two llama.cpp engines should
//!   produce identical token sequences given identical weights + input bytes.
//!   We compare `/v1/completions` (raw prompt, identical bytes) and
//!   `/v1/chat/completions` (real chat template) on both engines, plus ollama's
//!   native `/api/generate` / `/api/chat` as a cross-check.
//! - **Perf**: engine tokens/sec (server-side timings), wall tokens/sec, and
//!   latency. mullama timings come from the `timings` extension field added to
//!   the daemon; ollama timings come from native `/api/generate`.
//!
//! Every parity failure or perf gap is a candidate mullama bug to diagnose and
//! fix, then re-run.
//!
//! # Usage
//! ```text
//! mullama-bench \
//!   --mullama-url http://127.0.0.1:8080 \
//!   --ollama-url http://127.0.0.1:11434 \
//!   --models qwen2.5-0.5b,llama3.2-1b,qwen2.5-1.5b,llama3.2-3b \
//!   --prompt-file bench/prompts.jsonl \
//!   --runs 3 --warmup 1 --max-tokens 128 --temperature 0.0 \
//!   --mode both --report report.json
//! ```

use std::collections::BTreeMap;
use std::time::Instant;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Parity,
    Perf,
    Both,
    /// Replays a multi-turn agent trace and reports per-turn prefill/decode
    /// timings. This is the verification harness for the cross-turn KV
    /// reuse work: the win shows up as prompt_eval_ns collapsing on turn 2+
    /// (only the delta is prefilled) instead of staying flat at the full-history
    /// cost (the stock behaviour of re-prefilling every turn).
    AgentLoop,
}

impl Mode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "parity" => Ok(Mode::Parity),
            "perf" => Ok(Mode::Perf),
            "both" => Ok(Mode::Both),
            "agent-loop" => Ok(Mode::AgentLoop),
            other => Err(format!("invalid mode '{}' (parity|perf|both|agent-loop)", other)),
        }
    }
}

#[derive(Parser, Debug, Serialize)]
#[command(
    name = "mullama-bench",
    version,
    about = "Benchmark & parity test mullama vs ollama"
)]
struct Args {
    /// mullama daemon OpenAI base URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    mullama_url: String,

    /// ollama base URL.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_url: String,

    /// Comma-separated model aliases (must be loaded in mullama and present in ollama).
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,

    /// Path to a JSONL prompt file (one {"id","prompt"} object per line).
    #[arg(long, default_value = "bench/prompts.jsonl")]
    prompt_file: String,

    /// Measured runs per (model, prompt) after warmup.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Warmup runs to discard (avoids cold-load bias).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Max tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,

    /// Sampling temperature. Use 0.0 for strict parity (greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// RNG seed (for reproducible non-greedy runs; ignored at temp=0).
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Benchmark mode: parity, perf, or both.
    #[arg(long, default_value = "both")]
    mode: String,

    /// Write a full JSON report to this path.
    #[arg(long, default_value = "report.json")]
    report: String,

    /// Also hit ollama native /api/generate + /api/chat as a parity cross-check.
    #[arg(long, default_value_t = true)]
    native_crosscheck: bool,

    /// Return success even when strict parity comparisons differ.
    #[arg(long, default_value_t = false)]
    allow_parity_diffs: bool,

    /// Path to a JSONL agent-trace file for `agent-loop` mode. Each line is one
    /// trace: `{"id","turns":["user msg 1","user msg 2",...]}`. The bench replays
    /// the trace by sending the user turns incrementally, feeding the model's
    /// own assistant replies back into the history — a real agent loop.
    #[arg(long, default_value = "bench/trace.jsonl")]
    trace_file: String,

    /// Limit each agent trace to the first N user turns (0 = use the whole trace).
    #[arg(long, default_value_t = 0)]
    turns: usize,

    /// Max tokens generated per agent turn (agents emit short turns; keep this
    /// small so decode stays a minor slice and prefill dominates — the slice
    /// whose collapse proves the KV-reuse thesis).
    #[arg(long, default_value_t = 64)]
    agent_max_tokens: u32,

    /// Disable cross-turn KV reuse in agent-loop mode (send no `session` id).
    /// Use to capture the stock baseline (every turn re-prefills the full
    /// history) for comparison against the reuse-enabled run.
    #[arg(long, default_value_t = false)]
    no_kv_reuse: bool,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Prompt {
    id: String,
    prompt: String,
}

/// A multi-turn agent trace: an ordered list of user turns. The bench replays it
/// as a real agent loop — each user turn is appended to the running history, the
/// model generates an assistant reply, and that reply is fed back as context for
/// the next turn. With cross-turn KV reuse, turn N prefills only `user_turn_N`;
/// without it, turn N re-prefills the whole accumulated history.
#[derive(Debug, Clone, Deserialize)]
struct Trace {
    id: String,
    turns: Vec<String>,
}

/// One row of an agent-loop replay: per-turn prefill (prompt_eval_ns) and
/// decode (eval_ns) timings from the daemon's `timings` extension, plus the
/// prompt/completion token counts and client wall-clock. The KV-reuse win is
/// read off `prompt_eval_ns`: it should collapse on turn 2+ vs turn 1.
#[derive(Debug, Clone, Serialize)]
struct AgentLoopRecord {
    model: String,
    trace_id: String,
    turn: usize,
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_eval_ns: Option<u64>,
    eval_ns: Option<u64>,
    wall_secs: f64,
    /// The generated assistant text for this turn. Used to verify that the
    /// KV-reuse path is numerically identical to the stateless path: the same
    /// trace run with and without a session must produce identical per-turn
    /// text (same weights, same positions => same greedy tokens).
    text: String,
}

/// A single measured sample from one engine.
#[derive(Debug, Clone, Serialize)]
struct Sample {
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    /// Server-side prompt decode time (ns), if reported.
    prompt_eval_ns: Option<u64>,
    /// Server-side generation time (ns), if reported.
    eval_ns: Option<u64>,
    /// Client wall-clock seconds for the whole request.
    wall_secs: f64,
}

impl Sample {
    fn engine_tok_s(&self) -> Option<f64> {
        Some(self.completion_tokens as f64 / (self.eval_ns? as f64 / 1e9))
    }
    fn wall_tok_s(&self) -> f64 {
        if self.wall_secs > 0.0 {
            self.completion_tokens as f64 / self.wall_secs
        } else {
            0.0
        }
    }
}

/// Parity comparison between mullama and ollama for one (model, prompt, endpoint).
#[derive(Debug, Clone, Serialize)]
struct ParityRecord {
    model: String,
    prompt_id: String,
    endpoint: String, // "completions" | "chat"
    mullama_text: String,
    ollama_text: String,
    text_match: bool,
    token_match: bool,
    mullama_completion_tokens: u32,
    ollama_completion_tokens: u32,
    first_diff_char: Option<usize>,
    /// First token position (0-based, into the tokenized outputs) where the two
    /// engines diverge. `None` means one output's token stream is a prefix of
    /// the other's (i.e. no sampling divergence — only a length/truncation
    /// difference). Both engines' texts are tokenized with mullama's loaded
    /// model tokenizer (same GGUF), so the comparison is apples-to-apples.
    first_diff_token: Option<usize>,
}

/// Aggregated perf for one (model, endpoint, engine).
#[derive(Debug, Clone, Serialize)]
struct PerfRecord {
    model: String,
    endpoint: String,
    engine: String, // "mullama" | "ollama"
    runs: usize,
    engine_tok_s_mean: f64,
    engine_tok_s_p50: f64,
    wall_tok_s_mean: f64,
    wall_secs_mean: f64,
    completion_tokens_mean: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    config: Value,
    runtime: Value,
    parity: Vec<ParityRecord>,
    perf: Vec<PerfRecord>,
    agent_loop: Vec<AgentLoopRecord>,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async fn post_json(client: &reqwest::Client, url: &str, body: Value) -> Result<Value, String> {
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("POST {} failed: {}", url, e))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("read body {} failed: {}", url, e))?;
    if !status.is_success() {
        return Err(format!("POST {} -> {}: {}", url, status, text));
    }
    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("parse json {} failed: {} (body: {})", url, e, text))
}

async fn runtime_version(client: &reqwest::Client, base_url: &str) -> Value {
    let url = format!("{}/api/version", base_url.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(response) => response.json().await.unwrap_or(Value::Null),
        Err(error) => serde_json::json!({ "error": error.to_string() }),
    }
}

fn u32_at(v: &Value, path: &[&str]) -> Option<u32> {
    let mut cur = v;
    for p in path {
        cur = cur.get(*p)?;
    }
    cur.as_u64().map(|n| n as u32)
}

fn u64_at(v: &Value, path: &[&str]) -> Option<u64> {
    let mut cur = v;
    for p in path {
        cur = cur.get(*p)?;
    }
    cur.as_u64()
}

// ---------------------------------------------------------------------------
// Engine calls
// ---------------------------------------------------------------------------

/// mullama POST /v1/completions (raw prompt).
async fn mullama_completions(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    prompt: &str,
    args: &Args,
) -> Result<Sample, String> {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": false,
    });
    let start = Instant::now();
    let v = post_json(client, &format!("{}/v1/completions", base), body).await?;
    let wall = start.elapsed().as_secs_f64();

    let text = v
        .pointer("/choices/0/text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_tokens = u32_at(&v, &["usage", "prompt_tokens"]).unwrap_or(0);
    let completion_tokens = u32_at(&v, &["usage", "completion_tokens"]).unwrap_or(0);
    let (prompt_eval_ns, eval_ns) = timings_from_extension(&v);

    Ok(Sample {
        text,
        prompt_tokens,
        completion_tokens,
        prompt_eval_ns,
        eval_ns,
        wall_secs: wall,
    })
}

/// mullama POST /v1/chat/completions (single user message).
async fn mullama_chat(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    prompt: &str,
    args: &Args,
) -> Result<Sample, String> {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": false,
    });
    let start = Instant::now();
    let v = post_json(client, &format!("{}/v1/chat/completions", base), body).await?;
    let wall = start.elapsed().as_secs_f64();

    let text = v
        .pointer("/choices/0/message/content")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_tokens = u32_at(&v, &["usage", "prompt_tokens"]).unwrap_or(0);
    let completion_tokens = u32_at(&v, &["usage", "completion_tokens"]).unwrap_or(0);
    let (prompt_eval_ns, eval_ns) = timings_from_extension(&v);

    Ok(Sample {
        text,
        prompt_tokens,
        completion_tokens,
        prompt_eval_ns,
        eval_ns,
        wall_secs: wall,
    })
}

/// Pull (prompt_eval_ns, eval_ns) out of the mullama `timings` extension object.
fn timings_from_extension(v: &Value) -> (Option<u64>, Option<u64>) {
    let t = v.get("timings");
    (
        t.and_then(|x| x.get("prompt_eval_ns"))
            .and_then(|x| x.as_u64()),
        t.and_then(|x| x.get("eval_ns")).and_then(|x| x.as_u64()),
    )
}

/// Tokenize `text` with mullama's loaded model tokenizer via POST /v1/tokenize.
///
/// Both engines run the same GGUF, so the same tokenizer is the correct shared
/// yardstick for a token-level comparison. Returns the token-id stream that
/// the model's tokenizer assigns to the (already-detokenized) output text.
async fn mullama_tokenize(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    text: &str,
) -> Result<Vec<i32>, String> {
    let body = json!({ "model": model, "text": text });
    let v = post_json(client, &format!("{}/v1/tokenize", base), body).await?;
    Ok(v.get("tokens")
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_i64().map(|n| n as i32))
                .collect()
        })
        .unwrap_or_default())
}

/// Index of the first diverging token between two token streams, or `None` if
/// one stream is a prefix of the other (no mid-stream divergence).
fn first_diff_token(a: &[i32], b: &[i32]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    (a.len() != b.len()).then_some(n)
}

/// `true` if the two token streams agree on every shared position — i.e. the
/// shorter is a token-prefix of the longer. This is the real "no sampling
/// divergence" signal: a difference is purely length/truncation, not a flipped
/// argmax. Replaces the old count-based `tok_eq` which masked real divergences.
fn token_prefix_match(a: &[i32], b: &[i32]) -> bool {
    first_diff_token(a, b).is_none()
}

/// ollama OpenAI-compatible POST /v1/completions (raw prompt).
///
/// Kept for reference but NOT used for parity: ollama applies the model's chat
/// template to /v1/completions too, so this is not a true raw completion. The
/// parity comparison uses `ollama_generate_native` with `raw: true` instead.
#[allow(dead_code)]
async fn ollama_completions_openai(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    prompt: &str,
    args: &Args,
) -> Result<Sample, String> {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": false,
    });
    let start = Instant::now();
    let v = post_json(client, &format!("{}/v1/completions", base), body).await?;
    let wall = start.elapsed().as_secs_f64();

    let text = v
        .pointer("/choices/0/text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_tokens = u32_at(&v, &["usage", "prompt_tokens"]).unwrap_or(0);
    let completion_tokens = u32_at(&v, &["usage", "completion_tokens"]).unwrap_or(0);

    Ok(Sample {
        text,
        prompt_tokens,
        completion_tokens,
        prompt_eval_ns: None,
        eval_ns: None,
        wall_secs: wall,
    })
}

/// ollama OpenAI-compatible POST /v1/chat/completions (single user message).
async fn ollama_chat_openai(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    prompt: &str,
    args: &Args,
) -> Result<Sample, String> {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": false,
    });
    let start = Instant::now();
    let v = post_json(client, &format!("{}/v1/chat/completions", base), body).await?;
    let wall = start.elapsed().as_secs_f64();

    let text = v
        .pointer("/choices/0/message/content")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_tokens = u32_at(&v, &["usage", "prompt_tokens"]).unwrap_or(0);
    let completion_tokens = u32_at(&v, &["usage", "completion_tokens"]).unwrap_or(0);

    Ok(Sample {
        text,
        prompt_tokens,
        completion_tokens,
        prompt_eval_ns: None,
        eval_ns: None,
        wall_secs: wall,
    })
}

/// ollama native POST /api/generate (raw prompt) — source of truth for timings.
async fn ollama_generate_native(
    client: &reqwest::Client,
    base: &str,
    model: &str,
    prompt: &str,
    args: &Args,
) -> Result<Sample, String> {
    // `raw: true` makes ollama bypass the model's chat template and treat the
    // prompt as a literal completion. Without it, ollama wraps the prompt in the
    // chatml/llama3 template even for /api/generate (prompt_eval_count jumps by
    // ~8 wrapper tokens), so the "raw completions" comparison would be
    // templated-ollama vs raw-mullama — apples vs oranges. raw=true makes both
    // engines see identical bytes, which is what strict parity requires.
    let body = json!({
        "model": model,
        "prompt": prompt,
        "raw": true,
        "stream": false,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.max_tokens,
        },
    });
    let start = Instant::now();
    let v = post_json(client, &format!("{}/api/generate", base), body).await?;
    let wall = start.elapsed().as_secs_f64();

    let text = v
        .get("response")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_tokens = u32_at(&v, &["prompt_eval_count"])
        .or_else(|| u32_at(&v, &["prompt_count"]))
        .unwrap_or(0);
    let completion_tokens = u32_at(&v, &["eval_count"]).unwrap_or(0);
    let prompt_eval_ns = u64_at(&v, &["prompt_eval_duration"]);
    let eval_ns = u64_at(&v, &["eval_duration"]);

    Ok(Sample {
        text,
        prompt_tokens,
        completion_tokens,
        prompt_eval_ns,
        eval_ns,
        wall_secs: wall,
    })
}

// ---------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------

/// Normalize whitespace for fair text comparison (trailing/leading + collapse
/// runs of spaces/newlines). Greedy outputs should be byte-identical already,
/// but tiny tokenizer detok differences can surface as whitespace noise.
fn norm_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compare completion-token counts tolerating the EOG-counting convention
/// difference between mullama (excludes the stop token) and ollama (includes
/// it). Equal, or off-by-one with ollama one higher, both count as a match.
#[allow(dead_code)]
fn tok_eq(m: u32, o: u32) -> bool {
    m == o || m + 1 == o
}

fn first_diff(a: &str, b: &str) -> Option<usize> {
    let (a, b) = (norm_ws(a), norm_ws(b));
    let n = a.chars().count().min(b.chars().count());
    for (i, (ca, cb)) in a.chars().zip(b.chars()).take(n).enumerate() {
        if ca != cb {
            return Some(i);
        }
    }
    (a.chars().count() != b.chars().count()).then_some(n)
}

async fn run_parity(
    client: &reqwest::Client,
    args: &Args,
    prompts: &[Prompt],
    models: &[String],
    out: &mut Vec<ParityRecord>,
) -> Result<(), String> {
    for model in models {
        for p in prompts {
            // Endpoint: raw completions (identical bytes -> the rigorous check).
            // ollama via native /api/generate with raw=true so it does NOT wrap
            // the prompt in the chat template — both engines see the same bytes.
            let m = mullama_completions(client, &args.mullama_url, model, &p.prompt, args).await?;
            let o =
                ollama_generate_native(client, &args.ollama_url, model, &p.prompt, args).await?;
            let mt = mullama_tokenize(client, &args.mullama_url, model, &m.text).await
                .unwrap_or_default();
            let ot = mullama_tokenize(client, &args.mullama_url, model, &o.text).await
                .unwrap_or_default();
            out.push(ParityRecord {
                model: model.clone(),
                prompt_id: p.id.clone(),
                endpoint: "completions".into(),
                mullama_text: m.text.clone(),
                ollama_text: o.text.clone(),
                text_match: norm_ws(&m.text) == norm_ws(&o.text),
                // Real token-sequence parity: the two token streams agree on
                // every shared position (shorter is a prefix of longer). This
                // distinguishes pure truncation from a flipped-argmax sampling
                // divergence; see first_diff_token for the exact divergence
                // point. Server-side counts may still differ by the EOG/stop
                // convention, so they are reported separately below and no
                // longer drive token_match.
                token_match: token_prefix_match(&mt, &ot),
                mullama_completion_tokens: m.completion_tokens,
                ollama_completion_tokens: o.completion_tokens,
                first_diff_char: first_diff(&m.text, &o.text),
                first_diff_token: first_diff_token(&mt, &ot),
            });

            // Endpoint: chat completions (real chat template on both).
            let mc = mullama_chat(client, &args.mullama_url, model, &p.prompt, args).await?;
            let oc = ollama_chat_openai(client, &args.ollama_url, model, &p.prompt, args).await?;
            let mct = mullama_tokenize(client, &args.mullama_url, model, &mc.text).await
                .unwrap_or_default();
            let oct = mullama_tokenize(client, &args.mullama_url, model, &oc.text).await
                .unwrap_or_default();
            out.push(ParityRecord {
                model: model.clone(),
                prompt_id: p.id.clone(),
                endpoint: "chat".into(),
                mullama_text: mc.text.clone(),
                ollama_text: oc.text.clone(),
                text_match: norm_ws(&mc.text) == norm_ws(&oc.text),
                token_match: token_prefix_match(&mct, &oct),
                mullama_completion_tokens: mc.completion_tokens,
                ollama_completion_tokens: oc.completion_tokens,
                first_diff_char: first_diff(&mc.text, &oc.text),
                first_diff_token: first_diff_token(&mct, &oct),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Perf
// ---------------------------------------------------------------------------

fn pct(data: &mut [f64], q: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((q * (data.len() - 1) as f64).round() as usize).min(data.len() - 1);
    data[idx]
}

async fn run_perf(
    client: &reqwest::Client,
    args: &Args,
    prompts: &[Prompt],
    models: &[String],
    out: &mut Vec<PerfRecord>,
) -> Result<(), String> {
    for model in models {
        for p in prompts {
            // mullama via /v1/completions (has timings).
            let mut m_eng = Vec::new();
            let mut m_wall = Vec::new();
            let mut m_ctoks = Vec::new();
            let mut m_wall_secs = Vec::new();
            for _ in 0..args.warmup {
                let _ =
                    mullama_completions(client, &args.mullama_url, model, &p.prompt, args).await?;
            }
            for _ in 0..args.runs {
                let s =
                    mullama_completions(client, &args.mullama_url, model, &p.prompt, args).await?;
                m_eng.push(s.engine_tok_s().unwrap_or(0.0));
                m_wall.push(s.wall_tok_s());
                m_ctoks.push(s.completion_tokens as f64);
                m_wall_secs.push(s.wall_secs);
            }
            out.push(PerfRecord {
                model: model.clone(),
                endpoint: "completions".into(),
                engine: "mullama".into(),
                runs: args.runs,
                engine_tok_s_mean: m_eng.iter().sum::<f64>() / args.runs as f64,
                engine_tok_s_p50: pct(&mut m_eng.clone(), 0.5),
                wall_tok_s_mean: m_wall.iter().sum::<f64>() / args.runs as f64,
                wall_secs_mean: m_wall_secs.iter().sum::<f64>() / args.runs as f64,
                completion_tokens_mean: m_ctoks.iter().sum::<f64>() / args.runs as f64,
            });

            // ollama via native /api/generate (source of truth timings).
            let mut o_eng = Vec::new();
            let mut o_wall = Vec::new();
            let mut o_ctoks = Vec::new();
            let mut o_wall_secs = Vec::new();
            for _ in 0..args.warmup {
                let _ = ollama_generate_native(client, &args.ollama_url, model, &p.prompt, args)
                    .await?;
            }
            for _ in 0..args.runs {
                let s = ollama_generate_native(client, &args.ollama_url, model, &p.prompt, args)
                    .await?;
                o_eng.push(s.engine_tok_s().unwrap_or(0.0));
                o_wall.push(s.wall_tok_s());
                o_ctoks.push(s.completion_tokens as f64);
                o_wall_secs.push(s.wall_secs);
            }
            out.push(PerfRecord {
                model: model.clone(),
                endpoint: "completions".into(),
                engine: "ollama".into(),
                runs: args.runs,
                engine_tok_s_mean: o_eng.iter().sum::<f64>() / args.runs as f64,
                engine_tok_s_p50: pct(&mut o_eng.clone(), 0.5),
                wall_tok_s_mean: o_wall.iter().sum::<f64>() / args.runs as f64,
                wall_secs_mean: o_wall_secs.iter().sum::<f64>() / args.runs as f64,
                completion_tokens_mean: o_ctoks.iter().sum::<f64>() / args.runs as f64,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_parity_summary(records: &[ParityRecord]) {
    println!("\n=== Parity (greedy, temperature=0) ===");
    println!(
        "{:<18} {:<10} {:<8} {:<8} {:<10} {:<8}",
        "model", "endpoint", "text", "tokens", "m_toks", "o_toks"
    );
    let mut by_model: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in records {
        let pass = r.text_match && r.token_match;
        let e = by_model.entry(r.model.clone()).or_insert((0, 0));
        e.0 += if pass { 1 } else { 0 };
        e.1 += 1;
        println!(
            "{:<18} {:<10} {:<8} {:<8} {:<10} {:<8}{}",
            r.model,
            r.endpoint,
            if r.text_match { "OK" } else { "DIFF" },
            if r.token_match { "OK" } else { "DIFF" },
            r.mullama_completion_tokens,
            r.ollama_completion_tokens,
            if r.text_match {
                "".to_string()
            } else {
                format!(
                    "  diff_char@{} diff_tok@{}",
                    r.first_diff_char.unwrap_or(0),
                    r.first_diff_token
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "none".into())
                )
            },
        );
    }
    println!("\nparity pass-rate per model:");
    for (m, (ok, tot)) in &by_model {
        println!("  {}: {}/{}", m, ok, tot);
    }
}

fn print_perf_summary(records: &[PerfRecord]) {
    println!("\n=== Perf (engine tok/s, mean over runs) ===");
    println!(
        "{:<18} {:<10} {:<8} {:>12} {:>12} {:>10}",
        "model", "endpoint", "engine", "eng_tok/s", "wall_tok/s", "wall_s"
    );
    // group by (model, endpoint) for a mullama-vs-ollama side-by-side
    let mut groups: BTreeMap<(String, String), Vec<&PerfRecord>> = BTreeMap::new();
    for r in records {
        groups
            .entry((r.model.clone(), r.endpoint.clone()))
            .or_default()
            .push(r);
    }
    for ((model, endpoint), rs) in &groups {
        for r in rs {
            println!(
                "{:<18} {:<10} {:<8} {:>12.2} {:>12.2} {:>10.3}",
                model, endpoint, r.engine, r.engine_tok_s_mean, r.wall_tok_s_mean, r.wall_secs_mean,
            );
        }
        // ratio
        let m = rs.iter().find(|x| x.engine == "mullama");
        let o = rs.iter().find(|x| x.engine == "ollama");
        if let (Some(m), Some(o)) = (m, o) {
            if o.engine_tok_s_mean > 0.0 {
                let ratio = m.engine_tok_s_mean / o.engine_tok_s_mean;
                println!(
                    "{:<18} {:<10} {:<8} {:>12}",
                    "",
                    "",
                    "ratio",
                    format!("{:.2}x mullama/ollama", ratio),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

/// Replays one agent trace against mullama as a real agent loop: each user
/// turn is appended to the running history, the model's assistant reply is fed
/// back, and we record per-turn prefill/decode timings. The cross-turn KV-reuse
/// win reads off `prompt_eval_ns`: with reuse, turn 2+ prefills only the new
/// user turn (a small delta); without it, every turn re-prefills the whole
/// accumulated history.
async fn run_agent_loop(
    client: &reqwest::Client,
    args: &Args,
    model: &str,
    trace: &Trace,
    out: &mut Vec<AgentLoopRecord>,
) -> Result<(), String> {
    let url = format!("{}/v1/chat/completions", args.mullama_url.trim_end_matches('/'));
    let turns: Vec<&String> = if args.turns > 0 {
        trace.turns.iter().take(args.turns).collect()
    } else {
        trace.turns.iter().collect()
    };

    // Running conversation: the chat template is applied server-side, so we
    // send the full message history each turn and let the daemon decide what
    // to prefill (with cross-turn KV reuse, only the delta is actually computed).
    let mut history: Vec<Value> = Vec::new();
    // A stable session id per (model, trace) pins the context slot so its KV
    // persists across turns. Omitted when --no-kv-reuse captures the baseline.
    let session = if args.no_kv_reuse {
        None
    } else {
        Some(format!("{}:{}", model, trace.id))
    };
    for (i, user_turn) in turns.iter().enumerate() {
        history.push(json!({ "role": "user", "content": user_turn }));
        let mut body = json!({
            "model": model,
            "messages": history,
            "max_tokens": args.agent_max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "stream": false,
        });
        if let Some(ref sid) = session {
            body["session"] = json!(sid);
        }
        let start = Instant::now();
        let v = post_json(client, &url, body).await?;
        let wall = start.elapsed().as_secs_f64();

        let text = v
            .pointer("/choices/0/message/content")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let prompt_tokens = u32_at(&v, &["usage", "prompt_tokens"]).unwrap_or(0);
        let completion_tokens = u32_at(&v, &["usage", "completion_tokens"]).unwrap_or(0);
        let (prompt_eval_ns, eval_ns) = timings_from_extension(&v);

        out.push(AgentLoopRecord {
            model: model.to_string(),
            trace_id: trace.id.clone(),
            turn: i + 1,
            prompt_tokens,
            completion_tokens,
            prompt_eval_ns,
            eval_ns,
            wall_secs: wall,
            text: text.clone(),
        });

        // Feed the assistant reply back as context for the next turn.
        history.push(json!({ "role": "assistant", "content": text }));
    }
    Ok(())
}

fn print_agent_loop_summary(records: &[AgentLoopRecord]) {
    println!("\n=== Agent loop (per-turn prefill/decode) ===");
    println!(
        "{:<18} {:<10} {:>5} {:>10} {:>10} {:>12} {:>12} {:>9}",
        "model", "trace", "turn", "p_toks", "c_toks", "prefill_ms", "decode_ms", "wall_s",
    );
    // Group by (model, trace) so each trace's turn sequence stays together.
    let mut groups: BTreeMap<(String, String), Vec<&AgentLoopRecord>> = BTreeMap::new();
    for r in records {
        groups
            .entry((r.model.clone(), r.trace_id.clone()))
            .or_default()
            .push(r);
    }
    for ((model, trace_id), rs) in &groups {
        let mut first_prefill_ns: Option<u64> = None;
        for r in rs {
            let prefill_ms = r.prompt_eval_ns.map(|n| n as f64 / 1e6);
            let decode_ms = r.eval_ns.map(|n| n as f64 / 1e6);
            println!(
                "{:<18} {:<10} {:>5} {:>10} {:>10} {:>12.2} {:>12.2} {:>9.3}",
                model,
                trace_id,
                r.turn,
                r.prompt_tokens,
                r.completion_tokens,
                prefill_ms.unwrap_or(0.0),
                decode_ms.unwrap_or(0.0),
                r.wall_secs,
            );
            if r.turn == 1 {
                first_prefill_ns = r.prompt_eval_ns;
            }
        }
        // The headline number: how much later-turn prefill collapses vs turn 1.
        // With cross-turn KV reuse this ratio should be large (turn 1 full
        // prefill, turns 2+ only the delta); without it the ratio stays ~1.0
        // because every turn re-prefills the whole history.
        if let (Some(first), Some(last)) = (
            first_prefill_ns,
            rs.last().and_then(|r| r.prompt_eval_ns),
        ) {
            if last > 0 {
                let ratio = first as f64 / last as f64;
                println!(
                    "{:<18} {:<10} {:>5} {:>12}",
                    "",
                    "",
                    "",
                    format!("reuse ratio turn1/turnN: {:.1}x prefill saved", ratio),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mode = Mode::from_str(&args.mode)?;

    if args.models.is_empty() {
        return Err("no models given (--models)".into());
    }

    // Load prompts.
    let prompt_text = std::fs::read_to_string(&args.prompt_file)
        .map_err(|e| format!("read prompt file {}: {}", args.prompt_file, e))?;
    let prompts: Vec<Prompt> = prompt_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Prompt>(l).map_err(|e| format!("bad prompt line: {}", e)))
        .collect::<Result<_, _>>()?;

    // Load agent traces for agent-loop mode.
    let traces: Vec<Trace> = if mode == Mode::AgentLoop {
        let trace_text = std::fs::read_to_string(&args.trace_file)
            .map_err(|e| format!("read trace file {}: {}", args.trace_file, e))?;
        trace_text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Trace>(l).map_err(|e| format!("bad trace line: {}", e)))
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };

    println!(
        "mullama-bench: {} models, {} prompts, mode={:?}, runs={} warmup={} max_tokens={} temp={}",
        args.models.len(),
        prompts.len(),
        mode,
        args.runs,
        args.warmup,
        args.max_tokens,
        args.temperature,
    );
    if mode == Mode::AgentLoop {
        let total_turns: usize = traces.iter().map(|t| t.turns.len()).sum();
        println!(
            "  agent-loop: {} traces, {} total turns, max {} tok/turn",
            traces.len(),
            total_turns,
            args.agent_max_tokens,
        );
    }
    println!("  mullama: {}", args.mullama_url);
    println!("  ollama:  {}", args.ollama_url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;
    let runtime = serde_json::json!({
        "mullama": {
            "version": runtime_version(&client, &args.mullama_url).await,
            "llama_baseline": mullama::LLAMA_BASELINE,
        },
        "ollama": {
            "version": runtime_version(&client, &args.ollama_url).await,
        },
    });

    // Total work units for the progress bar.
    let total = match mode {
        Mode::Parity => args.models.len() * prompts.len() * 2,
        Mode::Perf => args.models.len() * prompts.len(),
        Mode::Both => args.models.len() * prompts.len() * 3,
        Mode::AgentLoop => {
            args.models.len()
                * traces
                    .iter()
                    .map(|t| if args.turns > 0 { args.turns.min(t.turns.len()) } else { t.turns.len() })
                    .sum::<usize>()
        }
    };
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} {elapsed} {msg}").unwrap(),
    );

    let mut parity_records = Vec::new();
    let mut perf_records = Vec::new();
    let mut agent_loop_records = Vec::new();

    // Run per-model so progress and partial reports stay meaningful.
    for model in &args.models {
        bar.set_message(model.clone());
        match mode {
            Mode::Parity => {
                run_parity(
                    &client,
                    &args,
                    &prompts,
                    std::slice::from_ref(model),
                    &mut parity_records,
                )
                .await?;
                bar.inc((prompts.len() * 2) as u64);
            }
            Mode::Perf => {
                run_perf(
                    &client,
                    &args,
                    &prompts,
                    std::slice::from_ref(model),
                    &mut perf_records,
                )
                .await?;
                bar.inc(prompts.len() as u64);
            }
            Mode::Both => {
                run_parity(
                    &client,
                    &args,
                    &prompts,
                    std::slice::from_ref(model),
                    &mut parity_records,
                )
                .await?;
                bar.inc((prompts.len() * 2) as u64);
                run_perf(
                    &client,
                    &args,
                    &prompts,
                    std::slice::from_ref(model),
                    &mut perf_records,
                )
                .await?;
                bar.inc(prompts.len() as u64);
            }
            Mode::AgentLoop => {
                for trace in &traces {
                    run_agent_loop(&client, &args, model, trace, &mut agent_loop_records).await?;
                    let n = if args.turns > 0 {
                        args.turns.min(trace.turns.len())
                    } else {
                        trace.turns.len()
                    };
                    bar.inc(n as u64);
                }
            }
        }
    }
    bar.finish_with_message("done");

    if matches!(mode, Mode::Parity | Mode::Both) {
        print_parity_summary(&parity_records);
    }
    if matches!(mode, Mode::Perf | Mode::Both) {
        print_perf_summary(&perf_records);
    }
    if mode == Mode::AgentLoop {
        print_agent_loop_summary(&agent_loop_records);
    }

    let parity_failures = parity_records
        .iter()
        .filter(|record| !record.text_match || !record.token_match)
        .count();

    let report = Report {
        config: serde_json::to_value(&args).unwrap_or(Value::Null),
        runtime,
        parity: parity_records,
        perf: perf_records,
        agent_loop: agent_loop_records,
    };
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)?)?;
    println!("\nreport written to {}", args.report);

    if parity_failures > 0 && !args.allow_parity_diffs {
        return Err(format!(
            "strict parity failed for {parity_failures} comparison(s); use --allow-parity-diffs only for exploratory runs"
        )
        .into());
    }

    Ok(())
}
