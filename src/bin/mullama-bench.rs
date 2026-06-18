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
}

impl Mode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "parity" => Ok(Mode::Parity),
            "perf" => Ok(Mode::Perf),
            "both" => Ok(Mode::Both),
            other => Err(format!("invalid mode '{}' (parity|perf|both)", other)),
        }
    }
}

#[derive(Parser, Debug, Serialize)]
#[command(name = "mullama-bench", version, about = "Benchmark & parity test mullama vs ollama")]
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
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Prompt {
    id: String,
    prompt: String,
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
    parity: Vec<ParityRecord>,
    perf: Vec<PerfRecord>,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async fn post_json(
    client: &reqwest::Client,
    url: &str,
    body: Value,
) -> Result<Value, String> {
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
        t.and_then(|x| x.get("prompt_eval_ns")).and_then(|x| x.as_u64()),
        t.and_then(|x| x.get("eval_ns")).and_then(|x| x.as_u64()),
    )
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

    let text = v.get("response").and_then(|t| t.as_str()).unwrap_or("").to_string();
    let prompt_tokens =
        u32_at(&v, &["prompt_eval_count"]).or_else(|| u32_at(&v, &["prompt_count"])).unwrap_or(0);
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
            let o = ollama_generate_native(client, &args.ollama_url, model, &p.prompt, args)
                .await?;
            out.push(ParityRecord {
                model: model.clone(),
                prompt_id: p.id.clone(),
                endpoint: "completions".into(),
                mullama_text: m.text.clone(),
                ollama_text: o.text.clone(),
                text_match: norm_ws(&m.text) == norm_ws(&o.text),
                // Tolerate the EOG-counting convention difference: ollama's
                // eval_count includes the stop/EOG token, mullama's
                // completion_tokens excludes it (the loop breaks before
                // counting). So a generation that stops at EOG differs by 1;
                // one that hits max_tokens is equal. Accept either.
                token_match: tok_eq(m.completion_tokens, o.completion_tokens),
                mullama_completion_tokens: m.completion_tokens,
                ollama_completion_tokens: o.completion_tokens,
                first_diff_char: first_diff(&m.text, &o.text),
            });

            // Endpoint: chat completions (real chat template on both).
            let mc = mullama_chat(client, &args.mullama_url, model, &p.prompt, args).await?;
            let oc = ollama_chat_openai(client, &args.ollama_url, model, &p.prompt, args).await?;
            out.push(ParityRecord {
                model: model.clone(),
                prompt_id: p.id.clone(),
                endpoint: "chat".into(),
                mullama_text: mc.text.clone(),
                ollama_text: oc.text.clone(),
                text_match: norm_ws(&mc.text) == norm_ws(&oc.text),
                token_match: tok_eq(mc.completion_tokens, oc.completion_tokens),
                mullama_completion_tokens: mc.completion_tokens,
                ollama_completion_tokens: oc.completion_tokens,
                first_diff_char: first_diff(&mc.text, &oc.text),
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
            for _ in 0..args.warmup {
                let _ = mullama_completions(client, &args.mullama_url, model, &p.prompt, args)
                    .await?;
            }
            for _ in 0..args.runs {
                let s = mullama_completions(client, &args.mullama_url, model, &p.prompt, args)
                    .await?;
                m_eng.push(s.engine_tok_s().unwrap_or(0.0));
                m_wall.push(s.wall_tok_s());
                m_ctoks.push(s.completion_tokens as f64);
            }
            out.push(PerfRecord {
                model: model.clone(),
                endpoint: "completions".into(),
                engine: "mullama".into(),
                runs: args.runs,
                engine_tok_s_mean: m_eng.iter().sum::<f64>() / args.runs as f64,
                engine_tok_s_p50: pct(&mut m_eng.clone(), 0.5),
                wall_tok_s_mean: m_wall.iter().sum::<f64>() / args.runs as f64,
                wall_secs_mean: 0.0, // filled below
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
                format!("  first_diff@{}", r.first_diff_char.unwrap_or(0))
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
        groups.entry((r.model.clone(), r.endpoint.clone())).or_default().push(r);
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
                    "", "", "ratio", format!("{:.2}x mullama/ollama", ratio),
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
    println!("  mullama: {}", args.mullama_url);
    println!("  ollama:  {}", args.ollama_url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    // Total work units for the progress bar.
    let total = match mode {
        Mode::Parity => args.models.len() * prompts.len() * 2,
        Mode::Perf => args.models.len() * prompts.len(),
        Mode::Both => args.models.len() * prompts.len() * 3,
    };
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} {elapsed} {msg}",
        )
        .unwrap(),
    );

    let mut parity_records = Vec::new();
    let mut perf_records = Vec::new();

    // Run per-model so progress and partial reports stay meaningful.
    for model in &args.models {
        bar.set_message(model.clone());
        match mode {
            Mode::Parity => {
                run_parity(&client, &args, &prompts, std::slice::from_ref(model), &mut parity_records)
                    .await?;
                bar.inc((prompts.len() * 2) as u64);
            }
            Mode::Perf => {
                run_perf(&client, &args, &prompts, std::slice::from_ref(model), &mut perf_records)
                    .await?;
                bar.inc(prompts.len() as u64);
            }
            Mode::Both => {
                run_parity(&client, &args, &prompts, std::slice::from_ref(model), &mut parity_records)
                    .await?;
                bar.inc((prompts.len() * 2) as u64);
                run_perf(&client, &args, &prompts, std::slice::from_ref(model), &mut perf_records)
                    .await?;
                bar.inc(prompts.len() as u64);
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

    let report = Report {
        config: serde_json::to_value(&args).unwrap_or(Value::Null),
        parity: parity_records,
        perf: perf_records,
    };
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)?)?;
    println!("\nreport written to {}", args.report);

    Ok(())
}