//! Agent file-access prefetch policy.
//!
//! Coding agents read files in predictable bursts: open `foo.rs`, then almost
//! always its siblings in the same module, then whatever it `use`s/`import`s.
//! Each such read becomes a turn whose prompt embeds the file's contents, and
//! (with cross-turn reuse) a prefill of that content. If we can *predict* the
//! next files an agent will read, the daemon's idle hydrator can pre-warm them
//! — tokenize + decode their contents into a free slot — so the turn that
//! actually reads them is a delta, not a cold prefill.
//!
//! This module is the *policy*: given the files an agent has touched so far
//! (extracted from conversation content), rank the files it is most likely to
//! touch next. It is deliberately pure and filesystem-light so it can be unit
//! tested deterministically:
//!
//! - **Path extraction** ([`extract_paths`]): scan text for path-like tokens
//!   (slash-separated, known source extensions) and existing files.
//! - **Sibling locality** ([`predict`]): files in the same directory as a
//!   recently-touched file, not yet touched, score high — agents sweep modules.
//! - **Import following** ([`imports_of`], [`predict`]): parse `use`/`mod`/
//!   `import`/`require`/`include` targets out of a touched file and resolve
//!   them to local paths; a file an open file depends on is a strong next-read
//!   signal.
//!
//! The predictor returns a ranked, de-duplicated candidate list. Wiring it to
//! actually warm KV is the hydrator's job (and is gated on idleness); the
//! policy here makes no I/O beyond reading directory listings and the touched
//! files' own text, and never blocks a request.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use dashmap::DashMap;

/// Per-session record of the source files an agent has touched, most-recent
/// last, bounded to the last `WINDOW` distinct paths. Fed from conversation
/// content at request time; read by the idle hydrator to predict (and warm)
/// the agent's next reads.
const WINDOW: usize = 16;

/// Tracks recently-touched files per session for the prefetch policy.
pub(crate) struct PrefetchObserver {
    by_session: DashMap<String, Vec<PathBuf>>,
}

impl PrefetchObserver {
    pub(crate) fn new() -> Self {
        Self {
            by_session: DashMap::new(),
        }
    }

    /// Record the source paths mentioned in `text` for `session` (e.g. the
    /// latest user turn). De-duplicates against the tail and bounds the history
    /// to the last [`WINDOW`] distinct paths, preserving recency order.
    pub(crate) fn observe(&self, session: &str, text: &str) {
        let found = extract_paths(text);
        if found.is_empty() {
            return;
        }
        let mut entry = self.by_session.entry(session.to_string()).or_default();
        for p in found {
            let path = PathBuf::from(p);
            // Move-to-end on repeat so recency stays meaningful.
            if let Some(pos) = entry.iter().position(|e| *e == path) {
                entry.remove(pos);
            }
            entry.push(path);
        }
        let len = entry.len();
        if len > WINDOW {
            entry.drain(0..len - WINDOW);
        }
    }

    /// The touched-file history for a session, most-recent last.
    pub(crate) fn history(&self, session: &str) -> Vec<PathBuf> {
        self.by_session
            .get(session)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// All sessions with recorded history (for the hydrator's candidate scan).
    pub(crate) fn sessions(&self) -> Vec<String> {
        self.by_session.iter().map(|e| e.key().clone()).collect()
    }
}

/// Source-file extensions we treat as prefetch candidates. Kept small and
/// code-focused; data/asset files aren't worth warming.
const SOURCE_EXTS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "c", "h", "cc", "cpp", "hpp",
    "java", "rb", "php", "cs", "swift", "kt", "scala", "sh", "toml", "md",
];

/// A scored prefetch candidate. Higher `score` = more likely next read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Candidate {
    pub path: PathBuf,
    pub score: f32,
    pub reason: PrefetchReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefetchReason {
    /// Imported/`use`d by a touched file.
    Import,
    /// Sibling in the same directory as a touched file.
    Sibling,
}

/// Extract path-like tokens from a blob of conversation text. Matches tokens
/// that contain a `/` and end in a known source extension, plus bare filenames
/// with a source extension. Returns de-duplicated, in first-seen order.
pub(crate) fn extract_paths(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for raw in text.split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`' | '(' | ')' | '<' | '>' | ',' | ';' | ':' )) {
        let tok = raw.trim_matches(|c: char| matches!(c, '.' | '!' | '?' | ']' | '[' | '{' | '}'));
        if tok.len() < 3 || tok.len() > 4096 {
            continue;
        }
        let has_ext = Path::new(tok)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| SOURCE_EXTS.contains(&e))
            .unwrap_or(false);
        if !has_ext {
            continue;
        }
        // Require either a path separator or that it looks like a real filename
        // (no spaces — already guaranteed by split). Bare `foo.rs` counts.
        if seen.insert(tok.to_string()) {
            out.push(tok.to_string());
        }
    }
    out
}

/// Parse the local module/import targets referenced by a source file's text.
/// Resolves a handful of common syntaxes to candidate sibling paths relative to
/// the file's own directory. Best-effort and language-agnostic: unresolved or
/// external imports (std, third-party crates) are simply dropped.
pub(crate) fn imports_of(file: &Path, text: &str) -> Vec<PathBuf> {
    let dir = file.parent().unwrap_or_else(|| Path::new("."));
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |p: PathBuf| {
        if seen.insert(p.clone()) {
            out.push(p);
        }
    };

    for line in text.lines() {
        let l = line.trim();
        match ext {
            "rs" => {
                // `mod foo;` / `pub mod foo;` / `pub(crate) mod foo;`. In Rust,
                // a `mod foo;` in `name.rs` resolves to `name/foo.rs` (or
                // `name/foo/mod.rs`) UNLESS the file is a crate/module root
                // (`mod.rs`, `lib.rs`, `main.rs`), where it's `foo.rs` in the
                // same dir. We try both layouts so either convention resolves.
                let l = l
                    .strip_prefix("pub(crate) ")
                    .or_else(|| l.strip_prefix("pub(super) "))
                    .or_else(|| l.strip_prefix("pub "))
                    .unwrap_or(l);
                if let Some(rest) = l.strip_prefix("mod ") {
                    if let Some(name) = rest.split(|c| c == ';' || c == ' ' || c == '{').next() {
                        let name = name.trim();
                        if is_ident(name) {
                            let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                            let is_root = matches!(stem, "mod" | "lib" | "main");
                            // Same-dir layout (root modules).
                            push(dir.join(format!("{name}.rs")));
                            push(dir.join(name).join("mod.rs"));
                            // Subdir-named-after-file layout (non-root modules).
                            if !is_root && !stem.is_empty() {
                                push(dir.join(stem).join(format!("{name}.rs")));
                                push(dir.join(stem).join(name).join("mod.rs"));
                            }
                        }
                    }
                }
            }
            "py" => {
                // `from .foo import x` / `import foo` -> foo.py in same dir.
                if let Some(rest) = l.strip_prefix("from ") {
                    let module = rest.split(" import").next().unwrap_or("").trim();
                    let module = module.trim_start_matches('.');
                    if is_ident(module) {
                        push(dir.join(format!("{module}.py")));
                    }
                } else if let Some(rest) = l.strip_prefix("import ") {
                    let module = rest.split([' ', ',']).next().unwrap_or("").trim();
                    if is_ident(module) {
                        push(dir.join(format!("{module}.py")));
                    }
                }
            }
            "js" | "ts" | "jsx" | "tsx" => {
                // `from './foo'` / `require('./foo')` -> ./foo.{js,ts}
                if let Some(spec) = relative_js_spec(l) {
                    for cand_ext in ["ts", "tsx", "js", "jsx"] {
                        push(dir.join(format!("{spec}.{cand_ext}")));
                    }
                    push(dir.join(&spec).join("index.ts"));
                    push(dir.join(&spec).join("index.js"));
                }
            }
            "c" | "h" | "cc" | "cpp" | "hpp" => {
                // `#include "foo.h"` (quoted = local).
                if let Some(inc) = l.strip_prefix("#include") {
                    let inc = inc.trim();
                    if let Some(name) = inc.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                        push(dir.join(name));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Predict the files an agent is most likely to read next, given the files it
/// has already touched (most-recent-last). `read_file` supplies a file's text
/// for import parsing and `list_dir` supplies a directory's entries for sibling
/// locality — both injected so the policy is testable without real I/O. Returns
/// a ranked, de-duplicated candidate list, highest score first, capped at
/// `limit`. Already-touched files are never returned.
pub(crate) fn predict(
    touched: &[PathBuf],
    limit: usize,
    read_file: &dyn Fn(&Path) -> Option<String>,
    list_dir: &dyn Fn(&Path) -> Vec<PathBuf>,
) -> Vec<Candidate> {
    let touched_set: HashSet<&Path> = touched.iter().map(|p| p.as_path()).collect();
    // Recency weight: the last-touched file's signals matter most. Weight decays
    // linearly from 1.0 (most recent) to ~0.3 (oldest) so a long history doesn't
    // drown the current focus.
    let n = touched.len().max(1);
    let mut scores: HashMap<PathBuf, (f32, PrefetchReason)> = HashMap::new();
    let mut bump = |path: PathBuf, add: f32, reason: PrefetchReason| {
        if touched_set.contains(path.as_path()) {
            return;
        }
        let e = scores.entry(path).or_insert((0.0, reason));
        e.0 += add;
        // Import beats sibling as the recorded reason if both fire.
        if reason == PrefetchReason::Import {
            e.1 = PrefetchReason::Import;
        }
    };

    for (i, file) in touched.iter().enumerate() {
        let recency = 0.3 + 0.7 * (i as f32 + 1.0) / n as f32;

        // Imports: strong signal (1.0 base).
        if let Some(text) = read_file(file) {
            for imp in imports_of(file, &text) {
                bump(imp, 1.0 * recency, PrefetchReason::Import);
            }
        }

        // Siblings: weaker signal (0.4 base), only source files.
        if let Some(parent) = file.parent() {
            for sib in list_dir(parent) {
                let is_src = sib
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| SOURCE_EXTS.contains(&e))
                    .unwrap_or(false);
                if is_src && sib != *file {
                    bump(sib, 0.4 * recency, PrefetchReason::Sibling);
                }
            }
        }
    }

    let mut ranked: Vec<Candidate> = scores
        .into_iter()
        .map(|(path, (score, reason))| Candidate { path, score, reason })
        .collect();
    // Sort by score desc, then path for determinism.
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    ranked.truncate(limit);
    ranked
}

/// Convenience: predict against the real filesystem. Reads files and directory
/// listings directly. Used by the daemon; tests use [`predict`] with injected
/// closures. Only returns candidates that exist on disk.
pub(crate) fn predict_fs(touched: &[PathBuf], limit: usize) -> Vec<Candidate> {
    let read = |p: &Path| std::fs::read_to_string(p).ok();
    let list = |p: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(p)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.is_file())
                    .collect()
            })
            .unwrap_or_default()
    };
    // Rank without truncating (usize::MAX), filter to files that actually exist,
    // THEN apply the limit — otherwise phantom import candidates (e.g. the
    // module layout we don't use) would consume slots ahead of real files.
    let mut existing: Vec<Candidate> = predict(touched, usize::MAX, &read, &list)
        .into_iter()
        .filter(|c| c.path.is_file())
        .collect();
    existing.truncate(limit);
    existing
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        && s.chars().next().map(|c| !c.is_numeric()).unwrap_or(false)
}

/// Pull the relative module specifier out of a JS/TS import/require line, if it
/// is a *local* one (`./` or `../`). Returns the spec without quotes/prefix,
/// e.g. `./foo` -> `foo`, `../bar/baz` -> `../bar/baz`.
fn relative_js_spec(line: &str) -> Option<String> {
    let start = line.find("from ").map(|i| i + 5).or_else(|| {
        line.find("require(").map(|i| i + 8)
    })?;
    let rest = &line[start..];
    let quote = rest.find(['\'', '"'])?;
    let after = &rest[quote + 1..];
    let end = after.find(['\'', '"'])?;
    let spec = &after[..end];
    if spec.starts_with("./") || spec.starts_with("../") {
        // Strip a single leading ./ for join; keep ../ as-is.
        Some(spec.strip_prefix("./").unwrap_or(spec).to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_paths_finds_source_files_and_dedups() {
        let text = "I edited src/main.rs and then src/lib.rs, also see src/main.rs again. \
                    Ignore notes.txt? No — and data.bin is skipped.";
        let paths = extract_paths(text);
        assert_eq!(paths, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn extract_paths_handles_bare_filenames_and_punctuation() {
        let text = "Open `mod.rs`. Then (handler.ts) and config.toml!";
        let paths = extract_paths(text);
        assert!(paths.contains(&"mod.rs".to_string()));
        assert!(paths.contains(&"handler.ts".to_string()));
        assert!(paths.contains(&"config.toml".to_string()));
    }

    #[test]
    fn imports_of_rust_mod_declarations() {
        let file = PathBuf::from("src/server.rs");
        let text = "mod kvstore;\nmod session;\npub mod handlers;\nuse std::sync::Arc;";
        let imps = imports_of(&file, text);
        assert!(imps.contains(&PathBuf::from("src/kvstore.rs")));
        assert!(imps.contains(&PathBuf::from("src/session.rs")));
        assert!(imps.contains(&PathBuf::from("src/handlers.rs")));
        // mod foo also offers foo/mod.rs.
        assert!(imps.contains(&PathBuf::from("src/handlers/mod.rs")));
        // std import is dropped.
        assert!(!imps.iter().any(|p| p.to_string_lossy().contains("std")));
    }

    #[test]
    fn imports_of_python_relative_and_plain() {
        let file = PathBuf::from("pkg/app.py");
        let text = "from .models import User\nimport helpers\nfrom os import path";
        let imps = imports_of(&file, text);
        assert!(imps.contains(&PathBuf::from("pkg/models.py")));
        assert!(imps.contains(&PathBuf::from("pkg/helpers.py")));
        // `from os import path` resolves to pkg/os.py (a local candidate); it
        // simply won't exist on disk and predict_fs filters it.
    }

    #[test]
    fn imports_of_js_relative_specifiers() {
        let file = PathBuf::from("web/app.ts");
        let text = "import { x } from './util';\nimport y from '../shared/api';\nimport z from 'react';";
        let imps = imports_of(&file, text);
        assert!(imps.contains(&PathBuf::from("web/util.ts")));
        assert!(imps.iter().any(|p| p.ends_with("shared/api.ts")));
        // external 'react' dropped.
        assert!(!imps.iter().any(|p| p.to_string_lossy().contains("react")));
    }

    #[test]
    fn imports_of_c_quoted_includes_only() {
        let file = PathBuf::from("src/main.c");
        let text = "#include \"util.h\"\n#include <stdio.h>";
        let imps = imports_of(&file, text);
        assert!(imps.contains(&PathBuf::from("src/util.h")));
        assert!(!imps.iter().any(|p| p.to_string_lossy().contains("stdio")));
    }

    #[test]
    fn predict_ranks_imports_above_siblings_and_excludes_touched() {
        let touched = vec![PathBuf::from("src/server.rs")];
        let files: HashMap<PathBuf, String> = [(
            PathBuf::from("src/server.rs"),
            "mod session;\nmod kvstore;".to_string(),
        )]
        .into_iter()
        .collect();
        let read = move |p: &Path| files.get(p).cloned();
        let list = |p: &Path| -> Vec<PathBuf> {
            if p == Path::new("src") {
                vec![
                    PathBuf::from("src/server.rs"),
                    PathBuf::from("src/session.rs"),
                    PathBuf::from("src/kvstore.rs"),
                    PathBuf::from("src/unrelated.rs"),
                ]
            } else {
                vec![]
            }
        };
        let preds = predict(&touched, 10, &read, &list);
        // server.rs itself excluded.
        assert!(!preds.iter().any(|c| c.path == PathBuf::from("src/server.rs")));
        // session & kvstore got BOTH an import and sibling bump -> top two.
        let top2: HashSet<_> = preds.iter().take(2).map(|c| c.path.clone()).collect();
        assert!(top2.contains(&PathBuf::from("src/session.rs")));
        assert!(top2.contains(&PathBuf::from("src/kvstore.rs")));
        // import reason recorded for them.
        let sess = preds.iter().find(|c| c.path == PathBuf::from("src/session.rs")).unwrap();
        assert_eq!(sess.reason, PrefetchReason::Import);
        // unrelated.rs only got a sibling bump -> ranked below, but present.
        let unrel = preds.iter().find(|c| c.path == PathBuf::from("src/unrelated.rs")).unwrap();
        assert_eq!(unrel.reason, PrefetchReason::Sibling);
        assert!(sess.score > unrel.score);
    }

    #[test]
    fn observer_tracks_recency_and_bounds_window() {
        let obs = PrefetchObserver::new();
        obs.observe("S", "open src/a.rs and src/b.rs");
        obs.observe("S", "now src/c.rs");
        // a, b, c in order.
        assert_eq!(
            obs.history("S"),
            vec![
                PathBuf::from("src/a.rs"),
                PathBuf::from("src/b.rs"),
                PathBuf::from("src/c.rs"),
            ]
        );
        // Re-touch a -> moves to end (most recent).
        obs.observe("S", "back to src/a.rs");
        assert_eq!(
            obs.history("S"),
            vec![
                PathBuf::from("src/b.rs"),
                PathBuf::from("src/c.rs"),
                PathBuf::from("src/a.rs"),
            ]
        );
        assert_eq!(obs.sessions(), vec!["S".to_string()]);
    }

    #[test]
    fn predict_fs_against_real_repo_finds_mod_children() {
        // Touch this very file's module parent (server.rs) and confirm the
        // policy predicts its real `mod` children from disk. Exercises the
        // filesystem path end-to-end (import parse + existence filter).
        let manifest = env!("CARGO_MANIFEST_DIR");
        let server = PathBuf::from(manifest).join("src/daemon/server.rs");
        if !server.is_file() {
            return; // repo layout changed; don't fail the suite
        }
        let preds = predict_fs(&[server], 32);
        let names: HashSet<String> = preds
            .iter()
            .filter_map(|c| c.path.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();
        // server.rs declares `mod prefetch;` `mod session;` `mod kvstore;` etc.
        assert!(names.contains("prefetch.rs"), "expected prefetch.rs, got {names:?}");
        assert!(names.contains("session.rs"), "expected session.rs, got {names:?}");
        // Every prediction must be a real file on disk.
        assert!(preds.iter().all(|c| c.path.is_file()));
    }

    #[test]
    fn predict_respects_limit_and_is_deterministic() {
        let touched = vec![PathBuf::from("a/x.rs")];
        let read = |_: &Path| None;
        let list = |_: &Path| -> Vec<PathBuf> {
            (0..10).map(|i| PathBuf::from(format!("a/f{i}.rs"))).collect()
        };
        let p1 = predict(&touched, 3, &read, &list);
        let p2 = predict(&touched, 3, &read, &list);
        assert_eq!(p1.len(), 3);
        assert_eq!(p1, p2, "prediction is deterministic for stable inputs");
    }
}