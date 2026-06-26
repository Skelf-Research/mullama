//! Prompt-lookup (n-gram) speculative decoding — single model, greedy-exact.
//!
//! Classic speculative decoding needs a small *draft* model to propose tokens
//! that a big *target* model verifies. Prompt-lookup decoding removes the draft
//! model entirely: it proposes the next tokens by finding where the current
//! suffix last occurred in the token history and copying what followed. This is
//! astonishingly effective for the workloads agents produce — code, file paths,
//! repeated identifiers, JSON, quoted context — where the continuation is very
//! often a verbatim repeat of something already in the prompt or output.
//!
//! The verification is a single batched target forward pass over the proposed
//! tokens (see [`Context::decode_batch_argmax`]). We accept the longest prefix
//! of the proposal that matches the target's own greedy argmax at each
//! position, then take the target's argmax at the first mismatch as the next
//! "real" token. Because every accepted token equals the target's greedy
//! choice, the output is **token-for-token identical** to plain greedy
//! decoding — speculation here is a pure latency optimization, never a quality
//! trade-off. That makes it safe to verify: run with and without it and the
//! token stream must match exactly.
//!
//! Speedup comes from amortizing K+1 tokens of generation into one target
//! forward pass whenever the n-gram guess is right. On a miss we fall back to
//! exactly one token (the target argmax), so the worst case is plain greedy
//! decoding plus the (cheap) n-gram lookup.

use std::collections::HashMap;

use crate::{token::TokenId, Context, Model, MullamaError};

/// Configuration for prompt-lookup speculative decoding.
#[derive(Debug, Clone)]
pub struct PromptLookupConfig {
    /// Length of the suffix n-gram to match against history. Longer = more
    /// precise matches (higher acceptance when it hits) but fewer hits.
    pub ngram: usize,
    /// Max tokens to propose per step (the draft length K).
    pub max_draft: usize,
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        // n=2 / K=8 are robust defaults from the prompt-lookup literature for
        // code/agent text.
        Self {
            ngram: 2,
            max_draft: 8,
        }
    }
}

/// Per-run speculative-decoding statistics, enough to report acceptance rate
/// and the realized speedup.
#[derive(Debug, Clone, Default)]
pub struct PromptLookupStats {
    /// Number of speculation rounds (target forward passes).
    pub rounds: usize,
    /// Tokens proposed by the n-gram drafter across all rounds.
    pub drafted: usize,
    /// Drafted tokens that matched the target's greedy choice (accepted).
    pub accepted: usize,
    /// Total tokens emitted (accepted drafts + the per-round argmax token).
    pub emitted: usize,
    /// Rounds where the drafter proposed nothing (no n-gram match).
    pub draft_misses: usize,
}

impl PromptLookupStats {
    /// Fraction of drafted tokens that were accepted (0.0 if none drafted).
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }

    /// Realized speedup vs plain greedy: tokens emitted per target forward
    /// pass. 1.0 means no benefit (every round emitted exactly one token);
    /// >1.0 means speculation amortized multiple tokens per pass.
    pub fn tokens_per_pass(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            self.emitted as f32 / self.rounds as f32
        }
    }
}

/// Propose up to `max_draft` continuation tokens for `context_tokens` by
/// finding the most recent earlier occurrence of its last `ngram` tokens and
/// copying what followed. Returns an empty vec when there's no match.
///
/// Pure and deterministic — unit tested without a model.
pub fn propose_draft(
    context_tokens: &[TokenId],
    ngram: usize,
    max_draft: usize,
) -> Vec<TokenId> {
    if ngram == 0 || max_draft == 0 || context_tokens.len() <= ngram {
        return Vec::new();
    }
    let suffix = &context_tokens[context_tokens.len() - ngram..];
    // Search backwards for the most recent earlier occurrence of `suffix`
    // (excluding the suffix's own position at the very end).
    let search_end = context_tokens.len() - ngram; // exclusive of the live suffix
    let mut i = search_end; // candidate match starts in [0, search_end)
    while i > 0 {
        i -= 1;
        if context_tokens[i..i + ngram] == *suffix {
            // Copy what followed this earlier occurrence.
            let after = i + ngram;
            let end = (after + max_draft).min(context_tokens.len());
            if after < end {
                return context_tokens[after..end].to_vec();
            }
        }
    }
    Vec::new()
}

/// Given a draft and the target's greedy argmax at each draft position plus the
/// position before it, compute the accepted tokens.
///
/// `argmax_at[j]` is the target's greedy choice for the token *following*
/// `draft[j-1]` (with `argmax_at[0]` being the choice that follows the last
/// real token — i.e. what the target would emit with no draft). We accept
/// `draft[j]` as long as it equals `argmax_at[j]`; at the first mismatch we
/// take `argmax_at[j]` as the corrected token and stop. The result is always
/// non-empty (at least the first argmax token), and identical to what greedy
/// decoding would have produced for these positions.
///
/// Returns `(accepted_tokens, n_draft_accepted)`.
pub fn accept_run(draft: &[TokenId], argmax_at: &[TokenId]) -> (Vec<TokenId>, usize) {
    // argmax_at has one more entry than draft: the leading "no-draft" argmax
    // plus one per drafted token.
    debug_assert_eq!(argmax_at.len(), draft.len() + 1);
    let mut out = Vec::with_capacity(draft.len() + 1);
    let mut n_accepted = 0;
    for (j, &d) in draft.iter().enumerate() {
        // The target's prediction *before* consuming draft[j] is argmax_at[j].
        if argmax_at[j] == d {
            out.push(d);
            n_accepted += 1;
        } else {
            // Mismatch: emit the target's own choice here and stop.
            out.push(argmax_at[j]);
            return (out, n_accepted);
        }
    }
    // All drafts accepted: also emit the argmax that follows the last draft.
    out.push(argmax_at[draft.len()]);
    (out, n_accepted)
}

/// Greedy generation accelerated with prompt-lookup speculation.
///
/// `context` must already hold the prompt's KV (decoded, seq 0). `prompt_tokens`
/// is the prompt token sequence (used to seed the n-gram history and KV
/// position). Generates up to `max_new` tokens greedily, returning the
/// generated token ids and the run's [`PromptLookupStats`]. The output equals
/// plain greedy decoding token-for-token.
pub fn generate_greedy(
    context: &mut Context,
    model: &Model,
    prompt_tokens: &[TokenId],
    max_new: usize,
    config: &PromptLookupConfig,
) -> Result<(Vec<TokenId>, PromptLookupStats), MullamaError> {
    let mut stats = PromptLookupStats::default();
    let mut history: Vec<TokenId> = prompt_tokens.to_vec();
    let mut generated: Vec<TokenId> = Vec::with_capacity(max_new);
    // Next KV position to write: the prompt already occupies [0, len).
    let mut next_pos = prompt_tokens.len() as i32;

    while generated.len() < max_new {
        // Propose a draft from the n-gram history.
        let remaining = max_new - generated.len();
        let draft = propose_draft(&history, config.ngram, config.max_draft.min(remaining));
        if draft.is_empty() {
            stats.draft_misses += 1;
        }
        stats.drafted += draft.len();

        // Build the verification batch: the last real token, followed by the
        // draft. Decoding this gives the target's argmax at each position:
        // argmax_at[0] follows the last real token, argmax_at[j+1] follows
        // draft[j]. We need the last real token's argmax, so prepend it.
        let last_real = *history.last().expect("history seeded by prompt");
        let mut batch: Vec<TokenId> = Vec::with_capacity(draft.len() + 1);
        batch.push(last_real);
        batch.extend_from_slice(&draft);

        // The last real token is already in the KV (decoded as part of the
        // prompt or a prior round), so re-decoding it here would duplicate it.
        // Instead we decode only the *new* positions (the draft) and obtain the
        // last-real argmax from the position already in the cache. To keep one
        // clean batched call, we roll the KV back by one and re-decode the last
        // real token together with the draft.
        context.kv_cache_seq_rm(0, next_pos - 1, -1);
        let argmax_at = context.decode_batch_argmax(&batch, next_pos - 1)?;
        stats.rounds += 1;

        let (accepted, n_draft_ok) = accept_run(&draft, &argmax_at);
        stats.accepted += n_draft_ok;

        // The KV now contains last_real + the full draft at [next_pos-1 ..].
        // Keep only last_real + the accepted tokens; drop the rest.
        // Accepted layout in KV positions (relative to next_pos-1):
        //   pos 0      = last_real (keep)
        //   pos 1..=n_draft_ok = accepted draft tokens (keep)
        //   the corrected/extra argmax token is NOT yet in the KV (it was the
        //   *prediction*, not a decoded token), so nothing to trim for it, but
        //   any drafted-but-rejected tokens beyond n_draft_ok ARE in the KV.
        let kept_end = next_pos - 1 + 1 + n_draft_ok as i32; // exclusive
        context.kv_cache_seq_rm(0, kept_end, -1);

        // Emit accepted tokens, stopping at EOG.
        let mut hit_eog = false;
        let mut emitted_this_round = 0usize;
        for &tok in &accepted {
            if generated.len() >= max_new {
                break;
            }
            if model.token_is_eog(tok) {
                hit_eog = true;
                break;
            }
            generated.push(tok);
            history.push(tok);
            emitted_this_round += 1;
        }
        stats.emitted += emitted_this_round;

        // The corrected/extra argmax token (the last element of `accepted` when
        // all drafts matched, or the mismatch token otherwise) was emitted but
        // is NOT yet in the KV — decode it so the next round continues from it.
        // We advance next_pos by the number of tokens now genuinely in the KV:
        // last_real was at next_pos-1; accepted draft tokens filled up to
        // kept_end. The final emitted token still needs decoding.
        next_pos = kept_end;
        if hit_eog {
            break;
        }
        if let Some(&last_emitted) = generated.last() {
            // Decode the final emitted token (the corrected/extra one) into KV.
            // If it was an accepted *draft* token it's already in KV; only the
            // trailing argmax token (beyond the accepted drafts) is missing.
            if emitted_this_round > n_draft_ok {
                context.decode_batch_argmax(&[last_emitted], next_pos)?;
                next_pos += 1;
            }
        }
    }

    Ok((generated, stats))
}

/// Plain greedy decoding (no speculation), for parity comparison. Decodes one
/// token at a time via the target's argmax. Returns the generated tokens.
pub fn generate_greedy_baseline(
    context: &mut Context,
    model: &Model,
    prompt_tokens: &[TokenId],
    max_new: usize,
) -> Result<Vec<TokenId>, MullamaError> {
    let mut generated = Vec::with_capacity(max_new);
    let mut next_pos = prompt_tokens.len() as i32;
    let mut last = *prompt_tokens.last().expect("non-empty prompt");
    // Roll back and re-decode the last prompt token to read its argmax.
    context.kv_cache_seq_rm(0, next_pos - 1, -1);
    for _ in 0..max_new {
        let am = context.decode_batch_argmax(&[last], next_pos - 1)?;
        let tok = am[0];
        if model.token_is_eog(tok) {
            break;
        }
        generated.push(tok);
        last = tok;
        next_pos += 1;
    }
    Ok(generated)
}

/// Build a HashMap n-gram index (unused by the linear scanner above, but handy
/// for very long histories). Kept minimal; the linear backward scan is fast
/// enough for agent-sized histories.
#[allow(dead_code)]
fn ngram_index(tokens: &[TokenId], ngram: usize) -> HashMap<&[TokenId], usize> {
    let mut idx = HashMap::new();
    if tokens.len() < ngram {
        return idx;
    }
    for i in 0..=tokens.len() - ngram {
        idx.insert(&tokens[i..i + ngram], i + ngram);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_draft_copies_after_last_match() {
        // history: a b c X a b ?  -> suffix "a b" last matched at index 0,
        // followed by c. Most-recent earlier match copies what followed it.
        let h: Vec<TokenId> = vec![1, 2, 3, 9, 1, 2];
        let d = propose_draft(&h, 2, 4);
        // suffix is [1,2] (the tail); earlier [1,2] at index 0 -> followed by
        // tokens [3,9,1,2], copied up to max_draft=4.
        assert_eq!(d, vec![3, 9, 1, 2]);
        // With a tighter cap it copies fewer.
        assert_eq!(propose_draft(&h, 2, 2), vec![3, 9]);
    }

    #[test]
    fn propose_draft_no_match_is_empty() {
        let h: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        assert!(propose_draft(&h, 2, 4).is_empty());
    }

    #[test]
    fn propose_draft_respects_max_and_ngram_bounds() {
        assert!(propose_draft(&[1, 2, 3], 0, 4).is_empty());
        assert!(propose_draft(&[1, 2, 3], 2, 0).is_empty());
        assert!(propose_draft(&[1, 2], 2, 4).is_empty()); // len == ngram
    }

    #[test]
    fn accept_run_all_match_emits_one_extra() {
        let draft = vec![5, 6, 7];
        // target argmax agrees on all three, plus a trailing token 8.
        let argmax = vec![5, 6, 7, 8];
        let (out, n) = accept_run(&draft, &argmax);
        assert_eq!(out, vec![5, 6, 7, 8]);
        assert_eq!(n, 3);
    }

    #[test]
    fn accept_run_mismatch_truncates_and_corrects() {
        let draft = vec![5, 6, 7];
        // target agrees on 5, then would emit 99 instead of 6.
        let argmax = vec![5, 99, 7, 8];
        let (out, n) = accept_run(&draft, &argmax);
        assert_eq!(out, vec![5, 99]); // accept 5, correct to 99, stop
        assert_eq!(n, 1);
    }

    #[test]
    fn accept_run_first_token_mismatch() {
        let draft = vec![5, 6];
        let argmax = vec![42, 6, 7];
        let (out, n) = accept_run(&draft, &argmax);
        assert_eq!(out, vec![42]);
        assert_eq!(n, 0);
    }

    #[test]
    fn stats_acceptance_and_speedup() {
        let mut s = PromptLookupStats::default();
        s.rounds = 4;
        s.drafted = 12;
        s.accepted = 9;
        s.emitted = 16;
        assert!((s.acceptance_rate() - 0.75).abs() < 1e-6);
        assert!((s.tokens_per_pass() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn empty_run_stats_are_zero_not_nan() {
        let s = PromptLookupStats::default();
        assert_eq!(s.acceptance_rate(), 0.0);
        assert_eq!(s.tokens_per_pass(), 0.0);
    }
}