use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use super::super::super::models::{LoadedModel, RequestGuard};
use super::super::super::protocol::{ResponseFormat, StreamChunk, Timings};
use super::super::session::common_prefix_len;
use super::super::Daemon;
use super::common::{generate_tokens, resolve_grammar, TokenSink};
use crate::token::TokenId;
use crate::{MullamaError, SamplerParams};

/// Cross-turn KV reuse inputs (see [`crate::daemon::server::session`]).
#[allow(private_interfaces)]
pub(crate) struct KvReuse {
    pub(crate) slot: usize,
    pub(crate) cached_tokens: Vec<TokenId>,
    /// Seq-state blob to hydrate the pinned slot's KV with on the first turn
    /// after a durable restore (fresh daemon). `None` for an in-memory hit —
    /// the KV is already in the slot.
    pub(crate) restore: Option<Vec<u8>>,
}

impl Daemon {
    /// Generate text without streaming.
    ///
    /// When `kv_reuse` is `Some`, the pinned context slot's KV cache is reused:
    /// only the new prompt suffix (beyond the cached prefix) is prefilled. The
    /// returned `Option<Vec<TokenId>>` is the updated cached-token sequence
    /// (prompt prefix + this turn's generated tokens) the caller writes back to
    /// the session store. When `kv_reuse` is `None`, the stock stateless path
    /// runs (clear + full decode) and `None` is returned.
    #[allow(private_interfaces)]
    pub async fn generate_text(
        &self,
        loaded: &LoadedModel,
        prompt: &str,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
        response_format: Option<&ResponseFormat>,
        kv_reuse: Option<KvReuse>,
    ) -> Result<(String, u32, u32, Timings, Option<Vec<TokenId>>, Option<Vec<u8>>), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let grammar_gbnf = resolve_grammar(response_format);
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        // Pin to the session's slot for reuse, else round-robin.
        let mut context = match &kv_reuse {
            Some(r) => loaded.acquire_context_at(r.slot).await,
            None => loaded.acquire_context().await,
        };
        let model = loaded.model.clone();

        let result = tokio::task::block_in_place(|| {
            let tokens = model.tokenize(prompt, add_bos, true)?;
            let prompt_tokens = tokens.len() as u32;

            let mut cached_tokens: Option<Vec<TokenId>> =
                kv_reuse.as_ref().map(|r| r.cached_tokens.clone());

            // Durable restore: hydrate the pinned slot's KV from the persisted
            // blob before the reuse logic runs. Uses the per-sequence
            // save/load (only the used positions, not the whole n_ctx
            // allocation — a full-context save would be ~hundreds of MB and
            // seconds of memcpy at n_ctx=8192). After load_state_seq the KV
            // holds the cached prefix at its original positions, so the prefix
            // match + seq_rm + delta-decode below behaves exactly as an
            // in-memory reuse hit. A failed load (incompatible build/format)
            // falls back to a full decode — never incorrect. The save side
            // uses the matching `save_state_seq`, so the blob format matches.
            if let Some(blob) = kv_reuse.as_ref().and_then(|r| r.restore.as_ref()) {
                context.kv_cache_clear();
                if context.load_state_seq(0, blob).is_err() {
                    // Restore refused by llama.cpp (state-version/format
                    // mismatch): treat as a fresh session. Never incorrect —
                    // the reuse path falls through to a full decode below.
                    cached_tokens = Some(Vec::new());
                    context.kv_cache_clear();
                }
            }

            let prompt_eval_ns = if let Some(reuse) = &kv_reuse {
                // Cross-turn reuse: keep the shared prefix, drop the divergent
                // tail from the KV, and decode only the new suffix. Positions
                // auto-continue from seq_pos_max+1 (llama_batch_get_one with
                // pos=null), so no explicit-position batch is required.
                let cached = cached_tokens.as_ref().unwrap_or(&reuse.cached_tokens);
                let l = common_prefix_len(cached, &tokens);
                let delta_empty = l >= tokens.len();
                if l == 0 || delta_empty {
                    // No reusable prefix, or the whole prompt is already cached.
                    // Either way there's no delta to prefill, so do a full
                    // re-decode via the *same* clear path as the stateless branch.
                    // (seq_rm(0,0,-1) + decode leaves residual context state that
                    // shifts greedy numerics; kv_cache_clear matches the
                    // stateless path bit-for-bit, preserving parity.)
                    context.kv_cache_clear();
                    let s = Instant::now();
                    context.decode(&tokens)?;
                    let ns = s.elapsed().as_nanos() as u64;
                    cached_tokens = Some(tokens.clone());
                    ns
                } else {
                    // Real reuse: keep prefix [0, l), drop the divergent tail
                    // [l, inf) from the KV, and decode only tokens[l..]. The
                    // prefix K/V was computed in a prior turn at the same
                    // positions, so this is numerically identical to a full
                    // decode while prefilling only the new suffix.
                    if l < cached.len() {
                        context.kv_cache_seq_rm(0, l as i32, -1);
                    }
                    let s = Instant::now();
                    context.decode(&tokens[l..])?;
                    let ns = s.elapsed().as_nanos() as u64;
                    if let Some(c) = cached_tokens.as_mut() {
                        c.truncate(l);
                        c.extend_from_slice(&tokens[l..]);
                    }
                    ns
                }
            } else {
                context.kv_cache_clear();
                let s = Instant::now();
                context.decode(&tokens)?;
                s.elapsed().as_nanos() as u64
            };

            // Snapshot the per-sequence KV state for durable persistence (caller
            // writes it back to the KV store via the session). Only meaningful
            // for the reuse path; the stateless branch returns None and isn't
            // pinned. Must be the seq variant to match the `load_state_seq`
            // restore path — only the used cells are written, not the whole
            // n_ctx allocation, so the blob is small (KB–low MB) and the load
            // is fast enough to preserve the delta-prefill win.
            let seq_state = if kv_reuse.is_some() {
                Some(context.save_state_seq(0))
            } else {
                None
            };

            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Repetition penalties include prompt history. Seed the base
            // sampler before adding grammar so prompt tokens affect penalties
            // without being consumed as generated grammar tokens.
            for &token in &tokens {
                sampler.accept(token);
            }

            if let Some(gbnf) = &grammar_gbnf {
                let grammar_sampler =
                    crate::sampling::Sampler::grammar(model.clone(), gbnf, "root")?;
                sampler.add(grammar_sampler);
            }

            let gen_result = generate_tokens(
                &mut *context,
                &model,
                &mut sampler,
                max_tokens,
                &stop_sequences,
                max_stop_len,
                &TokenSink::Buffer,
            )?;

            // Extend the cached sequence with this turn's decoded tokens so the
            // next turn's prefix match includes the assistant reply.
            if let Some(c) = cached_tokens.as_mut() {
                c.extend_from_slice(&gen_result.generated_tokens);
            }

            Ok::<_, MullamaError>((
                gen_result,
                prompt_tokens,
                prompt_eval_ns,
                cached_tokens,
                seq_state,
            ))
        })?;

        self.models.add_tokens(result.0.completion_tokens as u64);

        let timings = Timings {
            prompt_eval_ns: result.2,
            eval_ns: result.0.eval_ns,
            prompt_tokens: result.1,
            completion_tokens: result.0.completion_tokens,
        };
        Ok((
            result.0.generated,
            result.1,
            result.0.completion_tokens,
            timings,
            result.3,
            result.4,
        ))
    }

    /// Generate text with streaming.
    pub async fn generate_text_streaming(
        &self,
        loaded: Arc<LoadedModel>,
        prompt: String,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        let add_bos = loaded.model.add_bos_token();
        let model_for_tokenize = loaded.model.clone();
        let tokens =
            tokio::task::block_in_place(|| model_for_tokenize.tokenize(&prompt, add_bos, false))?;
        let prompt_tokens = tokens.len() as u32;

        let (setup, rx, request_id) = self.prepare_streaming(stop_sequences);
        let model = loaded.model.clone();

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;

            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();
                let mut sampler = sampler_params.build_chain(model.clone())?;
                for &token in &tokens {
                    sampler.accept(token);
                }
                context.decode(&tokens)?;

                generate_tokens(
                    &mut *context,
                    &model,
                    &mut sampler,
                    max_tokens,
                    &setup.stop_sequences,
                    setup.max_stop_len,
                    &TokenSink::Stream {
                        tx: &setup.tx,
                        request_id: setup.request_id_arc.clone(),
                        cancel_flag: &setup.cancel_flag,
                    },
                )
            });

            setup.finish(&result);
        });

        Ok((rx, prompt_tokens, request_id))
    }
}
