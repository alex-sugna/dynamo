// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{NvCreateCompletionRequest, NvCreateCompletionResponse};
use crate::{
    protocols::{
        common::{self, timing::RequestTracker},
        openai::nvext::{NvExtProvider, NvExtResponse, RawTokenLogprob, RawTopLogprob, TimingInfo},
    },
    types::TokenIdType,
};

impl NvCreateCompletionRequest {
    /// Enables usage tracking for non-streaming requests to comply with OpenAI API specification.
    ///
    /// According to OpenAI API spec, non-streaming completion responses (stream=false)
    /// must always include usage statistics. This method ensures `stream_options.include_usage`
    /// is set to `true` for non-streaming requests.
    ///
    /// Reference: https://platform.openai.com/docs/api-reference/completions/create
    ///
    /// # Arguments
    /// * `original_stream_flag` - The original value of the `stream` field before any internal processing
    pub fn enable_usage_for_nonstreaming(&mut self, original_stream_flag: bool) {
        if !original_stream_flag {
            // For non-streaming requests (stream=false), enable usage by default
            if self.inner.stream_options.is_none() {
                self.inner.stream_options =
                    Some(dynamo_async_openai::types::ChatCompletionStreamOptions {
                        include_usage: true,
                        continuous_usage_stats: false,
                    });
            } else if let Some(ref mut opts) = self.inner.stream_options {
                // If stream_options exists, ensure include_usage is true for non-streaming
                opts.include_usage = true;
            }
        }
    }

    // put this method on the request
    // inspect the request to extract options
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        // Enable tracking if:
        // 1. Client requested timing in extra_fields, OR
        // 2. query_instance_id annotation is present (needs worker_id tracking for response)
        let enable_tracking = self
            .nvext()
            .map(|nv| {
                nv.extra_fields
                    .as_ref()
                    .is_some_and(|fields| fields.iter().any(|f| f == "timing"))
                    || nv.annotations.as_ref().is_some_and(|annots| {
                        annots.iter().any(|a| a.starts_with("query_instance_id"))
                    })
            })
            .unwrap_or(false);

        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            continuous_usage_stats: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.continuous_usage_stats)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(0) > 0,
            enable_tracking,
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    pub enable_usage: bool,
    pub continuous_usage_stats: bool,
    pub enable_logprobs: bool,
    pub enable_tracking: bool,
}

/// Build the lossless raw-logprobs payload that rides through nvext for
/// the SMG-via-gRPC consumer. The OpenAI `Logprobs` shape that
/// `DeltaGenerator::create_logprobs` produces drops `token_id` on every
/// alternative — only the decoded strings remain. SMG needs the
/// token_ids to detokenize on its side. Returns None unless BOTH the
/// per-token chosen logprobs AND the per-position top-k are present.
pub(crate) fn build_raw_logprobs(
    log_probs: Option<&[f64]>,
    top_logprobs: Option<&[Vec<common::llm_backend::TopLogprob>]>,
    generated_token_ids: &[u32],
) -> Option<Vec<RawTokenLogprob>> {
    let lp = log_probs?;
    let tlp = top_logprobs?;
    // Defensive bounds — if the three slices desync (MTP /
    // speculation mismatch, same class as the IndexError in
    // handler_base._extract_logprobs), walk only the shortest.
    let n = lp.len().min(tlp.len()).min(generated_token_ids.len());
    if n == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let position_top: Vec<RawTopLogprob> = tlp[i]
            .iter()
            .map(|t| RawTopLogprob {
                token_id: t.token_id,
                logprob: t.logprob as f32,
            })
            .collect();
        out.push(RawTokenLogprob {
            token_id: generated_token_ids[i],
            logprob: lp[i] as f32,
            top_logprobs: position_top,
        });
    }
    Some(out)
}

pub struct DeltaGenerator {
    id: String,
    object: String,
    created: u32,
    model: String,
    system_fingerprint: Option<String>,
    usage: dynamo_async_openai::types::CompletionUsage,
    options: DeltaGeneratorOptions,
    tracker: Option<Arc<RequestTracker>>,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now: u32 = now.try_into().expect("timestamp exceeds u32::MAX");

        // Previously, our home-rolled CompletionUsage impl'd Default
        // PR !387 - https://github.com/64bit/async-openai/pull/387
        let usage = dynamo_async_openai::types::CompletionUsage {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };

        let completion_id = format!("cmpl-{request_id}");

        // Always create request tracker for per-worker metrics (TTFT, ITL per worker_id).
        // The enable_tracking option only controls whether timing info is included in the response.
        let tracker = Some(Arc::new(RequestTracker::new()));

        Self {
            id: completion_id,
            object: "text_completion".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            usage,
            options,
            tracker,
        }
    }

    /// Returns the request tracker if tracking is enabled, for sharing with PreprocessedRequest.
    pub fn tracker(&self) -> Option<Arc<RequestTracker>> {
        self.tracker.clone()
    }

    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: Vec<TokenIdType>,
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_async_openai::types::Logprobs> {
        if !self.options.enable_logprobs || logprobs.is_none() {
            return None;
        }

        let toks = tokens
            .into_iter()
            .zip(token_ids)
            .map(|(token, token_id)| (token.unwrap_or_default(), token_id))
            .collect::<Vec<(String, TokenIdType)>>();
        let tok_lps = toks
            .iter()
            .zip(logprobs.unwrap())
            .map(|(_, lp)| lp as f32)
            .collect::<Vec<f32>>();

        let top_lps = top_logprobs.map_or(vec![], |top_logprobs| {
            toks.iter()
                .zip(tok_lps.iter())
                .zip(top_logprobs.iter())
                .map(|(((t, tid), lp), top_lps)| {
                    let mut found_selected_token = false;
                    let mut converted_top_lps = top_lps
                        .iter()
                        .map(|top_lp| {
                            let top_t = top_lp.token.clone().unwrap_or_default();
                            let top_tid = top_lp.token_id;
                            found_selected_token = found_selected_token || top_tid == *tid;
                            dynamo_async_openai::types::TopLogprobs {
                                token: top_t,
                                logprob: top_lp.logprob as f32,
                                bytes: None,
                            }
                        })
                        .collect::<Vec<dynamo_async_openai::types::TopLogprobs>>();
                    if !found_selected_token {
                        // If the selected token is not in the top logprobs, add it
                        converted_top_lps.push(dynamo_async_openai::types::TopLogprobs {
                            token: t.clone(),
                            logprob: *lp,
                            bytes: None,
                        });
                    }
                    serde_json::to_value(converted_top_lps).unwrap()
                })
                .collect()
        });

        Some(dynamo_async_openai::types::Logprobs {
            tokens: toks.iter().map(|(t, _)| t.clone()).collect(),
            token_logprobs: tok_lps.into_iter().map(Some).collect(),
            text_offset: vec![],
            top_logprobs: top_lps,
        })
    }

    pub fn create_choice(
        &self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<dynamo_async_openai::types::CompletionFinishReason>,
        logprobs: Option<dynamo_async_openai::types::Logprobs>,
    ) -> NvCreateCompletionResponse {
        // todo - update for tool calling

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![dynamo_async_openai::types::Choice {
                text: text.unwrap_or_default(),
                index,
                finish_reason,
                logprobs,
            }],
            usage: if self.options.enable_usage && self.options.continuous_usage_stats {
                Some(self.get_usage())
            } else {
                None
            },
            nvext: None, // Will be populated by router layer if needed
        };

        NvCreateCompletionResponse { inner }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`NvCreateCompletionResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        let usage = self.get_usage();

        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
            nvext: None, // Will be populated by router layer if needed
        };

        NvCreateCompletionResponse { inner }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }

    /// Check if continuous usage tracking is enabled
    pub fn is_continuous_usage_enabled(&self) -> bool {
        self.options.continuous_usage_stats
    }

    pub fn get_usage(&self) -> dynamo_async_openai::types::CompletionUsage {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        usage
    }
}

impl crate::protocols::openai::DeltaGeneratorExt<NvCreateCompletionResponse> for DeltaGenerator {
    fn choice_from_postprocessor(
        &mut self,
        delta: common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateCompletionResponse> {
        // Aggregate token usage even if usage tracking is disabled for metrics tracking
        // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until context lengths exceed 4_294_967_295.
        let token_length: u32 = delta
            .token_ids
            .len()
            .try_into()
            .expect("token_ids length exceeds u32::MAX");

        self.usage.completion_tokens += token_length;

        // If backend provides completion_usage, use it to update usage stats
        // This is critical for prompt embeddings where prompt_tokens comes from
        // the embedding sequence length computed by the worker
        if let Some(completion_usage) = delta.completion_usage.as_ref() {
            // Update prompt_tokens from worker if provided (e.g., for embeddings)
            self.usage.prompt_tokens = completion_usage.prompt_tokens;

            // Propagate prompt token details if provided
            if let Some(prompt_details) = completion_usage.prompt_tokens_details.as_ref() {
                self.usage.prompt_tokens_details = Some(prompt_details.clone());
            }
        }

        // Snapshot the engine's generated token IDs before `create_logprobs`
        // consumes them. We need them later for the SMG-via-gRPC fallback that
        // injects them into nvext.token_ids.
        let generated_token_ids: Vec<u32> = delta.token_ids.clone();

        // Build the SMG-via-gRPC raw-logprobs side channel. The OpenAI
        // `Logprobs` shape that `create_logprobs` produces drops token_ids
        // on `top_logprobs` alternatives — only the decoded strings
        // remain. SMG's TrtllmService consumer needs token_ids to
        // detokenize via its own tokenizer, so we snapshot the
        // backend-side logprobs (with `token_id` preserved on every
        // entry) here, before `create_logprobs` consumes the fields.
        let raw_logprobs: Option<Vec<RawTokenLogprob>> = build_raw_logprobs(
            delta.log_probs.as_deref(),
            delta.top_logprobs.as_deref(),
            &generated_token_ids,
        );

        let logprobs = self.create_logprobs(
            delta.tokens,
            delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        let finish_reason = delta.finish_reason.map(Into::into);

        // create choice
        let index = delta.index.unwrap_or(0);
        let mut response = self.create_choice(index, delta.text.clone(), finish_reason, logprobs);

        // Get worker_id info from tracker (set by KvPushRouter based on phase)
        let worker_id_info = self.tracker.as_ref().and_then(|t| t.get_worker_info());

        // Prefer GAIE Stage 2 disaggregated_params token_ids (the prompt
        // echo for the prefill→decode handoff). When absent, fall back to
        // the generated token IDs from the engine — external routers like
        // SMG (driving us via TrtllmService gRPC) need raw token IDs to
        // detokenize on their side. HTTP/OpenAI clients ignore unknown
        // nvext fields, so this is additive.
        let token_ids = delta
            .disaggregated_params
            .as_ref()
            .and_then(|params| params.get("token_ids"))
            .and_then(|v| serde_json::from_value::<Vec<u32>>(v.clone()).ok())
            .or_else(|| {
                if !generated_token_ids.is_empty() {
                    Some(generated_token_ids)
                } else {
                    None
                }
            });
        let routed_experts = delta
            .disaggregated_params
            .as_ref()
            .and_then(|params| params.get("routed_experts"))
            .cloned();

        // Get timing info if this is the final response (has finish_reason)
        let timing_info: Option<TimingInfo> = if finish_reason.is_some() {
            self.tracker.as_ref().map(|tracker| {
                tracker.record_finish();
                tracker.get_timing_info()
            })
        } else {
            None
        };

        // Inject nvext if we have worker_id, token_ids, timing, routed
        // experts, or raw logprobs.
        if worker_id_info.is_some()
            || token_ids.is_some()
            || timing_info.is_some()
            || routed_experts.is_some()
            || raw_logprobs.is_some()
        {
            let nvext_response = NvExtResponse {
                worker_id: worker_id_info.clone(),
                timing: timing_info,
                token_ids: token_ids.clone(),
                routed_experts,
                raw_logprobs,
            };

            if let Ok(nvext_json) = serde_json::to_value(&nvext_response) {
                response.inner.nvext = Some(nvext_json);
                if let Some(ref info) = worker_id_info {
                    tracing::debug!(
                        "Injected worker_id into completions nvext: prefill={:?}, decode={:?}",
                        info.prefill_worker_id,
                        info.decode_worker_id
                    );
                }
                if let Some(ref tokens) = token_ids {
                    tracing::debug!(
                        "Injected token_ids into completions nvext: {} tokens",
                        tokens.len()
                    );
                }
            }
        }

        Ok(response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }

    fn create_usage_chunk(&self) -> NvCreateCompletionResponse {
        DeltaGenerator::create_usage_chunk(self)
    }

    fn is_usage_enabled(&self) -> bool {
        DeltaGenerator::is_usage_enabled(self)
    }

    fn is_continuous_usage_enabled(&self) -> bool {
        DeltaGenerator::is_continuous_usage_enabled(self)
    }

    fn get_usage(&self) -> dynamo_async_openai::types::CompletionUsage {
        DeltaGenerator::get_usage(self)
    }

    fn tracker(&self) -> Option<std::sync::Arc<crate::protocols::common::timing::RequestTracker>> {
        self.tracker.clone()
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests pinning the SMG-via-gRPC fallback that surfaces the
    //! engine's generated `delta.token_ids` into `NvExtResponse.token_ids`.
    //!
    //! Without these tests, regressions only surface in a 10-min cluster
    //! round-trip (xp launch → Dynamo build → engine warmup → smoke client).
    //! Local run: `cargo test -p dynamo-llm --lib --no-default-features
    //!     protocols::openai::completions::delta::tests`.
    use super::*;
    use crate::protocols::common::llm_backend::BackendOutput;
    use crate::protocols::openai::DeltaGeneratorExt as _;

    fn fresh_generator() -> DeltaGenerator {
        DeltaGenerator::new(
            "test-model".to_string(),
            DeltaGeneratorOptions::default(),
            "test-req-id".to_string(),
        )
    }

    fn backend_output_with_tokens(token_ids: Vec<u32>) -> BackendOutput {
        BackendOutput {
            token_ids,
            tokens: vec![],
            text: Some("ignored".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            stop_reason: None,
            index: Some(0),
            extra_args: None,
            completion_usage: None,
            disaggregated_params: None,
        }
    }

    /// Pin: the SMG-via-gRPC fallback in `choice_from_postprocessor`.
    /// When `delta.token_ids` is non-empty AND `disaggregated_params.token_ids`
    /// is absent (the common case), the postprocessor must surface the
    /// generated token IDs into `nvext.token_ids` on the response.
    #[test]
    fn choice_from_postprocessor_surfaces_generated_token_ids_to_nvext() {
        let mut dgen = fresh_generator();
        let backend = backend_output_with_tokens(vec![100, 200, 300]);
        let resp = dgen
            .choice_from_postprocessor(backend)
            .expect("postprocessor succeeds");

        let nvext = resp
            .inner
            .nvext
            .as_ref()
            .expect("nvext must be populated when token_ids present");
        let token_ids = nvext
            .get("token_ids")
            .expect("token_ids field must exist in nvext");
        let parsed: Vec<u32> = serde_json::from_value(token_ids.clone())
            .expect("token_ids deserializes as Vec<u32>");
        assert_eq!(parsed, vec![100u32, 200, 300]);
    }

    /// Pin: when `delta.token_ids` is empty (no generation yet, or annotation
    /// frame), nvext should not be injected with a stale token_ids field.
    #[test]
    fn choice_from_postprocessor_skips_nvext_token_ids_when_no_generation() {
        let mut dgen = fresh_generator();
        let backend = backend_output_with_tokens(vec![]);
        let resp = dgen
            .choice_from_postprocessor(backend)
            .expect("postprocessor succeeds");

        // nvext may be None entirely, or it may exist for other reasons
        // (worker_id, timing) but token_ids must not be populated.
        if let Some(nvext) = resp.inner.nvext.as_ref() {
            let has_token_ids = nvext
                .get("token_ids")
                .map(|v| !v.is_null())
                .unwrap_or(false);
            assert!(
                !has_token_ids,
                "token_ids must not be populated when delta has empty token_ids; got {nvext:?}"
            );
        }
    }

    /// Pin: GAIE Stage 2 disaggregated_params.token_ids takes precedence over
    /// the engine-generated fallback. Pre-existing behavior we must not break.
    #[test]
    fn choice_from_postprocessor_prefers_disaggregated_params_token_ids() {
        let mut dgen = fresh_generator();
        let mut backend = backend_output_with_tokens(vec![999, 999, 999]);
        backend.disaggregated_params = Some(serde_json::json!({
            "token_ids": [1u32, 2u32, 3u32],
        }));
        let resp = dgen
            .choice_from_postprocessor(backend)
            .expect("postprocessor succeeds");

        let nvext = resp.inner.nvext.as_ref().expect("nvext populated");
        let token_ids = nvext.get("token_ids").expect("token_ids set");
        let parsed: Vec<u32> = serde_json::from_value(token_ids.clone()).unwrap();
        assert_eq!(
            parsed,
            vec![1u32, 2, 3],
            "disaggregated_params.token_ids must win over delta.token_ids"
        );
    }
}
