// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error as StdError;
use std::sync::Arc;

use anyhow::{Error, Result};
use futures::{stream, stream::StreamExt};

use crate::{
    http::service::metrics::Metrics, model_card::ModelDeploymentCard, preprocessor::BackendOutput,
    protocols::common::llm_backend::PreprocessedRequest,
};

use dynamo_runtime::error::{self, BackendError, DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, Operator, ResponseStream,
    ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::{annotated::Annotated, maybe_error::MaybeError};

/// Check if an error chain indicates the request should be migrated.
fn is_migratable(err: &(dyn StdError + 'static)) -> bool {
    const MIGRATABLE: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    const NON_MIGRATABLE: &[ErrorType] = &[
        // Future: ErrorType::Cancelled, ErrorType::ValidationError, etc.
    ];
    if error::match_error_chain(err, MIGRATABLE, NON_MIGRATABLE) {
        return true;
    }

    // Migrate on two `KvSchedulerError` variants reachable on the decode-
    // side scheduler post-prefill:
    //
    //   * `NoEndpoints` — a decode worker is briefly absent from
    //     `RuntimeConfigWatch` (push_router `report_instance_down` from a
    //     stall, or a watch reconnect blip). The retry re-runs PrefillRouter
    //     → `compute_allowed_prefills` sees a smaller `live_partitions` set,
    //     excludes the dead partition, and routes through a different one.
    //
    //   * `AdmissionRejected` — a decode worker passed the prefill admit
    //     check, but by the time the request reaches the decode-side
    //     scheduler some decodes in the partition have filled (race between
    //     prefill execution and decode dispatch, ~prefill_latency seconds).
    //     The retry re-runs PrefillRouter and the per-role admission filter,
    //     which now picks a different partition with capacity. If the cluster
    //     is genuinely saturated everywhere, the retry will hit
    //     AdmissionRejected again at the prefill_router level and bubble out
    //     as 429 after `migration_limit` attempts — that's the intended
    //     terminal 429 path. We bound waste by `migration_limit` (currently
    //     2 → 3 attempts).
    //
    // NOTE: we make AdmissionRejected migratable globally rather than only
    // distinguishing "decode-side AdmissionRejected" from "prefill-router-
    // level AdmissionRejected" — the migration retry budget bounds the cost
    // of mis-treating a fleet-wide saturation as decode-local, and the
    // prefill_router admission rerun on retry sees the same fleet state and
    // fast-fails the same way. Net effect is identical to the typed split,
    // simpler code.
    // Typed downcast only — intentional. The chain structure asymmetry
    // between prefill-side and decode-side rejections is the load-bearing
    // distinction:
    //
    //   * Decode-side: `KvSchedulerError` reaches here NOT wrapped by
    //     `PrefillError`, so the typed downcast finds it. Variants
    //     `NoEndpoints` + `AdmissionRejected` are migratable — a different
    //     partition's decode may have capacity.
    //   * Prefill-side: `KvSchedulerError` is boxed inside `PrefillError::
    //     PrefillError(String, Box<dyn Error>(anyhow::Error))`, and the
    //     boxed anyhow::Error's trait-object concrete type erases the
    //     inner KvSchedulerError from typed downcast. Walk hits anyhow's
    //     trait-object layer and stops. is_migratable returns false →
    //     non-migratable → bubbles out as terminal 429 via
    //     `grpc/service/openai.rs::classify_setup_error`'s substring
    //     fallback. This is correct: prefill-side saturation is fleet-
    //     wide; immediate retry would hit the same fleet state.
    //
    // If the wrapping ever changes (e.g. PrefillError grows a typed
    // `Admission(KvSchedulerError)` variant), update this comment + the
    // companion classify_setup_error to keep the two paths in sync.
    let mut current: Option<&(dyn StdError + 'static)> = Some(err);
    while let Some(e) = current {
        if let Some(kse) =
            e.downcast_ref::<crate::kv_router::scheduler::KvSchedulerError>()
        {
            if matches!(
                kse,
                crate::kv_router::scheduler::KvSchedulerError::NoEndpoints
                    | crate::kv_router::scheduler::KvSchedulerError::AdmissionRejected { .. }
            ) {
                return true;
            }
        }
        current = e.source();
    }
    false
}

pub struct Migration {
    migration_limit: u32,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
}

impl Migration {
    pub fn new(migration_limit: u32, model_name: String, metrics: Arc<Metrics>) -> Arc<Self> {
        tracing::debug!("model {} migration limit {}", model_name, migration_limit);
        Arc::new(Self {
            migration_limit,
            model_name: Arc::new(model_name),
            metrics,
        })
    }

    pub fn from_mdc(
        mdc: &ModelDeploymentCard,
        migration_limit: u32,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        Self::new(migration_limit, mdc.display_name.clone(), metrics)
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for Migration
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>> {
        let (preprocessed_request, context) = request.transfer(());
        let engine_ctx = context.context();
        let engine_ctx_ = engine_ctx.clone();
        // [pin_trace:3/migration] Cross-link the internal UUID (`request_id`)
        // with the chatcmpl-* (`client_request_id`) that came in via gRPC
        // annotations. This is the bridge log: every downstream frontend log
        // uses request_id=UUID, but the decode worker logs request_id=chatcmpl-*.
        // To trace one probe end-to-end, find its chatcmpl_id here, then grep
        // both IDs across frontend + decode logs.
        let client_request_id: Option<String> = preprocessed_request
            .annotations
            .iter()
            .find_map(|s| s.strip_prefix("request_id=").map(|x| x.to_string()));
        tracing::info!(
            target: "dynamo::pin_trace",
            request_id = %engine_ctx_.id(),
            client_request_id = ?client_request_id,
            model = %self.model_name,
            migration_limit = self.migration_limit,
            routing_prefill_worker_id = ?preprocessed_request.routing.as_ref().and_then(|r| r.prefill_worker_id),
            routing_decode_worker_id = ?preprocessed_request.routing.as_ref().and_then(|r| r.decode_worker_id),
            routing_backend_instance_id = ?preprocessed_request.routing.as_ref().and_then(|r| r.backend_instance_id),
            decode_instance_id = ?preprocessed_request.decode_instance_id,
            "[pin_trace:3/migration] enter Migration::generate"
        );
        let retry_manager = RetryManager::build(
            engine_ctx,
            preprocessed_request,
            next,
            self.migration_limit,
            self.model_name.clone(),
            self.metrics.clone(),
        )
        .await?;
        let response_stream = stream::unfold(retry_manager, move |mut retry_manager| async move {
            retry_manager
                .next()
                .await
                .map(|response| (response, retry_manager))
        })
        .fuse();
        Ok(ResponseStream::new(Box::pin(response_stream), engine_ctx_))
    }
}

struct RetryManager {
    context: Arc<dyn AsyncEngineContext>,
    request: PreprocessedRequest,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>>,
    next_stream: Option<ManyOut<Annotated<BackendOutput>>>,
    retries_left: u32,
    /// Cumulative retries fired across this request's lifecycle. Used by the
    /// `[migration_summary]` log emitted whenever new_stream() recovers a
    /// request that needed at least one retry. Persists across multiple
    /// new_stream() invocations (initial attempt + mid-stream retries).
    retries_used: u32,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
}

impl RetryManager {
    pub async fn build(
        context: Arc<dyn AsyncEngineContext>,
        preprocessed_request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>>,
        retries_left: u32,
        model_name: Arc<String>,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        // If the caller pinned a specific worker (via routing.prefill_worker_id /
        // decode_worker_id / backend_instance_id, or top-level decode_instance_id),
        // their intent is "use exactly this worker." Migration retry would clear
        // the pin and re-pick a different worker — silently overriding the
        // targeting and (critically) making the e2e health probe pass even when
        // the targeted worker is dead. Force retries=0 for pinned requests so the
        // initial attempt is the only attempt and failures surface to the caller.
        let pinned_field = preprocessed_request
            .routing
            .as_ref()
            .and_then(|r| {
                if r.prefill_worker_id.is_some() {
                    Some("routing.prefill_worker_id")
                } else if r.decode_worker_id.is_some() {
                    Some("routing.decode_worker_id")
                } else if r.backend_instance_id.is_some() {
                    Some("routing.backend_instance_id")
                } else {
                    None
                }
            })
            .or_else(|| {
                preprocessed_request
                    .decode_instance_id
                    .map(|_| "request.decode_instance_id")
            });
        let effective_retries = match pinned_field {
            Some(field) => {
                // Always log when we see a pin, even when requested_retries was
                // already 0 — this confirms my pin-detection code is actually
                // running for the probe (the previous gated-on-retries>0 log
                // hid the "limit=0 + pinned" case, which is the dominant
                // failure mode being diagnosed right now).
                tracing::info!(
                    model = %model_name,
                    requested_retries = retries_left,
                    pinned_via = field,
                    "[migration] caller pinned a worker; effective retries=0 (preserves targeting for probe / EPP)"
                );
                0
            }
            None => retries_left,
        };
        let mut slf = Self {
            context,
            request: preprocessed_request,
            next_generate: next,
            next_stream: None,
            retries_left: effective_retries + 1, // +1 to account for the initial attempt
            retries_used: 0,
            model_name,
            metrics,
        };
        slf.new_stream().await?;
        Ok(slf)
    }

    pub async fn next(&mut self) -> Option<Annotated<BackendOutput>> {
        loop {
            let response_stream = match self.next_stream.as_mut() {
                Some(stream) => stream,
                None => {
                    tracing::error!("next() called with next_stream is None - should not happen");
                    return Some(Annotated::from_err(DynamoError::msg("next_stream is None")));
                }
            };
            if let Some(response) = response_stream.next().await {
                // Check if this is a migratable error that should trigger stream recreation.
                if let Some(err) = response.err() {
                    if is_migratable(&err) {
                        tracing::warn!(
                            request_id = %self.context.id(),
                            error = %err,
                            "Stream disconnected... recreating stream..."
                        );
                        self.metrics.inc_migration_ongoing_request(&self.model_name);
                        if let Err(err) = self.new_stream().await {
                            tracing::warn!(
                                request_id = %self.context.id(),
                                error = %format!("{err:#}"),
                                "Cannot recreate stream"
                            );
                        } else {
                            continue;
                        }
                    } else {
                        // Non-migratable error mid-stream. Before this log, the path was silent:
                        // we'd fall through to track_response + return without any indication
                        // that migration decided NOT to retry. Surfacing this is essential for
                        // diagnosing "why didn't migration recover" cases.
                        tracing::warn!(
                            request_id = %self.context.id(),
                            error = %err,
                            "[migration] mid-stream non-migratable error; propagating to client"
                        );
                    }
                }
                self.track_response(&response);
                return Some(response);
            }
            return None;
        }
    }

    async fn new_stream(&mut self) -> Result<()> {
        // If there's a stale stream, cancel the in-flight work on the prior
        // worker before we drop the consumer side. Best-effort: a hung worker
        // won't observe the cancel anyway, and a recovering worker will see
        // it and stop generating into a queue that's already detached.
        if let Some(prev) = self.next_stream.take() {
            prev.context().stop_generating();
        }

        // The field-clearing (bootstrap_info, prefill_result, prefill_worker_id,
        // decode_worker_id, backend_instance_id, decode_instance_id) used to live
        // here, outside the retry loop — which meant it fired on the initial
        // attempt too, destroying the caller's targeting before the routing
        // layer ever saw it. Health probes (which pin via backend_instance_id /
        // decode_instance_id) became silently broken: the cleared request was
        // routed to a different live worker, the probe got 200, the dead worker
        // looked healthy. Clearing belongs only on real retries — move it
        // inside the loop and gate on `is_retry`.
        let mut is_retry = false;
        let mut response_stream: Option<Result<ManyOut<Annotated<BackendOutput>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            if is_retry {
                self.retries_used += 1;
                // Clear disagg handoff state + pinned-worker routing hints so
                // the retry gets a fresh prefill → fresh bootstrap_info →
                // fresh decode pick, AND so the retry doesn't re-route via
                // the preselected_id path to the (now-dead) worker we just
                // failed on (kv_router/push_router.rs::select_worker — the
                // phase-dispatched preselected_id lookup bypasses
                // find_best_match's avail filter).
                self.request.bootstrap_info = None;
                self.request.prefill_result = None;
                if let Some(routing) = self.request.routing.as_mut() {
                    routing.prefill_worker_id = None;
                    routing.decode_worker_id = None;
                    routing.backend_instance_id = None;
                    // Don't clear allowed_worker_ids — that's the upstream
                    // client's intent (e.g. partitioning), not a per-attempt
                    // routing stamp.
                }
                self.request.decode_instance_id = None;
                tracing::info!(
                    model = %self.model_name,
                    retries_left = self.retries_left,
                    request_id = %self.context.id(),
                    "[migration] retry attempt: cleared pinned routing hints + disagg handoff state"
                );

                // Jittered backoff before re-admitting through the pipeline.
                // When a worker dies with N in-flight requests, all N hit the
                // retry path within the stall-timeout window (~10ms after
                // report_instance_down fires). Without jitter, all N rejoin
                // the admission filter simultaneously and burst-saturate the
                // surviving workers — most get rejected as 429 even though
                // there was capacity if requests had arrived spread out.
                //
                // `DYN_MIGRATION_RETRY_JITTER_MS` controls the jitter window.
                // Each retry sleeps `uniform(0, jitter_ms)` before its next
                // attempt at next_generate. Unset / 0 disables (current
                // behavior — no jitter). Sleeps are cancellable via the
                // request context: if the client disconnects mid-sleep we
                // exit the migration loop with an error.
                //
                // The sleep does NOT block the frontend tokio runtime:
                // tokio::time::sleep yields the executor. Other requests on
                // other tasks proceed normally. The request holds no
                // admission slot / GPU resource / routing decision during
                // the sleep — it's invisible to the admission filter until
                // next_generate.generate() actually runs.
                let jitter_ms = std::env::var("DYN_MIGRATION_RETRY_JITTER_MS")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                if jitter_ms > 0 {
                    use rand::Rng;
                    let delay_ms = rand::thread_rng().gen_range(0..jitter_ms);
                    tracing::debug!(
                        request_id = %self.context.id(),
                        retries_used = self.retries_used,
                        delay_ms,
                        jitter_window_ms = jitter_ms,
                        "[migration] retry jitter sleep"
                    );
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {}
                        _ = self.context.stopped() => {
                            tracing::info!(
                                request_id = %self.context.id(),
                                "[migration] context stopped during retry jitter; aborting"
                            );
                            return Err(Error::msg(format!(
                                "Context id {} stopped during retry jitter",
                                self.context.id()
                            )));
                        }
                    }
                }
            }
            is_retry = true;
            let request = Context::with_id(self.request.clone(), self.context.id().to_string());
            self.context.link_child(request.context());
            if self.context.is_stopped() || self.context.is_killed() {
                tracing::debug!("Abort creating new stream after context is stopped or killed");
                return Err(Error::msg(format!(
                    "Context id {} is stopped or killed",
                    self.context.id()
                )));
            }
            response_stream = Some(self.next_generate.generate(request).await);
            if let Some(err) = response_stream.as_ref().unwrap().as_ref().err() {
                if is_migratable(err.as_ref()) {
                    tracing::warn!(
                        request_id = %self.context.id(),
                        retries_left = self.retries_left,
                        error = %err,
                        "Creating new stream... retrying..."
                    );
                    self.metrics.inc_migration_new_request(&self.model_name);
                    continue;
                } else {
                    // The biggest silent-failure path we know about: an error
                    // returned synchronously from the pipeline that doesn't
                    // match the migratable allowlist. Examples observed in
                    // prod: "no endpoints available to route work" (kv_router
                    // scheduler when avail set is empty) and "No disaggregated
                    // params in prefill response" (prefill_router parse error).
                    // Both are TRANSIENT in nature (the cluster recovers) but
                    // were classified as fatal here — request fails with 500
                    // and no retry log fires. Surfacing this is the single
                    // most useful migration-side diagnostic.
                    tracing::warn!(
                        request_id = %self.context.id(),
                        retries_left = self.retries_left,
                        retries_used = self.retries_used,
                        error = %err,
                        error_chain = ?err,
                        "[migration] non-migratable error from pipeline; will NOT retry"
                    );
                }
            }
            break;
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                if self.retries_used > 0 {
                    // Single-line per-request recovery summary. Greppable as
                    // [migration_summary]; gives the count of cumulative
                    // retries across this request's lifecycle without
                    // having to correlate the per-attempt warn logs.
                    tracing::info!(
                        model = %self.model_name,
                        request_id = %self.context.id(),
                        retries_used = self.retries_used,
                        retries_left = self.retries_left,
                        "[migration_summary] recovered via retry"
                    );
                }
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate original error if any
            None => Err(Error::msg(
                "Migration limit exhausted", // should propagate original error if any
            )),
        }
    }

    fn track_response(&mut self, response: &Annotated<BackendOutput>) {
        if self.retries_left == 0 {
            return;
        }
        let llm_engine_output = match response.data.as_ref() {
            Some(output) => output,
            None => return,
        };
        if let Some(max_tokens) = self.request.stop_conditions.max_tokens {
            self.request.stop_conditions.max_tokens =
                Some(max_tokens.saturating_sub(llm_engine_output.token_ids.len() as u32));
        }
        for token_id in llm_engine_output.token_ids.iter() {
            self.request.token_ids.push(*token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::metrics::Metrics;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::error::{DynamoError, ErrorType};
    use dynamo_runtime::pipeline::AsyncEngine;
    use dynamo_runtime::pipeline::context::Controller;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::mpsc;

    const TEST_MODEL: &str = "test-model";

    // Helper to create a mock preprocessed request
    fn create_mock_request(max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    // Helper to create mock LLM engine output
    fn create_mock_output(token_id: u32) -> Annotated<BackendOutput> {
        Annotated::from_data(BackendOutput {
            token_ids: vec![token_id],
            tokens: vec![],
            text: Some(format!("token_{token_id}")),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            stop_reason: None,
            index: None,
            extra_args: None,
            disaggregated_params: None,
            completion_usage: None,
        })
    }

    #[derive(Debug, Clone)]
    enum MockBehavior {
        /// Always succeeds with all responses
        Success,
        /// Fails on first call with NoResponders error, then succeeds on subsequent calls
        FailThenSuccess,
        /// Succeeds initially, fails mid-stream with specific error, then succeeds on retry
        MidStreamFail { fail_after: usize },
        /// Succeeds initially, fails mid-stream with specific error, then always fails on retry attempts
        MidStreamFailAlways { fail_after: usize },
        /// Succeeds initially, fails mid-stream, then always fails with stream error on retry attempts
        MidStreamFailAlwaysStreamError { fail_after: usize },
        /// Always fails with NoResponders error (same as FailThenSuccess first call)
        AlwaysFail,
    }

    // Unified mock server streaming engine that can simulate different scenarios
    struct MockEngine {
        behavior: MockBehavior,
        num_responses: usize,
        token_offset: u32,
        call_count: Arc<AtomicU32>,
        context_id: String,
    }

    impl MockEngine {
        fn new(
            behavior: MockBehavior,
            num_responses: usize,
            token_offset: u32,
            context_id: String,
        ) -> Self {
            Self {
                behavior,
                num_responses,
                token_offset,
                call_count: Arc::new(AtomicU32::new(0)),
                context_id,
            }
        }
    }

    #[async_trait]
    impl
        AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, anyhow::Error>
        for MockEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
            let (preprocessed_request, context) = request.transfer(());

            // Assert that the context_id matches the expected one
            assert_eq!(
                context.id().to_string(),
                self.context_id,
                "Context ID mismatch"
            );

            // Calculate how many responses we've already generated based on request token_ids
            // Initial request has [1, 2, 3], so anything beyond that are generated responses
            let initial_tokens = 3; // [1, 2, 3]
            let responses_already_generated = preprocessed_request
                .token_ids
                .len()
                .saturating_sub(initial_tokens);

            // Assert that max_tokens reflects the expected remaining tokens
            let expected_max_tokens =
                self.num_responses
                    .saturating_sub(responses_already_generated) as u32;
            assert_eq!(
                preprocessed_request.stop_conditions.max_tokens,
                Some(expected_max_tokens),
                "max_tokens should be {} but got {:?}",
                expected_max_tokens,
                preprocessed_request.stop_conditions.max_tokens
            );

            match &self.behavior {
                MockBehavior::Success => {
                    // Always succeed with remaining responses
                    self.send_responses(responses_already_generated, self.num_responses)
                        .await
                }
                MockBehavior::FailThenSuccess => {
                    if call_num == 0 {
                        // First call - return "No responders available" error to trigger retry
                        return Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ));
                    } else {
                        // Subsequent calls - succeed with remaining responses
                        self.send_responses(responses_already_generated, self.num_responses)
                            .await
                    }
                }
                MockBehavior::MidStreamFail { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });
                    } else {
                        // Second call - send remaining responses from where we left off
                        tokio::spawn(async move {
                            for i in responses_already_generated..num_responses {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }

                    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                    let ctx = Arc::new(Controller::new(self.context_id.clone()));
                    Ok(dynamo_runtime::pipeline::ResponseStream::new(
                        Box::pin(stream),
                        ctx,
                    ))
                }
                MockBehavior::MidStreamFailAlways { fail_after } => {
                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        let (tx, rx) = mpsc::channel(1);
                        let token_offset = self.token_offset;
                        let fail_after = *fail_after;
                        let num_responses = self.num_responses;

                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - always fail with NoResponders error (same as AlwaysFail)
                        Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ))
                    }
                }
                MockBehavior::MidStreamFailAlwaysStreamError { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - immediately send stream error (no successful responses)
                        tokio::spawn(async move {
                            // Send the stream error immediately
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    }
                }
                MockBehavior::AlwaysFail => {
                    // Always fail with NoResponders error (same as FailThenSuccess first call)
                    Err(anyhow::anyhow!(
                        DynamoError::builder()
                            .error_type(ErrorType::CannotConnect)
                            .message("no responders")
                            .build()
                    ))
                }
            }
        }
    }

    impl MockEngine {
        async fn send_responses(
            &self,
            start: usize,
            end: usize,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let (tx, rx) = mpsc::channel(1);
            let token_offset = self.token_offset;

            tokio::spawn(async move {
                for i in start..end {
                    let response = create_mock_output(token_offset + 1 + i as u32);
                    if tx.send(response).await.is_err() {
                        break;
                    }
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let ctx = Arc::new(Controller::new(self.context_id.clone()));
            Ok(dynamo_runtime::pipeline::ResponseStream::new(
                Box::pin(stream),
                ctx,
            ))
        }
    }

    /// Test case 1: No migration needed
    /// Tests the normal case where the RetryManager successfully processes all responses
    /// from a single stream without any failures or need for retries/migration.
    /// Expected behavior: All 10 responses should be received successfully.
    #[tokio::test]
    async fn test_retry_manager_no_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            0,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 2: New request migration
    /// Tests the scenario where a worker becomes unreachable for new requests initially,
    /// triggering the RetryManager to retry the request. The MockEngine with FailThenSuccess
    /// fails on the first call with a "No responders available" error, then succeeds
    /// on subsequent calls, simulating a worker becoming available after initial failure.
    /// Expected behavior: All 10 responses should be received successfully after retry.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccess,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 3: Ongoing request migration
    /// Tests the scenario where a worker fails mid-stream during an ongoing request.
    /// This simulates a connection being lost after partial response delivery, requiring
    /// the RetryManager to detect the failure (via "Stream ended before generation completed" error),
    /// create a new stream, and continue from where it left off.
    /// Expected behavior: 5 responses from first stream + 5 responses from retry stream = 10 total.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received all 10 responses (5 from first stream + 5 from second stream)
        assert_eq!(responses.len(), 10);

        // Check that we received responses from both streams
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
    }

    /// Test case 4: New request migration - indefinite failure
    /// Tests the scenario where a worker becomes unreachable for new requests indefinitely.
    /// The RetryManager should exhaust all retries and return the original error from the first attempt.
    /// Expected behavior: Should receive an error after all retries are exhausted, with the original error.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(0);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::AlwaysFail,
            0,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        // Should fail to build due to initial stream creation failure after exhausting all 3 retries
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(error.to_string().contains("no responders"));
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 5: Ongoing request migration - indefinite failure
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests.
    /// The RetryManager should exhaust all retries and return the original stream disconnection error.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlways { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 3); // 2 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1); // initial ongoing failure retry
    }

    /// Test case 6: Ongoing request migration - indefinite failure with stream errors
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests,
    /// and all retry attempts also fail with stream errors instead of NATS errors.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure_stream_error() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlwaysStreamError { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
    }

    /// Test case 7: Request cancelled when creating new stream
    /// Tests the scenario where context.stop_generating() is called when creating a new stream.
    /// The RetryManager should detect that the context is stopped and abort creating new streams.
    /// Expected behavior: Should fail to build RetryManager with "Context is stopped or killed" error.
    #[tokio::test]
    async fn test_retry_manager_context_stopped_before_stream() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));

        // Stop the context before building RetryManager
        ctx.stop_generating();

        // Should fail to build due to stopped context
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(
                error
                    .to_string()
                    .contains(&format!("Context id {} is stopped or killed", context_id))
            );
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }
}
