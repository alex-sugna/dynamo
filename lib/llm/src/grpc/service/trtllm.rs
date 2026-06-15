// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TrtllmService gRPC servicer.
//!
//! Mirrors the surface defined by `lib/llm/src/grpc/protos/trtllm_service.proto`
//! (vendored from Together's smg/crates/grpc_client/proto/) so that external
//! routers like Shepherd Model Gateway can drive this Dynamo instance with
//! pre-tokenized input over the same contract they already use for TRT-LLM,
//! vLLM, and SGLang.
//!
//! Reference for structural patterns: `together-TensorRT-LLM/tensorrt_llm/grpc/grpc_servicer.py`.

use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::grpc::service::kserve::State;
use crate::grpc::service::openai::completion_response_stream;
use crate::protocols::openai::completions::NvCreateCompletionResponse;

// ---------------------------------------------------------------------------
// Generated proto modules.
//
// `trtllm.proto` declares `package trtllm;` and imports `common.proto` whose
// declared package is `smg.grpc.common`. prost emits cross-package references
// via `super::smg::grpc::common::T`, so the Rust module hierarchy must mirror
// the proto package hierarchy for references to resolve.
// ---------------------------------------------------------------------------

pub mod trtllm {
    tonic::include_proto!("trtllm");
}

pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}

pub use trtllm::trtllm_service_server::{TrtllmService, TrtllmServiceServer};

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

/// gRPC servicer that exposes the `TrtllmService` contract on top of Dynamo's
/// existing engine plane. Shares state with `KserveService` so the same
/// `ModelManager` and `Metrics` registries are used regardless of which
/// frontend the request entered through.
#[derive(Clone)]
pub struct TrtllmServiceImpl {
    state: Arc<State>,
}

impl TrtllmServiceImpl {
    pub fn new(state: Arc<State>) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &State {
        Arc::as_ref(&self.state)
    }
}

/// Resolve the model id for a request that arrived without one. The
/// SMG → Dynamo deployment serves a single model per Dynamo instance for M1;
/// multi-model resolution is M3. Errors with FailedPrecondition if no model is
/// registered yet (i.e. the worker hasn't joined via etcd discovery).
fn pick_default_model_id(state: &State) -> Result<String, Status> {
    state
        .manager()
        .model_display_names()
        .into_iter()
        .next()
        .ok_or_else(|| Status::failed_precondition("no model registered"))
}

/// Map a per-chunk engine-side error string to a tonic `Status`. Used
/// by the streaming loop where the error is already a `String` (from
/// `FinishReason::Error(String)` on the wire) and there is no source
/// chain to downcast. For stream-setup errors that still carry typed
/// info, see `classify_setup_error` in `grpc/service/openai.rs`.
///
/// Dispatch:
///   1. JSON `{"code": <4xx>, "message": "..."}` payload — surfaces
///      `InvalidRequestError` from the Python worker (prompt >
///      max_seq_len, bad sampling params). 4xx → 400. Walks all `{`
///      positions left-to-right and uses the first parse-able 4xx
///      payload, so a wrapped error with `{` in an earlier non-JSON
///      fragment doesn't block the real payload.
///   2. Substring match for "admission control rejected" → 429.
///      Fallback when typed downcast at stream-setup didn't fire and
///      the error reached us as a wrapped string. Stable on the
///      `KvSchedulerError::AdmissionRejected` Display text.
///   3. Anything else → 500.
///
/// Why two phases instead of one combined function:
/// the dec7ec981 revert was driven by the substring path "winning"
/// over the JSON path for a 400 payload that contained the admission
/// substring elsewhere in the message. Ordering JSON first + walking
/// braces eliminates the regression. The typed downcast lives in
/// `classify_setup_error` (only reachable where we still have
/// `anyhow::Error`); this function is only the string fallback.
fn engine_error_str_to_status(msg: &str) -> Status {
    #[derive(serde::Deserialize)]
    struct ErrorPayload {
        message: Option<String>,
        code: Option<u16>,
    }

    // (2) The Python worker emits structured errors as a JSON {code, message}
    // string inside FinishReason::Error. It reaches here wrapped by
    // DynamoError's Display, which prefixes "<error_type>: " (e.g. "Unknown:
    // {...}"). Find the FIRST `{` that opens a payload we can actually parse —
    // earlier versions sliced at first `{` unconditionally, which broke once
    // errors got wrapped with messages that contained `{` in earlier positions
    // (e.g. "{request_id: xyz} failed: {real-json-payload}"). Walk braces
    // left-to-right and try each one; first parse-able 4xx/5xx wins.
    //
    // 4xx: InvalidRequestError path (handler_base.py:1080) — client-side
    //      validation failure → InvalidArgument → SMG 400.
    // 5xx: _handle_errors path (py_executor.py) — server-side fault (KV
    //      transfer timeout, forward-pass crash, drafter error) → Internal →
    //      SMG 500. Locks the engine's HTTP intent into the wire format so
    //      tonic stream-cancellation races can't downgrade it to Code::Cancelled
    //      (which SMG's tonic_ext maps to 400 — wrong semantics for server
    //      faults; client treats it as malformed input and won't retry).
    let bytes = msg.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{'
            && let Ok(payload) = serde_json::from_str::<ErrorPayload>(&msg[i..])
            && let Some(code) = payload.code
        {
            let detail = payload.message.unwrap_or_else(|| msg.to_string());
            if (400..500).contains(&code) {
                return Status::invalid_argument(detail);
            }
            if (500..600).contains(&code) {
                return Status::internal(detail);
            }
            // Recognized envelope but code is out of HTTP range; keep walking.
        }
    }

    // (3) Substring fallback for AdmissionRejected — survives anyhow
    // wrapping that erases the typed error. Stable on the
    // `KvSchedulerError::AdmissionRejected` Display text.
    if msg.contains("admission control rejected") {
        return Status::resource_exhausted(msg.to_string());
    }

    // (4) Default.
    Status::internal(format!("engine error: {msg}"))
}

/// Cold-load budget for Llama-class models on B200 — model load + CUDA-graph
/// warmup typically completes in ~3 min; 5 min covers the long tail.
// Default bumped from 300s to 900s 2026-05-17. SMG's `GetModelInfo`
// call is single-shot at startup; on DeadlineExceeded, SMG caches the
// model under UNKNOWN_MODEL_ID for the rest of the SMG process's
// lifetime, stranding every client request that uses the real
// served-model-name ("Tokenizer not found for model ..."). Observed
// in krustykrab shadow at 15:44:30 — Kimi cold-load took ~7 min,
// longer than the prior 300s ceiling. 900s gives plenty of headroom
// for slow worker cold-starts. Long-term fix is SMG-side retry on
// the metadata-discovery workflow.
//
// Override via `DYN_MODEL_DISCOVERY_TIMEOUT_SECS` env var so we can
// nudge without a rebuild if a deployment hits an even slower
// cold-load.
const MODEL_DISCOVERY_TIMEOUT_DEFAULT_SECS: u64 = 900;

fn model_discovery_timeout() -> std::time::Duration {
    let secs = std::env::var("DYN_MODEL_DISCOVERY_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(MODEL_DISCOVERY_TIMEOUT_DEFAULT_SECS);
    std::time::Duration::from_secs(secs)
}
const MODEL_DISCOVERY_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(500);

/// Block until ModelManager has at least one model registered, then return
/// its display name. Returns `DeadlineExceeded` after `timeout` so the caller
/// gets a clean tonic Status rather than hanging forever.
async fn wait_for_model_id(
    state: &State,
    timeout: std::time::Duration,
) -> Result<String, Status> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if let Some(name) = state.manager().model_display_names().into_iter().next() {
            return Ok(name);
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(Status::deadline_exceeded(format!(
                "no model registered after waiting {}s; the dynamo worker did \
                 not join the frontend's ModelManager in time",
                timeout.as_secs()
            )));
        }
        tokio::time::sleep(MODEL_DISCOVERY_POLL_INTERVAL).await;
    }
}

// Tonic generates streaming response types that we have to name explicitly in
// the trait impl. Boxed dyn streams keep the implementation flexible.
type GenerateStream =
    Pin<Box<dyn Stream<Item = Result<trtllm::GenerateResponse, Status>> + Send + 'static>>;
type GetTokenizerStream = Pin<
    Box<dyn Stream<Item = Result<smg::grpc::common::GetTokenizerChunk, Status>> + Send + 'static>,
>;
type SubscribeKvEventsStream =
    Pin<Box<dyn Stream<Item = Result<smg::grpc::common::KvEventBatch, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl TrtllmService for TrtllmServiceImpl {
    type GenerateStream = GenerateStream;
    type GetTokenizerStream = GetTokenizerStream;
    type SubscribeKvEventsStream = SubscribeKvEventsStream;

    async fn generate(
        &self,
        request: Request<trtllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let proto_req = request.into_inner();
        let request_id = proto_req.request_id.clone();

        // SMG configures the served model name out-of-band. The proto request
        // does not carry it, so we resolve to the single registered model.
        // Multi-model on a single Dynamo instance is M3.
        let model_id = pick_default_model_id(self.state())?;

        let nv_request =
            request_translation::proto_to_completion_request(&proto_req, model_id)?;

        // Submit through the same engine plane the HTTP `/v1/completions`
        // handler uses. `completion_response_stream` returns a stream of
        // `Annotated<NvCreateCompletionResponse>` deltas plus an unused
        // ParsingOptions handle (we don't apply OpenAI parser logic here —
        // SMG owns response parsing).
        // `completion_response_stream` returns `Result<_, Status>` and
        // performs the typed `DynamoError`/`KvSchedulerError` → `Status`
        // mapping internally (see grpc/service/openai.rs::
        // completion_response_stream where the engine error is
        // downcast). Stream-setup errors arrive here pre-classified;
        // the bare `?` propagates them with the correct gRPC code.
        // Per-chunk errors (line ~280) are post-classification strings,
        // handled separately via `engine_error_str_to_status`.
        let (stream, _parsing_options) =
            completion_response_stream(self.state.clone(), nv_request).await?;

        // Stream loop:
        //
        //   - Engine emits a sequence of NvCreateCompletionResponse deltas:
        //     N text/token chunks (intermediate), then one chunk carrying
        //     finish_reason on choices[0], then (with include_usage) one
        //     trailing chunk with empty `choices` and populated `usage`.
        //   - Proto contract: emit GenerateStreamChunk for each intermediate
        //     delta, exactly one GenerateComplete at end-of-stream.
        //
        // We defer the Complete frame until the stream actually drains so that
        // the trailing usage chunk gets accumulated into the Complete's
        // prompt_tokens / completion_tokens. Emitting Complete on the first
        // chunk that has finish_reason (the obvious-but-wrong approach) yields
        // prompt_tokens=0 — caught by smg-dynamo-grpc-smoke.
        let proto_stream = async_stream::try_stream! {
            tokio::pin!(stream);
            let mut prompt_tokens: u32 = 0;
            let mut completion_tokens: u32 = 0;
            // Cached prompt tokens (KV-block-reuse hits). Populated by the
            // TRT-LLM worker into `completion_usage.prompt_tokens_details.
            // cached_tokens` on the finish frame, propagated by Dynamo's
            // delta.rs into `usage.prompt_tokens_details`. SMG's sim reads
            // this field to compute its cache-hit-rate metric — without it,
            // sim reports 0% even when the engine is reusing KV blocks.
            let mut cached_tokens: u32 = 0;
            // Accumulate all generated token IDs across the stream so we can
            // emit them in `GenerateComplete.output_token_ids` (cumulative,
            // per the proto contract). SMG's non-streaming chat-completion
            // path reads `complete.output_ids()` to detokenize the full
            // response — without this it sees zero tokens and emits empty
            // `message.content`. (The streaming path consumes per-chunk
            // `GenerateStreamChunk.token_ids` and doesn't need the cumulative
            // array, but we populate both for correctness.)
            let mut cumulative_token_ids: Vec<u32> = Vec::new();
            // Per-token logprobs (with token_id-preserving alternatives)
            // accumulate across chunks for the same reason cumulative
            // token_ids do — SMG's non-streaming consumer reads them off
            // the terminal Complete frame, not per-chunk. Each delta
            // brings its own slice via nvext.raw_logprobs.
            let mut cumulative_logprobs: Vec<crate::protocols::openai::nvext::RawTokenLogprob> =
                Vec::new();
            // Cache the chunk that carried finish_reason; we'll synthesize
            // the Complete frame from it once usage has had a chance to land.
            let mut finished_chunk: Option<NvCreateCompletionResponse> = None;

            while let Some(annotated) = stream.next().await {
                // Per-chunk errors arrive as `String` (from
                // `FinishReason::Error(String)` over the wire). Typed
                // downcast isn't available here; use the string variant
                // which handles JSON 4xx payloads + admission substring.
                let response = annotated
                    .ok()
                    .map_err(|e| engine_error_str_to_status(&e))?;
                let Some(nv_chunk) = response.data else {
                    // Annotation-only frames (request_id etc.) are not part of
                    // the TrtllmService contract; drop them.
                    continue;
                };

                if let Some(usage) = nv_chunk.inner.usage.as_ref() {
                    prompt_tokens = usage.prompt_tokens;
                    completion_tokens = usage.completion_tokens;
                    if let Some(details) = usage.prompt_tokens_details.as_ref() {
                        if let Some(c) = details.cached_tokens {
                            cached_tokens = c;
                        }
                    }
                }

                // Drain per-chunk delta token_ids and add them to the cumulative
                // accumulator. The helper is cheap, but extracting once lets us
                // reuse the count for the chunk-emission decision below.
                let chunk_token_ids = response_translation::extract_token_ids(&nv_chunk);
                let chunk_token_count = chunk_token_ids.len() as u32;
                cumulative_token_ids.extend_from_slice(&chunk_token_ids);

                // Same accumulator pattern for raw logprobs. Each delta
                // carries its slice via nvext.raw_logprobs; we append
                // them to the cumulative buffer in arrival order so the
                // terminal Complete frame contains one entry per emitted
                // token.
                if let Some(mut raw) = response_translation::extract_raw_logprobs(&nv_chunk) {
                    cumulative_logprobs.append(&mut raw);
                }

                let chunk_has_finish = nv_chunk
                    .inner
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.as_ref())
                    .is_some();

                // The engine's final delta typically carries BOTH new token
                // ids AND finish_reason on choices[0]. The streaming consumer
                // (SMG) reads tokens from per-chunk `GenerateStreamChunk.
                // token_ids` — not from the Complete frame's cumulative
                // array — so we must yield a Chunk for any delta that has
                // new tokens, regardless of whether it also carries
                // finish_reason. The trailing usage frame (empty choices,
                // populated `usage`) has `chunk_token_count == 0` and is
                // legitimately skipped.
                if chunk_token_count > 0 {
                    yield response_translation::nv_response_to_chunk(
                        &request_id,
                        &nv_chunk,
                        prompt_tokens,
                        cached_tokens,
                    );
                }

                if chunk_has_finish {
                    // Cache the finish-reason chunk for the terminal Complete
                    // frame. Defer the Complete itself until end-of-stream so
                    // any trailing usage chunk has a chance to land.
                    finished_chunk = Some(nv_chunk);
                }
            }

            // End-of-stream: synthesize the terminal Complete frame.
            // Prefer the cached finish-reason chunk so the proto carries the
            // engine's actual finish_reason; fall back to a synthetic frame
            // if the stream ended without ever surfacing one.
            let final_chunk = finished_chunk.unwrap_or_else(|| NvCreateCompletionResponse {
                inner: dynamo_async_openai::types::CreateCompletionResponse {
                    id: request_id.clone(),
                    choices: vec![],
                    created: 0,
                    model: String::new(),
                    system_fingerprint: None,
                    object: "text_completion".to_string(),
                    usage: None,
                    nvext: None,
                },
            });
            yield response_translation::nv_response_to_complete(
                &request_id,
                &final_chunk,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                cumulative_token_ids,
                cumulative_logprobs,
            );
        };

        Ok(Response::new(Box::pin(proto_stream)))
    }

    async fn embed(
        &self,
        _request: Request<trtllm::EmbedRequest>,
    ) -> Result<Response<trtllm::EmbedResponse>, Status> {
        // Embeddings via gRPC are out of scope for the SMG↔Dynamo M1.
        Err(Status::unimplemented(
            "Embed RPC is not implemented in this servicer",
        ))
    }

    async fn health_check(
        &self,
        _request: Request<trtllm::HealthCheckRequest>,
    ) -> Result<Response<trtllm::HealthCheckResponse>, Status> {
        // Frontend is "healthy" when at least one model is registered with the
        // ModelManager (i.e. a worker has joined via etcd discovery). The HTTP
        // path uses the same readiness signal in `check_ready`.
        let ready = !self.state.manager().model_display_names().is_empty();
        let status = if ready { "OK" } else { "NOT_READY" };
        Ok(Response::new(trtllm::HealthCheckResponse {
            status: status.to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<trtllm::AbortRequest>,
    ) -> Result<Response<trtllm::AbortResponse>, Status> {
        // Implementation lands in task #4. Requires a request_id ->
        // AsyncEngineContext registry so we can call .stop() on the in-flight
        // request. Until then, callers can rely on stream-drop cancellation.
        Err(Status::unimplemented(
            "Abort not yet wired to request registry",
        ))
    }

    async fn get_model_info(
        &self,
        _request: Request<trtllm::GetModelInfoRequest>,
    ) -> Result<Response<trtllm::GetModelInfoResponse>, Status> {
        // Block until ModelManager has a model registered, with a generous
        // timeout. External routers (SMG) call us once during worker
        // discovery to learn the served model name; if they get
        // FailedPrecondition because the dynamo.trtllm worker hasn't
        // finished CUDA-graph warmup + etcd registration yet, they cache
        // the failure and stick the tokenizer under UNKNOWN_MODEL_ID.
        // Blocking here puts the wait in the one place that knows when it
        // can be answered, instead of pushing client-side retry into every
        // consumer.
        let model_id = wait_for_model_id(self.state(), model_discovery_timeout()).await?;

        // Most of the structural fields (max_input_len, max_seq_len, vocab_size,
        // hidden_size, num_layers, num_heads) live behind ModelManager's MDC
        // surface; precise extraction is task #4 territory once the wiring
        // pattern is settled. For the skeleton we emit the model_id only;
        // SMG only requires it to learn the served model name.
        let resp = trtllm::GetModelInfoResponse {
            model_id,
            max_input_len: 0,
            max_seq_len: 0,
            max_batch_size: 0,
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            supported_features: vec!["guided_decoding".to_string()],
        };
        Ok(Response::new(resp))
    }

    async fn get_server_info(
        &self,
        _request: Request<trtllm::GetServerInfoRequest>,
    ) -> Result<Response<trtllm::GetServerInfoResponse>, Status> {
        // SMG reads `version` and `backend` to log/observe; the parallelism
        // fields are informational. Detailed parallelism extraction lives in
        // M3 (multi-model / multi-worker observability).
        let resp = trtllm::GetServerInfoResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: "dynamo".to_string(),
            tensor_parallel_size: 0,
            pipeline_parallel_size: 0,
            context_parallel_size: 0,
            world_size: 0,
        };
        Ok(Response::new(resp))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<smg::grpc::common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        // Streaming the HF tokenizer dir is an M3 polish item. For M1 SMG
        // operators ship the tokenizer to SMG directly via --model.
        Err(Status::unimplemented(
            "GetTokenizer not yet implemented; ship the tokenizer to SMG out-of-band",
        ))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg::grpc::common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        // Dynamo's internal `KvPushRouter` already handles cache-aware routing
        // between its own workers; SMG sees Dynamo as a single logical worker
        // and therefore has nothing to do with these events. Returning
        // Unimplemented is the documented contract for M1.
        //
        // SMG-side change in `model_gateway/src/worker/kv_event_monitor.rs`
        // gates subscription on `Backend::Dynamo` so this RPC is never
        // actually called by an SMG configured for the Dynamo backend.
        Err(Status::unimplemented(
            "Dynamo handles KV-aware routing internally; subscription not exposed",
        ))
    }
}

// ---------------------------------------------------------------------------
// Request translation: proto::GenerateRequest -> NvCreateCompletionRequest
//
// Pure functions, callable from the trait impl in task #4 and exercised by
// Layer 1 unit tests below without an engine.
// ---------------------------------------------------------------------------

pub(crate) mod request_translation {
    use super::trtllm;
    use crate::protocols::openai::common_ext::CommonExt;
    use crate::protocols::openai::completions::{
        MultiModalData, MultiModalImageItem, NvCreateCompletionRequest,
    };
    use crate::protocols::openai::nvext::NvExt;
    use dynamo_async_openai::types::{
        ChatCompletionStreamOptions,
        CreateCompletionRequest, Prompt, Stop, // re-exported through `dynamo_async_openai::types`
    };
    use tonic::Status;

    /// Translate a proto `GenerateRequest` into Dynamo's internal
    /// `NvCreateCompletionRequest`. Pre-tokenized input is carried as
    /// `Prompt::IntegerArray`, which the engine's preprocessor passes through
    /// without re-tokenizing (see
    /// `lib/llm/src/protocols/openai/completions.rs:59-79` and the
    /// `/v1/completions` HTTP handler).
    ///
    /// `model_id` is supplied by the caller because the proto request does not
    /// carry it — SMG configures the served model name out-of-band.
    pub fn proto_to_completion_request(
        req: &trtllm::GenerateRequest,
        model_id: String,
    ) -> Result<NvCreateCompletionRequest, Status> {
        let tokenized = req
            .tokenized
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("tokenized input is required"))?;

        if tokenized.input_token_ids.is_empty() {
            return Err(Status::invalid_argument(
                "tokenized.input_token_ids must be non-empty",
            ));
        }

        // Pre-tokenized prompt. `Prompt::IntegerArray` takes Vec<u32>, which
        // matches the proto's `repeated uint32` directly.
        let prompt_ids: Vec<u32> = tokenized.input_token_ids.clone();

        let sampling = req.sampling_config.as_ref();
        let output = req.output_config.as_ref();

        // Stop strings on proto are repeated string; OpenAI Stop is enum
        // String | StringArray. Use StringArray when any present.
        let stop = if req.stop.is_empty() {
            None
        } else if req.stop.len() == 1 {
            Some(Stop::String(req.stop[0].clone()))
        } else {
            Some(Stop::StringArray(req.stop.clone()))
        };

        // Always ask the engine for the trailing usage chunk regardless of
        // what the proto `req.streaming` flag says: `completion_response_stream`
        // unconditionally forces `inner.stream = Some(true)` on the request
        // (see lib/llm/src/grpc/service/openai.rs), so the engine ALWAYS
        // streams to us. Without `include_usage`, prompt_tokens/completion_tokens
        // arrive as zero in our accumulator and propagate as zero to the proto
        // Complete frame — caught by smg-dynamo-l3-chat asserting
        // `usage.prompt_tokens > 0` for non-streaming chats.
        let stream_options = Some(ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        });

        // Use struct-update syntax with Default so we don't break when fields
        // are added to CreateCompletionRequest. Only set what the proto carries.
        let inner = CreateCompletionRequest {
            model: model_id,
            prompt: Prompt::IntegerArray(prompt_ids),
            max_tokens: Some(req.max_tokens),
            temperature: sampling.and_then(|s| s.temperature),
            top_p: sampling.and_then(|s| s.top_p),
            n: sampling
                .map(|s| s.num_return_sequences.max(1) as u8)
                .or(Some(1)),
            stream: Some(req.streaming),
            stream_options,
            logprobs: output.and_then(|o| o.logprobs).map(|x| x.max(0) as u8),
            stop,
            presence_penalty: sampling.and_then(|s| s.presence_penalty),
            frequency_penalty: sampling.and_then(|s| s.frequency_penalty),
            user: Some(req.request_id.clone()),
            seed: sampling.and_then(|s| s.seed).map(|v| v as i64),
            ..Default::default()
        };

        // Common extensions cover the non-OpenAI sampling knobs plus guided
        // decoding glue.
        let mut common = CommonExt::default();
        if let Some(s) = sampling {
            common.top_k = s.top_k;
            common.min_p = s.min_p;
            common.repetition_penalty = s.repetition_penalty;
            common.min_tokens = s.min_tokens;
        }
        if req.ignore_eos {
            common.ignore_eos = Some(true);
        }
        if req.include_stop_token_in_output {
            common.include_stop_str_in_output = Some(true);
        }
        // SMG owns detokenization, stop-string matching, and tool-call
        // parsing on its side. Tell Backend to skip per-token decode_stream
        // (and the stop decoder that depends on it) — Backend's pass-through
        // forwards raw token_ids only. Without this we run HF's decoder once
        // here on the engine side and again in SMG: pure waste, measurable
        // in the gen-tok/s gap on Kimi MN.
        common.skip_detokenization = Some(true);

        // Forward per-message content hashes from SMG (set when SMG is
        // launched with `--enable-message-hash`) so the TRT engine can
        // record them in RequestStatistics.message_hashes. Without this
        // forwarding step the proto field is silently dropped at proto →
        // NvCreateCompletionRequest translation, and the engine logs
        // `statistics={...}` with no `message_hashes` key.
        if !req.message_hashes.is_empty() {
            common.message_hashes = Some(
                req.message_hashes
                    .iter()
                    .map(|mh| crate::protocols::openai::common_ext::MessageHashEntry {
                        role: mh.role.clone(),
                        hash: mh.hash.clone(),
                    })
                    .collect(),
            );
        }

        // Map proto guided decoding -> CommonExt guided_* fields.
        if let Some(gd) = req.guided_decoding.as_ref() {
            apply_guided_decoding(&mut common, gd)?;
        }

        // Carry request_id through nvext.annotations so the engine plane logs
        // it. NvExt itself has no request_id slot today.
        //
        // backend_instance_id / decode_instance_id passthrough: when SMG
        // forwards a request with `nvext.{backend,decode}_instance_id` set
        // on the OpenAI body (mirrors the pin-to-worker mechanism the
        // OpenAI HTTP path uses), we propagate both onto NvExt here. The
        // preprocessor at `lib/llm/src/preprocessor.rs` reads each from
        // NvExt and passes them through the routing path:
        //   - backend_instance_id pins the PREFILL leg (consumed by the
        //     prefill_router in disagg setups, or aggregated routing
        //     otherwise).
        //   - decode_instance_id pins the DECODE leg (consumed by
        //     KvPushRouter when prefill_router forwards the decode_req
        //     to the decode component). Required for /health/decode
        //     synth probes to actually test the worker they target —
        //     without this slot in the proto, SMG silently drops the
        //     field on its HTTP→gRPC translation and decode-side
        //     routing picks freely.
        let nvext = Some({
            let mut ext = NvExt::default();
            ext.annotations = Some(vec![format!("request_id={}", req.request_id)]);
            ext.backend_instance_id = req.backend_instance_id;
            ext.decode_instance_id = req.decode_instance_id;
            ext
        });

        // [pin_trace:1/gRPC_entry] First log in the trace chain. Confirms SMG
        // propagated the nvext fields onto the gRPC proto correctly. Cross-
        // links the chatcmpl-* request_id (from SMG / OpenAI body) with the
        // internal dispatch about to begin. Every downstream log in this
        // request's lifecycle should be findable by grepping the chatcmpl_id.
        tracing::info!(
            target: "dynamo::pin_trace",
            client_request_id = %req.request_id,
            nvext_backend_instance_id = ?req.backend_instance_id,
            nvext_decode_instance_id = ?req.decode_instance_id,
            "[pin_trace:1/grpc_entry] SMG→dynamo gRPC request received"
        );

        // Forward raw image bytes from the proto's MultimodalInput side
        // channel to the Python TRT-LLM worker. SMG packs decoded image
        // bytes into `multimodal_input.image_data`; the worker's
        // `multimodal_processor.process_openai_request` looks for
        // `multi_modal_data["image_url"]` as `[{"Url": "<data: URL>"}]`.
        // Base64-encoded data: URLs are decoded natively by the worker's
        // image_loader (image_loader.py:61-74), no NIXL/RDMA needed.
        //
        // Format the mime as image/jpeg unconditionally — JPEG/PNG/WEBP
        // are all sniffed by PIL.Image.open regardless of the URI's
        // declared media type, and the Python validator only restricts
        // formats: ["JPEG", "PNG", "WEBP"].
        // DIAG: log whether SMG actually populated multimodal_input before
        // we attempt to forward — distinguishes "SMG didn't pack" from
        // "we forwarded but the worker dropped".
        match req.multimodal_input.as_ref() {
            None => tracing::info!(
                target: "dynamo::mm_diag",
                request_id = %req.request_id,
                "MM-DIAG[rust]: req.multimodal_input is None",
            ),
            Some(mm) => tracing::info!(
                target: "dynamo::mm_diag",
                request_id = %req.request_id,
                image_count = mm.image_data.len(),
                first_image_bytes = mm.image_data.first().map(|b| b.len()).unwrap_or(0),
                "MM-DIAG[rust]: req.multimodal_input present",
            ),
        }
        let multi_modal_data = req
            .multimodal_input
            .as_ref()
            .filter(|mm| !mm.image_data.is_empty())
            .map(|mm| {
                use base64::Engine;
                MultiModalData {
                    image_url: mm
                        .image_data
                        .iter()
                        .map(|bytes| {
                            let b64 =
                                base64::engine::general_purpose::STANDARD.encode(bytes);
                            MultiModalImageItem::Url(format!(
                                "data:image/jpeg;base64,{b64}"
                            ))
                        })
                        .collect(),
                }
            });

        Ok(NvCreateCompletionRequest {
            inner,
            common,
            nvext,
            metadata: None,
            multi_modal_data,
            unsupported_fields: Default::default(),
        })
    }

    fn apply_guided_decoding(
        common: &mut CommonExt,
        gd: &trtllm::GuidedDecodingParams,
    ) -> Result<(), Status> {
        use trtllm::guided_decoding_params::GuideType;

        let guide_type = GuideType::try_from(gd.guide_type)
            .map_err(|_| Status::invalid_argument("invalid guide_type"))?;

        match guide_type {
            GuideType::Unspecified => Ok(()),
            GuideType::Json => {
                // Free-form valid JSON, no schema constraint.
                common.guided_json = Some(serde_json::Value::Null);
                Ok(())
            }
            GuideType::JsonSchema => {
                let schema: serde_json::Value = serde_json::from_str(&gd.guide).map_err(|e| {
                    Status::invalid_argument(format!("guided JSON schema is not valid JSON: {e}"))
                })?;
                common.guided_json = Some(schema);
                Ok(())
            }
            GuideType::Regex => {
                common.guided_regex = Some(gd.guide.clone());
                Ok(())
            }
            GuideType::EbnfGrammar => {
                common.guided_grammar = Some(gd.guide.clone());
                Ok(())
            }
            GuideType::StructuralTag => {
                // xgrammar structural-tag is a Dynamo extension; carry it as
                // grammar for now, leave routing to the engine's xgrammar path.
                common.guided_grammar = Some(gd.guide.clone());
                common.guided_decoding_backend = Some("xgrammar".to_string());
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Response translation: NvCreateCompletionResponse -> proto::GenerateResponse
//
// The translator reads the engine's per-chunk choice and maps it into either
// a streaming `GenerateStreamChunk` (delta tokens) or a final `GenerateComplete`
// frame. Engine-side token-id population is task #4 / the engine-plane bypass
// risk noted in the plan.
// ---------------------------------------------------------------------------

pub(crate) mod response_translation {
    use super::trtllm;
    use crate::protocols::openai::completions::NvCreateCompletionResponse;
    use crate::protocols::openai::nvext::RawTokenLogprob;

    /// Build a streaming chunk frame from a single Dynamo response delta.
    ///
    /// `request_id` echoes back the proto request_id. `prompt_tokens` is
    /// supplied separately so it doesn't need to be re-derived per chunk.
    ///
    /// Token IDs are surfaced into `nvext.token_ids` on the Dynamo
    /// response by `delta.rs::choice_from_postprocessor` (we modified the
    /// fallback there to use `delta.token_ids` when GAIE
    /// `disaggregated_params.token_ids` isn't set). Without this, the proto
    /// chunk would carry empty token_ids and the SMG client would have
    /// nothing to detokenize.
    pub fn nv_response_to_chunk(
        request_id: &str,
        nv: &NvCreateCompletionResponse,
        prompt_tokens: u32,
        cached_tokens: u32,
    ) -> trtllm::GenerateResponse {
        let choice = nv.inner.choices.first();
        let token_ids = extract_token_ids(nv);
        let logprobs = extract_raw_logprobs(nv).map(to_proto_token_logprobs).unwrap_or_default();
        let chunk = trtllm::GenerateStreamChunk {
            token_ids,
            sequence_index: choice.map(|c| c.index).unwrap_or(0),
            prompt_tokens,
            completion_tokens: nv
                .inner
                .usage
                .as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0),
            cached_tokens,
            logprobs,
        };
        trtllm::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(trtllm::generate_response::Response::Chunk(chunk)),
        }
    }

    /// Pull generated token IDs out of `NvExtResponse.token_ids`. The Dynamo
    /// completions postprocessor injects them under that path when the engine
    /// emits a non-empty `BackendOutput.token_ids` (see `delta.rs`).
    pub(crate) fn extract_token_ids(nv: &NvCreateCompletionResponse) -> Vec<u32> {
        nv.inner
            .nvext
            .as_ref()
            .and_then(|v| v.get("token_ids"))
            .and_then(|v| serde_json::from_value::<Vec<u32>>(v.clone()).ok())
            .unwrap_or_default()
    }

    /// Pull the lossless raw-logprobs payload out of `NvExtResponse.
    /// raw_logprobs`. The Dynamo completions postprocessor populates this
    /// (in `delta.rs::choice_from_postprocessor` via `build_raw_logprobs`)
    /// when the engine emitted per-token logprobs AND top-k alternatives.
    /// Empty / absent → returns None and the caller emits an empty
    /// `logprobs` Vec in the proto, matching the prior behavior of clients
    /// that don't request logprobs.
    pub(crate) fn extract_raw_logprobs(
        nv: &NvCreateCompletionResponse,
    ) -> Option<Vec<RawTokenLogprob>> {
        nv.inner
            .nvext
            .as_ref()
            .and_then(|v| v.get("raw_logprobs"))
            .and_then(|v| serde_json::from_value::<Vec<RawTokenLogprob>>(v.clone()).ok())
    }

    /// Convert the lossless intermediate `RawTokenLogprob` payload to the
    /// proto `TokenLogprob` shape SMG consumes off the gRPC stream.
    /// One-to-one — same shape, different types.
    pub(crate) fn to_proto_token_logprobs(raw: Vec<RawTokenLogprob>) -> Vec<trtllm::TokenLogprob> {
        raw.into_iter()
            .map(|r| trtllm::TokenLogprob {
                token_id: r.token_id,
                logprob: r.logprob,
                top_logprobs: r
                    .top_logprobs
                    .into_iter()
                    .map(|t| trtllm::TopLogprob {
                        token_id: t.token_id,
                        logprob: t.logprob,
                    })
                    .collect(),
            })
            .collect()
    }

    /// Build a final-completion frame at the end of a stream.
    ///
    /// `prompt_tokens` and `completion_tokens` are passed in explicitly because
    /// in OpenAI streaming with `include_usage`, the usage chunk arrives AFTER
    /// the chunk that carries `finish_reason` — so the caller accumulates them
    /// across the whole stream and hands them to us at end-of-stream. We do not
    /// re-read them from `nv.inner.usage` (which is typically None on the
    /// finish-reason chunk).
    ///
    /// `output_token_ids` is the cumulative array of generated token IDs (per
    /// the proto contract: "All output token IDs (cumulative, not delta)").
    /// SMG's non-streaming `/v1/chat/completions` path consumes this array via
    /// `complete.output_ids()` to detokenize the response. Streaming consumers
    /// have their tokens via per-chunk delta and ignore this field.
    pub fn nv_response_to_complete(
        request_id: &str,
        nv: &NvCreateCompletionResponse,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        output_token_ids: Vec<u32>,
        cumulative_logprobs: Vec<RawTokenLogprob>,
    ) -> trtllm::GenerateResponse {
        let choice = nv.inner.choices.first();
        let finish_reason = choice
            .and_then(|c| c.finish_reason.as_ref())
            .map(|fr| match fr {
                dynamo_async_openai::types::CompletionFinishReason::Stop => "stop",
                dynamo_async_openai::types::CompletionFinishReason::Length => "length",
                dynamo_async_openai::types::CompletionFinishReason::ContentFilter => {
                    "content_filter"
                }
            })
            .unwrap_or("stop")
            .to_string();
        let complete = trtllm::GenerateComplete {
            output_token_ids,
            sequence_index: choice.map(|c| c.index).unwrap_or(0),
            finish_reason,
            matched_stop: None,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            logprobs: to_proto_token_logprobs(cumulative_logprobs),
            prompt_logprobs: Vec::new(),
            perf_metrics: None,
            context_logits: None,
            generation_logits: None,
        };
        trtllm::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(trtllm::generate_response::Response::Complete(complete)),
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 1 unit tests: protocol translation. No engine, no GPU.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::{Prompt, Stop};

    fn build_request_with_tokens(token_ids: Vec<u32>) -> trtllm::GenerateRequest {
        trtllm::GenerateRequest {
            request_id: "req-1".to_string(),
            tokenized: Some(trtllm::TokenizedInput {
                original_text: "hello".to_string(),
                input_token_ids: token_ids,
                query_token_ids: vec![],
            }),
            sampling_config: None,
            output_config: None,
            max_tokens: 32,
            streaming: true,
            guided_decoding: None,
            embedding_bias: vec![],
            lora_config: None,
            prompt_tuning_config: None,
            multimodal_input: None,
            kv_cache_retention: None,
            disaggregated_params: None,
            lookahead_config: None,
            cache_salt_id: None,
            arrival_time: None,
            stop: vec![],
            stop_token_ids: vec![],
            ignore_eos: false,
            bad: vec![],
            bad_token_ids: vec![],
            include_stop_token_in_output: false,
            message_hashes: vec![],
        }
    }

    #[test]
    fn translation_emits_integer_array_prompt() {
        let req = build_request_with_tokens(vec![1u32, 2, 3]);
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        match &nv.inner.prompt {
            Prompt::IntegerArray(ids) => assert_eq!(ids, &vec![1u32, 2, 3]),
            other => panic!("expected IntegerArray, got {:?}", other),
        }
        assert_eq!(nv.inner.max_tokens, Some(32));
        assert_eq!(nv.inner.stream, Some(true));
        assert_eq!(nv.inner.user.as_deref(), Some("req-1"));
    }

    #[test]
    fn translation_rejects_empty_tokenized() {
        let req = build_request_with_tokens(vec![]);
        let err = request_translation::proto_to_completion_request(&req, "m".into()).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn translation_requires_tokenized_field() {
        let mut req = build_request_with_tokens(vec![1]);
        req.tokenized = None;
        let err = request_translation::proto_to_completion_request(&req, "m".into()).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn translation_maps_sampling_fields() {
        let mut req = build_request_with_tokens(vec![1]);
        req.sampling_config = Some(trtllm::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            top_k: Some(40),
            top_p: Some(0.9),
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: Some(42),
            temperature: Some(0.7),
            min_tokens: Some(5),
            beam_search_diversity_rate: None,
            repetition_penalty: Some(1.1),
            presence_penalty: Some(0.2),
            frequency_penalty: Some(0.3),
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            min_p: Some(0.05),
            beam_width_array: vec![],
        });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();

        // OpenAI-standard fields land on inner.
        assert_eq!(nv.inner.temperature, Some(0.7));
        assert_eq!(nv.inner.top_p, Some(0.9));
        assert_eq!(nv.inner.presence_penalty, Some(0.2));
        assert_eq!(nv.inner.frequency_penalty, Some(0.3));
        assert_eq!(nv.inner.seed, Some(42));

        // Non-standard knobs land on CommonExt.
        assert_eq!(nv.common.top_k, Some(40));
        assert_eq!(nv.common.min_p, Some(0.05));
        assert_eq!(nv.common.repetition_penalty, Some(1.1));
        assert_eq!(nv.common.min_tokens, Some(5));
    }

    #[test]
    fn translation_maps_stop_strings() {
        let mut req = build_request_with_tokens(vec![1]);
        req.stop = vec!["END".to_string(), "STOP".to_string()];
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        match nv.inner.stop {
            Some(Stop::StringArray(ref s)) => assert_eq!(s, &vec!["END".to_string(), "STOP".to_string()]),
            other => panic!("expected StringArray stop, got {:?}", other),
        }
    }

    #[test]
    fn translation_maps_single_stop_string() {
        let mut req = build_request_with_tokens(vec![1]);
        req.stop = vec!["END".to_string()];
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        match nv.inner.stop {
            Some(Stop::String(ref s)) => assert_eq!(s, "END"),
            other => panic!("expected single Stop::String, got {:?}", other),
        }
    }

    #[test]
    fn translation_maps_ignore_eos() {
        let mut req = build_request_with_tokens(vec![1]);
        req.ignore_eos = true;
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        assert_eq!(nv.common.ignore_eos, Some(true));
    }

    #[test]
    fn translation_maps_include_stop_token_in_output() {
        let mut req = build_request_with_tokens(vec![1]);
        req.include_stop_token_in_output = true;
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        assert_eq!(nv.common.include_stop_str_in_output, Some(true));
    }

    #[test]
    fn translation_maps_guided_json_schema() {
        let mut req = build_request_with_tokens(vec![1]);
        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
            guide_type: trtllm::guided_decoding_params::GuideType::JsonSchema as i32,
            guide: r#"{"type":"object","properties":{"x":{"type":"number"}}}"#.to_string(),
        });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        let schema = nv.common.guided_json.expect("schema set");
        assert_eq!(schema["type"], serde_json::json!("object"));
    }

    #[test]
    fn translation_maps_guided_regex() {
        let mut req = build_request_with_tokens(vec![1]);
        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
            guide_type: trtllm::guided_decoding_params::GuideType::Regex as i32,
            guide: r"^[A-Za-z]+$".to_string(),
        });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        assert_eq!(nv.common.guided_regex.as_deref(), Some("^[A-Za-z]+$"));
    }

    #[test]
    fn translation_passes_request_id_via_nvext_annotations() {
        let req = build_request_with_tokens(vec![1, 2]);
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        let annotations = nv
            .nvext
            .as_ref()
            .and_then(|n| n.annotations.as_ref())
            .expect("annotations populated");
        assert!(
            annotations.iter().any(|a| a.contains("req-1")),
            "expected request_id in annotations, got {annotations:?}",
        );
    }

    // ----------------------------------------------------------------------
    // Multimodal forwarding: SMG packs raw decoded image bytes into the
    // gRPC `MultimodalInput.image_data` side-channel; we wrap each as a
    // base64 `data:image/jpeg;base64,...` URI inside
    // `multi_modal_data.image_url`. The Python worker reads
    // request["multi_modal_data"]["image_url"] as `[{"Url": "..."}]`. These
    // tests pin the exact serde shape so a refactor that breaks the
    // `MultiModalImageItem::Url(s)` → `{"Url": s}` contract trips here
    // instead of silently dropping images in production.
    // ----------------------------------------------------------------------

    #[test]
    fn translation_multimodal_input_none_yields_no_multi_modal_data() {
        let req = build_request_with_tokens(vec![1]);
        // build_request_with_tokens already sets multimodal_input: None
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        assert!(
            nv.multi_modal_data.is_none(),
            "expected None when proto's multimodal_input is None",
        );
    }

    #[test]
    fn translation_multimodal_input_empty_image_data_yields_no_multi_modal_data() {
        let mut req = build_request_with_tokens(vec![1]);
        req.multimodal_input = Some(trtllm::MultimodalInput { image_data: vec![] });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        assert!(
            nv.multi_modal_data.is_none(),
            "expected None when image_data is empty (filter)",
        );
    }

    #[test]
    fn translation_multimodal_image_data_forwards_as_base64_data_url() {
        // Two synthetic images: small JPEG header bytes are sufficient — we
        // only assert serde-shape, not that the bytes are decodable.
        let img1: Vec<u8> = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, b'J', b'F', b'I', b'F'];
        let img2: Vec<u8> = vec![0xff, 0xd8, 0xff, 0xe1];
        let mut req = build_request_with_tokens(vec![1]);
        req.multimodal_input = Some(trtllm::MultimodalInput {
            image_data: vec![img1.clone(), img2.clone()],
        });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        let mm = nv.multi_modal_data.expect("multi_modal_data populated");
        assert_eq!(mm.image_url.len(), 2);

        // Recover each item's URL string via the enum's only variant.
        let url_of = |item: &MultiModalImageItem| -> String {
            let MultiModalImageItem::Url(s) = item;
            s.clone()
        };
        let u1 = url_of(&mm.image_url[0]);
        let u2 = url_of(&mm.image_url[1]);

        let prefix = "data:image/jpeg;base64,";
        assert!(u1.starts_with(prefix), "u1 = {u1:?}");
        assert!(u2.starts_with(prefix), "u2 = {u2:?}");

        // Decode the base64 payload back and compare to the original bytes
        // — proves the wrapper is faithful, not just well-formatted.
        use base64::Engine;
        let dec = |s: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(s.strip_prefix(prefix).unwrap())
                .unwrap()
        };
        assert_eq!(dec(&u1), img1);
        assert_eq!(dec(&u2), img2);
    }

    #[test]
    fn translation_multimodal_serde_matches_python_worker_contract() {
        // The Python worker (components/.../multimodal_processor.py) does:
        //     item["Url"]   for the URL-variant arm
        // i.e. it expects the Rust enum to serialize as {"Url": "<string>"}.
        // Pin that exact JSON shape here so a #[serde(...)] tweak on
        // MultiModalImageItem can't silently break the contract.
        let mut req = build_request_with_tokens(vec![1]);
        req.multimodal_input = Some(trtllm::MultimodalInput {
            image_data: vec![vec![0xff, 0xd8, 0xff]],
        });
        let nv = request_translation::proto_to_completion_request(&req, "m".into()).unwrap();
        let mm = nv.multi_modal_data.unwrap();
        let json = serde_json::to_value(&mm).unwrap();
        let url = json["image_url"][0]["Url"]
            .as_str()
            .expect("image_url[0].Url is a string — contract for Python worker");
        assert!(url.starts_with("data:image/jpeg;base64,"));
    }

    // ----------------------------------------------------------------------
    // extract_token_ids: pulls Vec<u32> out of the engine's NvExtResponse
    // serialization. delta.rs::choice_from_postprocessor injects them under
    // nvext.token_ids when delta.token_ids is non-empty (our fallback path
    // for the SMG-via-gRPC consumer). These tests pin the deserialization
    // against the exact JSON shape NvExtResponse serializes to.
    // ----------------------------------------------------------------------

    #[test]
    fn extract_token_ids_pulls_from_nvext_response_shape() {
        // Mirror the JSON shape produced by serializing NvExtResponse {
        //     worker_id: None, timing: None,
        //     token_ids: Some(vec![100, 200, 300]),
        //     routed_experts: None,
        // }
        // (skip_serializing_if drops the None fields)
        let nvext_json = serde_json::json!({
            "token_ids": [100u32, 200u32, 300u32],
        });
        let mut nv = NvCreateCompletionResponseFixture::with_text("ignored").0;
        nv.inner.nvext = Some(nvext_json);

        let extracted = response_translation::extract_token_ids(&nv);
        assert_eq!(extracted, vec![100u32, 200u32, 300u32]);
    }

    #[test]
    fn extract_token_ids_returns_empty_when_nvext_absent() {
        let nv = NvCreateCompletionResponseFixture::with_text("ignored").0;
        // nvext is None on the fixture by construction.
        assert_eq!(response_translation::extract_token_ids(&nv), Vec::<u32>::new());
    }

    #[test]
    fn extract_token_ids_returns_empty_when_token_ids_field_missing() {
        // nvext present but no token_ids field — e.g. only worker_id surfaced.
        let mut nv = NvCreateCompletionResponseFixture::with_text("ignored").0;
        nv.inner.nvext = Some(serde_json::json!({
            "worker_id": {"prefill_worker_id": 7u64},
        }));
        assert_eq!(response_translation::extract_token_ids(&nv), Vec::<u32>::new());
    }

    #[test]
    fn extract_token_ids_handles_full_nvext_response_shape() {
        // The fully-populated case — verify our extractor doesn't trip on
        // sibling fields in the same nvext object.
        let mut nv = NvCreateCompletionResponseFixture::with_text("ignored").0;
        nv.inner.nvext = Some(serde_json::json!({
            "worker_id": {"prefill_worker_id": 7u64, "decode_worker_id": 8u64},
            "timing": {"first_token_at": 1.5},
            "token_ids": [42u32, 99u32],
            "routed_experts": null,
        }));
        let extracted = response_translation::extract_token_ids(&nv);
        assert_eq!(extracted, vec![42u32, 99u32]);
    }

    #[test]
    fn nv_response_to_chunk_propagates_token_ids_from_nvext() {
        // End-to-end of the chunk translator: token_ids in nvext should
        // surface as GenerateStreamChunk.token_ids.
        let mut nv = NvCreateCompletionResponseFixture::with_text("hi").0;
        nv.inner.nvext = Some(serde_json::json!({"token_ids": [1u32, 2, 3]}));
        let frame = response_translation::nv_response_to_chunk("req-1", &nv, 5, 0);
        match frame.response {
            Some(trtllm::generate_response::Response::Chunk(c)) => {
                assert_eq!(c.token_ids, vec![1u32, 2, 3]);
                assert_eq!(c.prompt_tokens, 5);
            }
            other => panic!("expected Chunk variant, got {other:?}"),
        }
    }

    #[test]
    fn response_translation_emits_chunk_variant() {
        let nv = NvCreateCompletionResponseFixture::with_text("hello");
        let frame = response_translation::nv_response_to_chunk("req-1", &nv.0, 5, 0);
        assert_eq!(frame.request_id, "req-1");
        match frame.response {
            Some(trtllm::generate_response::Response::Chunk(c)) => {
                assert_eq!(c.prompt_tokens, 5);
                assert_eq!(c.sequence_index, 0);
            }
            other => panic!("expected Chunk variant, got {other:?}"),
        }
    }

    #[test]
    fn response_translation_emits_complete_variant_with_finish_reason() {
        let nv = NvCreateCompletionResponseFixture::with_finish(
            dynamo_async_openai::types::CompletionFinishReason::Length,
        );
        let frame = response_translation::nv_response_to_complete(
            "req-1",
            &nv.0,
            5,
            8,
            0,
            vec![10, 20, 30],
        );
        match frame.response {
            Some(trtllm::generate_response::Response::Complete(c)) => {
                assert_eq!(c.finish_reason, "length");
                assert_eq!(c.prompt_tokens, 5);
                assert_eq!(c.completion_tokens, 8);
                assert_eq!(c.output_token_ids, vec![10u32, 20, 30]);
            }
            other => panic!("expected Complete variant, got {other:?}"),
        }
    }

    #[test]
    fn nv_response_to_complete_carries_cumulative_output_token_ids() {
        // SMG non-streaming path reads complete.output_ids() — pin that
        // we propagate the accumulator caller hands us.
        let nv = NvCreateCompletionResponseFixture::with_finish(
            dynamo_async_openai::types::CompletionFinishReason::Stop,
        );
        let cumulative = vec![791u32, 6864, 315, 9822, 374, 12366, 13];
        let frame = response_translation::nv_response_to_complete(
            "req-1",
            &nv.0,
            15,
            7,
            0,
            cumulative.clone(),
        );
        match frame.response {
            Some(trtllm::generate_response::Response::Complete(c)) => {
                assert_eq!(c.output_token_ids, cumulative);
                assert_eq!(c.prompt_tokens, 15);
                assert_eq!(c.completion_tokens, 7);
            }
            other => panic!("expected Complete variant, got {other:?}"),
        }
    }

    #[test]
    fn nv_response_to_chunk_propagates_cached_tokens() {
        let nv = NvCreateCompletionResponseFixture::with_text("hi");
        let frame = response_translation::nv_response_to_chunk("req-1", &nv.0, 100, 64);
        match frame.response {
            Some(trtllm::generate_response::Response::Chunk(c)) => {
                assert_eq!(c.cached_tokens, 64);
            }
            other => panic!("expected Chunk variant, got {other:?}"),
        }
    }

    #[test]
    fn nv_response_to_complete_propagates_cached_tokens() {
        // Pin: sim's cache-hit-rate metric reads cached_tokens off the
        // Complete frame. Without this propagation sim reports 0% even when
        // the engine reports >0 reuse. Caught in the duolingo regression run.
        let nv = NvCreateCompletionResponseFixture::with_finish(
            dynamo_async_openai::types::CompletionFinishReason::Stop,
        );
        let frame = response_translation::nv_response_to_complete(
            "req-1",
            &nv.0,
            500,
            10,
            128,
            vec![],
        );
        match frame.response {
            Some(trtllm::generate_response::Response::Complete(c)) => {
                assert_eq!(c.cached_tokens, 128);
                assert_eq!(c.prompt_tokens, 500);
                assert_eq!(c.completion_tokens, 10);
            }
            other => panic!("expected Complete variant, got {other:?}"),
        }
    }

    // ----------------------------------------------------------------------
    // Test fixture helpers
    // ----------------------------------------------------------------------

    use crate::protocols::openai::completions::NvCreateCompletionResponse;

    struct NvCreateCompletionResponseFixture(NvCreateCompletionResponse);

    impl NvCreateCompletionResponseFixture {
        fn with_text(text: &str) -> Self {
            let inner = dynamo_async_openai::types::CreateCompletionResponse {
                id: "cmpl-1".to_string(),
                created: 0,
                model: "m".to_string(),
                object: "text_completion".to_string(),
                system_fingerprint: None,
                usage: None,
                nvext: None,
                choices: vec![dynamo_async_openai::types::Choice {
                    text: text.to_string(),
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
            };
            Self(NvCreateCompletionResponse { inner })
        }

        fn with_finish(reason: dynamo_async_openai::types::CompletionFinishReason) -> Self {
            let inner = dynamo_async_openai::types::CreateCompletionResponse {
                id: "cmpl-1".to_string(),
                created: 0,
                model: "m".to_string(),
                object: "text_completion".to_string(),
                system_fingerprint: None,
                usage: None,
                nvext: None,
                choices: vec![dynamo_async_openai::types::Choice {
                    text: String::new(),
                    index: 0,
                    logprobs: None,
                    finish_reason: Some(reason),
                }],
            };
            Self(NvCreateCompletionResponse { inner })
        }
    }

    // engine_error_to_status status-mapping tests
    //
    // Validates the branches the function discriminates:
    //   - JSON {code: 4xx, message} → InvalidArgument (HTTP 400)
    //   - JSON {code: 5xx, message} → Internal         (HTTP 500)
    //   - "admission control rejected" substring → ResourceExhausted (HTTP 429)
    //   - anything else → Internal (HTTP 500) — same status as 5xx envelope
    //     but with the default "engine error: ..." prefix
    #[test]
    fn engine_error_to_status_maps_invalid_request_json_to_invalid_argument() {
        // Shape the Python worker emits for InvalidRequestError reaches
        // this function wrapped by DynamoError's Display ("<error_type>:
        // <message>"). Cover both the plain-JSON case and the realistic
        // prefixed case.
        let plain = r#"{"code": 400, "message": "input token count exceeds max context length"}"#;
        let prefixed = format!("Unknown: {}", plain);
        for input in [plain, prefixed.as_str()] {
            let status = engine_error_str_to_status(input);
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "input={input}"
            );
            assert_eq!(
                status.message(),
                "input token count exceeds max context length",
                "input={input}"
            );
        }
    }

    #[test]
    fn engine_error_to_status_maps_admission_rejected_to_resource_exhausted() {
        let err = "admission control rejected: workers over threshold";
        let status = engine_error_str_to_status(err);
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn engine_error_to_status_falls_back_to_internal_on_plain_string() {
        // Bare string from Python plain-RequestError path (engine enqueue
        // wrap). No JSON, no admission substring → 500.
        let err = "engine.enqueue_request blew up";
        let status = engine_error_str_to_status(err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn engine_error_to_status_maps_5xx_json_to_internal_with_message() {
        // 5xx envelope from py_executor._handle_errors:
        //   {"code": 500, "message": "Request 1234 timed out"}
        // Engine is declaring "server-side fault" explicitly so the gRPC
        // layer doesn't have to guess. Internal → SMG 500. The message body
        // should be the JSON's message, not "engine error: {...}".
        let plain = r#"{"code": 500, "message": "Request 1234 timed out"}"#;
        let prefixed = format!("Unknown: {}", plain);
        for input in [plain, prefixed.as_str()] {
            let status = engine_error_str_to_status(input);
            assert_eq!(status.code(), tonic::Code::Internal, "input={input}");
            assert_eq!(status.message(), "Request 1234 timed out", "input={input}");
        }
        // 503 (server-side capacity) and 504 (upstream timeout) also map to
        // Internal — the gRPC contract doesn't distinguish further. SMG can
        // refine with its own status code mapping if needed.
        for code in [503, 504, 599] {
            let err = format!(r#"{{"code": {}, "message": "x"}}"#, code);
            assert_eq!(engine_error_str_to_status(&err).code(), tonic::Code::Internal);
        }
    }

    #[test]
    fn engine_error_to_status_ignores_out_of_range_codes() {
        // Codes outside [400, 600) aren't HTTP statuses — keep walking braces
        // and fall through to the default branch. Prevents arbitrary integer
        // payloads from upgrading themselves into a Status.
        for code in [200u16, 301, 100, 999] {
            let err = format!(r#"{{"code": {}, "message": "x"}}"#, code);
            let status = engine_error_str_to_status(&err);
            assert_eq!(status.code(), tonic::Code::Internal, "code={code}");
            assert!(
                status.message().starts_with("engine error: "),
                "code={code} msg={:?}",
                status.message()
            );
        }
    }

    #[test]
    fn engine_error_str_handles_multiple_braces_picks_parseable_one() {
        // Regression for the revert: a wrapped error message with `{` in
        // an earlier position must not block the parse-able JSON later.
        // First brace opens a non-JSON fragment; second opens the 400
        // payload. The walker must find the second.
        let err = r#"context: {request_id: xyz}; engine: {"code": 400, "message": "prompt too long"}"#;
        let status = engine_error_str_to_status(err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "prompt too long");
    }

    #[test]
    fn engine_error_str_admission_substring_does_not_shadow_400() {
        // Pathological mix: a wrapped message that contains BOTH a
        // valid 400 JSON and the "admission control rejected" substring.
        // The 400 JSON should win (it's more specific). This is the
        // direct regression scenario that caused the dec7ec981 revert.
        let err = r#"{"code": 400, "message": "prompt too long"} admission control rejected"#;
        let status = engine_error_str_to_status(err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }
}
