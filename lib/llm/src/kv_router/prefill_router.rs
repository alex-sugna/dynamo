// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

/// [PD_TIMING] Wall-clock seconds since the Unix epoch as f64.
/// Cross-pod log correlation: pair with the float timestamps logged by the
/// Python handler (handler_base.py) and the request statistics line emitted by
/// the TRT-LLM engine. Returns 0.0 on the (practically impossible) case of
/// the system clock being before UNIX_EPOCH.
#[inline]
fn pd_now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
use futures::{
    StreamExt,
    stream::{self},
};
use tokio::sync::{OwnedSemaphorePermit, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use dynamo_runtime::{
    component::Endpoint,
    engine::ResponseStream,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Context, ManyOut, Operator, PushRouter,
        RouterMode, ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated, maybe_error::MaybeError},
};

use crate::{
    discovery::{ModelManager, RuntimeConfigWatch},
    kv_router::protocols::WorkerId,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride, protocols::BlockExtraInfo},
    protocols::common::FinishReason,
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    protocols::common::preprocessor::{BootstrapInfo, PrefillResult},
    protocols::common::timing::{RequestPhase, RequestTracker, WORKER_TYPE_PREFILL},
};

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),

    /// Required worker ID not found in request headers (Direct routing mode)
    #[error(
        "Worker ID required in Direct routing mode but not found in request headers. \
             Expected x-prefill-instance-id to be set by external router (e.g., EPP)."
    )]
    MissingWorkerIdForDirectRouting,
}

/// Result of the prefill phase in `generate()`.
enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result
    Completed {
        result: PrefillResult,
        first_client_delta: Option<Annotated<LLMEngineOutput>>,
    },
    /// Prefill terminated the request itself (EOS / stop word on the first
    /// generated token). TRT-LLM emits no disaggregated_params and queues no
    /// KV transfer in this case — the response is already complete and
    /// decode is not needed. Forward the accumulated prefill outputs back to
    /// the client.
    TerminalInPrefill {
        outputs: Vec<Annotated<LLMEngineOutput>>,
    },
}

/// Result of `execute_prefill`: either a normal prefill (decode required) or
/// a terminal prefill (decode skipped).
enum ExecutePrefillResult {
    Normal {
        result: PrefillResult,
        first_client_delta: Option<Annotated<LLMEngineOutput>>,
        worker_info: Option<(u64, u32)>,
    },
    Terminal {
        outputs: Vec<Annotated<LLMEngineOutput>>,
    },
}

/// The inner router used by PrefillRouter
#[derive(Clone)]
enum InnerPrefillRouter {
    /// KV-aware routing using KvPushRouter
    KvRouter(Arc<KvPushRouter>),
    /// Simple routing (RoundRobin, Random, Direct)
    /// Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
    /// available in KV routing mode where the router has actual bookkeeping.
    SimpleRouter(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

impl InnerPrefillRouter {
    /// Generate with optional direct routing to specific worker.
    /// For KvRouter, target_worker is ignored since prefill_worker_id is already set on the request.
    /// For SimpleRouter, target_worker triggers direct routing via router.direct().
    async fn generate_to_worker(
        &self,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        match (self, target_worker) {
            // KvRouter: prefill_worker_id already set on request, KvPushRouter::select_worker uses it
            (InnerPrefillRouter::KvRouter(router), _) => router.generate(request).await,
            (InnerPrefillRouter::SimpleRouter(router), Some(worker_id)) => {
                router.direct(request, worker_id).await
            }
            (InnerPrefillRouter::SimpleRouter(router), None) => router.generate(request).await,
        }
    }

    /// Select next worker (for non-KV modes only)
    fn select_next_worker(&self) -> Option<u64> {
        match self {
            InnerPrefillRouter::SimpleRouter(router) => router.select_next_worker(),
            InnerPrefillRouter::KvRouter(_) => None,
        }
    }
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: `query_instance_id` annotation present → returns worker IDs without execution
/// - Pre-routed: `prefill_worker_id`/`decode_worker_id` set → routes to specified workers
/// - Normal: Worker IDs determined by router based on KV cache state
pub struct PrefillRouter {
    prefill_router: OnceLock<InnerPrefillRouter>,
    model_manager: Arc<ModelManager>,
    endpoint_id: OnceLock<EndpointId>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    decode_fallback: bool,
    /// Model name used to look up the worker monitor for prefill client registration
    model_name: String,
    /// Namespace used to look up the correct WorkerSet's worker monitor
    namespace: String,
    /// Watcher for the prefill endpoint's per-worker ModelRuntimeConfig. Used to
    /// look up the picked prefill's partition_group on the decode handoff so the
    /// decode scheduler can restrict to the same partition. None until activate().
    prefill_runtime_config_watch: OnceLock<RuntimeConfigWatch>,
    /// Watcher for the decode endpoint's per-worker ModelRuntimeConfig. Used to
    /// compute the set of partition_groups that currently have at least one
    /// live decode worker, so we can exclude prefill workers whose partition
    /// has no surviving decodes from selection (fail-fast at prefill pick
    /// instead of burning prefill compute on a doomed pipeline). Populated at
    /// construction by the discovery layer; None for `disabled()` (no decode
    /// endpoint to track).
    decode_runtime_config_watch: Option<RuntimeConfigWatch>,
}

impl PrefillRouter {
    fn extract_trtllm_prefill_first_client_delta(
        first_output: &Annotated<LLMEngineOutput>,
    ) -> Option<Annotated<LLMEngineOutput>> {
        let output = first_output.data.as_ref()?;
        if output.token_ids.is_empty() {
            return None;
        }

        let disaggregated_params = output.disaggregated_params.as_ref()?;
        let is_trtllm_prefill = disaggregated_params.get("opaque_state").is_some()
            || disaggregated_params.get("first_gen_tokens").is_some()
            || disaggregated_params.get("first_gen_log_probs").is_some();
        if !is_trtllm_prefill {
            return None;
        }

        let mut client_output = output.clone();
        client_output.finish_reason = None;
        client_output.stop_reason = None;
        client_output.disaggregated_params = None;

        Some(first_output.clone().transfer(Some(client_output)))
    }

    fn prepend_prefill_delta(
        decode_stream: ManyOut<Annotated<LLMEngineOutput>>,
        first_delta: Option<Annotated<LLMEngineOutput>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        let context = decode_stream.context();
        let combined_stream = stream::iter(first_delta).chain(decode_stream);
        ResponseStream::new(Box::pin(combined_stream), context)
    }

    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        decode_fallback: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            prefill_router: OnceLock::new(),
            model_manager,
            endpoint_id: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            router_mode,
            decode_fallback,
            model_name: String::new(), // Not used for disabled router
            namespace: String::new(),  // Not used for disabled router
            prefill_runtime_config_watch: OnceLock::new(),
            decode_runtime_config_watch: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        decode_fallback: bool,
        model_name: String,
        namespace: String,
        decode_runtime_config_watch: Option<RuntimeConfigWatch>,
    ) -> Arc<Self> {
        let prefill_router = OnceLock::new();
        let cancel_token = CancellationToken::new();

        let router = Arc::new(Self {
            prefill_router,
            model_manager: model_manager.clone(),
            endpoint_id: OnceLock::new(),
            cancel_token: cancel_token.clone(),
            router_mode,
            decode_fallback,
            model_name,
            namespace,
            prefill_runtime_config_watch: OnceLock::new(),
            decode_runtime_config_watch,
        });

        // Spawn background task to wait for activation
        let router_clone = router.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = activation_rx => {
                    let Ok(endpoint) = result else {
                        tracing::debug!("Prefill router activation channel closed without receiving endpoint");
                        return;
                    };

                    if let Err(e) = router_clone.activate(
                        endpoint,
                        model_manager,
                        kv_cache_block_size,
                        kv_router_config,
                    ).await {
                        tracing::error!(error = %e, "Failed to activate prefill router");
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Prefill router activation cancelled");
                }
            }
        });

        router
    }

    /// Activate the prefill router with the provided endpoint
    async fn activate(
        &self,
        endpoint: Endpoint,
        model_manager: Arc<ModelManager>,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> Result<()> {
        tracing::info!(
            router_mode = ?self.router_mode,
            "Activating prefill router"
        );

        // Store endpoint_id for later use in resolve_prefill_worker
        let _ = self.endpoint_id.set(endpoint.id());

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint
        // AND for the partition_group lookup at the decode handoff).
        // This must be done before creating the router so bootstrap info is available.
        let prefill_runtime_config_watch = model_manager
            .get_or_create_runtime_config_watcher(&endpoint)
            .await?;
        let _ = self
            .prefill_runtime_config_watch
            .set(prefill_runtime_config_watch);

        let inner_router = if self.router_mode.is_kv_routing() {
            // Create KV chooser using the endpoint (this is a prefill router)
            let kv_chooser = model_manager
                .kv_chooser_for(
                    &endpoint,
                    kv_cache_block_size,
                    kv_router_config,
                    WORKER_TYPE_PREFILL,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();

            // Register prefill client with worker monitor for TTFT metric cleanup in disaggregated mode
            if let Some(monitor) =
                model_manager.get_worker_monitor_for_namespace(&self.model_name, &self.namespace)
            {
                monitor.set_prefill_client(client.clone());
            }

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                client,
                RouterMode::KV,
                None, // busy_threshold
                None, // worker_monitor
            )
            .await?
            // Opt in to the prefill-side stall timeout knob
            // (DYN_STREAM_STALL_TIMEOUT_MS_PREFILL). Falls back to the base
            // DYN_STREAM_STALL_TIMEOUT_MS read by the constructor when unset.
            .with_stall_timeout_for("prefill");

            // Wrap it in KvPushRouter
            InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new(push_router, kv_chooser)))
        } else {
            // Create client for simple router
            let client = endpoint.client().await?;

            // Register prefill client with worker monitor for TTFT metric cleanup in disaggregated mode
            if let Some(monitor) =
                model_manager.get_worker_monitor_for_namespace(&self.model_name, &self.namespace)
            {
                monitor.set_prefill_client(client.clone());
            }

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                client,
                self.router_mode,
                None, // busy_threshold
                None, // worker_monitor
            )
            .await?
            // Same prefill-side stall timeout override as the KV-mode branch
            // above. The decode push (built in entrypoint/input/common.rs)
            // gets the "decode" override.
            .with_stall_timeout_for("prefill");

            InnerPrefillRouter::SimpleRouter(Arc::new(push_router))
        };

        // Set the router (ignore error if already set)
        let _ = self.prefill_router.set(inner_router);

        tracing::info!(
            router_mode = ?self.router_mode,
            "Prefill router activated successfully"
        );

        Ok(())
    }

    /// Select a prefill worker and resolve its bootstrap connection info.
    /// If preselected_worker is provided (GAIE Stage 2), use it directly.
    /// Otherwise, query for the best worker (KV mode) or select next worker (non-KV modes).
    async fn resolve_prefill_worker(
        &self,
        req: &PreprocessedRequest,
        preselected_worker: Option<u64>,
    ) -> Option<(u64, u32, BootstrapInfo)> {
        let endpoint_id = self.endpoint_id.get()?;
        self.prefill_router.get()?;

        // Worker selection
        let (worker_id, dp_rank) = if let Some(id) = preselected_worker {
            let dp_rank = req.routing.as_ref().and_then(|r| r.dp_rank).unwrap_or(0);
            tracing::debug!(
                worker_id = id,
                dp_rank = dp_rank,
                "Using pre-selected prefill worker for bootstrap"
            );
            (id, dp_rank)
        } else {
            // Use shared worker selection logic (update_states=false for peek behavior)
            // Extract LORA name and priority jump from routing hints
            let lora_name = req.routing.as_ref().and_then(|r| r.lora_name.clone());
            let priority_jump = req
                .routing
                .as_ref()
                .and_then(|r| r.priority_jump)
                .unwrap_or(0.0);
            let allowed_worker_ids = req
                .routing
                .as_ref()
                .and_then(|r| r.allowed_worker_ids.clone());
            let (routing_token_ids, block_mm_infos) = req.block_mm_routing_info();
            match self
                .query_prefill_worker(
                    routing_token_ids,
                    block_mm_infos,
                    false,
                    lora_name,
                    priority_jump,
                    allowed_worker_ids,
                )
                .await
            {
                Ok((worker_id, dp_rank)) => (worker_id, dp_rank),
                Err(_) => return None,
            }
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let endpoint = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)?;
        let host = endpoint.bootstrap_host?;
        let port = endpoint.bootstrap_port?;

        let bootstrap_room: u64 = rand::random_range(0..=i64::MAX as u64);

        tracing::debug!(
            worker_id = worker_id,
            dp_rank = dp_rank,
            bootstrap_host = %host,
            bootstrap_port = port,
            bootstrap_room = bootstrap_room,
            router_mode = ?self.router_mode,
            "Built bootstrap_info upfront before prefill"
        );

        Some((
            worker_id,
            dp_rank,
            BootstrapInfo {
                bootstrap_host: host,
                bootstrap_port: port,
                bootstrap_room,
            },
        ))
    }

    /// Execute prefill with the given router and extract structured result.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// If `phase_permit` is provided, it is dropped after the first output is received,
    /// allowing subsequent `set_phase` calls to proceed. This is used in the bootstrap
    /// optimization path to ensure `record_worker_full` completes before the phase changes.
    ///
    /// Returns either a normal prefill result (decode required) or a terminal
    /// prefill result (decode skipped — prefill emitted EOS / stop word on the
    /// first generated token, so TRT-LLM did not queue a KV transfer).
    async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_permit: Option<OwnedSemaphorePermit>,
        request_id: &str,
    ) -> Result<ExecutePrefillResult, PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        let mut prefill_response = router
            .generate_to_worker(request, target_worker)
            .await
            .map_err(|e| {
                // Embed the source error chain in the outer message so
                // grpc/service/openai.rs::classify_setup_error can match
                // the `KvSchedulerError::AdmissionRejected` Display text
                // ("admission control rejected request (retry after Ns)")
                // via substring fallback when typed downcast can't see
                // through the `anyhow → Box<dyn Error>` wrapping that
                // `Some(e.into())` produces. Companion to the typed
                // downcast in classify_setup_error — typed first, this
                // substring is the safety net. Mirrors the HTTP path's
                // belt-and-suspenders approach at http/service/openai.rs
                // ~line 234. The 400 path (JSON code payload) walks
                // braces left-to-right so a wrapped message with `{` in
                // the prefix won't shadow a legitimate 400 payload.
                PrefillError::PrefillError(
                    format!("failed to route to prefill worker: {e:#}"),
                    Some(e.into()),
                )
            })?;

        // Drop phase permit now - routing is complete, record_worker_full was called in select_worker.
        // This unblocks set_phase(Decode) in the main task without waiting for prefill output.
        drop(phase_permit);

        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        // [PD_TIMING] Frontend has the prefill first_output (contains
        // disaggregated_params + first token). Pair with the prefill pod's
        // `prefill_handler_yield` log on the same request_id to measure the
        // prefill-pod→frontend NATS hop. Pair with prefill stats' kv_send_start
        // to measure the full engine→frontend visibility latency.
        tracing::info!(
            "[PD_TIMING] event=frontend_first_output_received context_id={} ts={:.6}",
            request_id,
            pd_now_secs(),
        );

        if let Some(err) = first_output.err() {
            return Err(PrefillError::PrefillError(
                "Prefill router returned error in output".to_string(),
                Some(Box::new(err)),
            ));
        }

        // If prefill terminated the request itself (Stop / EoS / Length on the
        // first generated token) and emitted no disaggregated_params, there is
        // no KV transfer to coordinate and no work for decode. Collect the
        // remaining prefill outputs and forward them to the client directly.
        let terminal_in_prefill = first_output.data.as_ref().is_some_and(|o| {
            o.disaggregated_params.is_none()
                && matches!(
                    o.finish_reason,
                    Some(FinishReason::Stop)
                        | Some(FinishReason::EoS)
                        | Some(FinishReason::Length)
                )
        });
        if terminal_in_prefill {
            let mut outputs = vec![first_output];
            while let Some(next) = prefill_response.next().await {
                outputs.push(next);
            }
            return Ok(ExecutePrefillResult::Terminal { outputs });
        }

        let mut prompt_tokens_details = first_output
            .data
            .as_ref()
            .and_then(|o| o.completion_usage.as_ref())
            .and_then(|u| u.prompt_tokens_details.clone());

        while let Some(next) = prefill_response.next().await {
            if let Some(o) = next.data.as_ref()
                && prompt_tokens_details.is_none()
            {
                prompt_tokens_details = o
                    .completion_usage
                    .as_ref()
                    .and_then(|u| u.prompt_tokens_details.clone());
            }
        }

        // [PD_TIMING] Prefill response stream from the prefill pod has closed.
        // In healthy disagg this is microseconds after first_output (TRT-LLM
        // iterator ends right after the single yield). When kv-transfer hangs,
        // this is delayed until the prefill engine's kv_transfer_timeout fires
        // and the request is fully torn down. The delta to
        // `frontend_first_output_received` is the drain-wait cost.
        tracing::info!(
            "[PD_TIMING] event=frontend_prefill_stream_closed context_id={} ts={:.6}",
            request_id,
            pd_now_secs(),
        );

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };

        let Some(disaggregated_params) = output.disaggregated_params.clone() else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };
        let first_client_delta = Self::extract_trtllm_prefill_first_client_delta(&first_output);

        // Extract prefill worker ID and dp_rank from disaggregated_params
        let prefill_worker_info =
            disaggregated_params
                .get("worker_id")
                .and_then(|worker_id_json| {
                    let worker_id = worker_id_json
                        .get("prefill_worker_id")
                        .and_then(|v| v.as_u64())?;
                    let dp_rank = worker_id_json
                        .get("prefill_dp_rank")
                        .and_then(|v| v.as_u64())
                        .map(|r| r as u32)
                        .unwrap_or(0);
                    Some((worker_id, dp_rank))
                });
        // Extract cached_tokens from prefill's extra_args (actual KV cache hits)
        let cached_tokens = output
            .extra_args
            .as_ref()
            .and_then(|ea| ea.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        Ok(ExecutePrefillResult::Normal {
            result: PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
                cached_tokens,
            },
            first_client_delta,
            worker_info: prefill_worker_info,
        })
    }

    /// Spawn prefill as a background task.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// The `phase_permit` is passed to the spawned task and dropped after the first output,
    /// allowing the main task's `set_phase(Decode)` to proceed.
    fn spawn_prefill_task(
        &self,
        prefill_request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_permit: OwnedSemaphorePermit,
        request_id: String,
    ) {
        let router = self.prefill_router.get().cloned();
        // Capture current span to propagate trace context to the spawned task
        let span = tracing::Span::current();

        tokio::spawn(
            async move {
                match Self::execute_prefill(
                    router,
                    prefill_request,
                    target_worker,
                    Some(phase_permit),
                    &request_id,
                )
                .await
                {
                    Ok(_) => {
                        tracing::debug!("Prefill background task completed");
                    }
                    Err(e) => {
                        tracing::warn!("Prefill background task error: {e:?}");
                    }
                }
            }
            .instrument(span),
        );
    }

    /// Compute the set of prefill worker IDs whose `partition_group` has at
    /// least one live decode worker. If both runtime-config watches are
    /// populated, returns the intersection of (a) the caller's
    /// `allowed_worker_ids` (or "all" if None) and (b) prefills whose
    /// partition has decode coverage. Returns `None` (= no filter, existing
    /// behavior) when the decode watch isn't available — e.g. aggregated
    /// mode, the C bindings disabled-router path, or any future code path
    /// that skips passing the decode watch into `new()`.
    ///
    /// Coverage rule mirrors the scheduler's eligibility check:
    /// - prefill `partition_group = None` ⇒ wildcard, always allowed
    ///   (stamps a `None` partition on decode_req → scheduler doesn't filter)
    /// - prefill `partition_group = Some(g)` ⇒ allowed iff some decode has
    ///   `partition_group = Some(g)` OR some decode has `partition_group =
    ///   None` (wildcard decode)
    /// - no decodes at all ⇒ all prefills excluded (request will fail fast
    ///   with `NoEndpoints`; correct fail mode — no decode means no service)
    fn compute_allowed_prefills(
        &self,
        caller_allowed: Option<HashSet<WorkerId>>,
    ) -> Option<HashSet<WorkerId>> {
        let decode_watch = self.decode_runtime_config_watch.as_ref()?;
        let prefill_watch = self.prefill_runtime_config_watch.get()?;

        // Snapshot decode partitions
        let decode_snapshot = decode_watch.borrow();
        let mut live_partitions: HashSet<Option<String>> = HashSet::new();
        for cfg in decode_snapshot.values() {
            live_partitions.insert(cfg.partition_group.clone());
        }
        let wildcard_decode = live_partitions.contains(&None);
        drop(decode_snapshot);

        // Snapshot prefill partitions and apply the coverage rule
        let prefill_snapshot = prefill_watch.borrow();
        let total_prefills = prefill_snapshot.len();
        let mut allowed: HashSet<WorkerId> = HashSet::new();
        let mut excluded_partitions: HashSet<String> = HashSet::new();
        for (wid, cfg) in prefill_snapshot.iter() {
            let routable = match &cfg.partition_group {
                None => true, // wildcard prefill; stamps None
                Some(g) => wildcard_decode || live_partitions.contains(&Some(g.clone())),
            };
            if routable {
                allowed.insert(*wid);
            } else if let Some(g) = &cfg.partition_group {
                excluded_partitions.insert(g.clone());
            }
        }
        drop(prefill_snapshot);

        // Log every call so we can prove at runtime that the filter is running
        // AND what it currently sees, not just when it's excluding workers. The
        // earlier "only on exclusion" gate masked the case where the filter ran
        // but live_partitions still contained a partition whose decode had just
        // been reported down (watch propagation race) — i.e. the bug it was
        // supposed to catch was invisible.
        let excluded_count = total_prefills - allowed.len();
        let final_allowed_count = match &caller_allowed {
            Some(caller) => caller.intersection(&allowed).count(),
            None => allowed.len(),
        };
        tracing::info!(
            live_partitions = ?live_partitions,
            wildcard_decode,
            allowed_count = allowed.len(),
            excluded_count,
            excluded_partitions = ?excluded_partitions,
            total_prefills,
            caller_allowed_count = ?caller_allowed.as_ref().map(|s| s.len()),
            final_allowed_count,
            "[partition] compute_allowed_prefills snapshot"
        );

        // Intersect with caller's allowed set (e.g. EPP or migration retry exclusions)
        match caller_allowed {
            Some(caller) => Some(caller.intersection(&allowed).copied().collect()),
            None => Some(allowed),
        }
    }

    /// Query the best prefill worker without executing a request.
    /// Returns (worker_id, dp_rank).
    ///
    /// This is the shared worker selection logic used by both `resolve_prefill_worker`
    /// and `query_route`.
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<(u64, u32)> {
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        // Restrict to prefills whose partition has live decodes. When the
        // decode watch isn't available this returns the caller's set unchanged
        // (existing behavior preserved for non-disaggregated paths).
        let effective_allowed = self.compute_allowed_prefills(allowed_worker_ids);

        match prefill_router {
            InnerPrefillRouter::KvRouter(r) => {
                let (worker, _overlap) = r
                    .chooser
                    .find_best_match(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        lora_name,
                        priority_jump,
                        effective_allowed,
                        None, // partition_group: prefill picks are global; partitioning only constrains decode
                    )
                    .await?;
                Ok((worker.worker_id, worker.dp_rank))
            }
            InnerPrefillRouter::SimpleRouter(r) => {
                let worker_id = if update_states {
                    r.select_next_worker()
                } else {
                    r.peek_next_worker()
                }
                .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok((worker_id, 0))
            }
        }
    }

    /// Check if disaggregated mode is currently active (prefill router activated)
    pub fn is_activated(&self) -> bool {
        self.prefill_router.get().is_some()
    }
}

impl Drop for PrefillRouter {
    fn drop(&mut self) {
        tracing::debug!("Dropping PrefillRouter, cancelling background activation task");
        self.cancel_token.cancel();
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for PrefillRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        // Extract request data while preserving context
        let (mut req, context) = request.into_parts();
        let request_id = context.id().to_string();
        let engine_ctx = context.context();

        // [pin_trace:5/prefill_router_entry] First log inside PrefillRouter.
        // Confirms the pin survived Migration + Backend operators. If pin is
        // None here but Migration's [pin_trace:3/migration] showed it Some,
        // the loss is in the operator chain wiring between Migration→Backend
        // and PrefillRouter — investigate `link()` / forward_edge plumbing.
        tracing::info!(
            target: "dynamo::pin_trace",
            request_id = %request_id,
            decode_instance_id = ?req.decode_instance_id,
            routing_backend_instance_id = ?req.routing.as_ref().and_then(|r| r.backend_instance_id),
            routing_decode_worker_id = ?req.routing.as_ref().and_then(|r| r.decode_worker_id),
            tracker_phase = ?req.tracker.as_ref().map(|t| t.phase()),
            has_tracker = req.tracker.is_some(),
            prefill_router_activated = self.prefill_router.get().is_some(),
            "[pin_trace:5/prefill_router_entry] PrefillRouter::generate entry"
        );

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If prefill router is not activated (no prefill workers discovered),
        // this is aggregated mode — route directly to decode.
        //
        // [SMG-DYNAMO DEBUG] Loud log to diagnose smg-dynamo-pd-kimi-mn:
        // when SMG drives the gRPC servicer in cross-pod PD, every request
        // hit this branch and decode rejected with "Disaggregated params
        // are required for decode mode". Logging model_name/namespace +
        // request_id so we can match against the activator key registered
        // in watcher.rs:480/725 and confirm whether activation never fired
        // (cause #1) vs. fired under a different key (cause #2).
        if self.prefill_router.get().is_none() {
            tracing::warn!(
                request_id = %request_id,
                model_name = %self.model_name,
                namespace = %self.namespace,
                "PrefillRouter NOT activated; falling through to aggregated path \
                 (decode worker will see no disaggregated_params)"
            );
            return next.generate(context.map(|_| req)).await;
        }

        // Ensure tracker exists for routing decisions in disaggregated mode.
        // Create one if not provided by the upstream DeltaGenerator.
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.as_ref().unwrap();
        let prefill_phase_permit = tracker.set_phase(RequestPhase::Prefill).await;

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set)
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);
        // Clear decode_instance_id so it doesn't leak into prefill routing.
        // decode_instance_id targets decode workers, but the prefill router's
        // PushRouter is connected to the prefill component (prefill/generate).
        // If this ID leaks through, the prefill KvPushRouter will try to find
        // the decode worker in the prefill endpoint and fail.
        prefill_req.decode_instance_id = None;

        // Try to resolve prefill worker upfront: if we can get bootstrap info early,
        // spawn prefill in background and proceed to decode immediately.
        //
        // Accept both `prefill_worker_id` (set by upstream routers / EPP) and
        // `backend_instance_id` (the generic "pin to this worker" field used by
        // the e2e health-check probe at `system_status_server.rs::handle_health`).
        // Without honoring `backend_instance_id` here, the partition-liveness
        // filter below silently redirects the probe to a different prefill,
        // masking real worker-down conditions: the dead worker returns 200 from
        // its `/health/prefill` endpoint because dynamo rerouted to a live peer.
        // KvPushRouter::select_worker already does the same merge (line ~246:
        // `r.prefill_worker_id.or(r.backend_instance_id)`); mirror it here so
        // the prefill_router-level filter respects the same targeting contract.
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id.or(r.backend_instance_id));

        // In Direct routing mode, the prefill_worker_id must come from the request
        // headers (x-prefill-instance-id), set by the external router (e.g., EPP).
        if self.router_mode.is_direct_routing() && preselected_worker.is_none() {
            return Err(anyhow::anyhow!(
                PrefillError::MissingWorkerIdForDirectRouting
            ));
        }

        // Diagnostic: log every request that arrives with a caller-set pin
        // (prefill_worker_id or backend_instance_id). Tells us whether the pin
        // survived the upstream pipeline (Migration → preprocessor → backend).
        // For health probes the pin always points at the probe's own worker —
        // if this log fires for a dead-worker probe but the request still 200s,
        // the leak is downstream of prefill_router.
        if let Some(id) = preselected_worker {
            let via = prefill_req.routing.as_ref().and_then(|r| {
                if r.prefill_worker_id.is_some() {
                    Some("routing.prefill_worker_id")
                } else if r.backend_instance_id.is_some() {
                    Some("routing.backend_instance_id")
                } else {
                    None
                }
            });
            tracing::info!(
                request_id = %request_id,
                preselected_id = id,
                preselected_via = ?via,
                "[prefill_router] honoring caller pin; partition filter will be skipped"
            );
        }

        // Pin the partition-liveness filter onto the request so BOTH pick paths
        // honor it — the bootstrap optimization path (resolve_prefill_worker ->
        // query_prefill_worker) AND the fallback path (KvPushRouter::generate
        // picks internally from routing.allowed_worker_ids). Without this, the
        // fallback path bypasses the filter and can route to a prefill in a
        // partition whose decode is dead. Skip when preselected_worker is set —
        // operator intent (health checks) takes precedence.
        if preselected_worker.is_none() {
            if let Some(allowed) = self.compute_allowed_prefills(None) {
                let routing = prefill_req.routing_mut();
                let merged: HashSet<WorkerId> = match routing.allowed_worker_ids.take() {
                    Some(caller) => caller.intersection(&allowed).copied().collect(),
                    None => allowed,
                };
                tracing::info!(
                    request_id = %context.id(),
                    allowed_count = merged.len(),
                    "[partition] prefill_router applied liveness filter to routing.allowed_worker_ids"
                );
                routing.allowed_worker_ids = Some(merged);
            } else {
                // compute_allowed_prefills returned None — decode_runtime_config_watch
                // wasn't available (e.g. aggregated mode, C-bindings path). The fallback
                // path will pick freely; no liveness filtering. Worth surfacing because
                // an unset watch was a real failure mode during the rollout race.
                tracing::info!(
                    request_id = %context.id(),
                    "[partition] prefill_router SKIPPED liveness filter (decode_runtime_config_watch unavailable)"
                );
            }
        } else {
            tracing::info!(
                request_id = %context.id(),
                preselected = ?preselected_worker,
                "[partition] prefill_router SKIPPED liveness filter (caller pinned a worker)"
            );
        }

        let prefill_result = async {
            if let Some((worker_id, dp_rank, bootstrap_info)) = self
                .resolve_prefill_worker(&prefill_req, preselected_worker)
                .await
            {
                // Bootstrap optimization path: spawn prefill in background
                // We successfully used the peeked worker, so we must now advance the router state
                // to ensure the next request gets a different worker.
                if !self.router_mode.is_kv_routing()
                    && !self.router_mode.is_direct_routing()
                    && let Some(router) = self.prefill_router.get()
                {
                    router.select_next_worker();
                }

                let routing = prefill_req.routing_mut();
                routing.prefill_worker_id = Some(worker_id);
                routing.dp_rank = Some(dp_rank);
                prefill_req.bootstrap_info = Some(bootstrap_info.clone());

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                // Pass phase permit to spawned task - it drops after first output (record_worker_full complete)
                // This allows set_phase(Decode) below to proceed only after prefill routing is done
                self.spawn_prefill_task(
                    prefill_context,
                    Some(worker_id),
                    prefill_phase_permit,
                    request_id.clone(),
                );

                Ok(PrefillOutcome::Bootstrap(bootstrap_info))
            } else {
                // Original prefill path: wait for prefill to complete. Fires when
                // resolve_prefill_worker returns None (e.g., picked worker has no
                // bootstrap info yet, or query_prefill_worker errored). KvPushRouter
                // will pick internally — the filter still applies via the
                // routing.allowed_worker_ids we pinned above.
                tracing::info!(
                    request_id = %request_id,
                    "[partition] using original prefill path (resolve_prefill_worker returned None)"
                );

                // Drop the phase permit - we wait for completion
                // so there's no race with set_phase(Decode) below
                drop(prefill_phase_permit);

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                // In Direct mode, pass preselected_worker so execute_prefill uses
                // router.direct() instead of router.generate() (which bails in Direct mode).
                let _pd_exec_prefill_result = Self::execute_prefill(
                    self.prefill_router.get().cloned(),
                    prefill_context,
                    preselected_worker,
                    None,
                    &request_id,
                )
                .await?;

                // [PD_TIMING] execute_prefill has fully returned. Equivalent to
                // `frontend_prefill_stream_closed` plus a few μs of result unpacking.
                // Useful as the start-of-decode-dispatch wall-clock on this side.
                tracing::info!(
                    "[PD_TIMING] event=frontend_execute_prefill_returned context_id={} ts={:.6}",
                    request_id, pd_now_secs(),
                );

                match _pd_exec_prefill_result {
                    ExecutePrefillResult::Normal {
                        result,
                        first_client_delta,
                        worker_info,
                    } => {
                        let _ = worker_info; // not used; partition stamp reads tracker instead
                        Ok(PrefillOutcome::Completed {
                            result,
                            first_client_delta,
                        })
                    }
                    ExecutePrefillResult::Terminal { outputs } => {
                        Ok(PrefillOutcome::TerminalInPrefill { outputs })
                    }
                }
            }
        }
        .await;

        // Abort if cancelled during prefill
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!("Abort entering decode after context is stopped or killed");
            return Err(anyhow::anyhow!(
                "Context id {} is stopped or killed",
                engine_ctx.id()
            ));
        }

        // Handle prefill result
        match prefill_result {
            Ok(PrefillOutcome::TerminalInPrefill { outputs }) => {
                // Prefill terminated the request itself (EOS / stop word on
                // the first generated token). No KV transfer was queued and
                // decode has nothing to do — forward the accumulated prefill
                // outputs straight back to the client.
                tracing::debug!(
                    "Prefill terminated request without decode handoff ({} outputs)",
                    outputs.len()
                );
                if let Some(ref tracker) = req.tracker {
                    let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                }
                let stream = stream::iter(outputs);
                return Ok(ResponseStream::new(Box::pin(stream), engine_ctx));
            }
            Ok(outcome) => {
                tracing::debug!("Prefill completed, proceeding to decode");

                // Set phase to Decode for the decode request.
                // In bootstrap path, this blocks until the spawned prefill task drops its permit
                // (after first output / record_worker_full completes), ensuring correct phase for routing.
                if let Some(ref tracker) = req.tracker {
                    let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    // Permit is dropped immediately - decode proceeds, no need to hold it
                }

                let mut decode_req = req;
                // [pin_trace:6/decode_req_built] decode_req is the surviving
                // copy of the original PreprocessedRequest. Its pin must
                // still be Some(<target>) here (we only cleared it on the
                // prefill_req CLONE, not on req). If it's None here but was
                // Some at [pin_trace:5], something mutated req between
                // entry and prefill completion — investigate `execute_prefill`
                // or the bootstrap path.
                tracing::info!(
                    target: "dynamo::pin_trace",
                    request_id = %request_id,
                    decode_instance_id = ?decode_req.decode_instance_id,
                    routing_backend_instance_id = ?decode_req.routing.as_ref().and_then(|r| r.backend_instance_id),
                    "[pin_trace:6/decode_req_built] decode_req = req (post-prefill, pre-decode dispatch)"
                );
                let first_client_delta = match outcome {
                    PrefillOutcome::Bootstrap(info) => {
                        decode_req.bootstrap_info = Some(info);
                        None
                    }
                    PrefillOutcome::Completed {
                        result,
                        first_client_delta,
                    } => {
                        // Inject prefill's cached_tokens into decode request's extra_args
                        // This is the correct semantic value - decode worker won't see original prompt
                        if let Some(ct) = result.cached_tokens {
                            let mut extra = decode_req.extra_args.take()
                                .and_then(|v| if v.is_object() { Some(v) } else { None })
                                .unwrap_or_else(|| serde_json::json!({}));
                            extra["prefill_cached_tokens"] = serde_json::json!(ct);
                            decode_req.extra_args = Some(extra);
                        }
                        decode_req.prefill_result = Some(result);
                        first_client_delta
                    }
                    PrefillOutcome::TerminalInPrefill { .. } => {
                        unreachable!("TerminalInPrefill handled above")
                    }
                };

                // Restore original max_tokens for decode
                decode_req.stop_conditions.max_tokens = original_max_tokens;
                // Note: decode_instance_id is already set on req from preprocessing,
                // so it propagates naturally to decode_req via the clone above.

                // Clear backend_instance_id so it doesn't propagate to the decode router.
                // backend_instance_id targets prefill workers (registered under prefill/generate),
                // but the decode router's PushRouter is connected to the decode component
                // (e.g., tensorrt_llm/generate). If this ID leaks through, the decode
                // KvPushRouter will try to find the prefill worker in the decode endpoint
                // and fail with "instance_id not found".
                if let Some(ref mut r) = decode_req.routing {
                    r.backend_instance_id = None;
                }

                // Partitioning (opt-in via DYN_PARTITIONING_ENABLED=1): stamp the picked
                // prefill's partition_group onto decode_req so the decode scheduler restricts
                // its pick to workers in the same partition. Defensive: log every condition
                // path so we can tell from logs which gate is failing if behavior surprises.
                // Partitioning (opt-in via DYN_PARTITIONING_ENABLED=1): stamp the picked
                // prefill's partition_group onto decode_req so the decode scheduler restricts
                // its pick to the same partition. The picked worker_id is recorded on the
                // RequestTracker by KvPushRouter::generate via record_worker_full — read
                // it back here.
                let env_on =
                    std::env::var("DYN_PARTITIONING_ENABLED").as_deref() == Ok("1");
                let prefill_wid_opt = decode_req
                    .tracker
                    .as_ref()
                    .and_then(|t| t.prefill_worker_id());
                if env_on
                    && let Some(watch) = self.prefill_runtime_config_watch.get()
                    && let Some(prefill_wid) = prefill_wid_opt
                {
                    let group = watch
                        .borrow()
                        .get(&prefill_wid)
                        .and_then(|c| c.partition_group.clone());
                    tracing::info!(
                        request_id = %request_id, prefill_worker_id = prefill_wid,
                        partition_group = ?group,
                        "[partition] stamping decode_req"
                    );
                    if let Some(routing) = decode_req.routing.as_mut() {
                        routing.partition_group = group;
                    } else if group.is_some() {
                        let mut routing =
                            crate::protocols::common::preprocessor::RoutingHints::default();
                        routing.partition_group = group;
                        decode_req.routing = Some(routing);
                    }
                }

                // Set router_config_override for decode:
                // - overlap_score_weight = 0 (no KV cache overlap scoring for decode)
                // - assume_kv_reuse = false (generate random hashes since decode workers
                //   may already have blocks cached from prefill transfer)
                let existing_override = decode_req.router_config_override.take();
                decode_req.router_config_override = Some(RouterConfigOverride {
                    overlap_score_weight: Some(0.0),
                    assume_kv_reuse: Some(false),
                    ..existing_override.unwrap_or_default()
                });

                // [pin_trace:7/decode_dispatch] LAST checkpoint inside
                // PrefillRouter before handing off to KvPushRouter (decode).
                // After this log, the next code that touches the pin is
                // [pin_trace:8/kvpushrouter_entry]. If pin survives here but
                // not there, the loss is in `next.generate()` plumbing or
                // ServiceBackend / KvPushRouter::generate prelude.
                tracing::info!(
                    target: "dynamo::pin_trace",
                    request_id = %request_id,
                    decode_pin = ?decode_req.decode_instance_id,
                    routing_backend_instance_id = ?decode_req.routing.as_ref().and_then(|r| r.backend_instance_id),
                    routing_decode_worker_id = ?decode_req.routing.as_ref().and_then(|r| r.decode_worker_id),
                    routing_partition_group = ?decode_req.routing.as_ref().and_then(|r| r.partition_group.clone()),
                    tracker_phase = ?decode_req.tracker.as_ref().map(|t| t.phase()),
                    has_tracker = decode_req.tracker.is_some(),
                    "[pin_trace:7/decode_dispatch] handing decode_req to next.generate (KvPushRouter)"
                );

                // Map the modified request through with preserved context
                let decode_request = context.map(|_| decode_req);

                // [PD_TIMING] About to dispatch decode via the decode router
                // (`next.generate(...)`). This call publishes to NATS and returns
                // a stream handle; it does NOT wait for the decode worker to
                // start processing. The delta between this log and
                // `frontend_decode_stream_obtained` measures dispatch overhead
                // (route lookup + NATS publish + subscription setup).
                let pd_decode_dispatch_at = pd_now_secs();
                tracing::info!(
                    "[PD_TIMING] event=frontend_decode_dispatch context_id={} ts={:.6}",
                    request_id, pd_decode_dispatch_at,
                );

                let decode_stream = next.generate(decode_request).await?;

                let pd_decode_obtained_at = pd_now_secs();
                tracing::info!(
                    "[PD_TIMING] event=frontend_decode_stream_obtained context_id={} ts={:.6} dispatch_to_obtained_ms={:.3}",
                    request_id,
                    pd_decode_obtained_at,
                    (pd_decode_obtained_at - pd_decode_dispatch_at) * 1000.0,
                );

                let response_context = decode_stream.context();
                let prefill_cached_tokens = match &first_client_delta {
                    Some(delta) => delta.data.as_ref()
                        .and_then(|d| d.extra_args.as_ref())
                        .and_then(|ea| ea.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    None => None,
                };
                let mut first_decode_chunk = true;
                // Clone request_id into the stream closure for the first-chunk log
                let pd_request_id_for_decode = request_id.clone();
                let decode_stream = decode_stream.map(move |mut output| {
                    if first_decode_chunk {
                        first_decode_chunk = false;
                        // [PD_TIMING] First chunk from the decode worker has
                        // arrived at the frontend (success or error). The delta
                        // to `frontend_decode_stream_obtained` measures the time
                        // from NATS publish until the decode worker actually
                        // produced (and the frontend received) its first output.
                        tracing::info!(
                            "[PD_TIMING] event=frontend_first_decode_chunk context_id={} ts={:.6}",
                            pd_request_id_for_decode,
                            pd_now_secs(),
                        );
                        if let Some(prefill_cached_tokens) = prefill_cached_tokens
                            && let Some(ref mut data) = output.data
                        {
                            let mut extra = data.extra_args.take().unwrap_or_else(|| serde_json::json!({}));
                            extra["prefill_cached_tokens"] = serde_json::json!(prefill_cached_tokens);
                            data.extra_args = Some(extra);
                        }
                    }
                    output
                });

                // [PD_TIMING] Frontend is returning the wrapped ResponseStream
                // to its caller (the gRPC servicer). The first chunk SMG will
                // see is the prefill_delta, prepended below; subsequent chunks
                // are decode tokens (logged above on arrival).
                tracing::info!(
                    "[PD_TIMING] event=frontend_returning_stream context_id={} ts={:.6}",
                    request_id, pd_now_secs(),
                );

                Ok(Self::prepend_prefill_delta(
                    ResponseStream::new(Box::pin(decode_stream), response_context),
                    first_client_delta,
                ))
            }
            Err(PrefillError::NotActivated) => {
                if !self.decode_fallback {
                    tracing::error!(
                        "No prefill workers discovered yet and decode fallback is disabled. Failing request."
                    );
                    return Err(anyhow::anyhow!(PrefillError::NotActivated));
                }
                tracing::debug!("No prefill workers discovered yet, falling back to decode-only");
                // Clear backend_instance_id to prevent the decode router from trying
                // to find a prefill worker ID in its endpoint (see success path comment).
                let mut fallback_req = req;
                if let Some(ref mut r) = fallback_req.routing {
                    r.backend_instance_id = None;
                }
                next.generate(context.map(|_| fallback_req)).await
            }
            Err(e) => {
                if !self.decode_fallback {
                    tracing::error!(
                        request_id = %context.id(),
                        error = %e,
                        "Remote prefill failed and decode fallback is disabled. Failing request."
                    );
                    return Err(anyhow::anyhow!(e));
                }
                tracing::warn!(
                    request_id = %context.id(),
                    error = %e,
                    "Remote prefill failed, falling back to decode-only. This may impact performance in disaggregated deployments. Verify prefill workers are healthy and accessible."
                );
                // Clear backend_instance_id to prevent the decode router from trying
                // to find a prefill worker ID in its endpoint (see success path comment).
                let mut fallback_req = req;
                if let Some(ref mut r) = fallback_req.routing {
                    r.backend_instance_id = None;
                }
                next.generate(context.map(|_| fallback_req)).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::FinishReason;
    use dynamo_async_openai::types::CompletionUsage;
    use serde_json::json;

    fn make_output(
        token_ids: Vec<u32>,
        disaggregated_params: Option<serde_json::Value>,
        finish_reason: Option<FinishReason>,
    ) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            token_ids,
            tokens: Some(vec![Some("prefill".to_string())]),
            text: Some("prefill".to_string()),
            output_type: Default::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: Some(vec![-0.25]),
            top_logprobs: None,
            finish_reason,
            stop_reason: None,
            index: Some(0),
            disaggregated_params,
            extra_args: None,
            completion_usage: Some(CompletionUsage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        })
    }

    #[test]
    fn test_extract_trtllm_prefill_first_client_delta_sanitizes_terminal_fields() {
        let first_output = make_output(
            vec![42],
            Some(json!({
                "opaque_state": "Zm9v",
                "first_gen_tokens": [42],
            })),
            Some(FinishReason::Length),
        );

        let client_delta =
            PrefillRouter::extract_trtllm_prefill_first_client_delta(&first_output)
                .expect("expected TRTLLM prefill delta");
        let data = client_delta.data.expect("expected data");

        assert_eq!(data.token_ids, vec![42]);
        assert_eq!(data.text.as_deref(), Some("prefill"));
        assert_eq!(data.log_probs, Some(vec![-0.25]));
        assert!(data.finish_reason.is_none());
        assert!(data.stop_reason.is_none());
        assert!(data.disaggregated_params.is_none());
        assert_eq!(
            data.completion_usage.expect("usage").prompt_tokens,
            5
        );
    }

    #[test]
    fn test_extract_trtllm_prefill_first_client_delta_skips_non_trtllm_output() {
        let first_output = make_output(
            vec![7],
            Some(json!({
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 5000,
                "bootstrap_room": 1,
            })),
            None,
        );

        assert!(
            PrefillRouter::extract_trtllm_prefill_first_client_delta(&first_output)
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_prepend_prefill_delta_puts_prefill_chunk_before_decode_stream() {
        let first_delta = PrefillRouter::extract_trtllm_prefill_first_client_delta(&make_output(
            vec![101],
            Some(json!({
                "opaque_state": "Zm9v",
                "first_gen_tokens": [101],
            })),
            Some(FinishReason::Length),
        ));
        let decode_output = make_output(vec![102], None, Some(FinishReason::Stop));

        let ctx = Context::with_id((), "test-request".to_string()).context();
        let decode_stream =
            ResponseStream::new(Box::pin(stream::iter(vec![decode_output])), ctx);
        let mut combined = PrefillRouter::prepend_prefill_delta(decode_stream, first_delta);

        let first = combined.next().await.expect("missing prepended chunk");
        let first_data = first.data.expect("missing prepended data");
        assert_eq!(first_data.token_ids, vec![101]);
        assert!(first_data.finish_reason.is_none());

        let second = combined.next().await.expect("missing decode chunk");
        let second_data = second.data.expect("missing decode data");
        assert_eq!(second_data.token_ids, vec![102]);
        assert_eq!(second_data.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_cached_tokens_extracted_from_extra_args() {
        let output = Annotated {
            id: None,
            data: Some(LLMEngineOutput {
                token_ids: vec![1],
                text: Some("tok".to_string()),
                tokens: None,
                cum_log_probs: None,
                log_probs: None,
                top_logprobs: None,
                finish_reason: None,
                stop_reason: None,
                disaggregated_params: Some(json!({"opaque_state": "Zm9v", "first_gen_tokens": [1]})),
                index: None,
                extra_args: Some(json!({"cached_tokens": 17})),
                completion_usage: None,
                output_type: Default::default(),
                content_parts: None,
            }),
            event: None,
            comment: None,
            error: None,
        };

        let cached = output
            .data
            .as_ref()
            .and_then(|d| d.extra_args.as_ref())
            .and_then(|ea| ea.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        assert_eq!(cached, Some(17));
    }
}
