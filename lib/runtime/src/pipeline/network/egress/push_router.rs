// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{AsyncEngineContextProvider, ResponseStream};
use crate::error::{BackendError, DynamoError, ErrorType, match_error_chain};

/// Check if an error chain indicates the worker should be reported as down.
fn is_inhibited(err: &(dyn std::error::Error + 'static)) -> bool {
    const INHIBITED: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    match_error_chain(err, INHIBITED, &[])
}

/// Push `instance_id` into the recent-inhibitions window, prune entries older
/// than 5s, return the number of DISTINCT instance_ids currently in the window.
/// Used to detect "many distinct workers tripped the stall detector in a short
/// span" (== likely false-positive cascade, not actual correlated failure).
fn push_recent_inhibition(
    recent: &Arc<Mutex<VecDeque<(Instant, u64)>>>,
    instance_id: u64,
) -> usize {
    let mut q = recent.lock().unwrap();
    let now = Instant::now();
    q.push_back((now, instance_id));
    while let Some((t, _)) = q.front() {
        if now.duration_since(*t) > std::time::Duration::from_secs(5) {
            q.pop_front();
        } else {
            break;
        }
    }
    q.iter().map(|(_, id)| *id).collect::<HashSet<_>>().len()
}

/// Read the per-stream stall timeout from env (DYN_STREAM_STALL_TIMEOUT_MS).
/// Returns None when unset or when set to 0 (= disabled).
fn stall_timeout_from_env() -> Option<std::time::Duration> {
    let raw = std::env::var("DYN_STREAM_STALL_TIMEOUT_MS").ok()?;
    let ms: u64 = raw.trim().parse().ok()?;
    if ms == 0 {
        None
    } else {
        Some(std::time::Duration::from_millis(ms))
    }
}
use crate::{
    component::{Client, Endpoint},
    engine::{AsyncEngine, Data},
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn,
        error::{PipelineError, PipelineErrorExt},
    },
    protocols::maybe_error::MaybeError,
    traits::DistributedRuntimeProvider,
};
use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashSet, VecDeque},
    future::Future,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tokio_stream::StreamExt;
use tracing::Instrument;

/// Trait for monitoring worker load and determining busy state.
/// Implementations can define custom load metrics and busy thresholds.
#[async_trait]
pub trait WorkerLoadMonitor: Send + Sync {
    /// Start background monitoring of worker load.
    /// This should spawn background tasks that update the client's free instances.
    async fn start_monitoring(&self) -> anyhow::Result<()>;
}

#[derive(Clone)]
pub struct PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    // TODO: This shouldn't be pub, but lib/bindings/python/rust/lib.rs exposes it.
    /// The Client is how we gather remote endpoint information from etcd.
    pub client: Client,

    /// How we choose which instance to send traffic to.
    ///
    /// Setting this to KV means we never intend to call `generate` on this PushRouter. We are
    /// not using it as an AsyncEngine.
    /// Instead we will decide whether to call random/round_robin/direct ourselves and call them directly.
    /// dynamo-llm's KV Routing does this.
    router_mode: RouterMode,

    /// Number of round robin requests handled. Used to decide which server is next.
    round_robin_counter: Arc<AtomicU64>,

    /// The next step in the chain. PushRouter (this object) picks an instances,
    /// addresses it, then passes it to AddressedPushRouter which does the network traffic.
    addressed: Arc<AddressedPushRouter>,

    /// Threshold for determining when a worker is busy (0.0 to 1.0)
    /// If None, busy detection is disabled
    busy_threshold: Option<f64>,

    /// When false, `generate_with_fault_detection` skips fault detection logic:
    /// it won't call `report_instance_down` on errors, and it uses the raw discovery
    /// instance list instead of the filtered avail list. Use for recovery/query paths
    /// where transient failures are expected.
    fault_detection_enabled: bool,

    /// Per-chunk stall timeout. When `Some`, each `response_stream.next().await` is
    /// bounded by this duration; on expiry the router synthesizes a `Disconnected`
    /// error AND calls `report_instance_down` (same path as an explicit migratable
    /// error from the worker). Tuned via `DYN_STREAM_STALL_TIMEOUT_MS`. Default
    /// `None` (disabled) — set on routers that benefit from migration on hung
    /// workers (typically the prefill/decode push paths).
    stall_timeout: Option<std::time::Duration>,

    /// Label applied by `with_stall_timeout_for(role)`. Just for observability —
    /// gets stamped on stall + inhibit logs so we can grep "role=decode" vs
    /// "role=prefill" without inspecting endpoint paths.
    role: Option<String>,

    /// Sliding 5s window of `(timestamp, instance_id)` for inhibitions emitted
    /// from this router. When ≥3 DISTINCT instance_ids land in the window, we
    /// emit a cascade warning. Catches the false-positive failure mode where
    /// a too-tight stall timeout marks workers down faster than they recover.
    recent_inhibitions: Arc<Mutex<VecDeque<(Instant, u64)>>>,

    /// An internal Rust type. This says that PushRouter is generic over the T and U types,
    /// which are the input and output types of it's `generate` function. It allows the
    /// compiler to specialize us at compile time.
    _phantom: PhantomData<(T, U)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum RouterMode {
    #[default]
    RoundRobin,
    Random,
    KV,
    Direct,
}

impl RouterMode {
    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }

    pub fn is_direct_routing(&self) -> bool {
        *self == RouterMode::Direct
    }
}

async fn addressed_router(endpoint: &Endpoint) -> anyhow::Result<Arc<AddressedPushRouter>> {
    // Get network manager and create client (no mode checks!)
    let manager = endpoint.drt().network_manager();
    let req_client = manager.create_client()?;
    let resp_transport = endpoint.drt().tcp_server().await?;

    tracing::debug!(
        transport = req_client.transport_name(),
        "Creating AddressedPushRouter with request plane client"
    );

    AddressedPushRouter::new(req_client, resp_transport)
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    /// Create a new PushRouter without busy threshold (no busy detection)
    pub async fn from_client(client: Client, router_mode: RouterMode) -> anyhow::Result<Self> {
        Self::from_client_with_threshold(client, router_mode, None, None).await
    }

    /// Create a new PushRouter with fault detection disabled.
    ///
    /// Unlike `from_client`, this router will not call `report_instance_down` on
    /// transient errors, and `direct()` uses the raw discovery instance list instead
    /// of the filtered avail list. Use for recovery/query paths.
    pub async fn from_client_no_fault_detection(
        client: Client,
        router_mode: RouterMode,
    ) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;

        Ok(PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            busy_threshold: None,
            fault_detection_enabled: false,
            // Recovery/query paths: never stall-time-out, since transient
            // slowness on those paths is expected.
            stall_timeout: None,
            role: None,
            recent_inhibitions: Arc::new(Mutex::new(VecDeque::new())),
            _phantom: PhantomData,
        })
    }

    /// Create a new PushRouter with optional busy threshold and worker load monitor
    pub async fn from_client_with_threshold(
        client: Client,
        router_mode: RouterMode,
        busy_threshold: Option<f64>,
        worker_monitor: Option<Arc<dyn WorkerLoadMonitor>>,
    ) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;

        // Start worker monitor if provided and in dynamic mode
        if let Some(monitor) = worker_monitor.as_ref() {
            monitor.start_monitoring().await?;
        }

        let stall_timeout = stall_timeout_from_env();
        if let Some(d) = stall_timeout {
            tracing::info!(
                stall_timeout_ms = d.as_millis() as u64,
                "PushRouter stall detector enabled"
            );
        }
        let router = PushRouter {
            client: client.clone(),
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            busy_threshold,
            fault_detection_enabled: true,
            stall_timeout,
            role: None,
            recent_inhibitions: Arc::new(Mutex::new(VecDeque::new())),
            _phantom: PhantomData,
        };

        Ok(router)
    }

    /// Apply a role-specific override for the per-chunk stall timeout. Reads
    /// `DYN_STREAM_STALL_TIMEOUT_MS_<ROLE>` (case-insensitive); when set,
    /// replaces whatever value the constructor read from the base
    /// `DYN_STREAM_STALL_TIMEOUT_MS` env (including disabling the detector
    /// when the override is `0`). When unset, the base value is preserved.
    ///
    /// Builder-style so call sites that need a different timeout from the
    /// shared default can opt in without changing the constructor signature
    /// for every other PushRouter user. E.g. the prefill push router can
    /// pass `"prefill"`, the decode push router `"decode"`, while
    /// single-stage models pick up the base env-var as before.
    pub fn with_stall_timeout_for(mut self, role: &str) -> Self {
        // Record the role unconditionally — even when no override env is set
        // it's useful to have `role=decode` stamped on the stall/inhibit logs.
        self.role = Some(role.to_string());
        let key = format!("DYN_STREAM_STALL_TIMEOUT_MS_{}", role.to_ascii_uppercase());
        if let Ok(raw) = std::env::var(&key) {
            if let Ok(ms) = raw.trim().parse::<u64>() {
                self.stall_timeout = if ms == 0 {
                    None
                } else {
                    Some(std::time::Duration::from_millis(ms))
                };
                tracing::info!(
                    role = %role,
                    stall_timeout_ms = ms,
                    "PushRouter stall detector role override applied"
                );
            } else {
                tracing::warn!(
                    role = %role,
                    raw = %raw,
                    "ignoring un-parseable role-specific stall timeout"
                );
            }
        }
        self
    }

    /// Issue a request to the next available instance in a round-robin fashion
    pub async fn round_robin(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;

        let instance_id = {
            let instance_ids = self.client.instance_ids_avail();
            let count = instance_ids.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            instance_ids[counter % count]
        };
        tracing::trace!("round robin router selected {instance_id}");

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    /// Issue a request to a random endpoint
    pub async fn random(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let instance_id = {
            let instance_ids = self.client.instance_ids_avail();
            let count = instance_ids.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no instances found for endpoint {}",
                    self.client.endpoint.id()
                ));
            }
            let counter = rand::rng().random::<u64>() as usize;
            instance_ids[counter % count]
        };
        tracing::trace!("random router selected {instance_id}");

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    /// Issue a request to a specific endpoint
    pub async fn direct(
        &self,
        request: SingleIn<T>,
        instance_id: u64,
    ) -> anyhow::Result<ManyOut<U>> {
        // When fault detection is disabled, check the raw discovery list
        // (not filtered by report_instance_down) so transient failures
        // don't poison the instance for subsequent retries.
        let found = if self.fault_detection_enabled {
            self.client.instance_ids_avail().contains(&instance_id)
        } else {
            self.client.instance_ids().contains(&instance_id)
        };

        if !found {
            // [pin_trace:10a/direct_reject] Targeted worker NOT in availability
            // set. Either the pin's id has expired its etcd lease, or fault
            // detection removed it. Caller will get an Err which Migration may
            // try to retry (if migration_limit > 0 AND retries weren't already
            // forced to 0 by the pin-detect code in migration.rs).
            tracing::info!(
                target: "dynamo::pin_trace",
                request_id = %request.id(),
                instance_id,
                endpoint = %self.client.endpoint.id(),
                fault_detection_enabled = self.fault_detection_enabled,
                avail_count = self.client.instance_ids_avail().len(),
                total_count = self.client.instance_ids().len(),
                "[pin_trace:10a/direct_reject] target instance not in {} set — returning Err",
                if self.fault_detection_enabled { "avail" } else { "discovery" }
            );
            return Err(anyhow::anyhow!(
                "instance_id={instance_id} not found for endpoint {}",
                self.client.endpoint.id()
            ));
        }

        // [pin_trace:10b/direct_accept] Target IS in avail. About to compute
        // the NATS subject (or other transport address) and dispatch. Pair
        // this with the decode worker's `request_id=chatcmpl-*` log to verify
        // the worker that actually processed the request matches the pin.
        tracing::info!(
            target: "dynamo::pin_trace",
            request_id = %request.id(),
            instance_id,
            endpoint = %self.client.endpoint.id(),
            avail_count = self.client.instance_ids_avail().len(),
            "[pin_trace:10b/direct_accept] dispatching to pinned instance"
        );

        self.generate_with_fault_detection(instance_id, request)
            .await
    }

    /// Select the next worker according to the routing mode.
    /// Increments round-robin counter if applicable.
    /// Panics if called on Direct or KV mode - those have their own selection mechanisms.
    pub fn select_next_worker(&self) -> Option<u64> {
        let instance_ids = self.client.instance_ids_avail();
        let count = instance_ids.len();
        if count == 0 {
            return None;
        }

        match self.router_mode {
            RouterMode::RoundRobin => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
            }
            _ => {
                panic!(
                    "select_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
    }

    /// Peek the next worker according to the routing mode without incrementing the counter.
    /// Useful for checking if a worker is suitable before committing to it.
    pub fn peek_next_worker(&self) -> Option<u64> {
        let instance_ids = self.client.instance_ids_avail();
        let count = instance_ids.len();
        if count == 0 {
            return None;
        }

        match self.router_mode {
            RouterMode::RoundRobin => {
                // Just peek at the current counter value without incrementing
                let counter = self.round_robin_counter.load(Ordering::Relaxed) as usize;
                Some(instance_ids[counter % count])
            }
            RouterMode::Random => {
                // For random, peeking implies a fresh random selection since it's stateless.
                // Note: The caller must realize that select_next_worker() will pick a DIFFERENT random worker.
                let counter = rand::rng().random::<u64>() as usize;
                Some(instance_ids[counter % count])
            }
            _ => {
                panic!(
                    "peek_next_worker should not be called for {:?} routing mode",
                    self.router_mode
                )
            }
        }
    }

    /*
    pub async fn r#static(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let subject = self.client.endpoint.subject();
        tracing::debug!("static got subject: {subject}");
        let request = request.map(|req| AddressedRequest::new(req, subject));
        tracing::debug!("router generate");
        self.addressed.generate(request).await
    }
    */

    async fn generate_with_fault_detection(
        &self,
        instance_id: u64,
        request: SingleIn<T>,
    ) -> anyhow::Result<ManyOut<U>> {
        let request_id = request.id().to_string();
        let route_span = if matches!(self.router_mode, RouterMode::KV) {
            tracing::Span::none()
        } else {
            tracing::info_span!(
                "router.route_request",
                request_id = %request_id,
                worker_id = instance_id,
                router_mode = ?self.router_mode,
            )
        };

        // Check if all workers are busy (only if busy threshold is set and fault detection enabled)
        if self.fault_detection_enabled && self.busy_threshold.is_some() {
            let free_instances = self.client.instance_ids_free();
            if free_instances.is_empty() {
                // Check if we actually have any instances at all
                let all_instances = self.client.instance_ids();
                if !all_instances.is_empty() {
                    tracing::warn!(
                        instance_id,
                        total_workers = all_instances.len(),
                        "Rejecting request: all workers are busy"
                    );
                    return Err(PipelineError::ServiceOverloaded(
                        "All workers are busy, please retry later".to_string(),
                    )
                    .into());
                }
            }
        }

        // Get the address based on discovered transport type
        let address = {
            use crate::component::TransportType;

            // Get the instance and use its actual transport type
            let instances = self.client.instances();
            let instance = instances
                .iter()
                .find(|i| i.instance_id == instance_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("Instance {} not found in available instances", instance_id)
                })?;

            match &instance.transport {
                TransportType::Http(http_endpoint) => {
                    // [pin_trace:11/transport] Resolved HTTP transport for
                    // the pinned instance. The decode worker that processes
                    // the request must match `instance_id`.
                    tracing::info!(
                        target: "dynamo::pin_trace",
                        request_id = %request_id,
                        instance_id,
                        http_endpoint = %http_endpoint,
                        "[pin_trace:11/transport] dispatching via HTTP"
                    );
                    http_endpoint.clone()
                }
                TransportType::Tcp(tcp_endpoint) => {
                    tracing::info!(
                        target: "dynamo::pin_trace",
                        request_id = %request_id,
                        instance_id,
                        tcp_endpoint = %tcp_endpoint,
                        "[pin_trace:11/transport] dispatching via TCP"
                    );
                    tcp_endpoint.clone()
                }
                TransportType::Nats(subject) => {
                    // [pin_trace:11/transport] Resolved NATS subject for the
                    // pinned instance. Subjects are unique per
                    // (namespace,component,endpoint,instance_id) per
                    // `lib/runtime/src/transports/nats.rs:870` — only the
                    // pinned worker subscribes here, so if the WRONG worker
                    // logs receipt of this request_id, we have a NATS-level
                    // subject collision or routing anomaly.
                    tracing::info!(
                        target: "dynamo::pin_trace",
                        request_id = %request_id,
                        instance_id,
                        subject = %subject,
                        "[pin_trace:11/transport] dispatching via NATS"
                    );
                    subject.clone()
                }
            }
        };

        let request = request.map(|req| AddressedRequest::new(req, address));

        let stream: anyhow::Result<ManyOut<U>> = self
            .addressed
            .generate(request)
            .instrument(route_span)
            .await;
        match stream {
            Ok(stream) => {
                if !self.fault_detection_enabled {
                    return Ok(stream);
                }
                let engine_ctx = stream.context();
                let client = self.client.clone();
                // Helper: if a chunk carries any error, treat it as a
                // worker-down signal. We're inside the per-instance
                // generate path — any error here means *this* worker
                // failed to keep producing chunks. Mapping the error
                // chain to Disconnected (when it doesn't already match
                // the inhibited set explicitly) lets the migration
                // framework retry on a fresh worker instead of having
                // h2/tonic/transport-level errors fall through unhandled.
                // SIGKILL on a worker produces an "h2 protocol error"
                // chain that does NOT match the original migratable
                // allowlist; without rewriting, migration never fires.
                fn normalize_stream_err<U: MaybeError>(
                    res: U,
                    instance_id: u64,
                ) -> U {
                    if let Some(err) = res.err() {
                        if !is_inhibited(&err) {
                            // Wrap as Disconnected so migration's
                            // is_migratable check matches. Preserve the
                            // original message for forensics.
                            return U::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message(format!(
                                        "worker {} stream errored: {}",
                                        instance_id, err
                                    ))
                                    .build(),
                            );
                        }
                    }
                    res
                }

                if let Some(stall) = self.stall_timeout {
                    // Per-chunk stall detection: each .next().await is bounded
                    // by `stall`. On Elapsed: synthesize Disconnected AND mark
                    // the instance down — same recovery path as an explicit
                    // migratable error from the worker.
                    let role = self.role.clone();
                    let recent = self.recent_inhibitions.clone();
                    let stream = stream.timeout(stall).map(move |result| match result {
                        Ok(res) => {
                            if let Some(err) = res.err() {
                                tracing::debug!(
                                    "Reporting instance {instance_id} down due to stream error: {err}"
                                );
                                client.report_instance_down(instance_id);
                            }
                            normalize_stream_err(res, instance_id)
                        }
                        Err(_elapsed) => {
                            let distinct_in_window =
                                push_recent_inhibition(&recent, instance_id);
                            tracing::warn!(
                                role = ?role,
                                instance_id,
                                stall_ms = stall.as_millis() as u64,
                                distinct_inhibitions_5s = distinct_in_window,
                                "stream stalled; reporting instance down and synthesizing Disconnected"
                            );
                            if distinct_in_window >= 3 {
                                tracing::warn!(
                                    role = ?role,
                                    distinct_inhibitions_5s = distinct_in_window,
                                    stall_ms = stall.as_millis() as u64,
                                    "cascade detected: ≥3 distinct instances inhibited in 5s — \
                                     stall_timeout likely too tight for current load, or correlated worker failure"
                                );
                            }
                            client.report_instance_down(instance_id);
                            U::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message(format!(
                                        "stream stalled > {} ms on instance {} ",
                                        stall.as_millis(),
                                        instance_id
                                    ))
                                    .build(),
                            )
                        }
                    });
                    Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
                } else {
                    let stream = stream.map(move |res| {
                        if let Some(err) = res.err() {
                            tracing::debug!(
                                "Reporting instance {instance_id} down due to stream error: {err}"
                            );
                            client.report_instance_down(instance_id);
                        }
                        normalize_stream_err(res, instance_id)
                    });
                    Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
                }
            }
            Err(err) => {
                if self.fault_detection_enabled && is_inhibited(err.as_ref()) {
                    tracing::debug!("Reporting instance {instance_id} down due to error: {err}");
                    self.client.report_instance_down(instance_id);
                }
                Err(err)
            }
        }
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        match self.router_mode {
            RouterMode::Random => self.random(request).await,
            RouterMode::RoundRobin => self.round_robin(request).await,
            RouterMode::KV => {
                anyhow::bail!("KV routing should not call generate on PushRouter");
            }
            RouterMode::Direct => {
                anyhow::bail!(
                    "Direct routing should not call generate on PushRouter directly; use DirectRoutingRouter wrapper"
                );
            }
        }
    }
}
