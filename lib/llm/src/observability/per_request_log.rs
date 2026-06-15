// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request structured logging.
//!
//! `RequestLogScope` is the integration point: a handler creates one at
//! request entry, calls setter methods through the request lifecycle, and
//! the scope's `Drop` impl emits a single structured tracing event with
//! every collected field. Operators subscribe to the
//! `dynamo::observability::per_request` tracing target (e.g. via a
//! dedicated JSON file appender layer in the runtime's logging setup) to
//! capture one JSON line per request.
//!
//! All fields are optional except `request_id` and `model` so handlers
//! aren't forced to know everything up front. Unset fields are simply
//! omitted from the emitted event.
//!
//! # Sharing across stream pipelines
//!
//! The chat-completion handler hands a clone of the scope to the
//! streaming pipeline so per-token observations can update fields as
//! they arrive (ISL / OSL / cached_tokens / route / first_token).
//! The shared form is `Arc<std::sync::Mutex<RequestLogScope>>`; locks
//! are short and uncontended (one writer per request). The handler
//! holds its own clone for early-exit error paths; the stream
//! pipeline closure holds the other clone. Whichever drops *last*
//! triggers the structured-event emission via `Drop`.
//!
//! # Example
//! ```no_run
//! use dynamo_llm::observability::RequestLogScope;
//! # async fn handler() {
//! let mut scope = RequestLogScope::new("req-abc", "xp/dynamo-prod");
//! scope.mark_health_check(true);
//! scope.record_isl(3142);
//! scope.record_route(Some(12345), Some(67890));
//! scope.record_first_token();
//! scope.record_response(200, Some("stop"));
//! scope.record_osl(287);
//! // dropped here -> emits one tracing event with all collected fields
//! # }
//! ```

use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use tracing::Level;

/// Shared timestamp handle used by the e2e health-check bypass
/// (`/health/frontend`, `/health/prefill`, `/health/decode`). When the
/// recorded `Instant` is more recent than `DYN_E2E_LAST_HEALTHY_TIMEOUT`,
/// the probe short-circuits with 200 instead of running the synthetic
/// e2e completion. We only stamp it from `RequestLogScope::drop` when
/// the request produced real tokens, so degraded "200 with empty body"
/// completions can never mask a broken disagg path.
pub type LastSuccessfulRequestHandle = Arc<RwLock<Option<Instant>>>;

/// Display wrapper that renders `Some(v)` as `v` and `None` as `-`.
/// Avoids the `Some(...)`/`None` noise that the default Debug
/// formatter produces, while keeping the per-event emit allocation-free.
struct OrDash<T>(Option<T>);

impl<T: fmt::Display> fmt::Display for OrDash<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(v) => fmt::Display::fmt(v, f),
            None => f.write_str("-"),
        }
    }
}

/// Convenience type alias used throughout the chat-completion handler.
/// Cheap to clone (Arc); short-lock writes per token observation.
pub type SharedRequestLogScope = Arc<Mutex<RequestLogScope>>;

/// Wrap a [`RequestLogScope`] in `Arc<Mutex<...>>` for sharing across
/// the handler and the stream pipeline. Consumes the scope.
pub fn share(scope: RequestLogScope) -> SharedRequestLogScope {
    Arc::new(Mutex::new(scope))
}

/// Update the shared scope from an `LLMMetricAnnotation` arriving on
/// the response stream. Pulls the per-request fields visible to the
/// frontend at metric-observation time:
///
/// - `input_tokens` → `isl`
/// - `output_tokens` → `osl` (running total; final value lands when
///   the last chunk is observed)
/// - `cached_tokens` → `cached_tokens`
/// - `prefill_worker_id` → route info (decode_worker_id similarly)
/// - first chunk with `chunk_tokens > 0` → `first_token_at` (TTFT)
///
/// Lock duration is tiny — a few field writes — so this is safe to
/// call from a hot stream-processing closure on every chunk.
///
/// Call this *after* `process_response_and_observe_metrics` (it
/// reuses the same `LLMMetricAnnotation` already extracted).
pub fn update_from_metric_annotation(
    scope: &SharedRequestLogScope,
    metrics: &crate::preprocessor::LLMMetricAnnotation,
) {
    let Ok(mut s) = scope.lock() else {
        // Mutex poisoned — drop the update; we don't want to crash a
        // hot streaming path on observability.
        return;
    };
    s.record_isl(metrics.input_tokens as u64);
    s.record_osl(metrics.output_tokens as u64);
    if let Some(c) = metrics.cached_tokens {
        s.record_cached_tokens(c as u64);
    }
    if metrics.prefill_worker_id.is_some() || metrics.decode_worker_id.is_some() {
        s.record_route(metrics.prefill_worker_id, metrics.decode_worker_id);
    }
    if metrics.chunk_tokens > 0 {
        // record_first_token is idempotent — only the first call
        // stamps; later calls no-op.
        s.record_first_token();
    }
}

/// Tracing target for per-request structured events. A separate fmt layer
/// can be filtered to this target to write JSON lines to a dedicated sink.
pub const PER_REQUEST_TARGET: &str = "dynamo::observability::per_request";

/// What happened at the admission control gate (axis 2). `None` means the
/// request was admitted via the default path (admission control disabled
/// or didn't reject).
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    Admit,
    Reject {
        reason: &'static str,
        retry_after_secs: u32,
    },
}

/// RAII per-request log scope. Holds the fields collected during a
/// request's lifetime and emits a single structured tracing event on
/// drop.
pub struct RequestLogScope {
    request_id: String,
    model: String,
    is_health_check: bool,
    started_at: Instant,
    first_token_at: Option<Instant>,

    // Token / size accounting.
    isl: Option<u64>,
    osl: Option<u64>,
    cached_tokens: Option<u64>,

    // Routing decision (post-routing). Filled in once the request is
    // dispatched. Worker IDs are u64; we store i64 in the tracing event
    // because tracing-rs doesn't natively format u64 across all
    // subscriber layers reliably.
    prefill_worker_id: Option<u64>,
    decode_worker_id: Option<u64>,

    // Admission decision (axis 2). Default Admit when not explicitly set.
    admission: AdmissionDecision,

    // Response side.
    status: Option<u16>,
    finish_reason: Option<String>,

    // Set to true if we shouldn't emit the event on drop. Useful for
    // scopes that get superseded mid-request (rare).
    suppressed: bool,

    // If set, `Drop` stamps the shared timestamp when the request looks
    // genuinely successful (200 + osl > 0 + not a health-check probe).
    // The frontend's e2e probe (`/health/frontend`) reads this to skip
    // its synthetic completion when traffic has been flowing — but we
    // must not flip it on requests that returned an empty body, or the
    // probe will mask broken disagg state (see 2026-05-14 outage).
    last_successful_request: Option<LastSuccessfulRequestHandle>,
}

impl RequestLogScope {
    /// Create a new scope. `request_id` is the operator-facing handle for
    /// reconstructing one request's path; `model` is the served model
    /// name as the client sent it (`xp/dynamo-prod`, etc.).
    pub fn new(request_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            model: model.into(),
            is_health_check: false,
            started_at: Instant::now(),
            first_token_at: None,
            isl: None,
            osl: None,
            cached_tokens: None,
            prefill_worker_id: None,
            decode_worker_id: None,
            admission: AdmissionDecision::Admit,
            status: None,
            finish_reason: None,
            suppressed: false,
            last_successful_request: None,
        }
    }

    /// Attach the shared `last_successful_request` handle from the HTTP
    /// service state. The drop will stamp it iff the request finished
    /// with status 200 and emitted at least one token (osl > 0).
    pub fn with_last_successful_request(
        &mut self,
        handle: LastSuccessfulRequestHandle,
    ) -> &mut Self {
        self.last_successful_request = Some(handle);
        self
    }

    /// Tag this scope as a synthetic health-check probe (so an operator
    /// can filter `is_health_check=true` lines out of latency
    /// dashboards, or specifically include them when debugging probe
    /// behavior).
    pub fn mark_health_check(&mut self, on: bool) -> &mut Self {
        self.is_health_check = on;
        self
    }

    /// Record the input sequence length (post-tokenization). One-shot.
    pub fn record_isl(&mut self, isl: u64) -> &mut Self {
        self.isl = Some(isl);
        self
    }

    /// Record the final output sequence length when the response stream
    /// closes.
    pub fn record_osl(&mut self, osl: u64) -> &mut Self {
        self.osl = Some(osl);
        self
    }

    /// Record cached-prefix tokens (KV-cache hit at the prefill side).
    /// Useful for distinguishing cache-hit from cold prefill in the
    /// per-request log.
    pub fn record_cached_tokens(&mut self, cached: u64) -> &mut Self {
        self.cached_tokens = Some(cached);
        self
    }

    /// Record the routing decision. Pass `None` for the role that
    /// doesn't apply (aggregated mode → both populated; PD mode →
    /// distinct prefill + decode workers).
    pub fn record_route(
        &mut self,
        prefill_worker_id: Option<u64>,
        decode_worker_id: Option<u64>,
    ) -> &mut Self {
        if prefill_worker_id.is_some() {
            self.prefill_worker_id = prefill_worker_id;
        }
        if decode_worker_id.is_some() {
            self.decode_worker_id = decode_worker_id;
        }
        self
    }

    /// Stamp first-token timestamp so `ttft_ms` lands in the emitted
    /// event. Idempotent on the first call; subsequent calls are
    /// no-ops.
    pub fn record_first_token(&mut self) -> &mut Self {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        self
    }

    /// Record the response status + finish reason (e.g. `"stop"`,
    /// `"length"`, `"tool_calls"`).
    pub fn record_response(
        &mut self,
        status: u16,
        finish_reason: Option<impl Into<String>>,
    ) -> &mut Self {
        self.status = Some(status);
        if let Some(reason) = finish_reason {
            self.finish_reason = Some(reason.into());
        }
        self
    }

    /// Record an admission control decision. Default is `Admit`; set on
    /// reject to surface the reason in the per-request log.
    pub fn record_admission(&mut self, decision: AdmissionDecision) -> &mut Self {
        self.admission = decision;
        self
    }

    /// Suppress emission on drop. Useful when the scope is superseded
    /// (e.g. a wrapping handler that takes over emission).
    pub fn suppress(&mut self) -> &mut Self {
        self.suppressed = true;
        self
    }
}

impl Drop for RequestLogScope {
    fn drop(&mut self) {
        if self.suppressed {
            return;
        }

        // Stamp the e2e-health bypass timestamp iff the request actually
        // succeeded end-to-end. "Succeeded" means: not a synthetic
        // health-check probe, status 200 reached the client, and at
        // least one output token was streamed (osl > 0). All three are
        // required — without them, a degraded request that returned an
        // empty body to the client would still poison the bypass timer
        // and keep `/health/frontend` returning 200 for up to
        // DYN_E2E_LAST_HEALTHY_TIMEOUT seconds. We always run this
        // regardless of tracing-level filtering, since the bypass
        // semantics matter even when per-request logs are silenced.
        if !self.is_health_check
            && self.status == Some(200)
            && self.osl.unwrap_or(0) > 0
        {
            if let Some(handle) = &self.last_successful_request {
                if let Ok(mut slot) = handle.write() {
                    *slot = Some(Instant::now());
                }
            }
        }

        // If nobody is listening to PER_REQUEST_TARGET at INFO, this
        // becomes a no-op past the level filter.
        if !tracing::event_enabled!(target: PER_REQUEST_TARGET, Level::INFO) {
            return;
        }

        let now = Instant::now();
        let e2e_ms = now.duration_since(self.started_at).as_secs_f64() * 1000.0;
        let ttft_ms = self
            .first_token_at
            .map(|t| t.duration_since(self.started_at).as_secs_f64() * 1000.0);

        // Single human-friendly message line, |-grouped by category:
        //   [req] <id> <model> hc=<bool>
        //     | isl=<n> osl=<n> cached=<n>
        //     | prefill=<id> decode=<id>
        //     | ttft=<ms>ms e2e=<ms>ms
        //     | <status> <finish>
        //     | <admission_decision>[:<reason> retry=<n>s]
        // Unset Optional fields render as `-`, not `None`/`Some(N)`.
        // Operators grep on the `[req]` tag + use awk/sed on the
        // pipe-separated groups for aggregation.
        let admission_suffix = match &self.admission {
            AdmissionDecision::Admit => String::from("admit"),
            AdmissionDecision::Reject {
                reason,
                retry_after_secs,
            } => format!("reject:{} retry={}s", reason, retry_after_secs),
        };
        tracing::info!(
            target: PER_REQUEST_TARGET,
            "[req] {req} {model} hc={hc} | isl={isl} osl={osl} cached={cached} | prefill={prefill} decode={decode} | ttft={ttft}ms e2e={e2e:.1}ms | {status} {finish} | {admission}",
            req = self.request_id,
            model = self.model,
            hc = self.is_health_check,
            isl = OrDash(self.isl),
            osl = OrDash(self.osl),
            cached = OrDash(self.cached_tokens),
            prefill = OrDash(self.prefill_worker_id),
            decode = OrDash(self.decode_worker_id),
            ttft = OrDash(ttft_ms.map(|v| format!("{:.1}", v))),
            e2e = e2e_ms,
            status = OrDash(self.status),
            finish = OrDash(self.finish_reason.as_deref()),
            admission = admission_suffix,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_drop_is_no_op_when_target_unsubscribed() {
        // Without a subscriber, drop should not panic and should not
        // observably do anything.
        let mut scope = RequestLogScope::new("req-1", "test/model");
        scope.record_isl(42).record_route(Some(1), Some(2));
        drop(scope);
    }

    #[test]
    fn scope_records_admission_reject() {
        let mut scope = RequestLogScope::new("req-2", "test/model");
        scope.record_admission(AdmissionDecision::Reject {
            reason: "all_workers_overloaded",
            retry_after_secs: 2,
        });
        match scope.admission {
            AdmissionDecision::Reject {
                reason,
                retry_after_secs,
            } => {
                assert_eq!(reason, "all_workers_overloaded");
                assert_eq!(retry_after_secs, 2);
            }
            _ => panic!("expected Reject"),
        }
    }

    #[test]
    fn record_first_token_is_idempotent() {
        let mut scope = RequestLogScope::new("req-3", "test/model");
        scope.record_first_token();
        let first = scope.first_token_at;
        std::thread::sleep(std::time::Duration::from_millis(2));
        scope.record_first_token();
        assert_eq!(scope.first_token_at, first);
    }

    #[test]
    fn suppress_skips_emission() {
        let mut scope = RequestLogScope::new("req-4", "test/model");
        scope.suppress();
        // Drop is a no-op; we just verify the flag.
        assert!(scope.suppressed);
    }
}
