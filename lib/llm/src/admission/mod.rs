// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission control: reject requests with 429 when accepting them
//! would push the system into a bad state.
//!
//! Decoupled from upstream Dynamo via:
//! - One trait, [`AdmissionPolicy`], for the decision shape.
//! - One default impl, [`LoadAwarePolicy`], that handlers can opt into.
//! - One inline check at the head of the chat-completions handler.
//! - Env-driven config so we don't ship a Python CLI surface yet
//!   (frontend args wiring deferred until v2).
//!
//! On reject: HTTP 429 + `Retry-After` header + a structured body
//! exposing the [`RejectReason`] so clients (and dashboards) can break
//! out rejections by cause. The per-request [`RequestLogScope`] also
//! captures the decision for the per-request structured log.
//!
//! v1 in this module gates on total in-flight requests (a single
//! frontend-side atomic counter). v2 will swap [`LoadAwarePolicy`] for
//! per-worker scoring on `(active_tokens, active_uncached_tokens,
//! active_requests, kv_blocks_used / kv_blocks_total,
//! max_num_batched_tokens)` plus KV overlap from the radix tree, all
//! signals already present in `KvWorkerMonitor` and `lib/kv-router`.
//! The trait surface here is designed so the v1→v2 swap is one
//! `Box::new(...)`.
//!
//! [`RequestLogScope`]: crate::observability::RequestLogScope

use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicI64, Ordering};

use crate::observability::WindowedCounters;

mod prefill_throughput;
mod together_selector;
pub use prefill_throughput::{PrefillThroughput, prefill_throughput};
pub use together_selector::TogetherSelector;

/// Env vars driving v1 config (until the Python frontend args are
/// extended). All optional; absent or zero/empty disables.
const ENV_MAX_INFLIGHT: &str = "DYN_ADMISSION_MAX_INFLIGHT";
const ENV_RETRY_AFTER_SECS: &str = "DYN_ADMISSION_RETRY_AFTER_SECS";
const ENV_WINDOW: &str = "DYN_ADMISSION_WINDOW";

const DEFAULT_RETRY_AFTER_SECS: u32 = 2;
const DEFAULT_WINDOW: u64 = 100;

/// Return the process-wide [`LoadAwarePolicy`] if env config requests
/// admission control, or `None` if disabled. Initialized lazily on
/// first call. Returning `None` here is the documented "feature off"
/// state — callers skip the admission check entirely.
pub fn policy() -> Option<Arc<LoadAwarePolicy>> {
    static INSTANCE: OnceLock<Option<Arc<LoadAwarePolicy>>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let max_inflight: i64 = std::env::var(ENV_MAX_INFLIGHT)
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n: &i64| n > 0)?;
            let retry_after = std::env::var(ENV_RETRY_AFTER_SECS)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_RETRY_AFTER_SECS);
            tracing::info!(
                target: "dynamo::admission",
                max_inflight,
                retry_after_secs = retry_after,
                "admission control enabled"
            );
            Some(Arc::new(LoadAwarePolicy::new(max_inflight, retry_after)))
        })
        .clone()
}

/// Return the process-wide `[admission]` windowed counters. Lazy-init.
/// Always present (even when policy is disabled) — useful for
/// recording admit counts only.
pub fn counters() -> WindowedCounters {
    static INSTANCE: OnceLock<WindowedCounters> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let window = std::env::var(ENV_WINDOW)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_WINDOW);
            let c = WindowedCounters::new("admission", "requests", window);
            // Pre-register all reject reasons + admit so the first
            // emission shows them at zero — easier to dashboard.
            c.register("admit");
            c.register("reject_all_workers_overloaded");
            c.register("reject_request_too_large");
            c.register("reject_no_capacity_for_isl");
            c
        })
        .clone()
}

/// Reason a request was rejected. Each variant maps to a stable string
/// surfaced in the 429 response body and emitted as a label on the
/// `[admission]` windowed counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// All workers (or the global frontend cap) are at or above the
    /// configured load threshold for this request shape.
    AllWorkersOverloaded,
    /// The request itself is too large (e.g. ISL would exceed
    /// max_num_batched_tokens of every worker). Different from the
    /// "overloaded" case because retrying will not help.
    RequestTooLarge,
    /// No worker has KV-cache headroom for this request given its
    /// estimated KV footprint. Retrying *might* help if other requests
    /// drain.
    NoCapacityForIsl,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::AllWorkersOverloaded => "all_workers_overloaded",
            RejectReason::RequestTooLarge => "request_too_large",
            RejectReason::NoCapacityForIsl => "no_capacity_for_isl",
        }
    }
}

/// Outcome of an admission check.
#[derive(Debug, Clone)]
pub enum Decision {
    Admit,
    Reject {
        reason: RejectReason,
        retry_after_secs: u32,
    },
}

/// Read-only view of one worker's load. Fields are `Option` because
/// not every signal is always available (a worker that just registered
/// may not have published `ActiveLoad` yet, etc.). v1 doesn't read this
/// — v2 does.
#[derive(Debug, Clone, Default)]
pub struct WorkerSnapshot {
    pub worker_id: u64,
    pub role: WorkerRole,
    pub active_tokens: Option<u64>,
    pub active_uncached_tokens: Option<u64>,
    pub active_requests: Option<u32>,
    pub kv_blocks_used: Option<u64>,
    pub kv_blocks_total: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerRole {
    #[default]
    Aggregated,
    Prefill,
    Decode,
}

/// Inputs to [`AdmissionPolicy::decide`]. v1 mostly ignores the
/// per-request fields; v2 will use them for per-worker scoring +
/// KV-overlap.
#[derive(Debug, Clone, Default)]
pub struct AdmissionRequest<'a> {
    pub model: &'a str,
    /// Approximate input sequence length. v1 callers can pass `None`.
    pub estimated_isl: Option<u64>,
    /// Approximate output sequence length. v1 callers can pass `None`.
    pub estimated_osl: Option<u32>,
}

pub trait AdmissionPolicy: Send + Sync {
    /// Decide whether to admit the request. v1 treats `workers` as
    /// advisory — the simple in-flight policy ignores it. v2's
    /// per-worker policy will read every field.
    fn decide(&self, req: &AdmissionRequest<'_>, workers: &[WorkerSnapshot]) -> Decision;
}

/// v1 admission policy: a single global threshold on total in-flight
/// requests. Holds an atomic counter; bump on admit via
/// [`InflightAdmissionGuard`], drop counter on response close.
///
/// Configurable thresholds:
/// - `max_inflight`: when current inflight ≥ this, return
///   `Reject { AllWorkersOverloaded, retry_after_secs }`. `0` disables
///   the check (always admit).
///
/// Trait-shaped so v2 can swap in `LoadAwarePolicyV2` (per-worker
/// scoring) without changing call sites.
pub struct LoadAwarePolicy {
    inflight: Arc<AtomicI64>,
    max_inflight: i64,
    retry_after_secs: u32,
}

impl LoadAwarePolicy {
    pub fn new(max_inflight: i64, retry_after_secs: u32) -> Self {
        Self {
            inflight: Arc::new(AtomicI64::new(0)),
            max_inflight,
            retry_after_secs,
        }
    }

    /// Admit and acquire a guard that decrements the inflight counter
    /// on drop. Returns `None` if the policy rejects the request.
    /// Pair the guard's lifetime with the request's response — drop
    /// when the response is fully sent.
    pub fn try_admit(&self) -> Result<InflightAdmissionGuard, Decision> {
        if self.max_inflight <= 0 {
            // Counter still updated for accurate metrics, but never rejected.
            self.inflight.fetch_add(1, Ordering::AcqRel);
            return Ok(InflightAdmissionGuard {
                inflight: self.inflight.clone(),
                released: false,
            });
        }

        // Optimistic increment, then check; rollback if over the cap.
        let new = self.inflight.fetch_add(1, Ordering::AcqRel) + 1;
        if new > self.max_inflight {
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            return Err(Decision::Reject {
                reason: RejectReason::AllWorkersOverloaded,
                retry_after_secs: self.retry_after_secs,
            });
        }
        Ok(InflightAdmissionGuard {
            inflight: self.inflight.clone(),
            released: false,
        })
    }

    /// Current inflight count. Useful for tests / debug.
    pub fn current_inflight(&self) -> i64 {
        self.inflight.load(Ordering::Acquire)
    }
}

impl AdmissionPolicy for LoadAwarePolicy {
    fn decide(&self, _req: &AdmissionRequest<'_>, _workers: &[WorkerSnapshot]) -> Decision {
        // v1 doesn't use snapshot fields. The actual gate is in
        // try_admit (which mutates inflight count). decide() is the
        // read-only check used when we want to peek without acquiring.
        let current = self.current_inflight();
        if self.max_inflight > 0 && current >= self.max_inflight {
            Decision::Reject {
                reason: RejectReason::AllWorkersOverloaded,
                retry_after_secs: self.retry_after_secs,
            }
        } else {
            Decision::Admit
        }
    }
}

/// RAII counter for in-flight requests. Drop decrements; if the
/// request is short-circuited (e.g. validation error after admit), the
/// guard cleans up automatically.
pub struct InflightAdmissionGuard {
    inflight: Arc<AtomicI64>,
    released: bool,
}

impl InflightAdmissionGuard {
    /// Manually release. Use when you want to ensure the decrement
    /// happens before some other side effect.
    pub fn release(mut self) {
        if !self.released {
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            self.released = true;
        }
    }
}

impl Drop for InflightAdmissionGuard {
    fn drop(&mut self) {
        if !self.released {
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            self.released = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_under_threshold() {
        let p = LoadAwarePolicy::new(2, 1);
        let g1 = p.try_admit().expect("first should admit");
        let g2 = p.try_admit().expect("second should admit");
        assert_eq!(p.current_inflight(), 2);
        drop(g1);
        assert_eq!(p.current_inflight(), 1);
        drop(g2);
        assert_eq!(p.current_inflight(), 0);
    }

    #[test]
    fn reject_at_threshold() {
        let p = LoadAwarePolicy::new(1, 2);
        let _g = p.try_admit().expect("first admits");
        let err = p.try_admit().unwrap_err();
        match err {
            Decision::Reject {
                reason,
                retry_after_secs,
            } => {
                assert_eq!(reason, RejectReason::AllWorkersOverloaded);
                assert_eq!(retry_after_secs, 2);
            }
            _ => panic!("expected Reject"),
        }
        // Counter did NOT leak.
        assert_eq!(p.current_inflight(), 1);
    }

    #[test]
    fn zero_threshold_disables_check() {
        let p = LoadAwarePolicy::new(0, 1);
        // Can admit unboundedly.
        let mut guards = Vec::new();
        for _ in 0..1000 {
            guards.push(p.try_admit().expect("zero threshold = always admit"));
        }
        assert_eq!(p.current_inflight(), 1000);
    }

    #[test]
    fn release_is_idempotent() {
        let p = LoadAwarePolicy::new(10, 1);
        let g = p.try_admit().expect("admits");
        g.release();
        // No double-decrement on drop.
        assert_eq!(p.current_inflight(), 0);
    }

    #[test]
    fn decide_without_guard_does_not_mutate_counter() {
        let p = LoadAwarePolicy::new(1, 1);
        // decide() shouldn't touch the counter.
        for _ in 0..10 {
            let _ = p.decide(&AdmissionRequest::default(), &[]);
        }
        assert_eq!(p.current_inflight(), 0);
    }
}
