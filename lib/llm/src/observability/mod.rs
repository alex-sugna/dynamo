// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Observability primitives for production-readiness work.
//!
//! - [`RequestLogScope`]: RAII guard that captures per-request fields over a
//!   request's lifecycle and emits a single structured event on drop. Pair
//!   one with each top-level HTTP request handler so we get one log entry
//!   per request, keyed by `request_id`, with route + timing + outcome.
//! - [`WindowedCounters`]: subsystem helper that aggregates named counters
//!   over a fixed-size window (in arbitrary units — requests, iterations,
//!   seconds — declared by the subsystem) and emits one structured trace
//!   line per window, then resets. Format mirrors the kvblock-verifier /
//!   tcache trace style operators are already used to.
//!
//! Both are opt-in. Defaults are silent: a `RequestLogScope` does nothing
//! unless someone subscribes to its tracing target; a `WindowedCounters`
//! instance is a no-op until something calls `record`.

pub mod per_request_log;
pub mod router_decision;
pub mod windowed;

pub use per_request_log::{
    AdmissionDecision, LastSuccessfulRequestHandle, RequestLogScope, SharedRequestLogScope,
    share as share_request_log, update_from_metric_annotation,
};
pub use router_decision::ROUTER_DECISION_TARGET;
pub use windowed::WindowedCounters;
