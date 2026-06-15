// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-routing-decision structured tracing.
//!
//! Every time the KV scheduler picks a worker for a request, an event
//! is emitted on the [`ROUTER_DECISION_TARGET`] tracing target with
//! the chosen worker's identity, this request's footprint on it, and
//! the worker's live load. One line per pick, so a disaggregated
//! request produces two lines (prefill router + decode router).
//!
//! The chosen-only shape is deliberate: across many requests, every
//! actively-routed-to worker shows up regularly, and operators query
//! per-worker state via grep + aggregation. A worker that produces
//! no recent log entries is a worker that's not getting picked, which
//! is itself a useful signal.
//!
//! Field semantics on the emitted event:
//! - `request_id` — caller-supplied uuid, correlatable with
//!   `request_completed`.
//! - `worker_type` — `"prefill"` or `"decode"`.
//! - `worker_id` / `dp_rank` — chosen worker's identity.
//! - `isl_tokens` / `block_size` / `request_blocks` — request size in
//!   tokens / blocks.
//! - `request_cached_blocks` — blocks of *this* request's prefix
//!   already KV-computed on the chosen worker (overlap from radix tree).
//! - `request_uncached_blocks` — blocks this request would compute
//!   fresh on the chosen worker = `request_blocks - request_cached_blocks`.
//! - `active_cached_blocks` — total unique cached blocks held by all
//!   currently-active requests on the chosen worker.
//! - `active_uncached_blocks` — total uncached prefill compute (in
//!   blocks) pending across all active requests on the chosen worker.
//! - `active_request_count` — count of in-flight requests on the
//!   chosen worker.
//! - `kv_blocks_total` — chosen worker's total KV-cache capacity (from
//!   runtime config, may be `None` if the worker hasn't reported).
//! - `logit` — the score the selector used to make the pick (lower
//!   better in the default selector). `NaN` from custom selectors that
//!   don't compute logits.
//!
//! Note: `active_*` values are read from `slots` AFTER the selector
//! has run but BEFORE this request is booked into slots (the booking
//! happens further down in queue.rs). So they reflect the worker's
//! load *at the moment the router made the pick* and do NOT include
//! this request's contribution. To project post-admission state, add
//! `request_*` to `active_*`.

/// Tracing target for router-decision events. A dedicated fmt /
/// JSON layer can be filtered to this target if operators want a
/// separate sink.
pub const ROUTER_DECISION_TARGET: &str = "dynamo::observability::router_decision";
