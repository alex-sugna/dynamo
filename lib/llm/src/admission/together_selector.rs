// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! v2 admission filter — per-worker thresholds, plugged into the
//! existing [`WorkerSelector`] seam.
//!
//! The v1 admission gate ([`crate::admission::LoadAwarePolicy`]) is a
//! single global in-flight counter. v2 adds per-worker filtering so a
//! single overloaded worker can't pull the whole cluster off a cliff —
//! and, because per-worker active load is already shared across frontend
//! replicas via `--router-replica-sync` (see
//! [`crate::kv_router::sequence::create_multi_worker_sequences`]), the
//! filter makes consistent decisions across all frontends without any
//! additional cross-frontend admission state.
//!
//! `TogetherSelector` wraps an inner `WorkerSelector` (typically
//! [`crate::kv_router::scheduler::DefaultWorkerSelector`]). Filter rule:
//! a worker is **eligible** if at least one of its `dp_rank`s has both
//!
//! - `projected_uncached_blocks(w) ≤ MAX_ACTIVE_UNCACHED_BLOCKS_PER_WORKER`
//! - `pre_active_request_count(w)   <  MAX_INFLIGHT_PER_WORKER`
//!
//! where `projected_uncached_blocks(w) = ceil(prefill_tokens[w] / block_size)`
//! (post-this-request projected prefill, computed by the queue) and
//! `pre_active_request_count(w)` is the live count of in-flight requests
//! on that worker (also populated by the queue from
//! `ActiveSequencesMulti::active_request_counts`).
//!
//! Workers are filtered at WorkerId granularity (not dp_rank): the inner
//! `DefaultWorkerSelector` picks the best `dp_rank` among an eligible
//! worker by logit. If a worker has 4 ranks and 1 is hot, we leave the
//! worker in — the logit pick steers around the hot rank naturally.
//!
//! If the filter empties the candidate set, returns
//! [`KvSchedulerError::NoEndpoints`] — the existing reject path that
//! surfaces as a 5xx in OSS or, with our admission counters, a tracked
//! `reject_all_workers_overloaded`.
//!
//! Both thresholds are env-gated; either or both can be omitted, in
//! which case that axis is not enforced. If neither is set,
//! `from_env` returns the inner selector unwrapped (zero cost).

use std::collections::{HashMap, HashSet};

use crate::admission::prefill_throughput;
use crate::kv_router::WorkerSelector;
use crate::kv_router::protocols::{WorkerId, WorkerSelectionResult, WorkerWithDpRank};
use crate::kv_router::scheduler::{KvSchedulerError, SchedulingRequest};
use crate::local_model::runtime_config::ModelRuntimeConfig;

const ENV_MAX_ACTIVE_UNCACHED_BLOCKS_PER_WORKER: &str =
    "DYN_ADMISSION_MAX_ACTIVE_UNCACHED_BLOCKS_PER_WORKER";
/// Legacy / fallback cap (applied to both prefill and decode when role-
/// specific knobs are unset). Kept for backwards-compat with chart values
/// that haven't been split per-role yet.
const ENV_MAX_INFLIGHT_PER_WORKER: &str = "DYN_ADMISSION_MAX_INFLIGHT_PER_WORKER";
/// Role-specific caps. When set, override `ENV_MAX_INFLIGHT_PER_WORKER`
/// for the matching scheduler role. Prefill and decode have different
/// capacity envelopes (prefill is token-burst-bound, decode is concurrent-
/// stream-bound), so a single number can't express both — under whichever
/// role is tighter, the other is under-utilized.
const ENV_MAX_INFLIGHT_PER_PREFILL_WORKER: &str = "DYN_ADMISSION_MAX_INFLIGHT_PER_PREFILL_WORKER";
const ENV_MAX_INFLIGHT_PER_DECODE_WORKER: &str = "DYN_ADMISSION_MAX_INFLIGHT_PER_DECODE_WORKER";
const ENV_TTFT_SLO_MS: &str = "DYN_ADMISSION_TTFT_SLO_MS";

/// Worker-level admission filter. See module docs.
pub struct TogetherSelector {
    inner: Box<dyn WorkerSelector + Send + Sync>,
    max_active_uncached_blocks_per_worker: Option<usize>,
    max_inflight_per_worker: Option<usize>,
    /// Predicted-TTFT SLO in milliseconds. When `Some`, the filter
    /// rejects a worker if `prefill_tokens[w] × ms_per_uncached_token[w]`
    /// (estimated from the throughput tracker's rolling window) exceeds
    /// this budget. Cache hits naturally pass under load because their
    /// own `uncached_tokens ≈ 0` contribution shrinks the product. When
    /// the throughput estimate is `None` (cold-start, sparse window),
    /// this axis is skipped and the static inflight cap is the only
    /// gate. See module docs.
    ttft_slo_ms: Option<f64>,
    /// Which scheduler role this selector instance was constructed for.
    /// Surfaced on every rejection log so an operator can tell at a
    /// glance whether prefill or decode admission is the active
    /// bottleneck, without having to correlate by endpoint name.
    worker_type: &'static str,
}

impl TogetherSelector {
    /// Construct from env vars. Returns the inner selector unwrapped
    /// when no threshold env var is set, so opting out has zero
    /// runtime cost.
    ///
    /// `worker_type` is `WORKER_TYPE_PREFILL` or `WORKER_TYPE_DECODE`
    /// (see `protocols::common::timing`). The inflight cap is resolved
    /// in this precedence order:
    ///   1. role-specific env var (`*_PER_PREFILL_WORKER`/`*_PER_DECODE_WORKER`)
    ///   2. legacy `DYN_ADMISSION_MAX_INFLIGHT_PER_WORKER` fallback
    /// Other axes (uncached-blocks, TTFT SLO) are role-agnostic today.
    pub fn from_env(
        inner: Box<dyn WorkerSelector + Send + Sync>,
        worker_type: &'static str,
    ) -> Box<dyn WorkerSelector + Send + Sync> {
        let max_uncached = parse_env_usize(ENV_MAX_ACTIVE_UNCACHED_BLOCKS_PER_WORKER);
        let role_specific_inflight = match worker_type {
            "prefill" => parse_env_usize(ENV_MAX_INFLIGHT_PER_PREFILL_WORKER),
            "decode" => parse_env_usize(ENV_MAX_INFLIGHT_PER_DECODE_WORKER),
            _ => None,
        };
        let legacy_inflight = parse_env_usize(ENV_MAX_INFLIGHT_PER_WORKER);
        let max_inflight = role_specific_inflight.or(legacy_inflight);
        let ttft_slo_ms = parse_env_f64(ENV_TTFT_SLO_MS);

        if max_uncached.is_none() && max_inflight.is_none() && ttft_slo_ms.is_none() {
            return inner;
        }

        tracing::info!(
            target: "dynamo::admission",
            worker_type,
            max_active_uncached_blocks_per_worker = ?max_uncached,
            max_inflight_per_worker = ?max_inflight,
            role_specific_inflight = ?role_specific_inflight,
            legacy_inflight_fallback = ?legacy_inflight,
            ttft_slo_ms = ?ttft_slo_ms,
            "v2 per-worker admission filter enabled"
        );

        Box::new(Self {
            inner,
            max_active_uncached_blocks_per_worker: max_uncached,
            max_inflight_per_worker: max_inflight,
            ttft_slo_ms,
            worker_type,
        })
    }

    /// True if at least one of this worker's dp_ranks would still be
    /// under both thresholds after admitting `request`.
    fn worker_eligible(
        &self,
        worker_id: WorkerId,
        cfg: &ModelRuntimeConfig,
        request: &SchedulingRequest,
        block_size: u32,
        per_axis_counts: &mut RejectionAxisCounts,
    ) -> bool {
        let block_size_usize = block_size.max(1) as usize;
        let dp_end = cfg.data_parallel_start_rank + cfg.data_parallel_size;
        for dp_rank in cfg.data_parallel_start_rank..dp_end {
            let w = WorkerWithDpRank::new(worker_id, dp_rank);

            // Don't filter on data we don't have. `prefill_tokens` is
            // populated by `slots.potential_blocks_and_tokens` which
            // iterates `slots.workers` — that map can lag the outer
            // `workers_with_configs` watcher by a few ms on worker join,
            // and on cold-start it's empty entirely. Defaulting the
            // missing entry to `request.isl_tokens` (all-uncached worst
            // case) caused a real bug: a freshly-joined worker would be
            // excluded forever because its projected_uncached looked
            // like 1500+ blocks (single 50K-token request) vs a 200-block
            // threshold. With this default, "no data" → check passes
            // (the per-rank inflight check still applies, and once
            // slots syncs the worker shows up properly).
            let uncached_ok = match (
                self.max_active_uncached_blocks_per_worker,
                request.prefill_tokens.get(&w).copied(),
            ) {
                (Some(max), Some(t)) => t.div_ceil(block_size_usize) <= max,
                _ => true,
            };

            let pre_active = request
                .pre_active_request_count
                .get(&w)
                .copied()
                .unwrap_or(0);
            let inflight_ok = self
                .max_inflight_per_worker
                .is_none_or(|max| pre_active < max);

            // Predicted-TTFT axis. `prefill_tokens[w]` is
            // `new_tokens(isl, overlap) + active_tokens` — the queue
            // ahead of this request on rank w plus this request's own
            // uncached contribution. Multiplied by the worker's rolling
            // ms-per-uncached-token, that's the predicted TTFT the
            // client would observe. Cache hits have tiny
            // `new_tokens(isl, overlap)`, so under load they slip under
            // the SLO while large new-prefill requests don't — exactly
            // the "shed misses first, keep hits" behaviour we want.
            //
            // Skipped entirely when:
            // - SLO env var unset (`ttft_slo_ms` is None),
            // - throughput estimate not yet warmed (returns None;
            //   `min_tokens` worth of completions hasn't accumulated),
            // - no `prefill_tokens` entry for this rank (cold-start,
            //   matches the existing uncached-blocks defensive default).
            let ttft_ok = match (self.ttft_slo_ms, request.prefill_tokens.get(&w).copied()) {
                (Some(slo_ms), Some(tokens)) => {
                    match prefill_throughput().ms_per_uncached_token(w) {
                        Some(ms_per_tok) => (tokens as f64 * ms_per_tok) <= slo_ms,
                        None => true, // no estimate yet → don't gate
                    }
                }
                _ => true,
            };

            if uncached_ok && inflight_ok && ttft_ok {
                return true;
            }

            // Per-rank rejection observability. Each failed axis is
            // counted independently — if a rank fails on BOTH ttft AND
            // inflight, both counters tick (knowing two axes co-fire
            // is itself useful for tuning). DEBUG so high-traffic ops
            // don't drown in per-rank logs; the aggregate WARN at the
            // call site is the operator-visible summary.
            if !uncached_ok {
                per_axis_counts.uncached += 1;
                tracing::debug!(
                    target: "dynamo::admission",
                    axis = "uncached_blocks",
                    worker_id, dp_rank,
                    projected_blocks = request.prefill_tokens.get(&w).copied().unwrap_or(0).div_ceil(block_size_usize),
                    max_blocks = ?self.max_active_uncached_blocks_per_worker,
                    "v2 admission rank ineligible"
                );
            }
            if !inflight_ok {
                per_axis_counts.inflight += 1;
                tracing::debug!(
                    target: "dynamo::admission",
                    axis = "inflight",
                    worker_id, dp_rank,
                    pre_active,
                    max_inflight = ?self.max_inflight_per_worker,
                    "v2 admission rank ineligible"
                );
            }
            if !ttft_ok {
                per_axis_counts.ttft += 1;
                let tokens = request.prefill_tokens.get(&w).copied().unwrap_or(0);
                let ms_per_tok = prefill_throughput().ms_per_uncached_token(w);
                let predicted_ms = ms_per_tok.map(|m| tokens as f64 * m);
                tracing::debug!(
                    target: "dynamo::admission",
                    axis = "ttft_slo",
                    worker_id, dp_rank,
                    prefill_tokens = tokens,
                    ?predicted_ms,
                    slo_ms = ?self.ttft_slo_ms,
                    "v2 admission rank ineligible"
                );
            }
        }
        false
    }
}

/// Per-axis tally of why ranks were filtered out. Aggregated across
/// every worker × dp_rank evaluated for a single request; surfaced on
/// the top-level "all workers exceed" WARN so an operator can tell at
/// a glance which axis is the active bottleneck (inflight cap, TTFT
/// SLO, or uncached blocks) without grepping for per-rank DEBUG lines.
#[derive(Default, Debug)]
struct RejectionAxisCounts {
    uncached: u32,
    inflight: u32,
    ttft: u32,
}

impl WorkerSelector for TogetherSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let mut filtered: HashMap<WorkerId, ModelRuntimeConfig> =
            HashMap::with_capacity(workers.len());
        let mut filtered_out: HashSet<WorkerId> = HashSet::new();
        let mut per_axis_counts = RejectionAxisCounts::default();

        for (wid, cfg) in workers {
            if self.worker_eligible(*wid, cfg, request, block_size, &mut per_axis_counts) {
                filtered.insert(*wid, cfg.clone());
            } else {
                filtered_out.insert(*wid);
            }
        }

        if filtered.is_empty() {
            crate::admission::counters().incr("reject_all_workers_overloaded", 1);
            crate::admission::counters().record_event();
            // Attribute per-axis counters so dashboard/Loki can split
            // the rejection rate by which gate fired. Helpful when
            // tuning thresholds — e.g. if inflight=N is the only
            // non-zero counter, the cap is the active bottleneck and
            // raising it would admit more; if ttft is the largest,
            // we're SLO-bound and need more capacity not a higher cap.
            crate::admission::counters().incr("reject_axis_uncached", per_axis_counts.uncached as u64);
            crate::admission::counters().incr("reject_axis_inflight", per_axis_counts.inflight as u64);
            crate::admission::counters().incr("reject_axis_ttft", per_axis_counts.ttft as u64);
            tracing::warn!(
                target: "dynamo::admission",
                worker_type = self.worker_type,
                considered = workers.len(),
                request_id = ?request.maybe_request_id,
                isl = request.isl_tokens,
                axis_uncached = per_axis_counts.uncached,
                axis_inflight = per_axis_counts.inflight,
                axis_ttft = per_axis_counts.ttft,
                inflight_cap = ?self.max_inflight_per_worker,
                "v2 admission rejected: all workers exceed per-worker thresholds"
            );
            // AdmissionRejected (not NoEndpoints) so the http boundary
            // maps to 429 + Retry-After, distinguishing "we throttled
            // this on purpose" from "no workers exist". Default 2s
            // backoff matches v1's existing retry hint.
            return Err(KvSchedulerError::AdmissionRejected { retry_after_secs: 2 });
        }

        if !filtered_out.is_empty() {
            tracing::debug!(
                target: "dynamo::admission",
                kept = filtered.len(),
                filtered_out = filtered_out.len(),
                "v2 admission filtered overloaded workers"
            );
        }

        self.inner.select_worker(&filtered, request, block_size)
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

fn parse_env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
}
