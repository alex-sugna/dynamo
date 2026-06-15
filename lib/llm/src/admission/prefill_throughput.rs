// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-worker prefill throughput tracker.
//!
//! Maintains a sliding window of completed prefills per
//! `WorkerWithDpRank`, indexed by `(instant, uncached_tokens)`. Exposes
//! `ms_per_uncached_token(w)` as the inverse of windowed throughput —
//! i.e. how many wall-clock milliseconds of prefill work each uncached
//! token currently takes, accounting for batching.
//!
//! Used by [`crate::admission::together_selector::TogetherSelector`] to
//! predict TTFT for incoming requests:
//!
//! ```text
//! predicted_ttft[w] = prefill_tokens[w] × ms_per_uncached_token[w]
//! ```
//!
//! `prefill_tokens[w]` is the queue's already-computed
//! `new_tokens(isl, overlap) + active_tokens` — i.e. the queue ahead of
//! this request plus this request's own uncached contribution. No
//! separate queue tracking needed.
//!
//! Process-wide singleton via [`prefill_throughput`]. Record on prefill
//! completion (first-token observation); read on admission.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::kv_router::protocols::WorkerWithDpRank;

const ENV_PREFILL_WINDOW_MS: &str = "DYN_ADMISSION_PREFILL_WINDOW_MS";
const ENV_MIN_TOKENS_FOR_ESTIMATE: &str = "DYN_ADMISSION_MIN_TOKENS_FOR_ESTIMATE";

const DEFAULT_WINDOW_MS: u64 = 30_000;
const DEFAULT_MIN_TOKENS: u32 = 5_000;

pub struct PrefillThroughput {
    /// Per-worker FIFO of (completion_instant, uncached_tokens_completed).
    /// Reaped on every read or write so it stays bounded to `window`.
    samples: Mutex<HashMap<WorkerWithDpRank, VecDeque<(Instant, u32)>>>,
    window: Duration,
    min_tokens: u32,
}

impl PrefillThroughput {
    fn new() -> Self {
        let window_ms = parse_env_u64(ENV_PREFILL_WINDOW_MS).unwrap_or(DEFAULT_WINDOW_MS);
        let min_tokens = parse_env_u32(ENV_MIN_TOKENS_FOR_ESTIMATE).unwrap_or(DEFAULT_MIN_TOKENS);
        tracing::info!(
            target: "dynamo::admission",
            prefill_window_ms = window_ms,
            min_tokens_for_estimate = min_tokens,
            "prefill throughput tracker initialized"
        );
        Self {
            samples: Mutex::new(HashMap::new()),
            window: Duration::from_millis(window_ms),
            min_tokens,
        }
    }

    /// Record a completed prefill. Pure cache hits (uncached=0) give no
    /// throughput signal — they finish in ~0 GPU time regardless of
    /// engine load — so we skip them.
    pub fn record(&self, worker: WorkerWithDpRank, uncached_tokens: u32) {
        if uncached_tokens == 0 {
            return;
        }
        let now = Instant::now();
        let mut guard = self.samples.lock().unwrap();
        let entry = guard.entry(worker).or_default();
        entry.push_back((now, uncached_tokens));
        reap(entry, self.window, now);
    }

    /// Returns `None` when the window doesn't yet have at least
    /// `min_tokens` of completed prefills — caller should fall back to a
    /// static admission policy in that case.
    pub fn ms_per_uncached_token(&self, worker: WorkerWithDpRank) -> Option<f64> {
        let now = Instant::now();
        let mut guard = self.samples.lock().unwrap();
        let entry = guard.get_mut(&worker)?;
        reap(entry, self.window, now);
        let total: u32 = entry.iter().map(|(_, n)| n).sum();
        if total < self.min_tokens {
            return None;
        }
        let span_ms = now.duration_since(entry.front()?.0).as_secs_f64() * 1000.0;
        if span_ms < 1.0 {
            return None;
        }
        Some(span_ms / total as f64)
    }
}

fn reap(entry: &mut VecDeque<(Instant, u32)>, window: Duration, now: Instant) {
    while let Some(&(t, _)) = entry.front() {
        if now.duration_since(t) > window {
            entry.pop_front();
        } else {
            break;
        }
    }
}

/// Process-wide singleton. Constructed on first call.
pub fn prefill_throughput() -> &'static PrefillThroughput {
    static INSTANCE: OnceLock<PrefillThroughput> = OnceLock::new();
    INSTANCE.get_or_init(PrefillThroughput::new)
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}
fn parse_env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn returns_none_below_min_tokens() {
        // Use tiny min_tokens so the test is self-contained
        let t = PrefillThroughput {
            samples: Mutex::new(HashMap::new()),
            window: Duration::from_secs(60),
            min_tokens: 100,
        };
        t.record(worker(1), 50);
        assert!(t.ms_per_uncached_token(worker(1)).is_none());
        t.record(worker(1), 60); // total now 110, above min
        assert!(t.ms_per_uncached_token(worker(1)).is_some());
    }

    #[test]
    fn pure_cache_hits_ignored() {
        let t = PrefillThroughput {
            samples: Mutex::new(HashMap::new()),
            window: Duration::from_secs(60),
            min_tokens: 1,
        };
        t.record(worker(1), 0);
        // Window has no entries, so estimate is None
        assert!(t.ms_per_uncached_token(worker(1)).is_none());
    }

    #[test]
    fn old_samples_evicted() {
        let t = PrefillThroughput {
            samples: Mutex::new(HashMap::new()),
            window: Duration::from_millis(50),
            min_tokens: 10,
        };
        t.record(worker(1), 1000);
        std::thread::sleep(Duration::from_millis(80));
        // Sample now older than window; reap drops it
        assert!(t.ms_per_uncached_token(worker(1)).is_none());
    }
}
