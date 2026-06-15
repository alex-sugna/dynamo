// SPDX-FileCopyrightText: Copyright (c) 2026 Together AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Windowed subsystem counters.
//!
//! Subsystem code (admission, router, health, …) registers named
//! counters under a [`WindowedCounters`] instance, increments them as
//! events occur, and the helper emits one structured trace line per
//! window then resets all counters to zero. The format matches the
//! kvblock-verifier / tcache trace style:
//!
//! ```text
//! [admission] window=1000 | admit=987 reject_overload=12 reject_no_capacity=1
//! [router]    window=1000 | route_prefill_w0=489 route_prefill_w1=511
//! ```
//!
//! Window units are arbitrary (requests, iterations, seconds) — declared
//! by the subsystem at construction time via `unit_label`. The helper
//! emits when `record_event` has been called `window_size` times since
//! the last emission.
//!
//! All operations are lock-free on the hot path: counters are
//! [`AtomicU64`], the event count is also atomic. The emission path
//! takes a brief read lock to enumerate counter names; that path runs
//! at most once per window.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use tracing::Level;

/// Tracing target for windowed subsystem events. A single fmt layer
/// filtered to this target captures everything; subscribers split by
/// the `subsystem` field.
pub const WINDOWED_TARGET: &str = "dynamo::observability::windowed";

/// Aggregates named u64 counters; emits one structured trace line per
/// window of N recorded events, then resets all counters to zero.
///
/// Cheap to clone (internally `Arc`'d). Hand a clone to each subsystem
/// site that needs to record an event.
#[derive(Clone)]
pub struct WindowedCounters {
    inner: Arc<Inner>,
}

struct Inner {
    subsystem: &'static str,
    unit_label: &'static str,
    window_size: u64,
    counters: RwLock<BTreeMap<&'static str, Arc<AtomicU64>>>,
    events_in_window: AtomicU64,
    total_windows_emitted: AtomicU64,
}

impl WindowedCounters {
    /// Build a new windowed-counter set.
    ///
    /// - `subsystem`: appears as `[<subsystem>]` in the output line and as
    ///   the `subsystem` field on the tracing event. Use a short
    ///   identifier (`"admission"`, `"router"`, `"health"`).
    /// - `unit_label`: the unit `window=` is measured in (`"requests"`,
    ///   `"iterations"`, `"seconds"`). Free-form; emitted as a field
    ///   on the tracing event.
    /// - `window_size`: how many `record_event()` calls trigger an emit.
    ///
    /// Counters can be added at any time via `incr` — the first
    /// increment registers the name. To pre-register without
    /// incrementing (so the first emit always shows the counter even
    /// at value 0), call `register`.
    pub fn new(subsystem: &'static str, unit_label: &'static str, window_size: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                subsystem,
                unit_label,
                window_size,
                counters: RwLock::new(BTreeMap::new()),
                events_in_window: AtomicU64::new(0),
                total_windows_emitted: AtomicU64::new(0),
            }),
        }
    }

    /// Pre-register a counter so it appears in every emit even at zero.
    /// Idempotent.
    pub fn register(&self, name: &'static str) {
        let mut counters = self.inner.counters.write().unwrap();
        counters
            .entry(name)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
    }

    /// Increment the named counter by `delta`. Registers the counter on
    /// first use.
    pub fn incr(&self, name: &'static str, delta: u64) {
        // Fast path: counter already registered.
        {
            let counters = self.inner.counters.read().unwrap();
            if let Some(c) = counters.get(name) {
                c.fetch_add(delta, Ordering::Relaxed);
                return;
            }
        }
        // Slow path: register then increment.
        let mut counters = self.inner.counters.write().unwrap();
        let c = counters
            .entry(name)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(counters);
        c.fetch_add(delta, Ordering::Relaxed);
    }

    /// Record a single event. When `window_size` events have been
    /// recorded since the last emit, the helper emits the current
    /// snapshot and resets counters. Pass any positive `weight` if a
    /// single call accounts for more than one logical event.
    pub fn record_event(&self) {
        self.record_events(1);
    }

    /// Record `n` events at once.
    pub fn record_events(&self, n: u64) {
        if n == 0 {
            return;
        }
        let prev = self.inner.events_in_window.fetch_add(n, Ordering::Relaxed);
        let new = prev + n;
        if new >= self.inner.window_size {
            self.flush();
        }
    }

    /// Force-emit + reset, regardless of how full the window is. Useful
    /// for end-of-shutdown flush so partial windows aren't lost.
    pub fn flush(&self) {
        // Atomically take the events-in-window count.
        let events = self.inner.events_in_window.swap(0, Ordering::Relaxed);
        if events == 0 {
            return;
        }

        // Snapshot + reset all counters. Read lock is sufficient — we
        // swap each counter's value with 0 atomically, no need to hold
        // a write lock.
        let snapshot: Vec<(&'static str, u64)> = {
            let counters = self.inner.counters.read().unwrap();
            counters
                .iter()
                .map(|(name, c)| (*name, c.swap(0, Ordering::Relaxed)))
                .collect()
        };

        let window_idx = self.inner.total_windows_emitted.fetch_add(1, Ordering::Relaxed);

        if !tracing::event_enabled!(target: WINDOWED_TARGET, Level::INFO) {
            return;
        }

        // Build the human-friendly counter list (`k1=v1 k2=v2 ...`) so
        // the line matches the trace style operators are used to. Also
        // emit each counter as its own field on the tracing event so
        // structured/JSON subscribers can split them.
        let counters_str = snapshot
            .iter()
            .map(|(name, v)| format!("{}={}", name, v))
            .collect::<Vec<_>>()
            .join(" ");

        // Serialize counters as an inline JSON object for the
        // structured-format consumers; keeps the per-counter values
        // queryable without a custom Visit impl.
        let counters_json = serde_json::Value::Object(
            snapshot
                .iter()
                .map(|(name, v)| (name.to_string(), serde_json::Value::from(*v)))
                .collect(),
        );

        tracing::event!(
            target: WINDOWED_TARGET,
            Level::INFO,
            subsystem = self.inner.subsystem,
            unit = self.inner.unit_label,
            window = events,
            window_idx = window_idx,
            counters = %counters_json,
            "[{}] window={} | {}",
            self.inner.subsystem,
            events,
            counters_str
        );
    }

    /// Snapshot the current counter values without resetting. Intended
    /// for tests and ad-hoc inspection.
    pub fn snapshot(&self) -> BTreeMap<&'static str, u64> {
        let counters = self.inner.counters.read().unwrap();
        counters
            .iter()
            .map(|(name, c)| (*name, c.load(Ordering::Relaxed)))
            .collect()
    }

    /// How many windows have been emitted (for debugging / sanity in
    /// tests).
    pub fn windows_emitted(&self) -> u64 {
        self.inner.total_windows_emitted.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incr_registers_and_aggregates() {
        let w = WindowedCounters::new("test", "events", 1000);
        w.incr("foo", 1);
        w.incr("foo", 2);
        w.incr("bar", 5);
        let snap = w.snapshot();
        assert_eq!(snap.get("foo"), Some(&3));
        assert_eq!(snap.get("bar"), Some(&5));
    }

    #[test]
    fn flush_resets_counters() {
        let w = WindowedCounters::new("test", "events", 1000);
        w.incr("foo", 7);
        w.record_event();
        w.flush();
        let snap = w.snapshot();
        // After flush, counters are zeroed.
        assert_eq!(snap.get("foo"), Some(&0));
        assert_eq!(w.windows_emitted(), 1);
    }

    #[test]
    fn window_size_triggers_emit() {
        let w = WindowedCounters::new("test", "events", 3);
        w.incr("foo", 1);
        w.record_event();
        w.record_event();
        assert_eq!(w.windows_emitted(), 0);
        w.record_event();
        // Third event triggers an emit.
        assert_eq!(w.windows_emitted(), 1);
        // And resets foo.
        assert_eq!(w.snapshot().get("foo"), Some(&0));
    }

    #[test]
    fn record_events_accumulates() {
        let w = WindowedCounters::new("test", "events", 10);
        w.record_events(5);
        w.record_events(4);
        assert_eq!(w.windows_emitted(), 0);
        w.record_events(1);
        assert_eq!(w.windows_emitted(), 1);
    }

    #[test]
    fn flush_is_noop_when_empty() {
        let w = WindowedCounters::new("test", "events", 100);
        w.flush();
        assert_eq!(w.windows_emitted(), 0);
    }

    #[test]
    fn pre_register_appears_in_snapshot() {
        let w = WindowedCounters::new("test", "events", 100);
        w.register("a");
        w.register("b");
        let snap = w.snapshot();
        assert_eq!(snap.len(), 2);
        assert!(snap.contains_key("a"));
        assert!(snap.contains_key("b"));
    }
}
