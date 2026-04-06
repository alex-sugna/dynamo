// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;

use super::cache::PrefixCache;
use super::EncodeResult;
#[cfg(test)]
use super::CacheStats;
use crate::tokenizers::TikTokenTokenizer;
use crate::tokenizers::traits::Encoder;

pub(crate) struct TiktokenBackend {
    tokenizer: TikTokenTokenizer,
    boundary_strings: HashMap<u32, String>,
    state: Mutex<CacheState>,
}

#[derive(Default)]
struct CacheState {
    hits_exact: u64,
    hits_prefix: u64,
    misses: u64,
    cache: Option<PrefixCache>,
}

impl TiktokenBackend {
    pub(crate) fn new(
        tokenizer: TikTokenTokenizer,
        boundary_strings: HashMap<u32, String>,
        max_entries: usize,
    ) -> Self {
        let boundary_tokens = boundary_strings.keys().copied().collect();
        Self {
            tokenizer,
            boundary_strings,
            state: Mutex::new(CacheState {
                cache: Some(PrefixCache::new(max_entries, boundary_tokens)),
                ..CacheState::default()
            }),
        }
    }

    pub(crate) fn encode(&self, text: &str) -> Result<EncodeResult> {
        {
            let mut state = self.state.lock();
            let cache = state.cache.as_mut().expect("cache is initialized");
            if let Some(tokens) = cache.get_exact(text) {
                state.hits_exact += 1;
                return Ok(EncodeResult { tokens });
            }
        }

        let prefix_match = {
            let mut state = self.state.lock();
            let cache = state.cache.as_mut().expect("cache is initialized");
            cache.get_prefix(text)
        };

        if let Some((prefix_text, prefix_tokens)) = prefix_match {
            let suffix = &text[prefix_text.len()..];
            let suffix_ids = self.tokenizer.encode(suffix)?.token_ids().to_vec();

            let mut full_tokens = prefix_tokens;
            full_tokens.extend_from_slice(&suffix_ids);

            let mut state = self.state.lock();
            let cache = state.cache.as_mut().expect("cache is initialized");
            cache.put_with_rfind(text, &full_tokens, &self.boundary_strings);
            state.hits_prefix += 1;

            return Ok(EncodeResult { tokens: full_tokens });
        }

        let tokens = self.tokenizer.encode(text)?.token_ids().to_vec();
        let mut state = self.state.lock();
        let cache = state.cache.as_mut().expect("cache is initialized");
        cache.put_with_rfind(text, &tokens, &self.boundary_strings);
        state.misses += 1;

        Ok(EncodeResult { tokens })
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> CacheStats {
        let state = self.state.lock();
        let cache = state.cache.as_ref().expect("cache is initialized");
        CacheStats {
            hits_exact: state.hits_exact,
            hits_prefix: state.hits_prefix,
            misses: state.misses,
            cache_size: cache.len(),
            put_truncated: cache.put_stats.truncated,
            put_full: cache.put_stats.full,
            put_skipped: cache.put_stats.skipped,
        }
    }
}
