// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use linked_hash_map::LinkedHashMap;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub(crate) struct PutStats {
    pub(crate) truncated: u64,
    pub(crate) full: u64,
    pub(crate) skipped: u64,
}

pub(crate) struct PrefixCache {
    entries: LinkedHashMap<String, Vec<u32>>,
    max_entries: usize,
    boundary_tokens: HashSet<u32>,
    pub(crate) put_stats: PutStats,
}

impl PrefixCache {
    pub(crate) fn new(max_entries: usize, boundary_tokens: HashSet<u32>) -> Self {
        Self {
            entries: LinkedHashMap::new(),
            max_entries,
            boundary_tokens,
            put_stats: PutStats::default(),
        }
    }

    pub(crate) fn get_exact(&mut self, text: &str) -> Option<Vec<u32>> {
        self.entries.get_refresh(text).cloned()
    }

    pub(crate) fn get_prefix(&mut self, text: &str) -> Option<(String, Vec<u32>)> {
        let mut best: Option<(&str, &[u32])> = None;

        for (cached_text, cached_tokens) in self.entries.iter() {
            if text.starts_with(cached_text.as_str()) && cached_text.len() < text.len() {
                match best {
                    None => best = Some((cached_text.as_str(), cached_tokens.as_slice())),
                    Some((prev, _)) if cached_text.len() > prev.len() => {
                        best = Some((cached_text.as_str(), cached_tokens.as_slice()));
                    }
                    _ => {}
                }
            }
        }

        if let Some((prefix_text, prefix_tokens)) = best {
            let key = prefix_text.to_string();
            let tokens = prefix_tokens.to_vec();
            let _ = self.entries.get_refresh(&key);
            Some((key, tokens))
        } else {
            None
        }
    }

    pub(crate) fn put(&mut self, text: &str, tokens: &[u32], char_offsets: &[usize]) {
        assert_eq!(tokens.len(), char_offsets.len());

        let boundary_idx = tokens
            .iter()
            .enumerate()
            .rev()
            .find(|(_, tok)| self.boundary_tokens.contains(tok));

        let truncate_to = match boundary_idx {
            Some((idx, _)) => idx + 1,
            None => {
                self.put_stats.skipped += 1;
                return;
            }
        };

        if truncate_to == tokens.len() {
            self.put_stats.full += 1;
        } else {
            self.put_stats.truncated += 1;
        }

        let truncated_tokens = &tokens[..truncate_to];
        let truncated_text = &text[..char_offsets[truncate_to - 1]];

        while self.entries.len() >= self.max_entries {
            let _ = self.entries.pop_front();
        }

        self.entries
            .insert(truncated_text.to_string(), truncated_tokens.to_vec());
    }

    pub(crate) fn put_with_rfind(
        &mut self,
        text: &str,
        tokens: &[u32],
        boundary_strings: &HashMap<u32, String>,
    ) {
        let boundary = tokens
            .iter()
            .enumerate()
            .rev()
            .find(|(_, tok)| self.boundary_tokens.contains(tok));

        let (truncate_to, boundary_tok) = match boundary {
            Some((idx, tok)) => (idx + 1, *tok),
            None => {
                self.put_stats.skipped += 1;
                return;
            }
        };

        let truncated_tokens = &tokens[..truncate_to];
        let truncated_text = if truncate_to == tokens.len() {
            self.put_stats.full += 1;
            text
        } else {
            let Some(boundary_text) = boundary_strings.get(&boundary_tok) else {
                self.put_stats.skipped += 1;
                return;
            };
            match text.rfind(boundary_text) {
                Some(pos) => {
                    self.put_stats.truncated += 1;
                    &text[..pos + boundary_text.len()]
                }
                None => {
                    self.put_stats.skipped += 1;
                    return;
                }
            }
        };

        while self.entries.len() >= self.max_entries {
            let _ = self.entries.pop_front();
        }

        self.entries
            .insert(truncated_text.to_string(), truncated_tokens.to_vec());
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(max_entries: usize, boundary_ids: &[u32]) -> PrefixCache {
        PrefixCache::new(max_entries, boundary_ids.iter().copied().collect())
    }

    #[test]
    fn test_exact_match() {
        let mut cache = make_cache(10, &[100]);
        cache.entries.insert("hello".to_string(), vec![1, 2, 100]);

        assert_eq!(cache.get_exact("hello"), Some(vec![1, 2, 100]));
        assert_eq!(cache.get_exact("world"), None);
    }

    #[test]
    fn test_prefix_match_prefers_longest_prefix() {
        let mut cache = make_cache(10, &[100]);
        cache.entries.insert("hello".to_string(), vec![1, 100]);
        cache
            .entries
            .insert("hello world".to_string(), vec![1, 100, 2, 100]);

        let (prefix_text, prefix_tokens) = cache.get_prefix("hello world again").unwrap();
        assert_eq!(prefix_text, "hello world");
        assert_eq!(prefix_tokens, vec![1, 100, 2, 100]);
    }

    #[test]
    fn test_prefix_no_match_for_exact() {
        let mut cache = make_cache(10, &[100]);
        cache.entries.insert("hello".to_string(), vec![1, 2, 100]);

        assert!(cache.get_prefix("hello").is_none());
    }

    #[test]
    fn test_put_truncates_to_last_boundary() {
        let mut cache = make_cache(10, &[100]);
        let tokens = vec![1, 2, 100, 3, 4];
        let offsets = vec![1, 2, 3, 4, 5];

        cache.put("abcde", &tokens, &offsets);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_exact("abc"), Some(vec![1, 2, 100]));
        assert_eq!(cache.put_stats.truncated, 1);
        assert_eq!(cache.put_stats.full, 0);
    }

    #[test]
    fn test_put_full_when_last_token_is_boundary() {
        let mut cache = make_cache(10, &[100]);
        let tokens = vec![1, 2, 100];
        let offsets = vec![1, 2, 3];

        cache.put("abc", &tokens, &offsets);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_exact("abc"), Some(vec![1, 2, 100]));
        assert_eq!(cache.put_stats.full, 1);
        assert_eq!(cache.put_stats.truncated, 0);
    }

    #[test]
    fn test_put_skips_when_no_boundary_exists() {
        let mut cache = make_cache(10, &[100]);
        cache.put("abc", &[1, 2, 3], &[1, 2, 3]);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.put_stats.skipped, 1);
    }

    #[test]
    fn test_put_truncates_to_last_of_multiple_boundaries() {
        let mut cache = make_cache(10, &[100, 200]);
        let tokens = vec![1, 100, 2, 200, 3, 4];
        let offsets = vec![1, 2, 3, 4, 5, 6];

        cache.put("abcdef", &tokens, &offsets);

        assert_eq!(cache.get_exact("abcd"), Some(vec![1, 100, 2, 200]));
        assert!(cache.get_exact("abcdef").is_none());
        assert!(cache.get_exact("ab").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = make_cache(2, &[100]);
        cache.entries.insert("a".to_string(), vec![100]);
        cache.entries.insert("b".to_string(), vec![100]);

        cache.put("c", &[100], &[1]);

        assert_eq!(cache.len(), 2);
        assert!(cache.get_exact("a").is_none());
        assert!(cache.get_exact("b").is_some());
        assert!(cache.get_exact("c").is_some());
    }

    #[test]
    fn test_empty_input() {
        let mut cache = make_cache(10, &[100]);

        assert!(cache.get_exact("").is_none());
        assert!(cache.get_prefix("").is_none());
    }
}
