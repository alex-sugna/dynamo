// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod boundary;
mod cache;
mod tiktoken;
#[cfg(test)]
mod fuzzer;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

use crate::model_card::{ModelDeploymentCard, TokenizerKind};
use crate::tokenizers::tiktoken::load_special_token_boundaries;
use crate::tokenizers::{Encoding, TikTokenTokenizer};

use self::cache::PrefixCache;
use self::tiktoken::TiktokenBackend;

const DEFAULT_MAX_ENTRIES: usize = 1024;

struct EncodeResult {
    tokens: Vec<u32>,
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
struct CacheStats {
    hits_exact: u64,
    hits_prefix: u64,
    misses: u64,
    cache_size: usize,
    put_truncated: u64,
    put_full: u64,
    put_skipped: u64,
}

pub(crate) struct TokenizerCache {
    backend: Backend,
}

enum Backend {
    HuggingFace(HfBackend),
    Tiktoken(TiktokenBackend),
}

struct HfBackend {
    tokenizer: HfTokenizer,
    state: Mutex<HfCacheState>,
}

#[derive(Default)]
struct HfCacheState {
    hits_exact: u64,
    hits_prefix: u64,
    misses: u64,
    cache: Option<PrefixCache>,
}

impl TokenizerCache {
    pub(crate) fn from_model_card(mdc: &ModelDeploymentCard) -> Result<Option<Self>> {
        match &mdc.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(checked_file)) => {
                let path = checked_file.path().with_context(|| {
                    format!(
                        "Tokenizer cache requires a local tokenizer.json for {}",
                        mdc.display_name
                    )
                })?;
                Self::from_hf(path)
            }
            Some(TokenizerKind::TikTokenModel(checked_file)) => {
                let path = checked_file.path().with_context(|| {
                    format!(
                        "Tokenizer cache requires a local tiktoken model for {}",
                        mdc.display_name
                    )
                })?;
                Self::from_tiktoken(path)
            }
            None => Ok(None),
        }
    }

    pub(crate) fn encode(&self, text: &str) -> Result<Encoding> {
        let result = match &self.backend {
            Backend::HuggingFace(backend) => backend.encode(text)?,
            Backend::Tiktoken(backend) => backend.encode(text)?,
        };
        Ok(Encoding::Sp(result.tokens))
    }

    #[cfg(test)]
    fn stats(&self) -> CacheStats {
        match &self.backend {
            Backend::HuggingFace(backend) => backend.stats(),
            Backend::Tiktoken(backend) => backend.stats(),
        }
    }

    fn from_hf(path: &Path) -> Result<Option<Self>> {
        let boundary_tokens = boundary::discover_boundary_tokens(path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Failed discovering tokenizer boundaries for {}", path.display()))?;

        if boundary_tokens.is_empty() {
            return Ok(None);
        }

        let tokenizer = HfTokenizer::from_file(path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Failed to load HuggingFace tokenizer from {}", path.display()))?;

        Ok(Some(Self {
            backend: Backend::HuggingFace(HfBackend {
                tokenizer,
                state: Mutex::new(HfCacheState {
                    cache: Some(PrefixCache::new(DEFAULT_MAX_ENTRIES, boundary_tokens)),
                    ..HfCacheState::default()
                }),
            }),
        }))
    }

    fn from_tiktoken(path: &Path) -> Result<Option<Self>> {
        let boundary_strings = load_special_token_boundaries(path)
            .with_context(|| format!("Failed loading tiktoken boundary strings from {}", path.display()))?;

        if boundary_strings.is_empty() {
            return Ok(None);
        }

        let path_str = path
            .to_str()
            .with_context(|| format!("Invalid UTF-8 path for {}", path.display()))?;
        let tokenizer = TikTokenTokenizer::from_file_auto(path_str)
            .with_context(|| format!("Failed to load tiktoken tokenizer from {}", path.display()))?;

        Ok(Some(Self {
            backend: Backend::Tiktoken(TiktokenBackend::new(
                tokenizer,
                boundary_strings,
                DEFAULT_MAX_ENTRIES,
            )),
        }))
    }
}

impl HfBackend {
    fn encode(&self, text: &str) -> Result<EncodeResult> {
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
            let suffix_encoding = self
                .tokenizer
                .encode(suffix, false)
                .map_err(anyhow::Error::msg)
                .with_context(|| "Failed to encode HuggingFace suffix")?;

            let suffix_ids = suffix_encoding.get_ids();
            let suffix_offsets = suffix_encoding.get_offsets();
            let prefix_len = prefix_text.len();

            let mut full_tokens = prefix_tokens;
            let cached_tokens = full_tokens.len();
            full_tokens.extend_from_slice(suffix_ids);

            let mut full_offsets = Vec::with_capacity(full_tokens.len());
            full_offsets.extend(std::iter::repeat(prefix_len).take(cached_tokens));
            full_offsets.extend(suffix_offsets.iter().map(|(_, end)| prefix_len + end));

            let mut state = self.state.lock();
            let cache = state.cache.as_mut().expect("cache is initialized");
            cache.put(text, &full_tokens, &full_offsets);
            state.hits_prefix += 1;

            return Ok(EncodeResult { tokens: full_tokens });
        }

        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)
            .with_context(|| "Failed to encode HuggingFace prompt")?;
        let tokens = encoding.get_ids().to_vec();
        let offsets: Vec<usize> = encoding.get_offsets().iter().map(|(_, end)| *end).collect();

        let mut state = self.state.lock();
        let cache = state.cache.as_mut().expect("cache is initialized");
        cache.put(text, &tokens, &offsets);
        state.misses += 1;

        Ok(EncodeResult { tokens })
    }

    #[cfg(test)]
    fn stats(&self) -> CacheStats {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::path::PathBuf;

    use crate::tokenizers::traits::Encoder;

    const HF_MODEL_DIR: &str = "tests/data/sample-models/TinyLlama_v1.1";
    const TIKTOKEN_MODEL_DIR: &str = "tests/data/sample-models/mock-tiktoken";

    fn sample_model_dir(model_dir: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(model_dir)
    }

    fn load_mdc(model_dir: &str) -> ModelDeploymentCard {
        ModelDeploymentCard::load_from_disk(sample_model_dir(model_dir), None).unwrap()
    }

    fn hf_oracle(path: &Path, text: &str) -> Vec<u32> {
        HfTokenizer::from_file(path)
            .unwrap()
            .encode(text, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    fn tiktoken_oracle(path: &Path, text: &str) -> Vec<u32> {
        TikTokenTokenizer::from_file_auto(path.to_str().unwrap())
            .unwrap()
            .encode(text)
            .unwrap()
            .token_ids()
            .to_vec()
    }

    fn random_text(rng: &mut StdRng, max_len: usize) -> String {
        const ASCII: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t";
        const CJK: &[char] = &['你', '好', '世', '界', '測', '試', '語', '言'];
        const EMOJI: &[char] = &['😀', '🥹', '🚀', '✨', '🧠', '🔥'];

        let len = rng.random_range(1..=max_len);
        let choice = rng.random_range(0..4);
        match choice {
            0 => (0..len)
                .map(|_| ASCII[rng.random_range(0..ASCII.len())] as char)
                .collect(),
            1 => (0..len)
                .map(|_| CJK[rng.random_range(0..CJK.len())])
                .collect(),
            2 => (0..len)
                .map(|_| EMOJI[rng.random_range(0..EMOJI.len())])
                .collect(),
            _ => {
                let mut text = String::new();
                for _ in 0..len {
                    match rng.random_range(0..3) {
                        0 => text.push(ASCII[rng.random_range(0..ASCII.len())] as char),
                        1 => text.push(CJK[rng.random_range(0..CJK.len())]),
                        _ => text.push(EMOJI[rng.random_range(0..EMOJI.len())]),
                    }
                }
                text
            }
        }
    }

    fn build_hf_prompts(rng: &mut StdRng, turns: usize) -> Vec<String> {
        let mut prompt = "<s>Tokenizer cache system prompt.</s>".to_string();
        let mut prompts = vec![prompt.clone()];
        for _ in 0..turns {
            let user = random_text(rng, 48);
            let assistant = random_text(rng, 64);
            prompt.push_str(&format!(
                "<s>User: {user}\nAssistant: {assistant}</s>"
            ));
            prompts.push(prompt.clone());
        }
        prompts
    }

    fn build_tiktoken_prompts(rng: &mut StdRng, turns: usize) -> Vec<String> {
        let mut prompt = "<|im_start|>system\nTokenizer cache system prompt.<|im_end|>".to_string();
        let mut prompts = vec![prompt.clone()];
        for _ in 0..turns {
            let user = random_text(rng, 48);
            let assistant = random_text(rng, 64);
            prompt.push_str(&format!(
                "<|im_start|>user\n{user}<|im_end|><|im_start|>assistant\n{assistant}<|im_end|>"
            ));
            prompts.push(prompt.clone());
        }
        prompts
    }

    #[test]
    fn test_hf_tokenizer_cache_matches_oracle_and_hits_all_paths() {
        let mdc = load_mdc(HF_MODEL_DIR);
        let cache = TokenizerCache::from_model_card(&mdc).unwrap().unwrap();
        let tokenizer_path = sample_model_dir(HF_MODEL_DIR).join("tokenizer.json");
        let mut rng = StdRng::seed_from_u64(7);

        let plain_text = "plain text without tokenizer delimiters";
        let cached = cache.encode(plain_text).unwrap().token_ids().to_vec();
        let oracle = hf_oracle(&tokenizer_path, plain_text);
        assert_eq!(cached, oracle);

        for prompt in build_hf_prompts(&mut rng, 24) {
            let cached = cache.encode(&prompt).unwrap().token_ids().to_vec();
            let oracle = hf_oracle(&tokenizer_path, &prompt);
            assert_eq!(cached, oracle);

            let truncated_prompt = format!("{prompt}{}", random_text(&mut rng, 24));
            let cached = cache.encode(&truncated_prompt).unwrap().token_ids().to_vec();
            let oracle = hf_oracle(&tokenizer_path, &truncated_prompt);
            assert_eq!(cached, oracle);
        }

        let stats = cache.stats();
        // Exact-match behavior is covered by cache::tests::test_exact_match.
        assert!(stats.hits_prefix > 0);
        assert!(stats.misses > 0);
        assert!(stats.cache_size > 0);
        assert!(stats.put_truncated > 0);
        assert!(stats.put_full > 0);
        assert!(stats.put_skipped > 0);
    }

    #[test]
    fn test_tiktoken_tokenizer_cache_matches_oracle_and_hits_all_paths() {
        let mdc = load_mdc(TIKTOKEN_MODEL_DIR);
        let cache = TokenizerCache::from_model_card(&mdc).unwrap().unwrap();
        let tokenizer_path = sample_model_dir(TIKTOKEN_MODEL_DIR).join("tiktoken.model");
        let mut rng = StdRng::seed_from_u64(11);

        let plain_text = "plain text without tokenizer delimiters";
        let cached = cache.encode(plain_text).unwrap().token_ids().to_vec();
        let oracle = tiktoken_oracle(&tokenizer_path, plain_text);
        assert_eq!(cached, oracle);

        for prompt in build_tiktoken_prompts(&mut rng, 24) {
            let cached = cache.encode(&prompt).unwrap().token_ids().to_vec();
            let oracle = tiktoken_oracle(&tokenizer_path, &prompt);
            assert_eq!(cached, oracle);

            let truncated_prompt = format!("{prompt}{}", random_text(&mut rng, 24));
            let cached = cache.encode(&truncated_prompt).unwrap().token_ids().to_vec();
            let oracle = tiktoken_oracle(&tokenizer_path, &truncated_prompt);
            assert_eq!(cached, oracle);
        }

        let stats = cache.stats();
        // Exact-match behavior is covered by cache::tests::test_exact_match.
        assert!(stats.hits_prefix > 0);
        assert!(stats.misses > 0);
        assert!(stats.cache_size > 0);
        assert!(stats.put_truncated > 0);
        assert!(stats.put_full > 0);
        assert!(stats.put_skipped > 0);
    }
}
