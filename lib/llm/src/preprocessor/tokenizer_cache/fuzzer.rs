// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model_card::ModelDeploymentCard;
use crate::tokenizers::tiktoken::load_special_token_boundaries;
use crate::tokenizers::traits::Encoder;
use crate::tokenizers::TikTokenTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

const HF_MODEL_DIR: &str = "tests/data/sample-models/TinyLlama_v1.1";
const TIKTOKEN_MODEL_DIR: &str = "tests/data/sample-models/mock-tiktoken";
const HF_SEED: u64 = 7;
const TIKTOKEN_SEED: u64 = 11;
const FUZZER_ITERATIONS: usize = 1000;

fn sample_model_dir(model_dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(model_dir)
}

fn load_mdc(model_dir: &str) -> ModelDeploymentCard {
    ModelDeploymentCard::load_from_disk(sample_model_dir(model_dir), None).unwrap()
}

fn hf_oracle(tokenizer: &HfTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .unwrap()
        .get_ids()
        .to_vec()
}

fn tiktoken_oracle(tokenizer: &TikTokenTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text)
        .unwrap()
        .token_ids()
        .to_vec()
}

fn load_hf_added_token_contents(tokenizer_path: &Path) -> Vec<String> {
    let content = fs::read_to_string(tokenizer_path).unwrap();
    let parsed: Value = serde_json::from_str(&content).unwrap();
    parsed
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|token| token.get("content").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect()
}

fn load_tiktoken_special_token_contents(model_path: &Path) -> Vec<String> {
    let mut contents: Vec<String> = load_special_token_boundaries(model_path)
        .unwrap()
        .into_values()
        .collect();
    contents.sort();
    contents
}

fn random_text(rng: &mut StdRng, max_len: usize) -> String {
    const ASCII: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t!@#$%^&*()_+-=[]{}|;:',./<>?";
    const CJK: &[char] = &['你', '好', '世', '界', '測', '試', '語', '言'];
    const EMOJI: &[char] = &['😀', '🥹', '🚀', '✨', '🧠', '🔥'];
    const CYRILLIC: &[char] = &['Ж', 'Д', 'Й', 'Л', 'Ф', 'Я'];
    const ARABIC: &[char] = &['ا', 'ب', 'ت', 'ث', 'ج', 'ح'];

    let len = rng.random_range(1..=max_len);
    match rng.random_range(0..5) {
        0 => (0..len)
            .map(|_| ASCII[rng.random_range(0..ASCII.len())] as char)
            .collect(),
        1 => (0..len)
            .map(|_| CJK[rng.random_range(0..CJK.len())])
            .collect(),
        2 => (0..len)
            .map(|_| EMOJI[rng.random_range(0..EMOJI.len())])
            .collect(),
        3 => {
            let mut text = String::new();
            for _ in 0..len {
                match rng.random_range(0..4) {
                    0 => text.push(CYRILLIC[rng.random_range(0..CYRILLIC.len())]),
                    1 => text.push(ARABIC[rng.random_range(0..ARABIC.len())]),
                    2 => text.push(CJK[rng.random_range(0..CJK.len())]),
                    _ => text.push(ASCII[rng.random_range(0..ASCII.len())] as char),
                }
            }
            text
        }
        _ => {
            let weighted_ascii = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789    \n\t";
            (0..len)
                .map(|_| weighted_ascii[rng.random_range(0..weighted_ascii.len())] as char)
                .collect()
        }
    }
}

fn select_conversation_tokens(boundary_tokens: &[String]) -> Vec<String> {
    let conv_tokens: Vec<String> = boundary_tokens
        .iter()
        .filter(|token| {
            token.contains("User")
                || token.contains("Assistant")
                || token.contains("EOT")
                || token.to_lowercase().contains("begin")
                || token.to_lowercase().contains("end")
                || token.to_lowercase().contains("tool")
        })
        .cloned()
        .collect();

    if conv_tokens.is_empty() {
        boundary_tokens.iter().take(10).cloned().collect()
    } else {
        conv_tokens
    }
}

fn generate_conversation_segments(
    rng: &mut StdRng,
    boundary_tokens: &[String],
    size: &str,
) -> Vec<String> {
    let (max_turns, max_turn_len) = match size {
        "short" => (5, 200),
        "medium" => (10, 500),
        _ => (15, 2000),
    };

    let conv_tokens = select_conversation_tokens(boundary_tokens);
    let mut segments = Vec::new();
    let num_turns = rng.random_range(2..=max_turns);

    for _ in 0..num_turns {
        let turn_type = rng.random::<f64>();

        if turn_type < 0.15 && !conv_tokens.is_empty() {
            segments.push(conv_tokens[rng.random_range(0..conv_tokens.len())].clone());
        } else if turn_type < 0.30 && !conv_tokens.is_empty() {
            let token = &conv_tokens[rng.random_range(0..conv_tokens.len())];
            segments.push(format!(
                "{token}{}",
                random_text(rng, max_turn_len.min(200))
            ));
        } else if turn_type < 0.40 {
            segments.push(random_text(rng, max_turn_len.min(200)));
        } else if turn_type < 0.48 {
            const WHITESPACE_CASES: &[&str] = &["", " ", "\n", "\t", "  \n  "];
            segments.push(WHITESPACE_CASES[rng.random_range(0..WHITESPACE_CASES.len())].to_string());
        } else if turn_type < 0.58 && !boundary_tokens.is_empty() {
            let token = &boundary_tokens[rng.random_range(0..boundary_tokens.len())];
            segments.push(format!(
                "{}{token}{}",
                random_text(rng, 50),
                random_text(rng, 50)
            ));
        } else if turn_type < 0.75 && !conv_tokens.is_empty() {
            let token = &conv_tokens[rng.random_range(0..conv_tokens.len())];
            segments.push(format!("{token}{}", random_text(rng, max_turn_len)));
        } else if turn_type < 0.85 && conv_tokens.len() >= 2 {
            let first = &conv_tokens[rng.random_range(0..conv_tokens.len())];
            let second = &conv_tokens[rng.random_range(0..conv_tokens.len())];
            segments.push(format!(
                "{first}{}{second}{}",
                random_text(rng, max_turn_len.min(100)),
                random_text(rng, max_turn_len.min(100))
            ));
        } else {
            let turn_len = rng.random_range(1..=max_turn_len);
            segments.push(random_text(rng, turn_len));
        }
    }

    if !conv_tokens.is_empty() {
        segments.insert(0, conv_tokens[rng.random_range(0..conv_tokens.len())].clone());
    }
    segments.push(random_text(rng, 50));
    segments
}

fn assert_token_match(
    backend: &str,
    seed: u64,
    iteration: usize,
    text: &str,
    oracle_tokens: &[u32],
    cached_tokens: &[u32],
) {
    if oracle_tokens == cached_tokens {
        return;
    }

    let first_diff = oracle_tokens
        .iter()
        .zip(cached_tokens.iter())
        .position(|(oracle, cached)| oracle != cached)
        .unwrap_or_else(|| oracle_tokens.len().min(cached_tokens.len()));
    let preview: String = text.chars().take(200).collect();

    panic!(
        "{backend} tokenizer cache mismatch (seed={seed}, iteration={iteration}, text_len={}, first_diff={}, oracle_len={}, cached_len={})\npreview={preview:?}",
        text.len(),
        first_diff,
        oracle_tokens.len(),
        cached_tokens.len(),
    );
}

fn assert_stats(backend: &str, stats: CacheStats) {
    assert!(stats.hits_exact > 0, "{backend} fuzzer never exercised exact hits");
    assert!(stats.hits_prefix > 0, "{backend} fuzzer never exercised prefix hits");
    assert!(stats.misses > 0, "{backend} fuzzer never exercised misses");
    assert!(stats.put_truncated > 0, "{backend} fuzzer never exercised put_truncated");
    assert!(stats.put_full > 0, "{backend} fuzzer never exercised put_full");
    assert!(stats.put_skipped > 0, "{backend} fuzzer never exercised put_skipped");
}

#[test]
fn test_hf_tokenizer_cache_trt_style_fuzzer() {
    let mdc = load_mdc(HF_MODEL_DIR);
    let cache = TokenizerCache::from_model_card(&mdc).unwrap().unwrap();
    let tokenizer_path = sample_model_dir(HF_MODEL_DIR).join("tokenizer.json");
    let added_tokens = load_hf_added_token_contents(&tokenizer_path);
    let oracle_tokenizer = HfTokenizer::from_file(&tokenizer_path).unwrap();
    let mut rng = StdRng::seed_from_u64(HF_SEED);

    for iteration in 0..FUZZER_ITERATIONS {
        let size = match rng.random::<f64>() {
            r if r < 0.7 => "short",
            r if r < 0.9 => "medium",
            _ => "long",
        };

        let segments = generate_conversation_segments(&mut rng, &added_tokens, size);
        let mut text = String::new();

        for (segment_index, segment) in segments.into_iter().enumerate() {
            text.push_str(&segment);
            if text.is_empty() {
                continue;
            }

            let oracle = hf_oracle(&oracle_tokenizer, &text);
            let cached = cache.encode(&text).unwrap().token_ids().to_vec();
            assert_token_match("hf", HF_SEED, iteration, &text, &oracle, &cached);

            if segment_index == 0 {
                let cached_again = cache.encode(&text).unwrap().token_ids().to_vec();
                assert_token_match("hf", HF_SEED, iteration, &text, &oracle, &cached_again);
            }
        }

        if iteration % 50 == 0 {
            let plain = random_text(&mut rng, 200);
            let oracle = hf_oracle(&oracle_tokenizer, &plain);
            let cached = cache.encode(&plain).unwrap().token_ids().to_vec();
            assert_token_match("hf", HF_SEED, iteration, &plain, &oracle, &cached);
        }

        if !text.is_empty() {
            let oracle = hf_oracle(&oracle_tokenizer, &text);
            let cached = cache.encode(&text).unwrap().token_ids().to_vec();
            assert_token_match("hf", HF_SEED, iteration, &text, &oracle, &cached);
        }
    }

    assert_stats("hf", cache.stats());
}

#[test]
fn test_tiktoken_tokenizer_cache_trt_style_fuzzer() {
    let mdc = load_mdc(TIKTOKEN_MODEL_DIR);
    let cache = TokenizerCache::from_model_card(&mdc).unwrap().unwrap();
    let tokenizer_path = sample_model_dir(TIKTOKEN_MODEL_DIR).join("tiktoken.model");
    let boundary_tokens = load_tiktoken_special_token_contents(&tokenizer_path);
    let oracle_tokenizer = TikTokenTokenizer::from_file_auto(tokenizer_path.to_str().unwrap()).unwrap();
    let mut rng = StdRng::seed_from_u64(TIKTOKEN_SEED);

    for iteration in 0..FUZZER_ITERATIONS {
        let size = match rng.random::<f64>() {
            r if r < 0.7 => "short",
            r if r < 0.9 => "medium",
            _ => "long",
        };

        let segments = generate_conversation_segments(&mut rng, &boundary_tokens, size);
        let mut text = String::new();

        for (segment_index, segment) in segments.into_iter().enumerate() {
            text.push_str(&segment);
            if text.is_empty() {
                continue;
            }

            let oracle = tiktoken_oracle(&oracle_tokenizer, &text);
            let cached = cache.encode(&text).unwrap().token_ids().to_vec();
            assert_token_match(
                "tiktoken",
                TIKTOKEN_SEED,
                iteration,
                &text,
                &oracle,
                &cached,
            );

            if segment_index == 0 {
                let cached_again = cache.encode(&text).unwrap().token_ids().to_vec();
                assert_token_match(
                    "tiktoken",
                    TIKTOKEN_SEED,
                    iteration,
                    &text,
                    &oracle,
                    &cached_again,
                );
            }
        }

        if iteration % 50 == 0 {
            let plain = random_text(&mut rng, 200);
            let oracle = tiktoken_oracle(&oracle_tokenizer, &plain);
            let cached = cache.encode(&plain).unwrap().token_ids().to_vec();
            assert_token_match(
                "tiktoken",
                TIKTOKEN_SEED,
                iteration,
                &plain,
                &oracle,
                &cached,
            );
        }

        if !text.is_empty() {
            let oracle = tiktoken_oracle(&oracle_tokenizer, &text);
            let cached = cache.encode(&text).unwrap().token_ids().to_vec();
            assert_token_match(
                "tiktoken",
                TIKTOKEN_SEED,
                iteration,
                &text,
                &oracle,
                &cached,
            );
        }
    }

    assert_stats("tiktoken", cache.stats());
}
