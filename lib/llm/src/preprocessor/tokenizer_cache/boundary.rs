// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub(crate) fn discover_boundary_tokens(tokenizer_json_path: &Path) -> Result<HashSet<u32>, String> {
    let content = std::fs::read_to_string(tokenizer_json_path)
        .map_err(|error| format!("Failed to read tokenizer.json: {error}"))?;

    let parsed: serde_json::Value =
        serde_json::from_str(&content).map_err(|error| format!("Failed to parse JSON: {error}"))?;

    let Some(added_tokens) = parsed.get("added_tokens").and_then(|value| value.as_array()) else {
        return Ok(HashSet::new());
    };

    let tokenizer = HfTokenizer::from_file(tokenizer_json_path)
        .map_err(|error| format!("Failed to load tokenizer.json: {error}"))?;

    let mut boundary_ids = HashSet::new();
    for token in added_tokens {
        let Some(content) = token.get("content").and_then(|value| value.as_str()) else {
            continue;
        };
        let encoding = tokenizer
            .encode(content, false)
            .map_err(|error| format!("Failed to encode added token {content:?}: {error}"))?;
        if encoding.get_ids().len() == 1 {
            boundary_ids.insert(encoding.get_ids()[0]);
        }
    }

    Ok(boundary_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_discover_boundary_tokens_reads_tokenizer_ids() {
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct/tokenizer.json");
        let boundary_ids = discover_boundary_tokens(&tokenizer_path).unwrap();
        assert!(boundary_ids.contains(&0));
        assert!(boundary_ids.contains(&6));
        assert!(boundary_ids.contains(&7));
        assert!(boundary_ids.contains(&9));
    }

    #[test]
    fn test_discover_boundary_tokens_returns_empty_without_added_tokens() {
        let dir = tempdir().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        fs::write(&tokenizer_path, r#"{"model": {"vocab": {}}}"#).unwrap();

        let boundary_ids = discover_boundary_tokens(&tokenizer_path).unwrap();
        assert!(boundary_ids.is_empty());
    }

    #[test]
    fn test_missing_file_is_error() {
        let result = discover_boundary_tokens(Path::new("/nonexistent/tokenizer.json"));
        assert!(result.is_err());
    }
}
