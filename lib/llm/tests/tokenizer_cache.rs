// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::PromptFormatter;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::tokenizers::tiktoken::TikTokenTokenizer;
use dynamo_llm::tokenizers::traits::Encoder;
use tokenizers::Tokenizer as HfTokenizer;

const HF_MODEL_DIR: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
const TIKTOKEN_MODEL_DIR: &str = "tests/data/sample-models/mock-tiktoken";

fn sample_model_dir(model_dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(model_dir)
}

fn make_chat_request(messages: &str, model: &str) -> NvCreateChatCompletionRequest {
    let messages: Vec<dynamo_async_openai::types::ChatCompletionRequestMessage> =
        serde_json::from_str(messages).unwrap();
    let inner = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .build()
        .unwrap();

    NvCreateChatCompletionRequest {
        inner,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        request_id: None,
        rid: None,
        media_io_kwargs: None,
        unsupported_fields: Default::default(),
    }
}

fn build_tiktoken_preprocessor() -> (Arc<OpenAIPreprocessor>, PathBuf) {
    let model_dir = sample_model_dir(TIKTOKEN_MODEL_DIR);
    let mdc = ModelDeploymentCard::load_from_disk(&model_dir, None).unwrap();
    let tokenizer = mdc.tokenizer().unwrap();
    let formatter = match PromptFormatter::no_op() {
        PromptFormatter::OAI(formatter) => formatter,
    };
    (
        OpenAIPreprocessor::new_with_parts(mdc, formatter, tokenizer).unwrap(),
        model_dir,
    )
}

#[tokio::test]
async fn test_hf_preprocessor_tokenizer_cache_matches_direct_tokenization() {
    let model_dir = sample_model_dir(HF_MODEL_DIR);
    let mdc = ModelDeploymentCard::load_from_disk(&model_dir, None).unwrap();
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).unwrap();
    let direct_tokenizer = HfTokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let request1 = make_chat_request(
        r#"[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain tokenizer cache."}
        ]"#,
        "test-model",
    );
    let prompt1 = preprocessor.apply_template(&request1).unwrap().unwrap();
    let (preprocessed1, _) = preprocessor.preprocess_request(&request1, None).await.unwrap();
    let oracle1 = direct_tokenizer
        .encode(prompt1.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(preprocessed1.token_ids, oracle1);

    let request2 = make_chat_request(
        r#"[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain tokenizer cache."},
            {"role": "assistant", "content": "It reuses previously tokenized prefixes."},
            {"role": "user", "content": "Why does that help TTFT?"}
        ]"#,
        "test-model",
    );
    let prompt2 = preprocessor.apply_template(&request2).unwrap().unwrap();
    let (preprocessed2, _) = preprocessor.preprocess_request(&request2, None).await.unwrap();
    let oracle2 = direct_tokenizer
        .encode(prompt2.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(preprocessed2.token_ids, oracle2);

    let (preprocessed2_repeat, _) = preprocessor.preprocess_request(&request2, None).await.unwrap();
    assert_eq!(preprocessed2_repeat.token_ids, oracle2);
}

#[test]
fn test_tiktoken_preprocessor_tokenizer_cache_matches_direct_tokenization() {
    let (preprocessor, model_dir) = build_tiktoken_preprocessor();
    let tokenizer_path = model_dir.join("tiktoken.model");
    let direct_tokenizer = TikTokenTokenizer::from_file_auto(tokenizer_path.to_str().unwrap()).unwrap();

    let base_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nExplain tokenizer cache.<|im_end|>";
    let full_text = format!(
        "{base_text}<|im_start|>assistant\nIt reuses tokenized prefixes.<|im_end|><|im_start|>user\nWhy does that reduce latency?<|im_end|>"
    );

    let base_encoding = preprocessor.tokenize(base_text).unwrap();
    let base_oracle = direct_tokenizer.encode(base_text).unwrap();
    assert_eq!(base_encoding.token_ids(), base_oracle.token_ids());

    let full_encoding = preprocessor.tokenize(&full_text).unwrap();
    let full_oracle = direct_tokenizer.encode(&full_text).unwrap();
    assert_eq!(full_encoding.token_ids(), full_oracle.token_ids());

    let full_repeat = preprocessor.tokenize(&full_text).unwrap();
    assert_eq!(full_repeat.token_ids(), full_oracle.token_ids());
}
