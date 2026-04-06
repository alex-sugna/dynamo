// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::PromptFormatter;
use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;
use dynamo_llm::tokenizers::tiktoken::TikTokenTokenizer;
use dynamo_llm::tokenizers::traits::Encoder;

const HF_MODEL_DIR: &str = "tests/data/sample-models/TinyLlama_v1.1";
const TIKTOKEN_MODEL_DIR: &str = "tests/data/sample-models/mock-tiktoken";

const DEFAULT_TRIALS: usize = 20;
const HF_TOKEN_TARGETS: &[usize] = &[20_000, 50_000, 100_000, 150_000, 200_000];
const TIKTOKEN_TOKEN_TARGETS: &[usize] = &[20_000, 50_000, 100_000];

const SYSTEM_PROMPT: &str = concat!(
    "You are a helpful AI assistant. You answer questions accurately and concisely. ",
    "You can discuss a wide range of topics including science, technology, history, ",
    "and current events. Please be thorough in your responses."
);

const USER_TURNS: [&str; 16] = [
    "Can you explain quantum computing in detail? I'd like to understand the fundamentals and how it differs from classical computing.",
    "That's interesting. How does quantum entanglement relate to quantum error correction?",
    "What are the practical applications of this in cryptography and drug discovery?",
    "Could you provide some specific examples of companies working on quantum computing?",
    "I see. What about the potential downsides or limitations of current quantum hardware?",
    "How has the field evolved over the past decade? What breakthroughs have there been?",
    "What do experts currently think about the timeline for fault-tolerant quantum computing?",
    "Can you summarize the key points we've discussed about quantum computing?",
    "Let's switch topics. Tell me about the latest developments in large language models.",
    "How do transformer architectures work at a fundamental level?",
    "What are the key challenges in scaling these models to trillions of parameters?",
    "How does mixture of experts help with efficiency?",
    "What about inference optimization techniques like quantization and speculative decoding?",
    "Can you explain KV cache and why it matters for serving performance?",
    "What's the difference between prefill and decode phases?",
    "How does disaggregated serving work in practice?",
];

const ASSISTANT_TURNS: [&str; 16] = [
    "Great question! Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers. ",
    "Quantum entanglement is crucial for quantum error correction. When qubits are entangled, measuring one instantly reveals information about the other, regardless of distance. ",
    "The practical applications are vast. In cryptography, Shor's algorithm can factor large numbers exponentially faster, threatening RSA encryption. In drug discovery, quantum simulation can model molecular interactions. ",
    "Several major companies are leading the charge. IBM has their Condor processor with over 1000 qubits. Google achieved quantum supremacy with Sycamore. IonQ uses trapped ion technology. ",
    "Current quantum hardware faces significant challenges. Decoherence limits computation time. Error rates are still too high for many practical applications. Cryogenic cooling requirements make systems expensive. ",
    "The field has seen remarkable progress. In 2019, Google demonstrated quantum supremacy. IBM has steadily increased qubit counts. Error correction codes have improved dramatically. ",
    "Expert opinions vary widely. Some predict fault-tolerant quantum computing within a decade. Others are more conservative, citing the enormous engineering challenges that remain. ",
    "To summarize: quantum computing uses qubits that can exist in superposition, enabling parallel computation. Key challenges include decoherence, error rates, and scaling. ",
    "Large language models have seen explosive growth. GPT-4, Claude, Gemini, and Llama represent different approaches to scaling intelligence through massive pretraining on text data. ",
    "Transformers use self-attention to weigh the importance of different parts of the input. The key innovation is the attention mechanism: Q, K, V projections compute relevance scores. ",
    "Scaling challenges include memory bandwidth, communication overhead in distributed training, and the quadratic cost of attention with sequence length. ",
    "Mixture of Experts activates only a subset of parameters per token. This allows models to have many parameters while keeping compute constant per forward pass. ",
    "Quantization reduces precision from FP16 to INT8 or INT4, trading tiny accuracy loss for 2-4x memory savings. Speculative decoding uses a small model to draft tokens verified by the large model. ",
    "KV cache stores the key and value projections from previous tokens. Without it, each new token would require recomputing attention over the entire sequence. ",
    "Prefill processes all input tokens in parallel while decode generates one token at a time. They have very different hardware utilization profiles. ",
    "Disaggregated serving separates prefill and decode into different GPU pools. Prefill workers handle context processing while decode workers handle autoregressive generation. ",
];

#[derive(Clone, Copy, Debug)]
struct Percentiles {
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    mean: f64,
}

#[derive(Debug)]
struct BenchmarkResult {
    target_tokens: usize,
    actual_base_tokens: usize,
    actual_full_tokens: usize,
    num_turns: usize,
    text_chars: usize,
    uncached: Percentiles,
    cached: Percentiles,
    speedup_p50: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Hf,
    Tiktoken,
    Both,
}

fn sample_model_dir(model_dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(model_dir)
}

fn build_preprocessor(model_dir: &str) -> Arc<OpenAIPreprocessor> {
    let model_dir = sample_model_dir(model_dir);
    let mdc = ModelDeploymentCard::load_from_disk(&model_dir, None).unwrap();
    let tokenizer = mdc.tokenizer().unwrap();
    let formatter = match PromptFormatter::no_op() {
        PromptFormatter::OAI(formatter) => formatter,
    };
    OpenAIPreprocessor::new_with_parts(mdc, formatter, tokenizer).unwrap()
}

fn build_hf_conversation(num_turns: usize) -> String {
    let mut parts = vec![format!("<s>System\n{SYSTEM_PROMPT}</s>")];
    for i in 0..num_turns {
        parts.push(format!("<s>User\n{}</s>", USER_TURNS[i % USER_TURNS.len()]));
        parts.push(format!(
            "<s>Assistant\n{}</s>",
            ASSISTANT_TURNS[i % ASSISTANT_TURNS.len()].repeat(8)
        ));
    }
    parts.concat()
}

fn build_tiktoken_conversation(num_turns: usize) -> String {
    let mut parts = vec![format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")];
    for i in 0..num_turns {
        parts.push(format!(
            "<|im_start|>user\n{}<|im_end|>",
            USER_TURNS[i % USER_TURNS.len()]
        ));
        parts.push(format!(
            "<|im_start|>assistant\n{}<|im_end|>",
            ASSISTANT_TURNS[i % ASSISTANT_TURNS.len()].repeat(8)
        ));
    }
    parts.concat()
}

fn hf_token_count(tokenizer: &HuggingFaceTokenizer, text: &str) -> usize {
    tokenizer.encode(text).unwrap().token_ids().len()
}

fn tiktoken_token_count(tokenizer: &TikTokenTokenizer, text: &str) -> usize {
    tokenizer.encode(text).unwrap().token_ids().len()
}

fn find_turns_for_token_count<FBuild, FCount>(
    target_tokens: usize,
    max_turns: usize,
    build: FBuild,
    count_tokens: FCount,
) -> usize
where
    FBuild: Fn(usize) -> String,
    FCount: Fn(&str) -> usize,
{
    let mut lo = 1;
    let mut hi = max_turns;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if count_tokens(&build(mid)) < target_tokens {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn percentile(sorted_samples: &[f64], pct: f64) -> f64 {
    let n = sorted_samples.len();
    let index = ((n as f64) * pct).floor() as usize;
    sorted_samples[index.min(n - 1)]
}

fn summarize(samples: &[f64]) -> Percentiles {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean = sorted.iter().sum::<f64>() / (sorted.len() as f64);
    Percentiles {
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
        mean,
    }
}

fn format_seconds(seconds: f64) -> String {
    if seconds < 0.001 {
        format!("{:.0}us", seconds * 1_000_000.0)
    } else if seconds < 1.0 {
        format!("{:.2}ms", seconds * 1000.0)
    } else {
        format!("{seconds:.3}s")
    }
}

fn print_results(label: &str, results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(85));
    println!(
        "{:>8}  {:>10}  {:>14}  {:>14}  {:>9}",
        "Tokens", "Chars", "Uncached p50", "Cached p50", "Speedup"
    );
    println!("{}", "-".repeat(60));
    for result in results {
        println!(
            "{:>8}  {:>10}  {:>14}  {:>14}  {:>8.1}x",
            result.actual_full_tokens,
            result.text_chars,
            format_seconds(result.uncached.p50),
            format_seconds(result.cached.p50),
            result.speedup_p50,
        );
    }

    println!("\n{}", "=".repeat(85));
    println!("{label} detailed stats:");
    for result in results {
        println!(
            "\n  {} tokens (target {}, base {}, {} turns, {} chars):",
            result.actual_full_tokens,
            result.target_tokens,
            result.actual_base_tokens,
            result.num_turns,
            result.text_chars
        );
        println!(
            "    Uncached: p50={}  p90={}  p95={}  p99={}  mean={}",
            format_seconds(result.uncached.p50),
            format_seconds(result.uncached.p90),
            format_seconds(result.uncached.p95),
            format_seconds(result.uncached.p99),
            format_seconds(result.uncached.mean),
        );
        println!(
            "    Cached:   p50={}  p90={}  p95={}  p99={}  mean={}",
            format_seconds(result.cached.p50),
            format_seconds(result.cached.p90),
            format_seconds(result.cached.p95),
            format_seconds(result.cached.p99),
            format_seconds(result.cached.mean),
        );
        println!("    Speedup:  {:.1}x", result.speedup_p50);
    }
}

fn run_hf_benchmark(num_trials: usize) {
    let tokenizer_path = sample_model_dir(HF_MODEL_DIR).join("tokenizer.json");
    let direct = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();

    println!("HF tokenizer: {}", tokenizer_path.display());
    println!("Benchmark: incremental single-turn encoding (warm cache + append one turn)");
    println!("{}", "=".repeat(85));

    let mut results = Vec::new();
    for &target_tokens in HF_TOKEN_TARGETS {
        print!("\n  Benchmarking ~{}K tokens...", target_tokens / 1000);
        let num_turns = find_turns_for_token_count(
            target_tokens,
            500,
            build_hf_conversation,
            |text| hf_token_count(&direct, text),
        );
        let base_text = build_hf_conversation(num_turns);
        let full_text = build_hf_conversation(num_turns + 1);
        let actual_base_tokens = hf_token_count(&direct, &base_text);
        let actual_full_tokens = hf_token_count(&direct, &full_text);

        let mut uncached_times = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            let start = Instant::now();
            let _ = direct.encode(black_box(&full_text)).unwrap();
            uncached_times.push(start.elapsed().as_secs_f64());
        }

        let mut cached_times = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            let preprocessor = build_preprocessor(HF_MODEL_DIR);
            let _ = preprocessor.tokenize(&base_text).unwrap();
            let start = Instant::now();
            let _ = preprocessor.tokenize(black_box(&full_text)).unwrap();
            cached_times.push(start.elapsed().as_secs_f64());
        }

        let uncached = summarize(&uncached_times);
        let cached = summarize(&cached_times);
        let result = BenchmarkResult {
            target_tokens,
            actual_base_tokens,
            actual_full_tokens,
            num_turns,
            text_chars: full_text.len(),
            uncached,
            cached,
            speedup_p50: uncached.p50 / cached.p50,
        };
        println!(
            " {} tokens, uncached={}, cached={}, speedup={:.1}x",
            result.actual_full_tokens,
            format_seconds(result.uncached.p50),
            format_seconds(result.cached.p50),
            result.speedup_p50
        );
        results.push(result);
    }

    print_results("HF", &results);
}

fn run_tiktoken_benchmark(num_trials: usize) {
    let tokenizer_path = sample_model_dir(TIKTOKEN_MODEL_DIR).join("tiktoken.model");
    let direct = TikTokenTokenizer::from_file_auto(tokenizer_path.to_str().unwrap()).unwrap();

    println!("Tiktoken model: {}", tokenizer_path.display());
    println!("Benchmark: incremental single-turn encoding (warm cache + append one turn)");
    println!("{}", "=".repeat(85));

    let mut results = Vec::new();
    for &target_tokens in TIKTOKEN_TOKEN_TARGETS {
        print!("\n  Benchmarking ~{}K tokens...", target_tokens / 1000);
        let num_turns = find_turns_for_token_count(
            target_tokens,
            500,
            build_tiktoken_conversation,
            |text| tiktoken_token_count(&direct, text),
        );
        let base_text = build_tiktoken_conversation(num_turns);
        let full_text = build_tiktoken_conversation(num_turns + 1);
        let actual_base_tokens = tiktoken_token_count(&direct, &base_text);
        let actual_full_tokens = tiktoken_token_count(&direct, &full_text);

        let mut uncached_times = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            let start = Instant::now();
            let _ = direct.encode(black_box(&full_text)).unwrap();
            uncached_times.push(start.elapsed().as_secs_f64());
        }

        let mut cached_times = Vec::with_capacity(num_trials);
        for _ in 0..num_trials {
            let preprocessor = build_preprocessor(TIKTOKEN_MODEL_DIR);
            let _ = preprocessor.tokenize(&base_text).unwrap();
            let start = Instant::now();
            let _ = preprocessor.tokenize(black_box(&full_text)).unwrap();
            cached_times.push(start.elapsed().as_secs_f64());
        }

        let uncached = summarize(&uncached_times);
        let cached = summarize(&cached_times);
        let result = BenchmarkResult {
            target_tokens,
            actual_base_tokens,
            actual_full_tokens,
            num_turns,
            text_chars: full_text.len(),
            uncached,
            cached,
            speedup_p50: uncached.p50 / cached.p50,
        };
        println!(
            " {} tokens, uncached={}, cached={}, speedup={:.1}x",
            result.actual_full_tokens,
            format_seconds(result.uncached.p50),
            format_seconds(result.cached.p50),
            result.speedup_p50
        );
        results.push(result);
    }

    print_results("Tiktoken", &results);
}

fn parse_args() -> (Mode, usize) {
    let mut mode = Mode::Both;
    let mut trials = DEFAULT_TRIALS;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--hf" => mode = Mode::Hf,
            "--tiktoken" => mode = Mode::Tiktoken,
            "--both" => mode = Mode::Both,
            "--trials" => {
                let value = args.next().expect("--trials requires a value");
                trials = value.parse().expect("--trials must be an integer");
            }
            "--help" | "-h" => {
                println!("Usage: cargo bench -p dynamo-llm --bench tokenizer_cache -- [--hf|--tiktoken|--both] [--trials N]");
                std::process::exit(0);
            }
            other if other.starts_with('-') => panic!("Unknown argument: {other}"),
            _ => {}
        }
    }

    (mode, trials)
}

fn main() {
    let (mode, trials) = parse_args();
    match mode {
        Mode::Hf => run_hf_benchmark(trials),
        Mode::Tiktoken => run_tiktoken_benchmark(trials),
        Mode::Both => {
            run_hf_benchmark(trials);
            println!();
            run_tiktoken_benchmark(trials);
        }
    }
}
