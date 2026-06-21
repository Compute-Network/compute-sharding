use anyhow::Result;
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct PrefixCase {
    name: &'static str,
    seed_prompt: &'static str,
    probe_prompt: &'static str,
    unrelated_prompt: &'static str,
}

#[derive(Debug, Clone)]
struct GenerationRun {
    prompt_tokens: usize,
    ttft_ms: u128,
    total_ms: u128,
    finish_reason: String,
    text: String,
    token_ids: Vec<u32>,
}

const DEFAULT_CASES: &[PrefixCase] = &[
    PrefixCase {
        name: "short",
        seed_prompt: "The capital of France is",
        probe_prompt: "The capital of France was",
        unrelated_prompt: "Birds can fly because",
    },
    PrefixCase {
        name: "medium",
        seed_prompt: "In one sentence, explain why the sky looks blue during the day.",
        probe_prompt: "In one sentence, explain why the sky looks red at sunset.",
        unrelated_prompt: "In one sentence, explain why leaves turn yellow in autumn.",
    },
    PrefixCase {
        name: "long",
        seed_prompt: "Answer in one concise sentence. A user is benchmarking a staged local language model and wants to know why continuation tokens can be faster than the first generated token when per-request decode caches are reused across steps. Explain the main reason.",
        probe_prompt: "Answer in one concise sentence. A user is benchmarking a staged local language model and wants to know why continuation tokens can be faster than the first generated token when per-request decode caches are reused across steps. Summarize the main reason.",
        unrelated_prompt: "Answer in one concise sentence. A user is benchmarking a staged local language model and wants to know how quantized FFN kernels affect local inference speed. Summarize the main reason.",
    },
];

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let stage1_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json")
    });
    let stage2_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json")
    });
    let vocab_path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"));
    let max_tokens = args
        .get(4)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    let layer_cap = args.get(5).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(6).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(7)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let prompt_mode = args
        .get(8)
        .and_then(|value| GemmaPromptMode::parse(value))
        .unwrap_or_default();

    println!("=== Real 2-Stage Gemma Prefix Bench ===");
    println!("stage 1    : {}", stage1_path.display());
    println!("stage 2    : {}", stage2_path.display());
    println!("max tokens : {}", max_tokens);
    println!("layer cap  : {:?}", layer_cap);
    println!("vocab cap  : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!("prompt mode: {}", prompt_mode.as_str());
    println!();

    for case in DEFAULT_CASES {
        println!("=== Case: {} ===", case.name);
        println!("seed      : {:?}", case.seed_prompt);
        println!("probe     : {:?}", case.probe_prompt);
        println!("unrelated : {:?}", case.unrelated_prompt);

        let (head_unrelated, tail_unrelated) = load_backend_pair(
            &stage1_path,
            &stage2_path,
            &vocab_path,
            layer_cap,
            vocab_cap,
            disable_ple,
        )?;
        let _ = run_once(
            &head_unrelated,
            &tail_unrelated,
            case.unrelated_prompt,
            max_tokens,
            prompt_mode,
            &format!("{}-unrelated-seed", case.name),
        )?;
        let unrelated = run_once(
            &head_unrelated,
            &tail_unrelated,
            case.probe_prompt,
            max_tokens,
            prompt_mode,
            &format!("{}-unrelated-probe", case.name),
        )?;

        let (head_shared, tail_shared) = load_backend_pair(
            &stage1_path,
            &stage2_path,
            &vocab_path,
            layer_cap,
            vocab_cap,
            disable_ple,
        )?;
        let _ = run_once(
            &head_shared,
            &tail_shared,
            case.seed_prompt,
            max_tokens,
            prompt_mode,
            &format!("{}-shared-seed", case.name),
        )?;
        let shared = run_once(
            &head_shared,
            &tail_shared,
            case.probe_prompt,
            max_tokens,
            prompt_mode,
            &format!("{}-shared-probe", case.name),
        )?;

        println!("probe toks : {}", shared.prompt_tokens);
        println!(
            "unrelated  : ttft={}ms total={}ms finish={} text={:?} token_ids={:?}",
            unrelated.ttft_ms,
            unrelated.total_ms,
            unrelated.finish_reason,
            unrelated.text,
            unrelated.token_ids
        );
        println!(
            "shared     : ttft={}ms total={}ms finish={} text={:?} token_ids={:?}",
            shared.ttft_ms, shared.total_ms, shared.finish_reason, shared.text, shared.token_ids
        );
        println!();
    }

    Ok(())
}

fn load_backend_pair(
    stage1_path: &Path,
    stage2_path: &Path,
    vocab_path: &Path,
    layer_cap: Option<usize>,
    vocab_cap: Option<usize>,
    disable_ple: bool,
) -> Result<(RealGemmaBackend, RealGemmaBackend)> {
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    let mut head = RealGemmaBackend::new(stage1_path);
    head.set_debug_layer_cap(layer_cap);
    head.set_debug_vocab_cap(vocab_cap);
    head.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = RealGemmaBackend::new(stage2_path);
    tail.set_debug_layer_cap(layer_cap);
    tail.set_debug_vocab_cap(vocab_cap);
    tail.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(vocab_path, sp)?;
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    Ok((head, tail))
}

fn run_once(
    head: &RealGemmaBackend,
    tail: &RealGemmaBackend,
    prompt: &str,
    max_tokens: u32,
    prompt_mode: GemmaPromptMode,
    request_id: &str,
) -> Result<GenerationRun> {
    let eos_token_id = tail.eos_token_id().or_else(|| head.eos_token_id());
    let mut prompt_token_ids = head.tokenize_prompt_mode(prompt, prompt_mode);
    let prompt_tokens = prompt_token_ids.len();
    let mut generated_token_ids = Vec::with_capacity(max_tokens as usize);
    let mut finish_reason = "length".to_string();
    let total_start = Instant::now();
    let mut ttft_ms = 0u128;
    let mut text = String::new();

    for step in 0..max_tokens as usize {
        let step_start = Instant::now();
        let step_token_ids: Vec<u32> = if step == 0 {
            prompt_token_ids.clone()
        } else {
            vec![*generated_token_ids.last().unwrap()]
        };
        let head_output = head.begin_token_ids(request_id, &step_token_ids, Some(1), 0)?;
        let tail_output = tail.continue_forward(head_output)?;
        let sample = tail.sample_tail(tail_output)?;
        let step_ms = step_start.elapsed().as_millis();

        let Some(&next_token_id) = sample.token_ids.first() else {
            break;
        };

        if ttft_ms == 0 {
            ttft_ms = step_ms;
        }

        if eos_token_id == Some(next_token_id) {
            finish_reason = "stop".to_string();
            break;
        }

        generated_token_ids.push(next_token_id);
        prompt_token_ids.push(next_token_id);
        text = tail.decode_token_ids(&generated_token_ids);
    }

    let total_ms = total_start.elapsed().as_millis();
    if text.is_empty() && !generated_token_ids.is_empty() {
        text = tail.decode_token_ids(&generated_token_ids);
    }

    head.clear_decode_session(request_id);
    tail.clear_decode_session(request_id);

    Ok(GenerationRun {
        prompt_tokens,
        ttft_ms,
        total_ms,
        finish_reason,
        text,
        token_ids: generated_token_ids,
    })
}
