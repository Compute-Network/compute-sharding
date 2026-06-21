use anyhow::Result;
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::{RealForwardProfile, RealGemmaBackend};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct PromptCase {
    name: &'static str,
    prompt: &'static str,
}

#[derive(Debug, Clone)]
struct GenerationRun {
    prompt_tokens: usize,
    ttft_ms: u128,
    ttft_head_ms: u128,
    ttft_tail_ms: u128,
    ttft_sample_ms: u128,
    total_ms: u128,
    continuation_tok_s: f64,
    finish_reason: String,
    text: String,
    token_ids: Vec<u32>,
    first_head_profile: Option<RealForwardProfile>,
    first_tail_profile: Option<RealForwardProfile>,
}

const DEFAULT_CASES: &[PromptCase] = &[
    PromptCase {
        name: "short",
        prompt: "Reply with one word. What is the capital of France?",
    },
    PromptCase {
        name: "medium",
        prompt: "In one sentence, explain why the sky looks blue during the day.",
    },
    PromptCase {
        name: "long",
        prompt: "Answer in one concise sentence. A user is benchmarking a staged local language model and wants to know why continuation tokens can be faster than the first generated token when per-request decode caches are reused across steps. Explain the main reason.",
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
        .unwrap_or(4)
        .max(1);
    let warm_runs = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let layer_cap = args.get(6).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(7).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(8)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let prompt_mode = args
        .get(9)
        .and_then(|value| GemmaPromptMode::parse(value))
        .unwrap_or_default();
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real 2-Stage Gemma Generate Sweep ===");
    println!("stage 1    : {}", stage1_path.display());
    println!("stage 2    : {}", stage2_path.display());
    println!("max tokens : {}", max_tokens);
    println!("warm runs  : {}", warm_runs);
    println!("layer cap  : {:?}", layer_cap);
    println!("vocab cap  : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!("prompt mode: {}", prompt_mode.as_str());
    println!();

    let mut head = RealGemmaBackend::new(&stage1_path);
    head.set_debug_layer_cap(layer_cap);
    head.set_debug_vocab_cap(vocab_cap);
    head.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        head.load_tokenizer(&vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = RealGemmaBackend::new(&stage2_path);
    tail.set_debug_layer_cap(layer_cap);
    tail.set_debug_vocab_cap(vocab_cap);
    tail.set_debug_disable_ple(disable_ple);
    if vocab_path.exists() {
        let sp = if scores_path.exists() {
            Some(scores_path.as_path())
        } else {
            None
        };
        tail.load_tokenizer(&vocab_path, sp)?;
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    for case in DEFAULT_CASES {
        println!("=== Case: {} ===", case.name);
        println!("prompt       : {:?}", case.prompt);
        let cold = run_once(
            &head,
            &tail,
            case.prompt,
            max_tokens,
            prompt_mode,
            &format!("gen-sweep-{}-cold", case.name),
        )?;
        println!("cold         :");
        print_run(&cold);
        print_profile_once("head cold", cold.first_head_profile.as_ref());
        print_profile_once("tail cold", cold.first_tail_profile.as_ref());

        let mut warm = Vec::with_capacity(warm_runs);
        for idx in 0..warm_runs {
            warm.push(run_once(
                &head,
                &tail,
                case.prompt,
                max_tokens,
                prompt_mode,
                &format!("gen-sweep-{}-warm-{idx}", case.name),
            )?);
        }

        if warm.is_empty() {
            println!();
            continue;
        }

        println!("warm         :");
        print_ms_series("ttft", warm.iter().map(|run| run.ttft_ms).collect());
        print_ms_series(
            "ttft head",
            warm.iter().map(|run| run.ttft_head_ms).collect(),
        );
        print_ms_series(
            "ttft tail",
            warm.iter().map(|run| run.ttft_tail_ms).collect(),
        );
        print_ms_series(
            "ttft sample",
            warm.iter().map(|run| run.ttft_sample_ms).collect(),
        );
        print_ms_series("total", warm.iter().map(|run| run.total_ms).collect());
        print_f64_series(
            "cont tok/s",
            warm.iter().map(|run| run.continuation_tok_s).collect(),
        );
        let first = &warm[0];
        let deterministic = warm.iter().all(|run| {
            run.finish_reason == first.finish_reason
                && run.text == first.text
                && run.token_ids == first.token_ids
        });
        println!("prompt toks  : {}", first.prompt_tokens);
        println!("finish       : {}", first.finish_reason);
        println!(
            "deterministic: {}",
            if deterministic { "PASS" } else { "FAIL" }
        );
        println!("text         : {:?}", first.text);
        println!("token ids    : {:?}", first.token_ids);
        print_profile_avg(
            "head ttft",
            warm.iter()
                .filter_map(|run| run.first_head_profile.as_ref())
                .collect(),
        );
        print_profile_avg(
            "tail ttft",
            warm.iter()
                .filter_map(|run| run.first_tail_profile.as_ref())
                .collect(),
        );
        println!();
    }

    Ok(())
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
    let mut ttft_head_ms = 0u128;
    let mut ttft_tail_ms = 0u128;
    let mut ttft_sample_ms = 0u128;
    let mut text = String::new();
    let mut first_head_profile = None;
    let mut first_tail_profile = None;

    for step in 0..max_tokens as usize {
        let step_token_ids: Vec<u32> = if step == 0 {
            prompt_token_ids.clone()
        } else {
            vec![*generated_token_ids.last().unwrap()]
        };
        let head_start = Instant::now();
        let head_output = head.begin_token_ids(request_id, &step_token_ids, Some(1), 0)?;
        let head_ms = head_start.elapsed().as_millis();
        let head_profile = head.last_forward_profile();

        let tail_start = Instant::now();
        let tail_output = tail.continue_forward(head_output)?;
        let tail_ms = tail_start.elapsed().as_millis();
        let tail_profile = tail.last_forward_profile();

        let sample_start = Instant::now();
        let sample = tail.sample_tail(tail_output)?;
        let sample_ms = sample_start.elapsed().as_millis();
        let step_ms = head_ms + tail_ms + sample_ms;

        let Some(&next_token_id) = sample.token_ids.first() else {
            break;
        };

        if ttft_ms == 0 {
            ttft_ms = step_ms;
            ttft_head_ms = head_ms;
            ttft_tail_ms = tail_ms;
            ttft_sample_ms = sample_ms;
            first_head_profile = head_profile;
            first_tail_profile = tail_profile;
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
    let continuation_tokens = generated_token_ids.len().saturating_sub(1);
    let continuation_ms = total_ms.saturating_sub(ttft_ms);
    let continuation_tok_s = if continuation_tokens == 0 || continuation_ms == 0 {
        0.0
    } else {
        continuation_tokens as f64 / (continuation_ms as f64 / 1_000.0)
    };

    head.clear_decode_session(request_id);
    tail.clear_decode_session(request_id);

    Ok(GenerationRun {
        prompt_tokens,
        ttft_ms,
        ttft_head_ms,
        ttft_tail_ms,
        ttft_sample_ms,
        total_ms,
        continuation_tok_s,
        finish_reason,
        text,
        token_ids: generated_token_ids,
        first_head_profile,
        first_tail_profile,
    })
}

fn print_run(run: &GenerationRun) {
    println!("  prompt toks: {}", run.prompt_tokens);
    println!("  ttft       : {}ms", run.ttft_ms);
    println!(
        "  ttft split : head={}ms tail={}ms sample={}ms",
        run.ttft_head_ms, run.ttft_tail_ms, run.ttft_sample_ms
    );
    println!("  total      : {}ms", run.total_ms);
    println!("  cont tok/s : {:.2}", run.continuation_tok_s);
    println!("  finish     : {}", run.finish_reason);
    println!("  text       : {:?}", run.text);
    println!("  token ids  : {:?}", run.token_ids);
}

fn print_ms_series(label: &str, mut values: Vec<u128>) {
    values.sort_unstable();
    let min = values.first().copied().unwrap_or(0);
    let max = values.last().copied().unwrap_or(0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<u128>() / values.len() as u128;
    println!("  {label:<11}: min={min}ms median={median}ms avg={avg}ms max={max}ms");
}

fn print_f64_series(label: &str, mut values: Vec<f64>) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(0.0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    println!("  {label:<11}: min={min:.2} median={median:.2} avg={avg:.2} max={max:.2}");
}

fn print_profile_avg(label: &str, profiles: Vec<&RealForwardProfile>) {
    if profiles.is_empty() {
        println!("  {label:<11}: <none>");
        return;
    }
    let len = profiles.len() as u128;
    let avg =
        |f: fn(&RealForwardProfile) -> u128| profiles.iter().map(|p| f(p)).sum::<u128>() / len;
    println!(
        "  {label:<11}: attn={}ms (qkv={}ms core={}ms out={}ms) ffn={}ms (gate+up={}ms down={}ms) ple={}ms",
        avg(|p| p.attn_micros) / 1_000,
        avg(|p| p.attn_qkv_micros) / 1_000,
        avg(|p| p.attn_core_micros) / 1_000,
        avg(|p| p.attn_out_micros) / 1_000,
        avg(|p| p.ffn_micros) / 1_000,
        avg(|p| p.ffn_gate_up_micros) / 1_000,
        avg(|p| p.ffn_down_micros) / 1_000,
        avg(|p| p.ple_micros) / 1_000,
    );
}

fn print_profile_once(label: &str, profile: Option<&RealForwardProfile>) {
    let Some(profile) = profile else {
        println!("  {label:<11}: <none>");
        return;
    };
    println!(
        "  {label:<11}: attn={}ms (qkv={}ms core={}ms out={}ms) ffn={}ms (gate+up={}ms down={}ms) ple={}ms",
        profile.attn_micros / 1_000,
        profile.attn_qkv_micros / 1_000,
        profile.attn_core_micros / 1_000,
        profile.attn_out_micros / 1_000,
        profile.ffn_micros / 1_000,
        profile.ffn_gate_up_micros / 1_000,
        profile.ffn_down_micros / 1_000,
        profile.ple_micros / 1_000,
    );
}
