use anyhow::Result;
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::{RealForwardProfile, RealGemmaBackend};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct GenerationRun {
    ttft_ms: u128,
    total_ms: u128,
    continuation_tok_s: f64,
    continuation_head_ms: Vec<u128>,
    continuation_tail_ms: Vec<u128>,
    continuation_sample_ms: Vec<u128>,
    continuation_head_profiles: Vec<RealForwardProfile>,
    continuation_tail_profiles: Vec<RealForwardProfile>,
    finish_reason: String,
    text: String,
    token_ids: Vec<u32>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let stage1_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json")
    });
    let stage2_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json")
    });
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Hello".to_string());
    let vocab_path = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab.json"));
    let max_tokens = args
        .get(5)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(16)
        .max(1);
    let warm_runs = args
        .get(6)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    let layer_cap = args.get(7).and_then(|value| value.parse::<usize>().ok());
    let vocab_cap = args.get(8).and_then(|value| value.parse::<usize>().ok());
    let disable_ple = args
        .get(9)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "yes" | "disable_ple" | "no_ple"
            )
        })
        .unwrap_or(false);
    let stop_sequences = args
        .get(10)
        .map(|value| parse_stop_sequences(value))
        .unwrap_or_default();
    let prompt_mode = args
        .get(11)
        .and_then(|value| GemmaPromptMode::parse(value))
        .unwrap_or_default();
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real 2-Stage Gemma Generate Bench ===");
    println!("stage 1    : {}", stage1_path.display());
    println!("stage 2    : {}", stage2_path.display());
    println!("prompt     : {:?}", prompt);
    println!("max tokens : {}", max_tokens);
    println!("warm runs  : {}", warm_runs);
    println!("layer cap  : {:?}", layer_cap);
    println!("vocab cap  : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!("stop seqs  : {:?}", stop_sequences);
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

    let cold = run_once(
        &head,
        &tail,
        &prompt,
        max_tokens,
        &stop_sequences,
        prompt_mode,
        "gen-bench-cold",
    )?;
    println!("=== Cold Run ===");
    print_run(&cold);
    println!();

    let mut warm = Vec::with_capacity(warm_runs);
    for idx in 0..warm_runs {
        warm.push(run_once(
            &head,
            &tail,
            &prompt,
            max_tokens,
            &stop_sequences,
            prompt_mode,
            &format!("gen-bench-warm-{idx}"),
        )?);
    }

    if warm.is_empty() {
        return Ok(());
    }

    println!("=== Warm Summary ===");
    print_ms_series("ttft", warm.iter().map(|run| run.ttft_ms).collect());
    print_ms_series("total", warm.iter().map(|run| run.total_ms).collect());
    print_f64_series(
        "cont tok/s",
        warm.iter().map(|run| run.continuation_tok_s).collect(),
    );
    print_ms_series(
        "cont head",
        warm.iter()
            .map(|run| avg_u128(&run.continuation_head_ms))
            .collect(),
    );
    print_ms_series(
        "cont tail",
        warm.iter()
            .map(|run| avg_u128(&run.continuation_tail_ms))
            .collect(),
    );
    print_ms_series(
        "cont sample",
        warm.iter()
            .map(|run| avg_u128(&run.continuation_sample_ms))
            .collect(),
    );
    print_profile_avg(
        "head cont",
        warm.iter()
            .flat_map(|run| run.continuation_head_profiles.iter())
            .collect(),
    );
    print_profile_avg(
        "tail cont",
        warm.iter()
            .flat_map(|run| run.continuation_tail_profiles.iter())
            .collect(),
    );

    let first = &warm[0];
    let deterministic = warm.iter().all(|run| {
        run.finish_reason == first.finish_reason
            && run.text == first.text
            && run.token_ids == first.token_ids
    });
    println!(
        "deterministic : {}",
        if deterministic { "PASS" } else { "FAIL" }
    );
    println!("finish reason : {}", first.finish_reason);
    println!("text          : {:?}", first.text);
    println!("token ids     : {:?}", first.token_ids);

    Ok(())
}

fn run_once(
    head: &RealGemmaBackend,
    tail: &RealGemmaBackend,
    prompt: &str,
    max_tokens: u32,
    stop_sequences: &[String],
    prompt_mode: GemmaPromptMode,
    request_id: &str,
) -> Result<GenerationRun> {
    let eos_token_id = tail.eos_token_id().or_else(|| head.eos_token_id());
    let mut prompt_token_ids = head.tokenize_prompt_mode(prompt, prompt_mode);
    let mut generated_token_ids = Vec::with_capacity(max_tokens as usize);
    let mut finish_reason = "length".to_string();
    let total_start = Instant::now();
    let mut ttft_ms = 0u128;
    let mut continuation_head_ms = Vec::new();
    let mut continuation_tail_ms = Vec::new();
    let mut continuation_sample_ms = Vec::new();
    let mut continuation_head_profiles = Vec::new();
    let mut continuation_tail_profiles = Vec::new();
    let mut text = String::new();

    for step in 0..max_tokens as usize {
        let step_start = Instant::now();
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
        let step_ms = step_start.elapsed().as_millis();

        let Some(&next_token_id) = sample.token_ids.first() else {
            break;
        };

        if ttft_ms == 0 {
            ttft_ms = step_ms;
        } else {
            continuation_head_ms.push(head_ms);
            continuation_tail_ms.push(tail_ms);
            continuation_sample_ms.push(sample_ms);
            if let Some(profile) = head_profile {
                continuation_head_profiles.push(profile);
            }
            if let Some(profile) = tail_profile {
                continuation_tail_profiles.push(profile);
            }
        }

        if eos_token_id == Some(next_token_id) {
            finish_reason = "stop".to_string();
            break;
        }

        generated_token_ids.push(next_token_id);
        prompt_token_ids.push(next_token_id);
        text = tail.decode_token_ids(&generated_token_ids);

        if let Some(trimmed) = trim_at_stop_sequence(&text, stop_sequences) {
            text = trimmed;
            finish_reason = "stop".to_string();
            break;
        }
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
        ttft_ms,
        total_ms,
        continuation_tok_s,
        continuation_head_ms,
        continuation_tail_ms,
        continuation_sample_ms,
        continuation_head_profiles,
        continuation_tail_profiles,
        finish_reason,
        text,
        token_ids: generated_token_ids,
    })
}

fn print_run(run: &GenerationRun) {
    println!("ttft       : {}ms", run.ttft_ms);
    println!("total      : {}ms", run.total_ms);
    println!("cont tok/s : {:.2}", run.continuation_tok_s);
    println!("finish     : {}", run.finish_reason);
    println!("text       : {:?}", run.text);
    println!("token ids  : {:?}", run.token_ids);
}

fn print_ms_series(label: &str, mut values: Vec<u128>) {
    values.sort_unstable();
    let min = values.first().copied().unwrap_or(0);
    let max = values.last().copied().unwrap_or(0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<u128>() / values.len() as u128;
    println!("{label:<13}: min={min}ms median={median}ms avg={avg}ms max={max}ms");
}

fn print_f64_series(label: &str, mut values: Vec<f64>) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(0.0);
    let median = values[values.len() / 2];
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    println!("{label:<13}: min={min:.2} median={median:.2} avg={avg:.2} max={max:.2}");
}

fn avg_u128(values: &[u128]) -> u128 {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<u128>() / values.len() as u128
    }
}

fn print_profile_avg(label: &str, profiles: Vec<&RealForwardProfile>) {
    if profiles.is_empty() {
        return;
    }
    let len = profiles.len() as u128;
    let avg =
        |f: fn(&RealForwardProfile) -> u128| profiles.iter().map(|p| f(p)).sum::<u128>() / len;
    println!(
        "{label:<13}: attn={}ms (qkv={}ms core={}ms out={}ms) ffn={}ms (gate+up={}ms down={}ms) ple={}ms",
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

fn parse_stop_sequences(value: &str) -> Vec<String> {
    value
        .split('|')
        .map(str::to_string)
        .filter(|item| !item.is_empty())
        .collect()
}

fn trim_at_stop_sequence(text: &str, stop_sequences: &[String]) -> Option<String> {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()?;
    Some(text[..stop_at].to_string())
}

#[cfg(test)]
mod tests {
    use super::{parse_stop_sequences, trim_at_stop_sequence};

    #[test]
    fn parse_stop_sequences_splits_pipe_list() {
        assert_eq!(
            parse_stop_sequences("<END>|STOP"),
            vec![String::from("<END>"), String::from("STOP")]
        );
    }

    #[test]
    fn trim_at_stop_sequence_uses_earliest_match() {
        assert_eq!(
            trim_at_stop_sequence("Paris,<END>a", &[String::from("<END>"), String::from(",")]),
            Some(String::from("Paris"))
        );
    }
}
