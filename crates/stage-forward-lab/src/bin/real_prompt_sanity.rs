use anyhow::{Result, bail};
use stage_forward_lab::prompt_suite::{
    ValidationPromptSuiteMode, expectation_matches, validation_prompt_cases,
};
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct GenerationRun {
    finish_reason: String,
    text: String,
    token_ids: Vec<u32>,
    ttft_ms: u128,
    total_ms: u128,
}

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
    let prompt_mode = args
        .get(4)
        .and_then(|value| GemmaPromptMode::parse(value))
        .unwrap_or_default();
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
    let suite_mode = args
        .get(8)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let scores_path = vocab_path
        .parent()
        .map(|parent| parent.join("vocab_scores.json"))
        .unwrap_or_else(|| PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json"));

    println!("=== Real Prompt Sanity ===");
    println!("stage 1    : {}", stage1_path.display());
    println!("stage 2    : {}", stage2_path.display());
    println!("prompt mode: {}", prompt_mode.as_str());
    println!("layer cap  : {:?}", layer_cap);
    println!("vocab cap  : {:?}", vocab_cap);
    println!("disable PLE: {}", disable_ple);
    println!("suite mode : {}", suite_mode.as_str());
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

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        println!("=== Case: {} ===", case.name);
        println!("prompt       : {:?}", case.prompt);
        let first = run_once(
            &head,
            &tail,
            case.prompt,
            1,
            case.stop_sequences,
            prompt_mode,
            &format!("sanity-{}-first", case.name),
        )?;
        let continuation = run_once(
            &head,
            &tail,
            case.prompt,
            case.max_tokens,
            case.stop_sequences,
            prompt_mode,
            &format!("sanity-{}-cont", case.name),
        )?;

        let first_ok = expectation_matches(case.first_token_expectation, &first.text);
        let continuation_ok =
            expectation_matches(case.continuation_expectation, &continuation.text);
        failed |= !(first_ok && continuation_ok);

        println!(
            "first token  : ttft={}ms total={}ms finish={} ok={} text={:?} token_ids={:?}",
            first.ttft_ms,
            first.total_ms,
            first.finish_reason,
            pass_fail(first_ok),
            first.text,
            first.token_ids
        );
        println!(
            "continuation : ttft={}ms total={}ms finish={} ok={} text={:?} token_ids={:?}",
            continuation.ttft_ms,
            continuation.total_ms,
            continuation.finish_reason,
            pass_fail(continuation_ok),
            continuation.text,
            continuation.token_ids
        );
        println!();
    }

    if failed {
        bail!("prompt sanity suite failed");
    }

    println!("overall: PASS");
    Ok(())
}

fn run_once(
    head: &RealGemmaBackend,
    tail: &RealGemmaBackend,
    prompt: &str,
    max_tokens: u32,
    stop_sequences: &[&str],
    prompt_mode: GemmaPromptMode,
    request_id: &str,
) -> Result<GenerationRun> {
    let eos_token_id = tail.eos_token_id().or_else(|| head.eos_token_id());
    let mut prompt_token_ids = head.tokenize_prompt_mode(prompt, prompt_mode);
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

    head.clear_decode_session(request_id);
    tail.clear_decode_session(request_id);

    Ok(GenerationRun {
        finish_reason,
        text,
        token_ids: generated_token_ids,
        ttft_ms,
        total_ms,
    })
}

fn trim_at_stop_sequence(text: &str, stop_sequences: &[&str]) -> Option<String> {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()?;
    Some(text[..stop_at].to_string())
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_at_stop_sequence_uses_earliest_match() {
        assert_eq!(
            trim_at_stop_sequence("Paris, France.", &[",", "."]).as_deref(),
            Some("Paris")
        );
    }
}
