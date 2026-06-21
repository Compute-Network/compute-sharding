#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Result, bail};
use llama_stage_backend::{LlamaStageBackend, greedy_single_node_completion, resolve_model_arg};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> (std::path::PathBuf, u32, Vec<String>) {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 6u32;

    if args.get(idx).map(|s| s.as_str()) == Some("--max-tokens") {
        if let Some(raw) = args.get(idx + 1) {
            if let Ok(parsed) = raw.parse::<u32>() {
                max_tokens = parsed.max(1);
            }
        }
        idx += 2;
    }

    let prompts = if args.len() > idx {
        args[idx..].to_vec()
    } else {
        default_prompts()
    };

    (model_path, max_tokens, prompts)
}

fn main() -> Result<()> {
    let (model_path, max_tokens, prompts) = parse_args();

    let mut head = LlamaStageBackend::new(&model_path)?;
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = LlamaStageBackend::new(&model_path)?;
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    for (idx, prompt) in prompts.iter().enumerate() {
        let request_id = format!("two-stage-decode-{idx}");

        let t_baseline = Instant::now();
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let baseline_ms = t_baseline.elapsed().as_millis();

        let prompt_tokens = head.tokenize(prompt)?;

        let t_head = Instant::now();
        let mut stage_tensor = head.begin_prompt_session(&request_id, prompt, Some(max_tokens))?;
        let head_prefill_ms = t_head.elapsed().as_millis();
        let transfer_bytes = stage_tensor.bytes.len();

        let t_tail = Instant::now();
        let mut tail_forward =
            tail.continue_forward_with_tokens(stage_tensor, prompt_tokens, true)?;
        let prompt_tail_ms = t_tail.elapsed().as_millis();

        let mut generated_text = String::new();
        let mut generated_token_ids = Vec::new();
        let mut sample_ms_total = 0u128;
        let mut head_decode_ms_total = 0u128;
        let mut tail_decode_ms_total = prompt_tail_ms;

        for step in 0..max_tokens {
            let t_sample = Instant::now();
            let sampled = tail.sample_tail_token(tail_forward.clone())?;
            sample_ms_total += t_sample.elapsed().as_millis();
            generated_text.push_str(&sampled.piece);
            generated_token_ids.push(sampled.token_id);

            if sampled.is_eog || step + 1 >= max_tokens {
                break;
            }

            let t_head_step = Instant::now();
            stage_tensor =
                head.continue_head_tokens(&request_id, vec![sampled.token_id], Some(max_tokens))?;
            head_decode_ms_total += t_head_step.elapsed().as_millis();

            let t_tail_step = Instant::now();
            tail_forward =
                tail.continue_forward_with_tokens(stage_tensor, vec![sampled.token_id], false)?;
            tail_decode_ms_total += t_tail_step.elapsed().as_millis();
        }

        let matches = baseline.text == generated_text
            && baseline.completion_tokens == generated_token_ids.len() as u32
            && baseline.token_ids == generated_token_ids;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("max_tokens={max_tokens}");
        println!("baseline_text={:?}", baseline.text);
        println!("two_stage_text={:?}", generated_text);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("two_stage_token_ids={:?}", generated_token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("two_stage_tokens={}", generated_token_ids.len());
        println!("baseline_ms={baseline_ms}");
        println!("head_prefill_ms={head_prefill_ms}");
        println!("head_decode_ms={head_decode_ms_total}");
        println!("transfer_bytes={transfer_bytes}");
        println!("tail_decode_ms={tail_decode_ms_total}");
        println!("sample_ms={sample_ms_total}");
        println!(
            "ttft_ms={}",
            head_prefill_ms + prompt_tail_ms + sample_ms_total
        );
        println!("match={matches}");
        println!();

        if !matches {
            bail!("two-stage decode diverged from baseline on case {idx}");
        }
    }

    println!("overall=PASS");
    Ok(())
}
