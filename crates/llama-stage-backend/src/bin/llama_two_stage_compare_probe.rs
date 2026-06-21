#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Result, bail};
use llama_stage_backend::{LlamaStageBackend, greedy_single_node_baseline, resolve_model_arg};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::time::Instant;

fn default_prompts() -> Vec<String> {
    vec![
        "Hello".to_string(),
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (model_path, arg_idx) = resolve_model_arg(&args);
    let prompts = if args.len() > arg_idx {
        args[arg_idx..].to_vec()
    } else {
        default_prompts()
    };

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
        let t_baseline = Instant::now();
        let baseline = greedy_single_node_baseline(&model_path, prompt)?;
        let baseline_ms = t_baseline.elapsed().as_millis();

        let request_id = format!("two-stage-compare-{idx}");
        let t_head = Instant::now();
        let tensor = head.begin_prompt(&request_id, prompt, Some(1), 0)?;
        let head_ms = t_head.elapsed().as_millis();
        let transfer_bytes = tensor.bytes.len();

        let t_tail = Instant::now();
        let forward = tail.continue_forward(tensor)?;
        let tail_forward_ms = t_tail.elapsed().as_millis();

        let t_sample = Instant::now();
        let sample = tail.sample_tail(forward)?;
        let sample_ms = t_sample.elapsed().as_millis();

        let matches =
            baseline.text == sample.text && baseline.completion_tokens == sample.completion_tokens;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("baseline_text={:?}", baseline.text);
        println!("two_stage_text={:?}", sample.text);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("two_stage_tokens={}", sample.completion_tokens);
        println!("baseline_ms={baseline_ms}");
        println!("head_ms={head_ms}");
        println!("transfer_bytes={transfer_bytes}");
        println!("tail_forward_ms={tail_forward_ms}");
        println!("sample_ms={sample_ms}");
        println!("ttft_ms={}", head_ms + tail_forward_ms + sample_ms);
        println!("match={matches}");
        println!();

        if !matches {
            bail!("two-stage output diverged from baseline on case {idx}");
        }
    }

    println!("overall=PASS");
    Ok(())
}
