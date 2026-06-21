#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::Result;
use llama_stage_backend::{LlamaStageBackend, resolve_model_arg};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (model_path, arg_idx) = resolve_model_arg(&args);
    let prompt = args
        .get(arg_idx)
        .cloned()
        .unwrap_or_else(|| "Hello".to_string());

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

    let t_head = Instant::now();
    let tensor = head.begin_prompt("two-stage-probe", &prompt, Some(1), 0)?;
    let head_ms = t_head.elapsed().as_millis();
    let transfer_bytes = tensor.bytes.len();

    let t_tail = Instant::now();
    let tail_forward = tail.continue_forward(tensor)?;
    let tail_forward_ms = t_tail.elapsed().as_millis();

    let t_sample = Instant::now();
    let sample = tail.sample_tail(tail_forward)?;
    let sample_ms = t_sample.elapsed().as_millis();

    println!("head_ms={head_ms}");
    println!("transfer_bytes={transfer_bytes}");
    println!("tail_forward_ms={tail_forward_ms}");
    println!("sample_ms={sample_ms}");
    println!("ttft_ms={}", head_ms + tail_forward_ms + sample_ms);
    println!("text={:?}", sample.text);
    println!("completion_tokens={}", sample.completion_tokens);
    Ok(())
}
