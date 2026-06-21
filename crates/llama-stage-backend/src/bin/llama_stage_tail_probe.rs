#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::Result;
use llama_stage_backend::{LlamaStageBackend, resolve_model_arg};
use stage_forward_lab::{StageForwardBackend, StageLayout, StageTensor};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (model_path, arg_idx) = resolve_model_arg(&args);
    let tensor_path = args
        .get(arg_idx)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("head-stage-tensor.json"));

    let tensor: StageTensor = serde_json::from_slice(&fs::read(&tensor_path)?)?;

    let mut backend = LlamaStageBackend::new(&model_path)?;
    backend.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    let t0 = Instant::now();
    let forward = backend.continue_forward(tensor)?;
    let forward_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    let sample = backend.sample_tail(forward)?;
    let sample_ms = t1.elapsed().as_millis();

    println!("tail_forward_ms={forward_ms}");
    println!("tail_sample_ms={sample_ms}");
    println!("text={:?}", sample.text);
    println!("completion_tokens={}", sample.completion_tokens);
    Ok(())
}
