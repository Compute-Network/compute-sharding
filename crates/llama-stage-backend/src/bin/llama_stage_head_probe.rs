#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::Result;
use llama_stage_backend::{LlamaStageBackend, resolve_model_arg};
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (model_path, arg_idx) = resolve_model_arg(&args);
    let prompt = args
        .get(arg_idx)
        .cloned()
        .unwrap_or_else(|| "Hello".to_string());
    let output_path = args.get(arg_idx + 1).map(PathBuf::from);

    let mut backend = LlamaStageBackend::new(&model_path)?;
    backend.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    let t0 = Instant::now();
    let tensor = backend.begin_prompt("head-probe", &prompt, Some(1), 0)?;
    let elapsed = t0.elapsed().as_millis();

    eprintln!("head_ms={}", elapsed);
    eprintln!("hidden_dim={}", tensor.hidden_dim);
    eprintln!("bytes={}", tensor.bytes.len());

    let json = serde_json::to_string_pretty(&tensor)?;
    if let Some(path) = output_path {
        fs::write(&path, json)?;
        eprintln!("wrote={}", path.display());
    } else {
        println!("{json}");
    }

    Ok(())
}
