#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::Result;
use llama_stage_backend::{greedy_single_node_baseline, resolve_model_arg};
use std::env;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (model_path, arg_idx) = resolve_model_arg(&args);
    let prompt = args
        .get(arg_idx)
        .cloned()
        .unwrap_or_else(|| "Hello".to_string());

    let t0 = Instant::now();
    let sample = greedy_single_node_baseline(model_path, &prompt)?;
    let elapsed = t0.elapsed().as_millis();

    println!("baseline_ms={elapsed}");
    println!("text={:?}", sample.text);
    println!("completion_tokens={}", sample.completion_tokens);
    Ok(())
}
