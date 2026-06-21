#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::{default_gemma_model_path, greedy_single_node_completion};
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_gemma_model_path);
    if !model_path.exists() {
        bail!("model not found: {}", model_path.display());
    }
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_tokens: u32 = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(256);

    let started = Instant::now();
    let completion =
        greedy_single_node_completion(&model_path, &prompt, max_tokens).context("completion")?;
    let elapsed = started.elapsed().as_secs_f64();
    let tps = completion.completion_tokens as f64 / elapsed;

    println!("model={}", model_path.display());
    println!("tokens={}", completion.completion_tokens);
    println!("elapsed={elapsed:.3}s");
    println!("tps={tps:.2}");
    println!("text={:?}", completion.text);
    Ok(())
}
