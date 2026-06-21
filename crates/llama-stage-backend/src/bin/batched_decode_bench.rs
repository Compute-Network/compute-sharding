#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Phase 0 spike: validate the assumption that speculative decoding relies on.
// If llama_decode(batch=k) costs ~the same as llama_decode(batch=1), then a
// k-token speculative window amortizes the per-call setup overhead — that's
// the lift mechanism. If batch=k costs ~k × batch=1, there's no per-call
// overhead to amortize and speculative decoding gives nothing.
//
// Single-process, single Metal context. Times pure decode loops at varying
// batch sizes. The model and prefill prompt are constant across runs; only
// the batch size of the decode-loop varies.
use anyhow::{Context, Result, bail};
use llama_stage_backend::{LlamaStageBackend, default_gemma_model_path};
use std::time::Instant;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_gemma_model_path);
    if !model_path.exists() {
        bail!("model not found: {}", model_path.display());
    }
    eprintln!("[bench] model = {}", model_path.display());

    // Total decoded tokens stays constant across configs so we compare the
    // same amount of *work* under different batch shapes.
    let total_tokens: usize = std::env::var("TOTAL_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let prompt = "The capital of France is";

    for &batch_size in &[1usize, 2, 4, 8, 16] {
        if total_tokens % batch_size != 0 {
            continue;
        }
        let iterations = total_tokens / batch_size;
        let elapsed_ms =
            run_bench(&model_path, prompt, batch_size, iterations).context("bench run")?;
        let per_token_ms = elapsed_ms as f64 / total_tokens as f64;
        let per_call_ms = elapsed_ms as f64 / iterations as f64;
        println!(
            "batch={batch_size:<2} iters={iterations:<2} total={elapsed_ms:>5}ms  per-call={per_call_ms:>6.1}ms  per-token={per_token_ms:>6.1}ms"
        );
    }

    Ok(())
}

fn run_bench(
    model_path: &std::path::Path,
    prompt: &str,
    batch_size: usize,
    iterations: usize,
) -> Result<u128> {
    // Each config gets a fresh backend so KV state and Metal warmup don't
    // bleed between runs. The first decode of each config absorbs cold-start
    // cost; we warm up with one prefill+decode before timing.
    let backend = LlamaStageBackend::single_node_for_bench(model_path.to_path_buf())?;
    let (model_ptr, session_ctx, prefill_token) = backend.bench_prefill_and_seed(prompt)?;

    // Warmup: one decode of the requested batch size, untimed.
    backend.bench_decode_batch(session_ctx, prefill_token, batch_size)?;

    let t = Instant::now();
    for _ in 0..iterations {
        backend.bench_decode_batch(session_ctx, prefill_token, batch_size)?;
    }
    let elapsed = t.elapsed().as_millis();

    backend.bench_cleanup(model_ptr, session_ctx);
    Ok(elapsed)
}
