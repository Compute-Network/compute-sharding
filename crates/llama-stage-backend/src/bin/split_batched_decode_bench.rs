#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Phase 3 prerequisite: does llama_decode_head / llama_decode_tail amortize
// per-call setup cost when fed batch_size > 1, the way the unsplit
// llama_decode does in batched_decode_bench?
//
// If yes: speculative decoding in split mode is a win — every k-batched call
// replaces k sequential per-token calls.
// If no: spec decode in split mode is pointless and the entire follow-on
// implementation would be wasted effort.
//
// Builds head + tail in the same process (no TCP), prefills both with the
// same prompt, then times decode_head and decode_tail at increasing batch
// sizes. Same total token count per config so the comparison is apples to
// apples.
//
// Usage: split_batched_decode_bench [model.gguf]
//        HEAD_MODEL=head.gguf TAIL_MODEL=tail.gguf split_batched_decode_bench
use anyhow::{Result, bail};
use llama_stage_backend::{
    LlamaStageBackend, StageNodeConfig, StagePayloadKind as PayloadKind,
    StageTensorPayload as StageTensor, build_stage_backend, default_gemma_model_path,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_gemma_model_path);
    let head_model_path = std::env::var("HEAD_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path.clone());
    let tail_model_path = std::env::var("TAIL_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_path.clone());
    if !head_model_path.exists() {
        bail!("head model not found: {}", head_model_path.display());
    }
    if !tail_model_path.exists() {
        bail!("tail model not found: {}", tail_model_path.display());
    }
    eprintln!("[bench] model      = {}", model_path.display());
    eprintln!("[bench] head model = {}", head_model_path.display());
    eprintln!("[bench] tail model = {}", tail_model_path.display());

    // Match production split for a full GGUF: 42 layers total, head 0..=20,
    // tail 21..=41. Reindexed shard GGUFs each start at layer 0, so default
    // the tail shard to 0..=20 when TAIL_MODEL is explicitly supplied.
    // Override via env if needed.
    let head_end: u32 = std::env::var("HEAD_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let has_tail_model_override = std::env::var_os("TAIL_MODEL").is_some();
    let default_tail_start = if has_tail_model_override { 0 } else { 21 };
    let default_tail_end = if has_tail_model_override { 20 } else { 41 };
    let tail_start: u32 = std::env::var("TAIL_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_tail_start);
    let tail_end: u32 = std::env::var("TAIL_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_tail_end);

    eprintln!("[bench] split: head 0..={head_end}, tail {tail_start}..={tail_end}");

    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());

    // Total decoded tokens stays constant across batch configs so per-token
    // numbers compare equal amounts of work.
    let total_tokens: usize = std::env::var("TOTAL_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    eprintln!("[bench] prompt={prompt:?} total_decoded_tokens={total_tokens}");

    let head = build_stage_backend(&StageNodeConfig {
        model_path: head_model_path,
        stage_id: "bench-head".to_string(),
        start_layer: 0,
        end_layer: head_end,
        is_head: true,
        is_tail: false,
    })?;
    let tail = build_stage_backend(&StageNodeConfig {
        model_path: tail_model_path,
        stage_id: "bench-tail".to_string(),
        start_layer: tail_start,
        end_layer: tail_end,
        is_head: false,
        is_tail: true,
    })?;

    // Warm both stages with a prefill at batch_size = prompt_token_count.
    // After this, both KV caches contain the prompt and are ready for decode.
    let prompt_tokens = head.tokenize(&prompt)?;
    eprintln!("[bench] prompt token count = {}", prompt_tokens.len());

    let request_id = "bench-1".to_string();
    let head_tensor = head.continue_head_tokens(&request_id, prompt_tokens.clone(), Some(64))?;
    let tail_tensor =
        tail.continue_forward_with_tokens(head_tensor, prompt_tokens.clone(), false)?;
    drop(tail_tensor); // discard prefill output; we only need KV state

    // The seed token for decode must be a valid id; use the last prompt token
    // again (semantically meaningless — the bench cares about timing only).
    let seed_token = *prompt_tokens.last().unwrap();

    // Get hidden_dim by running a single decode_head and capturing the tensor
    // size. This will mirror what spec decode would actually ship.
    let probe = head.continue_head_tokens(&request_id, vec![seed_token], None)?;
    let hidden_dim = probe.hidden_dim;
    let probe_bytes_per_token = probe.bytes.len();
    eprintln!("[bench] hidden_dim={hidden_dim} bytes_per_token={probe_bytes_per_token}");

    // Send the probe through tail to keep both KV caches in sync (and discard).
    let _ = tail.continue_forward_with_tokens(probe, vec![seed_token], false)?;

    println!();
    println!("=== HEAD (decode_head) ===");
    bench_head(
        &head,
        &request_id,
        seed_token,
        total_tokens,
        &[1, 2, 4, 8, 16],
    )?;

    println!();
    println!("=== TAIL (decode_tail) ===");
    bench_tail(
        &head,
        &tail,
        &request_id,
        seed_token,
        total_tokens,
        &[1, 2, 4, 8, 16],
    )?;

    Ok(())
}

fn bench_head(
    head: &LlamaStageBackend,
    request_id: &str,
    seed_token: i32,
    total_tokens: usize,
    batch_sizes: &[usize],
) -> Result<()> {
    for &batch_size in batch_sizes {
        if total_tokens % batch_size != 0 {
            continue;
        }
        let iterations = total_tokens / batch_size;

        // Warmup: one untimed decode_head at this batch size to absorb any
        // first-call cost (Metal pipeline bind, graph build, etc.).
        let tokens: Vec<i32> = vec![seed_token; batch_size];
        let _ = head.continue_head_tokens(request_id, tokens.clone(), None)?;

        let t = Instant::now();
        for _ in 0..iterations {
            let _ = head.continue_head_tokens(request_id, tokens.clone(), None)?;
        }
        let elapsed_ms = t.elapsed().as_millis();
        let per_call = elapsed_ms as f64 / iterations as f64;
        let per_token = elapsed_ms as f64 / total_tokens as f64;
        println!(
            "  batch={batch_size:<2} iters={iterations:<2} total={elapsed_ms:>5}ms  per-call={per_call:>6.1}ms  per-token={per_token:>6.1}ms"
        );
    }
    Ok(())
}

fn bench_tail(
    head: &LlamaStageBackend,
    tail: &LlamaStageBackend,
    request_id: &str,
    seed_token: i32,
    total_tokens: usize,
    batch_sizes: &[usize],
) -> Result<()> {
    for &batch_size in batch_sizes {
        if total_tokens % batch_size != 0 {
            continue;
        }
        let iterations = total_tokens / batch_size;

        // Tail needs hidden states from head — produce them once at the
        // requested batch size to use as a fixed input across iterations.
        let tokens: Vec<i32> = vec![seed_token; batch_size];
        let head_tensor = head.continue_head_tokens(request_id, tokens.clone(), None)?;

        // Warmup tail at this batch size.
        let warmup_input = clone_tensor(&head_tensor);
        let _ = tail.continue_forward_with_tokens(warmup_input, tokens.clone(), false)?;

        let t = Instant::now();
        for _ in 0..iterations {
            let input = clone_tensor(&head_tensor);
            let _ = tail.continue_forward_with_tokens(input, tokens.clone(), false)?;
        }
        let elapsed_ms = t.elapsed().as_millis();
        let per_call = elapsed_ms as f64 / iterations as f64;
        let per_token = elapsed_ms as f64 / total_tokens as f64;
        println!(
            "  batch={batch_size:<2} iters={iterations:<2} total={elapsed_ms:>5}ms  per-call={per_call:>6.1}ms  per-token={per_token:>6.1}ms"
        );
    }
    Ok(())
}

fn clone_tensor(src: &StageTensor) -> StageTensor {
    StageTensor {
        request_id: src.request_id.clone(),
        kind: PayloadKind::HiddenState,
        stage_trace: src.stage_trace.clone(),
        hidden_dim: src.hidden_dim,
        bytes: src.bytes.clone(),
        prompt_text: None,
        max_tokens: src.max_tokens,
        continuation: src.continuation.clone(),
        transient: src.transient.clone(),
        carry: src.carry.clone(),
    }
}
