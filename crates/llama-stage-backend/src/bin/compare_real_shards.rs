#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

use anyhow::{Context, Result, bail};
use llama_stage_backend::LlamaStageBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::path::PathBuf;

fn usage(bin: &str) -> String {
    format!(
        "usage: {bin} <full.gguf> <head.gguf> <tail.gguf> <prompt> [head_end tail_start tail_end]"
    )
}

fn f32s(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("hidden byte length {} is not divisible by 4", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn diff_stats(lhs: &[f32], rhs: &[f32]) -> Result<(f64, f32, f32, f32)> {
    if lhs.len() != rhs.len() {
        bail!("hidden lengths differ: {} vs {}", lhs.len(), rhs.len());
    }
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut lhs_sq = 0.0f64;
    let mut rhs_sq = 0.0f64;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        let diff = (a - b).abs();
        sum_abs += diff as f64;
        max_abs = max_abs.max(diff);
        lhs_sq += (a as f64) * (a as f64);
        rhs_sq += (b as f64) * (b as f64);
    }
    let n = lhs.len().max(1) as f64;
    Ok((
        sum_abs / n,
        max_abs,
        (lhs_sq / n).sqrt() as f32,
        (rhs_sq / n).sqrt() as f32,
    ))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!(
            "{}",
            usage(
                args.first()
                    .map(String::as_str)
                    .unwrap_or("compare_real_shards")
            )
        );
    }

    let full_path = PathBuf::from(&args[1]);
    let head_path = PathBuf::from(&args[2]);
    let tail_path = PathBuf::from(&args[3]);
    let prompt = &args[4];
    let head_end: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(12);
    let tail_start: u32 = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(head_end + 1);
    let tail_end: u32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(34);
    let local_tail_end = tail_end
        .checked_sub(tail_start)
        .context("tail_start must be <= tail_end")?;

    let mut full_head = LlamaStageBackend::new(&full_path)?;
    full_head.load_layout(StageLayout {
        model_id: "full-head".into(),
        stage_id: "full-head".into(),
        start_layer: 0,
        end_layer: head_end,
        is_head: true,
        is_tail: false,
    })?;
    let mut full_tail = LlamaStageBackend::new(&full_path)?;
    full_tail.load_layout(StageLayout {
        model_id: "full-tail".into(),
        stage_id: "full-tail".into(),
        start_layer: tail_start,
        end_layer: tail_end,
        is_head: false,
        is_tail: true,
    })?;
    let mut real_head = LlamaStageBackend::new(&head_path)?;
    real_head.load_layout(StageLayout {
        model_id: "real-head".into(),
        stage_id: "real-head".into(),
        start_layer: 0,
        end_layer: head_end,
        is_head: true,
        is_tail: false,
    })?;
    let mut real_tail = LlamaStageBackend::new(&tail_path)?;
    real_tail.load_layout(StageLayout {
        model_id: "real-tail".into(),
        stage_id: "real-tail".into(),
        start_layer: 0,
        end_layer: local_tail_end,
        is_head: false,
        is_tail: true,
    })?;

    let full_tokens = full_head.tokenize(prompt)?;
    let real_tokens = real_head.tokenize(prompt)?;
    println!("prompt_tokens_full={full_tokens:?}");
    println!("prompt_tokens_real={real_tokens:?}");
    println!("prompt_tokens_match={}", full_tokens == real_tokens);

    let full_hidden = full_head.begin_prompt_session("full", prompt, Some(1))?;
    let real_hidden = real_head.begin_prompt_session("real", prompt, Some(1))?;
    println!("hidden_dim_full={}", full_hidden.hidden_dim);
    println!("hidden_dim_real={}", real_hidden.hidden_dim);
    println!("hidden_bytes_full={}", full_hidden.bytes.len());
    println!("hidden_bytes_real={}", real_hidden.bytes.len());
    println!(
        "head_hidden_bytes_equal={}",
        full_hidden.bytes == real_hidden.bytes
    );

    let full_vec = f32s(&full_hidden.bytes)?;
    let real_vec = f32s(&real_hidden.bytes)?;
    let (mean_abs, max_abs, full_rms, real_rms) = diff_stats(&full_vec, &real_vec)?;
    println!("head_hidden_mean_abs_diff={mean_abs:.8}");
    println!("head_hidden_max_abs_diff={max_abs:.8}");
    println!("head_hidden_full_rms={full_rms:.8}");
    println!("head_hidden_real_rms={real_rms:.8}");

    let full_tail_tensor =
        full_tail.continue_forward_with_tokens(full_hidden, full_tokens, true)?;
    let real_tail_tensor =
        real_tail.continue_forward_with_tokens(real_hidden, real_tokens, true)?;
    let full_sample = full_tail.sample_tail_token(full_tail_tensor)?;
    let real_sample = real_tail.sample_tail_token(real_tail_tensor)?;
    println!("full_tail_token_id={}", full_sample.token_id);
    println!("full_tail_piece={:?}", full_sample.piece);
    println!("full_tail_is_eog={}", full_sample.is_eog);
    println!("real_tail_token_id={}", real_sample.token_id);
    println!("real_tail_piece={:?}", real_sample.piece);
    println!("real_tail_is_eog={}", real_sample.is_eog);
    println!(
        "tail_sample_match={}",
        full_sample.token_id == real_sample.token_id
    );

    Ok(())
}
