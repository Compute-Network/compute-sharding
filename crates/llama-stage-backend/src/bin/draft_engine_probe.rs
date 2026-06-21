#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Phase 2 verification: load the draft model standalone, prefill a prompt,
// generate k tokens greedily. Prints the draft completion side-by-side with
// the target's full single-node greedy completion so we can eyeball whether
// the draft is "close enough" to be useful as a spec-decode draft.
//
// Acceptance is informal here — the bench in batched_decode_bench already
// proved that batched verification will give a 2-6× lift IF acceptance is
// reasonable. This probe just sanity-checks that drafts aren't garbage.
//
// Usage: draft_engine_probe <draft.gguf> <target.gguf> [prompt] [k]
use anyhow::{Context, Result, bail};
use llama_stage_backend::{DraftEngine, greedy_single_node_completion};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let draft_path = PathBuf::from(
        args.next()
            .context("usage: probe <draft.gguf> <target.gguf> [prompt] [k]")?,
    );
    let target_path = PathBuf::from(
        args.next()
            .context("usage: probe <draft.gguf> <target.gguf> [prompt] [k]")?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let k: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    if !draft_path.exists() {
        bail!("draft not found: {}", draft_path.display());
    }
    if !target_path.exists() {
        bail!("target not found: {}", target_path.display());
    }

    eprintln!("[probe] draft  = {}", draft_path.display());
    eprintln!("[probe] target = {}", target_path.display());
    eprintln!("[probe] prompt = {prompt:?}");
    eprintln!("[probe] k      = {k}");

    let mut draft = DraftEngine::load(&draft_path)?;
    let prompt_tokens = draft.tokenize(&prompt)?;
    eprintln!("[probe] prompt token count = {}", prompt_tokens.len());

    if prompt_tokens.is_empty() {
        bail!("tokenized prompt is empty");
    }

    let prefill_t = Instant::now();
    // Prefill all but the last prompt token; the last one becomes the seed
    // for greedy_step_k so its logit is fresh.
    draft.prefill(&prompt_tokens[..prompt_tokens.len() - 1])?;
    let prefill_ms = prefill_t.elapsed().as_millis();

    let last_prompt_token = *prompt_tokens.last().unwrap();
    let step_t = Instant::now();
    let draft_ids = draft.greedy_step_k(last_prompt_token, k)?;
    let step_ms = step_t.elapsed().as_millis();
    let per_token_ms = step_ms as f64 / k as f64;

    println!("draft prefill_ms={prefill_ms} step_ms={step_ms} per_token_ms={per_token_ms:.2}");
    println!("draft token ids: {draft_ids:?}");

    // Compare against the target's full greedy completion of the same prompt.
    let target_t = Instant::now();
    let target = greedy_single_node_completion(&target_path, &prompt, k)?;
    let target_ms = target_t.elapsed().as_millis();
    println!(
        "target full_ms={target_ms} per_token_ms={:.2}",
        target_ms as f64 / k as f64
    );
    println!("target token ids: {:?}", target.token_ids);
    println!("target text: {:?}", target.text);

    let matches = draft_ids
        .iter()
        .zip(target.token_ids.iter())
        .take_while(|(a, b)| **a == **b)
        .count();
    println!(
        "[result] {matches}/{} draft tokens match target prefix ({}%)",
        draft_ids.len(),
        if draft_ids.is_empty() {
            0
        } else {
            matches * 100 / draft_ids.len()
        }
    );

    Ok(())
}
