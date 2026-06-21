#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Phase 3 integration probe for tail-side verify_batch_at_tail + rollback_kv.
//
// Builds head + tail in one process (no TCP), generates a "ground-truth"
// greedy continuation by single-stepping, then re-runs the same prefix as
// a single batched verify call with two scenarios:
//
//   1. Drafts exactly match the ground truth → expects accepted_count == k
//      and the (k+1)-th token equals the next ground-truth token (the bonus
//      sampled past the last accepted draft).
//
//   2. Drafts diverge after the first token → expects accepted_count == 1
//      and the bonus token equals the ground-truth token at the divergence
//      position (target's own pick).
//
// After the partial-accept scenario, the probe rolls back the head's KV the
// same way the gateway will, then issues a single-token decode and confirms
// the tail picks the next ground-truth token — proving that the verify
// path's KV rollback left the cache in a state consistent with what
// non-spec decode would have produced.
//
// Usage: spec_verify_probe [model.gguf]
use anyhow::{Result, bail};
use llama_stage_backend::{
    LlamaStageBackend, StageNodeConfig, StageTensorPayload, build_stage_backend,
    default_gemma_model_path,
};
use std::path::PathBuf;

const K: usize = 4;
// We need T_1..T_(k+2): k drafts to test full-accept, plus T_(k+2) as the
// bonus sampled past the last accepted draft.
const GROUND_TRUTH_LEN: usize = K + 2;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_gemma_model_path);
    if !model_path.exists() {
        bail!("model not found: {}", model_path.display());
    }
    eprintln!("[probe] model = {}", model_path.display());

    let head_end: u32 = std::env::var("HEAD_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let tail_start: u32 = std::env::var("TAIL_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);
    let tail_end: u32 = std::env::var("TAIL_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(41);

    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());

    let head = build_stage_backend(&StageNodeConfig {
        model_path: model_path.clone(),
        stage_id: "probe-head".to_string(),
        start_layer: 0,
        end_layer: head_end,
        is_head: true,
        is_tail: false,
    })?;
    let tail = build_stage_backend(&StageNodeConfig {
        model_path: model_path.clone(),
        stage_id: "probe-tail".to_string(),
        start_layer: tail_start,
        end_layer: tail_end,
        is_head: false,
        is_tail: true,
    })?;

    let prompt_tokens = head.tokenize(&prompt)?;
    eprintln!(
        "[probe] prompt tokens = {} ({prompt_tokens:?})",
        prompt_tokens.len()
    );

    let request_id = "probe-truth".to_string();

    // === Phase A: ground truth via single-step decode ===
    prefill(&head, &tail, &request_id, &prompt_tokens)?;
    let first_sample = tail
        .cached_tail_sample(&request_id)
        .ok_or_else(|| anyhow::anyhow!("no cached sample after prefill"))?;

    let mut ground_truth: Vec<i32> = vec![first_sample.token_id];
    let mut last_tok = first_sample.token_id;
    for _ in 1..GROUND_TRUTH_LEN {
        let head_hidden = head.continue_head_tokens(&request_id, vec![last_tok], None)?;
        let _ = tail.continue_forward_with_tokens(head_hidden, vec![last_tok], false)?;
        let s = tail
            .cached_tail_sample(&request_id)
            .ok_or_else(|| anyhow::anyhow!("missing cached sample at single-step"))?;
        ground_truth.push(s.token_id);
        last_tok = s.token_id;
    }
    eprintln!("[probe] ground truth (k+2 = {GROUND_TRUTH_LEN}): {ground_truth:?}");

    // === Phase B: full-accept verify ===
    prefill(&head, &tail, &request_id, &prompt_tokens)?;
    let n_pos_tail_before = tail.session_n_pos(&request_id);
    let n_pos_head_before = head.session_n_pos(&request_id);
    eprintln!("[probe] post-prefill n_pos: head={n_pos_head_before} tail={n_pos_tail_before}");
    if n_pos_tail_before != prompt_tokens.len() as i32 {
        bail!(
            "tail n_pos after prefill expected {}, got {n_pos_tail_before}",
            prompt_tokens.len()
        );
    }

    let last_token = ground_truth[0];
    let drafts_full: Vec<i32> = ground_truth[1..=K].to_vec();
    let batch_tokens: Vec<i32> = std::iter::once(last_token)
        .chain(drafts_full.iter().copied())
        .collect();
    eprintln!("[probe] full-accept batch tokens: {batch_tokens:?}");

    let head_batch_hidden =
        head.continue_head_tokens(&request_id, batch_tokens.clone(), Some(64))?;
    assert_hidden_shape(&head_batch_hidden, batch_tokens.len())?;

    let outcome = tail.verify_batch_at_tail(
        &request_id,
        head_batch_hidden,
        last_token,
        drafts_full.clone(),
        false,
    )?;
    eprintln!(
        "[probe] full-accept outcome: accepted={} ids={:?} eog={}",
        outcome.accepted_count, outcome.accepted_token_ids, outcome.is_eog
    );

    if outcome.accepted_count != K as u32 {
        bail!(
            "full-accept: expected accepted_count={K}, got {}",
            outcome.accepted_count
        );
    }
    if outcome.accepted_token_ids.len() != K + 1 {
        bail!(
            "full-accept: expected {} returned ids, got {}",
            K + 1,
            outcome.accepted_token_ids.len()
        );
    }
    for (i, want) in drafts_full.iter().enumerate() {
        if outcome.accepted_token_ids[i] != *want {
            bail!(
                "full-accept: position {i} got {} want {want}",
                outcome.accepted_token_ids[i]
            );
        }
    }
    let bonus_full = outcome.accepted_token_ids[K];
    if bonus_full != ground_truth[K + 1] {
        bail!(
            "full-accept bonus mismatch: got {bonus_full} want {} (ground_truth[{}])",
            ground_truth[K + 1],
            K + 1
        );
    }

    let n_pos_tail_after_full = tail.session_n_pos(&request_id);
    let expected_full = n_pos_tail_before + K as i32 + 1;
    if n_pos_tail_after_full != expected_full {
        bail!("full-accept tail n_pos: got {n_pos_tail_after_full} want {expected_full}");
    }
    eprintln!("[probe] full-accept PASSED");

    // === Phase C: partial-accept verify (drafts diverge after position 0) ===
    prefill(&head, &tail, &request_id, &prompt_tokens)?;
    let n_pos_tail_pre_partial = tail.session_n_pos(&request_id);
    let n_pos_head_pre_partial = head.session_n_pos(&request_id);

    // Drafts: T_2 (correct) then 3 deliberate junk ids that the target
    // model will never sample (high vocab indices reserved for added tokens
    // / specials). We don't care about the exact ids — only that they
    // mismatch the target's greedy pick at that position.
    let drafts_partial: Vec<i32> = vec![ground_truth[1], 99_999, 99_998, 99_997];
    let batch_tokens_p: Vec<i32> = std::iter::once(ground_truth[0])
        .chain(drafts_partial.iter().copied())
        .collect();
    eprintln!("[probe] partial-accept batch tokens: {batch_tokens_p:?} drafts: {drafts_partial:?}");

    let head_batch_hidden_p =
        head.continue_head_tokens(&request_id, batch_tokens_p.clone(), Some(64))?;
    assert_hidden_shape(&head_batch_hidden_p, batch_tokens_p.len())?;

    let outcome_p = tail.verify_batch_at_tail(
        &request_id,
        head_batch_hidden_p,
        ground_truth[0],
        drafts_partial.clone(),
        false,
    )?;
    eprintln!(
        "[probe] partial-accept outcome: accepted={} ids={:?}",
        outcome_p.accepted_count, outcome_p.accepted_token_ids
    );

    if outcome_p.accepted_count != 1 {
        bail!(
            "partial-accept: expected accepted_count=1, got {}",
            outcome_p.accepted_count
        );
    }
    if outcome_p.accepted_token_ids.len() != 2 {
        bail!(
            "partial-accept: expected 2 returned ids, got {}",
            outcome_p.accepted_token_ids.len()
        );
    }
    if outcome_p.accepted_token_ids[0] != ground_truth[1] {
        bail!(
            "partial-accept: accepted draft mismatch ({} vs ground_truth[1]={})",
            outcome_p.accepted_token_ids[0],
            ground_truth[1]
        );
    }
    let bonus_p = outcome_p.accepted_token_ids[1];
    if bonus_p != ground_truth[2] {
        bail!(
            "partial-accept bonus mismatch: got {bonus_p} want {} (ground_truth[2])",
            ground_truth[2]
        );
    }

    let n_pos_tail_after_partial = tail.session_n_pos(&request_id);
    let expected_partial = n_pos_tail_pre_partial + 1 + 1; // last_token + 1 accepted
    if n_pos_tail_after_partial != expected_partial {
        bail!("partial-accept tail n_pos: got {n_pos_tail_after_partial} want {expected_partial}");
    }
    eprintln!("[probe] partial-accept PASSED");

    // === Phase D: head rollback + post-rollback continuation matches ground truth ===
    // The gateway-side recipe: head got the same k+1 batched tokens, so its
    // KV grew by k+1. After the tail returned accepted=1, the gateway tells
    // the head to roll back to (n_pos_head_pre_partial + 1 + 1).
    let head_keep = n_pos_head_pre_partial + outcome_p.accepted_count as i32 + 1;
    eprintln!(
        "[probe] head rollback: pre_batch={n_pos_head_pre_partial} keep={head_keep} (current={})",
        head.session_n_pos(&request_id)
    );
    head.rollback_kv(&request_id, head_keep as u32)?;
    if head.session_n_pos(&request_id) != head_keep {
        bail!(
            "head n_pos after rollback: got {} want {head_keep}",
            head.session_n_pos(&request_id)
        );
    }

    // Now decode bonus_p (= ground_truth[2]) through head+tail. The result
    // should match the next ground-truth token.
    let post_head_hidden = head.continue_head_tokens(&request_id, vec![bonus_p], None)?;
    let _ = tail.continue_forward_with_tokens(post_head_hidden, vec![bonus_p], false)?;
    let next_sample = tail
        .cached_tail_sample(&request_id)
        .ok_or_else(|| anyhow::anyhow!("no cached sample after post-rollback decode"))?;
    if next_sample.token_id != ground_truth[3] {
        bail!(
            "post-rollback continuation mismatch: got {} want ground_truth[3]={}",
            next_sample.token_id,
            ground_truth[3]
        );
    }
    eprintln!(
        "[probe] post-rollback continuation PASSED (decoded {} matches ground_truth[3])",
        next_sample.token_id
    );

    println!("\nALL PHASE 3 CHECKS PASSED.");
    Ok(())
}

fn prefill(
    head: &LlamaStageBackend,
    tail: &LlamaStageBackend,
    request_id: &str,
    prompt_tokens: &[i32],
) -> Result<()> {
    head.clear_decode_session(request_id)?;
    tail.clear_decode_session(request_id)?;
    let head_hidden = head.continue_head_tokens(request_id, prompt_tokens.to_vec(), Some(64))?;
    let _ = tail.continue_forward_with_tokens(head_hidden, prompt_tokens.to_vec(), false)?;
    Ok(())
}

fn assert_hidden_shape(tensor: &StageTensorPayload, expected_tokens: usize) -> Result<()> {
    let expected_bytes = expected_tokens * tensor.hidden_dim * 4;
    if tensor.bytes.len() != expected_bytes {
        bail!(
            "hidden shape mismatch: bytes={} expected={} (tokens={} hidden_dim={})",
            tensor.bytes.len(),
            expected_bytes,
            expected_tokens,
            tensor.hidden_dim
        );
    }
    Ok(())
}
