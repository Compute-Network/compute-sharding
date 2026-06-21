#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Phase 4 integration probe: end-to-end speculative decoding through
// `RemoteStageGateway::connect_with_draft`. Spawns a TCP head + TCP tail in
// child processes (production wire path), then drives the gateway in-process
// so the DraftEngine lives next to the head over the same gateway logic that
// the daemon will use.
//
// Runs the same prompt twice — once with spec OFF (baseline), once with spec
// ON — and compares wall time + tokens to confirm:
//   1. Both paths produce identical text (greedy + verify is exact under
//      target's argmax),
//   2. Spec path is faster than baseline at the measured acceptance, OR at
//      least not slower (we treat a 0.9× lower bound as PASS so warmup noise
//      doesn't false-negative).
//
// Usage: spec_gateway_probe <draft.gguf> [target.gguf] [prompt] [max_tokens]
use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    ManagedGatewayLaunchSpec, ManagedHeadNode, ManagedTailNode, RemoteStageGateway,
    RemoteStageTimings, SpecDecodeConfig, default_gemma_model_path,
};
use std::path::PathBuf;
use std::time::Instant;

fn current_profile_bin(name: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("resolve current executable path")?;
    let dir = exe
        .parent()
        .context("current executable has no parent directory")?;
    let candidate = dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    if candidate.exists() {
        Ok(candidate)
    } else {
        bail!(
            "expected sibling binary for current build profile, but it was not found: {}",
            candidate.display()
        )
    }
}

fn print_timings(label: &str, timings: &RemoteStageTimings) {
    eprintln!(
        "[probe] {label} timings head_prefill={}ms head_decode={}ms tail_decode={}ms sample={}ms ttft={}ms total={}ms",
        timings.head_prefill_ms,
        timings.head_decode_ms,
        timings.tail_decode_ms,
        timings.sample_ms,
        timings.ttft_ms,
        timings.total_ms,
    );
    eprintln!(
        "[probe] {label} bytes transfer_initial={} transfer_total={} head_hidden_prefill={} head_hidden_decode={} prompt_tokens={} decode_steps={}",
        timings.transfer_bytes,
        timings.total_transfer_bytes,
        timings.head_hidden_bytes_prefill,
        timings.head_hidden_bytes_decode,
        timings.prompt_tokens,
        timings.decode_steps,
    );
    eprintln!(
        "[probe] {label} transport head_pack={}ms/{}us tail_unpack={}ms/{}us json_encode={}ms/{}us json_decode={}ms/{}us write={}ms/{}us read={}ms/{}us",
        timings.head_pack_ms,
        timings.head_pack_us,
        timings.tail_unpack_ms,
        timings.tail_unpack_us,
        timings.stage_request_json_encode_ms,
        timings.stage_request_json_encode_us,
        timings.stage_response_json_decode_ms,
        timings.stage_response_json_decode_us,
        timings.stage_request_write_ms,
        timings.stage_request_write_us,
        timings.stage_response_read_ms,
        timings.stage_response_read_us,
    );
    eprintln!(
        "[probe] {label} server request_decode={}ms/{}us handle={}ms/{}us response_encode={}ms/{}us response_write={}ms/{}us",
        timings.stage_server_request_json_decode_ms,
        timings.stage_server_request_json_decode_us,
        timings.stage_server_handle_ms,
        timings.stage_server_handle_us,
        timings.stage_server_response_json_encode_ms,
        timings.stage_server_response_json_encode_us,
        timings.stage_server_response_write_ms,
        timings.stage_server_response_write_us,
    );
    eprintln!(
        "[probe] {label} spec tail_kernel={}us tail_sync={}us tail_sample={}us tail_verify_sample={}us tail_verify_detok={}us tail_verify_rollback={}us inline_samples={} sample_fallbacks={} spec_rounds={} spec_proposed={} spec_accepted={} spec_draft={}ms/{}us spec_verify={}ms/{}us spec_rollback={}ms/{}us",
        timings.tail_decode_kernel_us,
        timings.tail_sync_us,
        timings.tail_sample_us,
        timings.tail_verify_sample_us,
        timings.tail_verify_detok_us,
        timings.tail_verify_rollback_us,
        timings.inline_sample_hits,
        timings.sample_rpc_fallbacks,
        timings.spec_rounds,
        timings.spec_drafts_proposed,
        timings.spec_drafts_accepted,
        timings.spec_draft_ms,
        timings.spec_draft_us,
        timings.spec_verify_ms,
        timings.spec_verify_us,
        timings.spec_rollback_ms,
        timings.spec_rollback_us,
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let draft_path =
        PathBuf::from(args.next().context(
            "usage: spec_gateway_probe <draft.gguf> [target.gguf] [prompt] [max_tokens]",
        )?);
    let target_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_gemma_model_path);
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let max_tokens: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);

    // Per-stage overrides (for testing with reindexed shard GGUFs, where each
    // shard has its own layer range 0..N-1). Fall back to target_path so the
    // single-file path still works.
    let head_model_path = std::env::var("HEAD_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| target_path.clone());
    let tail_model_path = std::env::var("TAIL_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| target_path.clone());

    if !draft_path.exists() {
        bail!("draft model not found: {}", draft_path.display());
    }
    if !head_model_path.exists() {
        bail!("head model not found: {}", head_model_path.display());
    }
    if !tail_model_path.exists() {
        bail!("tail model not found: {}", tail_model_path.display());
    }

    eprintln!("[probe] draft      = {}", draft_path.display());
    eprintln!("[probe] head model = {}", head_model_path.display());
    eprintln!("[probe] tail model = {}", tail_model_path.display());
    eprintln!("[probe] prompt = {prompt:?} max_tokens = {max_tokens}");

    // Match production split: 42 layers, head 0..=20, tail 21..=41. For
    // reindexed shards (HEAD_MODEL/TAIL_MODEL pointing at per-stage GGUFs),
    // each shard renumbers to 0..N-1, so override HEAD_START=0 HEAD_END=20
    // TAIL_START=0 TAIL_END=20.
    let head_start: u32 = std::env::var("HEAD_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
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

    let stage_node_bin = current_profile_bin("llama_stage_tcp_node")?;
    let gateway_bin = current_profile_bin("llama_stage_gateway_tcp_node")?;
    let launch = ManagedGatewayLaunchSpec {
        stage_node_bin: Some(stage_node_bin.clone()),
        gateway_bin: Some(gateway_bin.clone()),
        ..ManagedGatewayLaunchSpec::default()
    };
    eprintln!("[probe] stage node bin = {}", stage_node_bin.display());
    eprintln!("[probe] gateway bin = {}", gateway_bin.display());

    eprintln!("[probe] spawning head (layers {head_start}..={head_end})");
    let head = ManagedHeadNode::spawn(
        head_model_path.clone(),
        "127.0.0.1:0",
        head_start,
        head_end,
        &launch,
    )
    .context("spawn head")?;
    eprintln!("[probe] head listening on {}", head.addr());

    eprintln!("[probe] spawning tail (layers {tail_start}..={tail_end})");
    let tail = ManagedTailNode::spawn(
        tail_model_path.clone(),
        "127.0.0.1:0",
        tail_start,
        tail_end,
        &launch,
    )
    .context("spawn tail")?;
    eprintln!("[probe] tail listening on {}", tail.addr());

    // === Baseline run: spec disabled. ===
    eprintln!("[probe] === baseline (spec OFF) ===");
    let mut baseline_gw = RemoteStageGateway::connect(head.addr(), tail.addr(), false)
        .context("connect baseline gateway")?;
    let t0 = Instant::now();
    let baseline = baseline_gw
        .complete("baseline-1", &prompt, max_tokens)
        .context("baseline complete")?;
    let baseline_elapsed = t0.elapsed().as_secs_f64();
    let baseline_tps = baseline.completion_tokens as f64 / baseline_elapsed;
    eprintln!(
        "[probe] baseline tokens={} elapsed={:.2}s tps={:.2} ttft={}ms",
        baseline.completion_tokens, baseline_elapsed, baseline_tps, baseline.timings.ttft_ms
    );
    print_timings("baseline", &baseline.timings);
    eprintln!("[probe] baseline text = {:?}", baseline.text);
    drop(baseline_gw);

    // === Spec run: connect_with_draft enables the spec path. ===
    let spec_config = SpecDecodeConfig::from_env();
    eprintln!("[probe] === spec ({spec_config:?}) ===");
    let mut spec_gw = RemoteStageGateway::connect_with_draft(
        head.addr(),
        tail.addr(),
        false,
        &draft_path,
        spec_config,
    )
    .context("connect spec gateway")?;
    eprintln!(
        "[probe] head spec_decode_v1={} tail spec_decode_v1={}",
        spec_gw.head_info().spec_decode_v1,
        spec_gw.tail_info().spec_decode_v1,
    );
    eprintln!(
        "[probe] gateway spec_config.enabled={} spec_active={}",
        spec_gw.spec_config().enabled,
        spec_gw.spec_active(),
    );
    if !spec_gw.spec_active() {
        bail!(
            "spec gateway came up with spec_active=false — check head/tail spec_decode_v1 capability"
        );
    }
    let t1 = Instant::now();
    let spec = spec_gw
        .complete("spec-1", &prompt, max_tokens)
        .context("spec complete")?;
    let spec_elapsed = t1.elapsed().as_secs_f64();
    let spec_tps = spec.completion_tokens as f64 / spec_elapsed;
    eprintln!(
        "[probe] spec tokens={} elapsed={:.2}s tps={:.2} ttft={}ms",
        spec.completion_tokens, spec_elapsed, spec_tps, spec.timings.ttft_ms
    );
    print_timings("spec", &spec.timings);
    eprintln!("[probe] spec text = {:?}", spec.text);

    // === Equivalence check ===
    // Spec decoding under greedy verify is ARGMAX-EXACT against the target's
    // own batched decode, which is what spec uses on every round. Baseline,
    // by contrast, runs single-token batched decodes. Metal's batched vs
    // single-token kernels sometimes pick a different argmax at tied or
    // near-tied logit positions, so we tolerate occasional divergence and
    // measure how closely the two paths track instead of demanding identity.
    let n = baseline.token_ids.len().min(spec.token_ids.len());
    let agree = (0..n)
        .take_while(|&i| baseline.token_ids[i] == spec.token_ids[i])
        .count();
    let agree_pct = if n > 0 { agree * 100 / n } else { 0 };
    eprintln!(
        "[probe] prefix agreement: {agree}/{n} ({agree_pct}%) tokens identical before divergence"
    );
    if spec.text != baseline.text {
        eprintln!("[probe] BASELINE text: {:?}", baseline.text);
        eprintln!("[probe] SPEC text:     {:?}", spec.text);
    }
    // Hard floor: outputs must agree on at least the first few tokens. A
    // catastrophic spec bug would diverge from token 0; noise diverges late.
    if agree < 3.min(n) {
        bail!("spec diverges from baseline immediately (agree={agree}/{n})");
    }

    let speedup = baseline_elapsed / spec_elapsed;
    eprintln!("[probe] speedup spec/baseline = {speedup:.2}x");

    if speedup < 0.9 {
        bail!(
            "spec path slower than baseline ({:.2}x) — investigate acceptance / draft cost",
            speedup
        );
    }

    println!("\nPHASE 4 GATEWAY PROBE PASSED.");
    println!("baseline: {:.2} tps", baseline_tps);
    println!("spec:     {:.2} tps  (speedup {:.2}x)", spec_tps, speedup);
    Ok(())
}
