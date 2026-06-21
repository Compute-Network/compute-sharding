#![allow(
    clippy::collapsible_if,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of
)]

// Integration probe for the new ManagedTailNode + ManagedHeadGatewayStack split.
// Spawns:
//   - tail-only stage worker on 127.0.0.1:0 (simulates the remote tail machine)
//   - head + gateway pointing at the tail worker (simulates the head machine)
// Connects a GatewayServiceClient and runs a complete() round-trip.
use anyhow::{Context, Result};
use llama_stage_backend::{
    GatewayServiceClient, ManagedGatewayLaunchSpec, ManagedHeadGatewayStack, ManagedTailNode,
    RemoteStageTimings, default_gemma_model_path,
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
        anyhow::bail!(
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
        "[probe] {label} transport head_pack={}ms/{}us tail_unpack={}ms/{}us json_encode={}ms/{}us json_decode={}ms/{}us write={}ms/{}us read={}ms/{}us inline_samples={} sample_fallbacks={}",
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
        timings.inline_sample_hits,
        timings.sample_rpc_fallbacks,
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
}

fn main() -> Result<()> {
    // Per-stage model paths can be overridden via env to test sharded GGUFs.
    let head_model_path = std::env::var("HEAD_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_gemma_model_path());
    let tail_model_path = std::env::var("TAIL_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_gemma_model_path());
    if !head_model_path.exists() {
        anyhow::bail!("head model not found: {}", head_model_path.display());
    }
    if !tail_model_path.exists() {
        anyhow::bail!("tail model not found: {}", tail_model_path.display());
    }
    eprintln!("[probe] head model = {}", head_model_path.display());
    eprintln!("[probe] tail model = {}", tail_model_path.display());

    // Default: full gemma-4-e4b-q4 (total_layers=42, head=0..=20, tail=21..=41).
    // For renumbered per-stage shards, override via env (each shard reindexes
    // its layers to 0..N-1, so e.g. tail shard wants TAIL_START=0 TAIL_END=20).
    fn env_layer(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    let head_start = env_layer("HEAD_START", 0);
    let head_end = env_layer("HEAD_END", 20);
    let tail_start = env_layer("TAIL_START", 21);
    let tail_end = env_layer("TAIL_END", 41);

    let stage_node_bin = current_profile_bin("llama_stage_tcp_node")?;
    let gateway_bin = current_profile_bin("llama_stage_gateway_tcp_node")?;
    let launch_spec = ManagedGatewayLaunchSpec {
        stage_node_bin: Some(stage_node_bin.clone()),
        gateway_bin: Some(gateway_bin.clone()),
        ..ManagedGatewayLaunchSpec::default()
    };
    eprintln!("[probe] stage node bin = {}", stage_node_bin.display());
    eprintln!("[probe] gateway bin = {}", gateway_bin.display());

    eprintln!("[probe] spawning tail worker (layers {tail_start}-{tail_end})");
    let t0 = Instant::now();
    let tail = ManagedTailNode::spawn(
        tail_model_path.clone(),
        "127.0.0.1:0",
        tail_start,
        tail_end,
        &launch_spec,
    )
    .context("spawning tail node")?;
    eprintln!(
        "[probe] tail listening on {} ({:.1}s)",
        tail.addr(),
        t0.elapsed().as_secs_f64()
    );

    eprintln!(
        "[probe] spawning head + gateway (head layers {head_start}-{head_end}, tail={})",
        tail.addr()
    );
    let t1 = Instant::now();
    let stack = ManagedHeadGatewayStack::spawn_with_remote_tail(
        head_model_path.clone(),
        head_start,
        head_end,
        tail.addr(),
        false,
        &launch_spec,
    )
    .context("spawning head + gateway")?;
    eprintln!(
        "[probe] gateway ready at {} ({:.1}s)",
        stack.gateway_addr(),
        t1.elapsed().as_secs_f64()
    );

    eprintln!("[probe] connecting client to gateway");
    let mut client =
        GatewayServiceClient::connect(stack.gateway_addr()).context("connect gateway client")?;

    let prompt = "The capital of France is".to_string();
    eprintln!("[probe] complete prompt={prompt:?}");
    let t2 = Instant::now();
    let completion = client
        .complete("split-probe-1".to_string(), prompt.clone(), 24)
        .context("complete request")?;
    let total = t2.elapsed().as_secs_f64();

    eprintln!(
        "[probe] tokens={} text={:?}",
        completion.completion_tokens, completion.text
    );
    print_timings("baseline", &completion.timings);
    let tps = if total > 0.0 {
        completion.completion_tokens as f64 / total
    } else {
        0.0
    };
    eprintln!("[probe] elapsed={:.2}s tps={:.2}", total, tps);

    if completion.completion_tokens == 0 {
        anyhow::bail!("no tokens generated");
    }
    Ok(())
}
