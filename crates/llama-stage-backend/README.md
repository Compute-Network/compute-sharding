# llama-stage-backend

Thin Rust wrapper around a vendored, patched `llama.cpp` split-decode path for the Gemma 4 two-stage milestone.

Current milestone:
- model: `gemma-4-E4B-it-Q4_K_M.gguf`
- split: head `0-20`, tail `21-41`
- wire contract: existing `StageTensor`
- protocol version: `1`

Default model resolution:
1. `~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf`
2. `models/gemma-4-E4B-it-Q4_K_M.gguf`

## Sidecar install

Build the sidecars:

```bash
cargo build -p llama-stage-backend --bins
```

Install the stage-node and gateway-node binaries into `~/.compute/bin`:

```bash
cargo run -p llama-stage-backend --bin llama_stage_install_sidecars
```

Or install into another directory:

```bash
cargo run -p llama-stage-backend --bin llama_stage_install_sidecars -- --bin-dir /opt/compute/bin
```

Installed binaries:
- `~/.compute/bin/llama_stage_tcp_node`
- `~/.compute/bin/llama_stage_gateway_tcp_node`

## Probes

Single-node baseline:

```bash
cargo run -p llama-stage-backend --bin llama_single_node_baseline_probe -- "The capital of France is"
```

Head-only probe:

```bash
cargo run -p llama-stage-backend --bin llama_stage_head_probe -- "The capital of France is" /tmp/llama-stage-tensor.json
```

Tail-only probe:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tail_probe -- /tmp/llama-stage-tensor.json
```

Integrated two-stage probe:

```bash
cargo run -p llama-stage-backend --bin llama_two_stage_lan_probe -- "The capital of France is"
```

Persistent stdio stage node:

```bash
cargo run -p llama-stage-backend --bin llama_stage_stdio_node -- --stage-id stage-1 --start-layer 0 --end-layer 20 --head
```

Persistent TCP stage node:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_node -- --bind 127.0.0.1:9001 --stage-id stage-1 --start-layer 0 --end-layer 20 --head
```

Tail TCP stage node:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_node -- --bind 127.0.0.1:9002 --stage-id stage-2 --start-layer 21 --end-layer 41 --tail
```

Deterministic baseline-vs-two-stage suite:

```bash
cargo run -p llama-stage-backend --bin llama_two_stage_compare_probe
```

Deterministic multi-token decode comparison:

```bash
cargo run -p llama-stage-backend --bin llama_two_stage_decode_compare_probe -- --max-tokens 4
```

Process-boundary stdio roundtrip comparison:

```bash
cargo run -p llama-stage-backend --bin llama_stage_stdio_roundtrip_probe -- --max-tokens 4
```

Process-boundary TCP roundtrip comparison:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_roundtrip_probe -- --max-tokens 4
```

The same local TCP roundtrip with a forced reconnect after the prompt boundary:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_roundtrip_probe -- --max-tokens 4 --reconnect-after-prompt
```

Remote-address TCP comparison against independently started services:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_remote_probe -- --head 127.0.0.1:9001 --tail 127.0.0.1:9002 --max-tokens 4
```

Or force a reconnect after the prompt-to-tail handoff while keeping KV state on the remote nodes:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_remote_probe -- --head 127.0.0.1:9001 --tail 127.0.0.1:9002 --max-tokens 4 --reconnect-after-prompt
```

Interleaved multi-session proof against one head/tail node pair:

```bash
cargo run -p llama-stage-backend --bin llama_stage_tcp_interleave_probe -- --max-tokens 4
```

Long-lived gateway process on top of the head/tail TCP nodes:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_stdio -- --head 127.0.0.1:9001 --tail 127.0.0.1:9002
```

Long-lived TCP gateway service on top of the head/tail TCP nodes:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_node -- --bind 127.0.0.1:9003 --head 127.0.0.1:9001 --tail 127.0.0.1:9002
```

Long-lived stdio client for a remote TCP gateway:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_client_stdio -- --gateway 127.0.0.1:9003
```

Gateway sequential completion proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_stdio_probe -- --max-tokens 4
```

Gateway sequential proof with reconnect after prompt handoff:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_stdio_probe -- --max-tokens 4 --reconnect-after-prompt
```

Gateway interleaved proof through the same gateway process:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_stdio_probe -- --max-tokens 4 --interleave
```

TCP gateway roundtrip proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_roundtrip_probe -- --max-tokens 4
```

TCP gateway roundtrip proof with reconnect after prompt handoff:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_roundtrip_probe -- --max-tokens 4 --reconnect-after-prompt
```

TCP gateway interleaved proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_roundtrip_probe -- --max-tokens 4 --interleave
```

Remote-address TCP gateway proof against an already-running gateway service:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_remote_probe -- --gateway 127.0.0.1:9003 --max-tokens 4
```

Remote-address TCP gateway interleaved proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_remote_probe -- --gateway 127.0.0.1:9003 --max-tokens 4 --interleave
```

Gateway client stdio proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_client_stdio_probe -- --max-tokens 4
```

Gateway client stdio proof with reconnect after prompt handoff:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_client_stdio_probe -- --max-tokens 4 --reconnect-after-prompt
```

Gateway client stdio interleaved proof:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_client_stdio_probe -- --max-tokens 4 --interleave
```

Or with explicit prompts:

```bash
cargo run -p llama-stage-backend --bin llama_two_stage_compare_probe -- ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf "Hello" "The capital of France is"
```

`overall=PASS` means the one-token two-stage split path matched the single-node baseline on the tested prompts.

For the multi-token probe, `overall=PASS` means the staged decode loop matched the single-node baseline on:
- emitted text
- sampled token ids
- completion token count

For the stdio roundtrip probe, the same match guarantee holds, but the head and tail stages run as separate child processes speaking JSON over stdio.

For the TCP roundtrip probe, the same match guarantee holds, but the head and tail stages run as separate socket services speaking the same JSON-line request/response contract over TCP.

For the remote probe, the client does not spawn anything locally. It connects to already-running head and tail TCP stage services and verifies multi-token text and token-id parity against the single-node baseline.

Current service contract:
- decode session state is keyed by `request_id`
- request-boundary reconnect is supported
- interleaved independent decode sessions on the same node pair are supported
- concurrent multi-client accept is not addressed yet; the TCP node still serves one connected client at a time
- the gateway is the intended single warm client for those nodes
- both stdio and TCP gateway surfaces are now available
- the intended upstream caller surface is now the gateway client, not direct stage-node TCP

## Two-machine LAN run

Machine A: start the head node from the installed sidecar:

```bash
~/.compute/bin/llama_stage_tcp_node \
  --model ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --bind 0.0.0.0:9001 \
  --stage-id stage-1 \
  --start-layer 0 \
  --end-layer 20 \
  --head
```

Machine B: start the tail node and the gateway from the installed sidecars:

```bash
~/.compute/bin/llama_stage_tcp_node \
  --model ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --bind 0.0.0.0:9002 \
  --stage-id stage-2 \
  --start-layer 21 \
  --end-layer 41 \
  --tail
```

```bash
~/.compute/bin/llama_stage_gateway_tcp_node \
  --bind 0.0.0.0:9003 \
  --head <MACHINE_A_IP>:9001 \
  --tail 127.0.0.1:9002
```

Then, from either machine, validate the remote gateway end to end:

```bash
cargo run -p llama-stage-backend --bin llama_stage_gateway_tcp_remote_probe -- \
  --gateway <MACHINE_B_IP>:9003 \
  --max-tokens 4
```

For the real `compute-app` relay path, point the daemon at the remote gateway:

```toml
[experimental]
stage_mode_enabled = true
stage_backend = "llama-stage-gateway"
stage_gateway_addr = "<MACHINE_B_IP>:9003"
stage_gateway_autostart = false
```

And validate with the `compute-app` relay harness:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -p compute-daemon --bin llama_stage_gateway_relay_ws_roundtrip -- \
  --gateway <MACHINE_B_IP>:9003 \
  --max-tokens 4
```
