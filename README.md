# Compute Sharding

Compute Sharding is a standalone CLI for the open two-stage sharded inference work behind Compute's split-backend experiments.

The first public target is intentionally narrow:

- model: `gemma-4-e4b-q4`
- split: head layers `0-20`, tail layers `21-41`
- artifacts: ComputeNet Hugging Face GGUF stage shards
- runtime: patched `llama.cpp` stage sidecars
- pairing: local peer orchestrator that selects the lowest-latency compatible counterpart
- UI: Compute-branded terminal interface with spinning globe, peer list, logs, and a test chat tab

This repo is for the sharding path while Compute continues building the larger MoE/hybrid system.

## Quickstart

Install from source:

```bash
cargo install --path .
```

Download the validated shards:

```bash
compute-sharding download both
```

Run a tail node:

```bash
compute-sharding serve \
  --role tail \
  --bind 0.0.0.0:8787 \
  --stage-bind 0.0.0.0:9202 \
  --public-addr http://YOUR_TAIL_HOST:8787 \
  --public-stage-addr YOUR_TAIL_HOST:9202
```

Run a head node and let it pick the lowest-latency tail from the peers you provide:

```bash
compute-sharding serve \
  --role head \
  --peer http://YOUR_TAIL_HOST:8787 \
  --bind 0.0.0.0:8787 \
  --public-addr http://YOUR_HEAD_HOST:8787
```

Open the TUI on the head node:

```bash
compute-sharding tui --gateway 127.0.0.1:9300
```

Send a one-shot test prompt:

```bash
compute-sharding chat --gateway 127.0.0.1:9300 "Explain distributed inference in one sentence."
```

## Sidecars

The CLI expects these patched backend binaries when `serve` is allowed to spawn stages:

- `llama_stage_tcp_node`
- `llama_stage_gateway_tcp_node`

Search order:

1. `--sidecar-dir`
2. `$COMPUTE_SHARDING_SIDECAR_DIR`
3. `~/.compute/bin`
4. sibling `compute-backend/target/release`
5. sibling `compute-backend/target/debug`

Run:

```bash
compute-sharding info
```

to print the model catalog and sidecar paths.

## Networking

Each node runs a tiny local orchestrator:

- `GET /health`
- `GET /peers`
- `POST /register`
- `POST /chat`

Head nodes select the lowest-latency reachable tail that advertises the same model. The tail stage TCP address must be reachable by the head. For home internet, that usually means port forwarding, a VPN, or a tunnel.

## Current Boundary

This is not an arbitrary sharding framework yet. The public release focuses on the validated dual-shard path so contributors can reproduce and improve the real split-backend work without the full Compute app stack.

See [docs/architecture.md](docs/architecture.md) and [docs/results.md](docs/results.md).
