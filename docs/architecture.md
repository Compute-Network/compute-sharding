# Architecture

Compute Sharding separates the public orchestration harness from the patched native backend.

## Pieces

```text
head node
  compute-sharding orchestrator
  llama_stage_tcp_node       head layers 0-20
  llama_stage_gateway_tcp_node
        |
        | TCP hidden-state handoff
        v
tail node
  compute-sharding orchestrator
  llama_stage_tcp_node       tail layers 21-41
```

The orchestrator handles discovery, peer registration, latency probing, and gateway startup. The native sidecars do the actual GGUF/llama.cpp work.

## Model

Default model:

- `gemma-4-e4b-q4`
- head shard: `gemma-4-e4b-q4-head-0-20.gguf`
- tail shard: `gemma-4-e4b-q4-tail-21-41.gguf`
- Hugging Face repo: `ComputeNet-sh/gemma-4-e4b-q4-gguf-stages`

The split is fixed in this first release because it matches the validated path from the Compute app work.

## Pairing

Each node advertises:

- node id
- role: `head` or `tail`
- model id
- shard kind
- orchestrator URL
- stage TCP address
- gateway TCP address, for head nodes

Head nodes measure HTTP latency to candidate tail orchestrators and select the lowest-latency compatible tail. Once selected, the head starts the local gateway against:

- local head stage address
- remote tail stage address

## Why Layer Sharding First

MoE expert routing can be more efficient at high concurrency, but it also creates many more routing decisions and network exchanges per generated token. Layer sharding is the practical open path first because it has fewer network round trips and directly maps to the backend already validated by Compute.

The future direction is hybrid: pipeline by layer across the open network, then use MoE/expert sharding inside very low-latency clusters.
