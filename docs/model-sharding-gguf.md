# GGUF Model Sharding Guide

This guide documents how the public Compute Gemma 4 E4B shards were made, and
what needs to change when adapting the same approach to another GGUF model
family such as Qwen or GLM.

The important point: these shards are model-aware GGUF rewrites, not byte-range
file splits. Each shard is a valid GGUF with local layer numbering and metadata
that matches the tensors it contains.

## Current Public Shards

The first release uses:

- source model: `gemma-4-E4B-it-Q4_K_M.gguf`
- architecture: `gemma4`
- total layers: `42`
- head shard: original layers `0-20`
- tail shard: original layers `21-41`
- output artifacts:
  - `gemma-4-e4b-q4-head-0-20.gguf`
  - `gemma-4-e4b-q4-tail-21-41.gguf`

The published artifacts live at:

```text
https://huggingface.co/ComputeNet-sh/gemma-4-e4b-q4-gguf-stages
```

## Reproduce The Gemma 4 Shards

Build the sharding writer:

```bash
cargo build --release -p stage-forward-lab --bin write_gemma4_stage_ggufs
```

Run it against the full GGUF:

```bash
mkdir -p out/gemma-4-e4b-q4-shards

cargo run --release -p stage-forward-lab --bin write_gemma4_stage_ggufs -- \
  ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf \
  out/gemma-4-e4b-q4-shards \
  20 21 41
```

The writer emits:

```text
out/gemma-4-e4b-q4-shards/head-0-20.gguf
out/gemma-4-e4b-q4-shards/tail-21-41.gguf
```

Rename them to the public catalog names:

```bash
mv out/gemma-4-e4b-q4-shards/head-0-20.gguf \
  out/gemma-4-e4b-q4-shards/gemma-4-e4b-q4-head-0-20.gguf

mv out/gemma-4-e4b-q4-shards/tail-21-41.gguf \
  out/gemma-4-e4b-q4-shards/gemma-4-e4b-q4-tail-21-41.gguf
```

Inspect the outputs:

```bash
cargo run --release -- inspect-model --metadata --tensors \
  out/gemma-4-e4b-q4-shards/gemma-4-e4b-q4-head-0-20.gguf

cargo run --release -- inspect-model --metadata --tensors \
  out/gemma-4-e4b-q4-shards/gemma-4-e4b-q4-tail-21-41.gguf
```

## What The Writer Does

The Gemma 4 writer is:

```text
crates/stage-forward-lab/src/bin/write_gemma4_stage_ggufs.rs
```

For each shard it:

1. Parses the source GGUF and verifies `general.architecture = gemma4`.
2. Chooses a contiguous original layer range.
3. Copies global metadata, then rewrites `gemma4.block_count` to the local
   layer count.
4. Adjusts `gemma4.attention.shared_kv_layers` for Gemma 4 shared-KV layout.
5. Slices any metadata arrays whose length equals the full layer count.
6. Includes only tensors needed by that shard.
7. Renumbers layer tensors from global names like `blk.21.*` to local names
   like `blk.0.*` inside the tail shard.
8. Slices Gemma-specific per-layer tensors:
   - `per_layer_model_proj.weight`
   - `per_layer_token_embd.weight`
9. Preserves the original GGUF quantization bytes where possible.
10. Rewrites the GGUF header, tensor table, aligned offsets, and tensor data.

That is why the runtime can load the head shard as a local `0-20` model and the
tail shard as a local `0-20` model, while the orchestrator still advertises the
original global ranges.

## Porting The Method To Another Model

For another GGUF model family, the process is the same but the architecture
adapter changes.

Use this checklist:

1. Start from a single working GGUF that already loads in llama.cpp.
2. Inspect metadata:

```bash
cargo run --release -- inspect-model --metadata --tensors /path/to/model.gguf
```

3. Identify:
   - architecture name from `general.architecture`
   - layer count metadata key
   - hidden size metadata key
   - layer tensor naming pattern
   - tokenizer and output tensors that must be present on one or both shards
   - any cross-layer, shared-KV, sliding-window, MoE, or per-layer auxiliary
     tensors that cannot be blindly copied
4. Write an architecture-specific shard writer based on
   `write_gemma4_stage_ggufs.rs`.
5. Keep shard ranges contiguous for the first version.
6. Rewrite each shard so local layer tensors start at `blk.0.*`.
7. Slice any architecture-specific tensors that are indexed by original layer.
8. Preserve quantized tensor bytes unless a tensor has to be row-sliced.
9. Add a catalog entry in `src/models.rs`.
10. Run a single-node baseline and a split-stage correctness probe before
    benchmarking speed.

For dense transformer models, layer sharding is usually straightforward once
the tensor map is known. For MoE models, the first practical version should keep
all experts for a layer inside the same layer shard. Splitting individual
experts across machines is a different routing problem and needs a separate
runtime path.

## Agent Prompt

You can point Codex, Claude, or another coding agent at this repository and use:

```text
Read docs/model-sharding-gguf.md and the Gemma 4 shard writer at
crates/stage-forward-lab/src/bin/write_gemma4_stage_ggufs.rs.

Create a shard writer for <MODEL_ARCHITECTURE> GGUF that produces two valid
stage GGUFs with local layer numbering. Start with contiguous layer ranges.
Preserve quantized tensor bytes where possible. Identify and correctly slice
any metadata arrays or tensors indexed by layer. Add catalog entries and a
minimal correctness probe before doing performance work.
```

Replace `<MODEL_ARCHITECTURE>` with the GGUF architecture reported by
`inspect-model`.

## Boundaries

This repo currently ships the validated Gemma 4 E4B dual-shard path. The method
can be adapted to other llama.cpp-supported GGUF architectures, but each model
family needs an architecture-aware tensor map and runtime validation. In other
words: Gemma was chosen for accessibility and reproducibility; the same pattern
is meant to extend to Qwen, GLM, and other GGUF-backed models once their tensor
layout and llama.cpp support are handled.
