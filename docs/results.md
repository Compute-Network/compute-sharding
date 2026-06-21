# Sharding Results

These are the headline results from the Compute sharding work that motivated this standalone release.

| Test Family | Result | Meaning |
|---|---:|---|
| Early split baseline | `2-4 tok/s` | Initial split execution proved the handoff path but was not fast enough. |
| First optimized split | `~20 tok/s` | Moving more of the heavy path into the patched backend removed a large amount of orchestration overhead. |
| Patched llama.cpp split gateway | `41.9-50.6 tok/s` | The validated split-backend path crossed the 40 TPS mark. |
| Later lab peaks | `61-79 tok/s` | Further backend tuning showed the ceiling was much higher than the first working prototype. |

The key change was keeping the expensive model execution inside patched `llama.cpp`/GGML and using the Compute layer as the coordinator, downloader, peer selector, and gateway harness.

This repo packages the public, reproducible part of that work: shard download, local orchestration, peer latency selection, gateway launch, and test chat.
