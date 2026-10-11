---
summary: Where everything above ends up. Keeping the device busy, and keeping its state right.
---

## Async execution

With async scheduling, the device runs ahead of the scheduler, so state on the device and state on the host can disagree. The vllm-rlt bug in [#109](https://github.com/ThinkFlowLab/vllm-rlt/pull/109) was exactly this: a request that resumed in a new slot read the token another request had left in that slot's device buffer.

## Memory

- Fixed Step-Audio2's KV cache allocation on the CI GPU ([#6839](https://github.com/vllm-project/vllm-omni/pull/6839))

## Tools

CUDA graphs, torch.compile, and profiling of multi-stage TTS pipelines.
