---
summary: GPU kernels in Triton and CUDA, and the number formats they run on.
---

## Kernels

- Fixed vLLM-Omni's Triton SnakeBeta activation kernel for bf16 and fp16 inputs ([#3472](https://github.com/vllm-project/vllm-omni/pull/3472))

## Quantization

All in vLLM-Omni:

- FP8 for the Qwen2.5-Omni thinker language model ([#3466](https://github.com/vllm-project/vllm-omni/pull/3466), in review)
- Loading Comfy INT8 ConvRot checkpoints for MiniMax-H3 ([#6894](https://github.com/vllm-project/vllm-omni/pull/6894), in review)
- Initialize component-level quantization base state ([#5103](https://github.com/vllm-project/vllm-omni/pull/5103))

## Learning

Most of this layer is still study: CUDA GEMM from a naive kernel up to tensor cores, Triton, and FlashAttention. Notes land on the blog under this layer.
