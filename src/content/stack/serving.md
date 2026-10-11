---
summary: Inference engines for omni-modal models, from request scheduling down to the KV cache.
---

## vLLM-Omni

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) extends vLLM to speech, audio, image and video models, which usually run as multi-stage pipelines. I have contributed since December 2025.

### Models

- Integrated Step-Audio2 ([#464](https://github.com/vllm-project/vllm-omni/pull/464))
- Stable Audio 3, a text-to-audio diffusion model ([#3973](https://github.com/vllm-project/vllm-omni/pull/3973), in review)

### Fixes

- Duplex sessions: keep cancellation in session control queues ([#7782](https://github.com/vllm-project/vllm-omni/pull/7782)), refresh client defaults after a session update ([#7784](https://github.com/vllm-project/vllm-omni/pull/7784)), keep queued input reservations across a clear ([#7783](https://github.com/vllm-project/vllm-omni/pull/7783), in review)
- Stage pipeline: default missing input sources for legacy stage configs ([#5498](https://github.com/vllm-project/vllm-omni/pull/5498))
- CI: fix Step-Audio2 KV cache allocation on the CI GPU ([#6839](https://github.com/vllm-project/vllm-omni/pull/6839))

### Now

- Generalizing eager MTP across TTS models, from the Q4 2026 roadmap ([#8264](https://github.com/vllm-project/vllm-omni/issues/8264))

## vllm-rlt

[vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt) is a vLLM-based engine for looped transformers. In its refactor I own M5: the executor, the worker and the model runner.

- Documented who owns what across the worker and model runner, what I found, and the migration plan ([#110](https://github.com/ThinkFlowLab/vllm-rlt/pull/110))
- Found a bug in async scheduling: a preempted request that resumes in a new slot reads another request's token. Reproduced it on GPU and fixed it ([#109](https://github.com/ThinkFlowLab/vllm-rlt/pull/109))

## semantic-router

Member of two working groups in [semantic-router](https://github.com/vllm-project/semantic-router): router models and inference runtime, and developer experience.

- Offline compatibility receipts for the Candle CPU runtime ([#3851](https://github.com/vllm-project/semantic-router/pull/3851))
- A test that pins the vLLM-Omni image generation request contract ([#3144](https://github.com/vllm-project/semantic-router/pull/3144)), plus fixes to the agent smoke test and Podman socket test ([#3154](https://github.com/vllm-project/semantic-router/pull/3154), [#3161](https://github.com/vllm-project/semantic-router/pull/3161))
