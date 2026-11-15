# Inkling-2-Small on 3× RTX PRO 6000 (sm120) at TP3

Patches and a launch script for serving [mel-bw/Inkling-2-Small-EXL3-3.17bpw](https://huggingface.co/mel-bw/Inkling-2-Small-EXL3-3.17bpw) with vLLM on three 96 GB Blackwell workstation cards. Text and image input are supported.

Tested on 3× RTX PRO 6000 Blackwell Max-Q (250 W, PCIe Gen4 x16, no P2P), Ampere Altra (aarch64, 2.2 GHz), CUDA 13.0, Python 3.12.

## Performance (no speculative decoding; see [DSpark](#dspark-speculative-decoding-work-in-progress))

Single stream, prefix caching on.

| | BF16 KV | FP8 KV | BF16 KV + optional patches | FP8 KV + optional patches |
|---|---|---|---|---|
| Prefill, 8k-token prompt | 4,412 tok/s | 4,463 tok/s | 5,418 tok/s | 5,410 tok/s |
| Prefill, 32k-token prompt | 4,400 tok/s | 4,460 tok/s | 5,367 tok/s | 5,435 tok/s |
| Decode | 86.1 tok/s | 85.3 tok/s | 106.3 tok/s | 106.5 tok/s |
| KV cache capacity | 936,461 tokens | 1,885,836 tokens | 858,852 tokens | 1,729,548 tokens |
| Context window | 850,000 tokens | 1,000,000 tokens | 850,000 tokens | 1,000,000 tokens |

"Optional patches" is `patches/vllm/optional/0004` and `0005` plus prefill CUDA graphs (`GRAPHS=1`). Prefill is measured on unique random-token prompts (no prefix-cache hits); decode is the mean over four chat prompts, 512 tokens each. KV capacity is what vLLM allocates at `--gpu-memory-utilization 0.98`.

The optional patches target this test system: GPUs without P2P and a slow aarch64 host CPU. On systems with P2P, or with a faster x86 host CPU, the gains from the optional patches may be smaller.

Quality, from the Local Inference Lab [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) profiles, 10 concurrent requests against `--max-num-seqs 4`. Scores are correct answers out of completed runs.

| | BF16 KV + optional patches | BF16 KV + optional patches + DSpark (`SPEC_K=7`) | FP8 KV + optional patches + DSpark (`SPEC_K=7`) |
|---|---|---|---|
| Estonia | 29/30 | 28/30 | 29/30 |
| LAVD | 30/30 | 30/30 | not run yet |
| Hotel lights | 24/24 | 21/21 | not run yet |

## Base versions

| Component | Base | Patches |
|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) `efe12305f064b9c5301231d66eb42b7c0db5895f` | `patches/vllm/0001`–`0003`, in order; optionally `patches/vllm/optional/0004`–`0005` |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` | `patches/exllamav3/` |
| Decode-once MoE kernel | `xmoe_do` from [diffbot's recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), plus [`do_gemm.diff`](https://github.com/meldavy/dsv41f-1017-sm120-tp3/blob/main/patches/kernel/do_gemm.diff) | `patches/kernel/` |
| FlashInfer | `flashinfer-python==0.7.3` from PyPI | none |
| torch / triton / transformers | `2.14.0+cu130` / `3.8.0` / `5.22.2` | none |

## Patches

Required to run the model.

**vLLM**

| Patch | What it does |
|---|---|
| `0001-pr67325-inkling2-small.diff` | Inkling 2 Small model support, [#67325](https://github.com/vllm-project/vllm/pull/67325) as of `d664cd1e`. |
| `0002-pr55078-pr51560-inkling-triton-fallback.diff` | Triton attention fallback for Inkling on SM8x ([#55078](https://github.com/vllm-project/vllm/pull/55078)) and SM12x ([#51560](https://github.com/vllm-project/vllm/pull/51560)), merged on top of 0001. |
| `0003-inkling2-sm120-tp3.diff` | SM120 kernels, EXL3 experts, TP3, fixed-size conv state, FP8 KV cache, vision and DSpark on SM120. Details below. |

What 0003 changes:

- **SM120 relative attention.** The PR's relative attention is a `tml_fa4` kernel that runs only on SM100, and 0002's Triton kernels (#51560, #55078) are for Inkling 1, which lays out its relative logits differently. `ops/triton_rel_attention_sm12x.py` follows their design (paged Triton, split-KV decode), reads Inkling 2's sheared relative logits, and returns the fp32 output and LSE that `gate_sink` expects. Selected in `ops/attention.py` on SM12x.
- **SM120 shared memory.** SM120 allows 99 KB of shared memory per block against 227 KB on SM100, so the PR's GEMM configs don't fit. `ops/linear.py` caps tiles and pipeline stages, disables TMA, and falls back to torch where CuTe-DSL is unsupported.
- **Fixed-size conv state.** The PR keeps the short-conv history per token in the paged KV cache (24 KiB per token per layer), which caps context near 65k on this setup. `conv_state.py` keeps a fixed per-request state on vLLM's Mamba state path (`MambaBase` and the ShortConv backend), so prefix caching works through `--mamba-cache-mode align`. Wired through `sconv.py`, `ops/sconv.py`, `attention.py` and `model.py`.
- **EXL3 experts.** The NVFP4 experts don't fit in 3× 96 GB alongside a 1M-token context. `exl3_moe.py` runs EXL3 experts, sliced for TP, on turboderp's ExLlamaV3 cooperative MoE kernel for decode and diffbot's decode-once grouped GEMM for prefill, the same pairing as the [DS4.1 recipe](https://github.com/meldavy/dsv41f-1017-sm120-tp3). Enabled with `INKLING2_EXL3=1`.
- **TP3 head padding.** 8 KV heads don't divide by 3. This follows the local-inference-lab KK fork's TP3 padding for DeepSeek V4.1, done at load time: `tp_pad.py`, called from a config hook in `model_executor/models/config.py`, pads 48 query / 8 KV heads to 54 / 9 with zero-padded heads for the target and DSpark models. Outputs are unchanged.
- **FP8 KV cache.** vLLM's `--kv-cache-dtype fp8` (e4m3), implemented in the SM120 attention kernels. See [KV cache: BF16 and FP8](#kv-cache-bf16-and-fp8).
- **Vision on SM120.** The encoder's `tml_fa4` attention runs only on SM100. Elsewhere it uses vLLM's `MMEncoderAttention` (`vision.py`).
- **DSpark on SM120.** The draft model's block attention is a Gluon kernel that runs only on SM100. `ops/block_attn_sm12x.py` is a split-KV Triton replacement. The cross-rank argmax uses FlashInfer's NVLink multicast when available and an all-gather of per-rank maxima otherwise (`dspark.py`). The conv state follows vLLM's speculative short-conv layout. See [DSpark](#dspark-speculative-decoding-work-in-progress).

**ExLlamaV3 and the decode-once kernel.** Both patches add Inkling 2's soft-capped SwiGLU, `4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`, as activation id 5. In ExLlamaV3 that is the cooperative MoE kernel and `hadamard_inner.cuh`; in the decode-once kernel it is `xmoe/quant/hadamard_inner.cuh`.

## Optional patches

Two patches in `patches/vllm/optional/`, applied in order on top of 0003. Outputs are unchanged.

Each patch's effect, BF16 KV cache, single stream (same method as the performance table):

| | 0003 | + 0004 | + 0004 + 0005 |
|---|---|---|---|
| Prefill, 8k-token prompt | 4,412 tok/s | 4,296 tok/s | 5,418 tok/s |
| Prefill, 32k-token prompt | 4,400 tok/s | 4,353 tok/s | 5,367 tok/s |
| Decode | 86.1 tok/s | 98.8 tok/s | 106.3 tok/s |

The "+ 0004" and "+ 0004 + 0005" columns use prefill CUDA graphs (`GRAPHS=1`).

**`0004-sm120-low-batch-kernels.diff`**: decode kernels for SM120, independent of the GPU interconnect.

| Change | Files | What it addresses |
|---|---|---|
| Low-M GEMV | `ops/gemv_sm12x.py`, `ops/linear.py` | The PR sends M ≤ 16 GEMMs to FlashInfer's CuTe-DSL low-M kernels, which don't run on SM120, and its cuBLAS fallback runs well below memory bandwidth. A Triton split-K GEMV handles up to 16 rows, including the router. |
| Fused routing | `exl3_moe.py` | The PR's router (sigmoid, bias, top-8, gather, normalize) runs as separate torch ops. For decode-size batches a single Triton kernel computes the same math, re-zeroes the persistent split-K router-logit buffer it reads (no zero-fill kernel), and writes the experts' fp16 input (no conversion kernel). |
| Cooperative-kernel rows | `exl3_moe.py` | `INKLING2_COOP_MAX_ROWS` (default 8) sets how many rows run on ExLlamaV3's cooperative MoE kernel. `serve.sh` sets 32 with DSpark, so verify batches of up to 4 requests stay on it. |

**`0005-no-p2p-allreduce-overlap.diff`**: tensor-parallel communication for GPUs without P2P. At startup it checks CUDA peer access between the TP ranks; with peer access, both custom all-reduces stay off and all reductions use vLLM's all-reduce. `INKLING2_HOST_AR=1` and `INKLING2_DMA_AR=1` force them on, `=0` forces them off.

| Change | Files | What it addresses |
|---|---|---|
| Host-mapped all-reduce | `ops/host_allreduce.py` | vLLM's custom all-reduce and FlashInfer's Lamport all-reduce need P2P, and vLLM's doesn't support TP3. Without P2P, NCCL uses its SHM transport at about 18 µs per small call. Decode-size tensors (up to 64 KB) instead use a one-shot all-reduce in host-mapped memory with NCCL's LL protocol (4-byte data + 4-byte flag per 8-byte word), at about 6 µs per call. |
| Copy-engine all-reduce | `ops/dma_allreduce.py` | NCCL's SHM transport tops out near 9.4 GB/s here. Prefill tensors of 4 MB or more use a reduce-scatter + all-gather through pinned host memory on the copy engines, synchronized with stream memory operations (`cuStreamWriteValue64` / `cuStreamWaitValue64`), near the PCIe limit. CUDA-graph safe. |
| Two-microbatch prefill overlap | `ubatch.py`, `model.py`, `attention.py`, `sconv.py`, `conv_state.py` | Same idea as DeepSeek's two-batch overlap and vLLM's dual-batch overlap (DBO), but vLLM's DBO overlaps expert-parallel all-to-all, not TP all-reduce. Prefill runs as two microbatches staggered per layer, so each one's all-reduce overlaps the other's compute. |

**Prefill CUDA graphs** are a launch option (`GRAPHS=1`): prefill is captured at 256 to 4096 tokens to remove host launch overhead. With 0005, the overlap's all-reduce launch and wait run as eager breaks through vLLM's breakable CUDA graphs.

## KV cache: BF16 and FP8

`KV_DTYPE=auto` (BF16) or `KV_DTYPE=fp8` in `serve.sh`.

With FP8, K and V are stored as e4m3 with a scale of 1.0. Values are cast in registers when the attention prologue writes them, clamped to ±448, and converted back to BF16 in registers after each load in the attention kernels; there is no separate conversion pass. Inkling's V is RMS-normalized and K is norm-clipped before the cache write, so both sit well inside the e4m3 range. The DSpark draft layers use the same cache dtype.

| | BF16 | FP8 |
|---|---|---|
| KV cache capacity (optional patches) | 858,852 tokens | 1,729,548 tokens |
| Negative log-likelihood, 8k-token real text | 1.0351 / 1.0353 (two runs) | 1.0363 / 1.0366 (two runs) |
| Mean \|Δ log p\| against a BF16 reference run | 0.030 (BF16 run-to-run) | 0.040 |
| Top-1 token agreement with a BF16 reference run | 98.6% (BF16 run-to-run) | 98.0–98.1% |
| Decode | 106.3 tok/s | 106.5 tok/s |
| Prefill, 32k-token prompt | 5,367 tok/s | 5,435 tok/s |
## DSpark speculative decoding (work in progress)

`SPEC=1` in `serve.sh` enables the checkpoint's DSpark draft model (`mtp.safetensors`). The draft model always fills its trained 8-position block (the accepted token plus 7 masked positions); `SPEC_K` (1–7, default 3) sets how many of its drafts the main model verifies per step with one request in flight; `SPEC_K_BATCH` (default 2) applies when two or more requests share a step, since the verify cost grows with every request. Output quality matches decoding without DSpark: token log-probabilities along DSpark-decoded text agree with a plain prefill over the same tokens to within the same margin as plain decoding.

**`SPEC_K=3` is the recommended starting value.** Raise it when your drafts are accepted more often (code, structured output) or when a verify step is cheap relative to a decode step on your hardware (faster GPUs, P2P interconnect), and measure with your own workload. On this system, single stream, BF16 KV, optional patches (one sweep; separate sessions vary by about ±5%):

| `SPEC_K` | 7 | 4 | 3 | Without DSpark |
|---|---|---|---|---|
| Decode, mean of four prompts | 117.6 tok/s | 130.3 tok/s | 130.8 tok/s | 104.8 tok/s |
| Decode, prose prompt | 74.3 tok/s | 83.6 tok/s | 95.8 tok/s | 105.0 tok/s |

All further numbers in this section use `SPEC_K=3` and the optional patches. Single stream, 512 tokens per prompt:

| Prompt | Decode, BF16 KV | Decode, FP8 KV | Accepted tokens per step (BF16 / FP8) |
|---|---|---|---|
| Python LRU cache | 130.1 tok/s | 126.3 tok/s | 2.32 / 2.25 |
| History of Tallinn | 90.9 tok/s | 92.6 tok/s | 1.62 / 1.64 |
| Train word problem | 153.4 tok/s | 146.9 tok/s | 2.75 / 2.63 |
| 40 words starting with "st" | 124.9 tok/s | 147.6 tok/s | 2.30 / 2.65 |
| Without DSpark | 106.3 tok/s | 106.5 tok/s | — |

Single-stream DSpark speed follows acceptance, which depends on the exact generated text; that text varies slightly between server sessions (prefill reductions are not bit-deterministic), so the per-prompt numbers above vary by about ±5% between sessions. The steady-state c=4 numbers below do not.

Concurrency: c=1 runs the four prompts one at a time (`SPEC_K=3`). c=4 (`SPEC_K_BATCH=2`) keeps four requests in flight for 90 s (each finished request is immediately replaced, prompts mixed), measured three times; ranges in parentheses.

| | BF16 KV | BF16 KV + DSpark | FP8 KV | FP8 KV + DSpark |
|---|---|---|---|---|
| c=1, per request | 106.3 tok/s | 124.8 tok/s | 106.5 tok/s | 128.4 tok/s |
| c=4, per request | 55.5 tok/s (55.4–55.6) | 56.4 tok/s (56.1–56.9) | 55.5 tok/s (55.3–55.6) | 56.4 tok/s (56.0–56.8) |
| c=4, total | 214.7 tok/s (214.4–214.9) | 213.7 tok/s (213.1–214.1) | 214.8 tok/s (214.6–215.1) | 214.6 tok/s (213.7–215.7) |

Capacity and prefill:

| | BF16 KV | BF16 KV + DSpark | FP8 KV | FP8 KV + DSpark |
|---|---|---|---|---|
| KV cache capacity | 858,852 tokens | 519,922 tokens | 1,729,548 tokens | 1,038,326 tokens |
| Prefill, 32k-token prompt | 5,367 tok/s | 5,306 tok/s | 5,435 tok/s | 5,164 tok/s |

Draft acceptance by position, over the Estonia and LAVD reasoning benchmarks (161k verify steps):

| Draft position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Accepted | 64% | 50% | 42% | 37% | 33% | 29% | 27% |

Status:

- A verify step checks `SPEC_K` + 1 tokens, and each token routes to 8 of 384 experts, so the expert weights a step reads grow almost linearly with `SPEC_K`. The cost of a verify step against a single-token decode step sets how many accepted tokens per step DSpark needs to be faster.
- Under sustained 4-request load, DSpark with `SPEC_K_BATCH=2` matches decoding without it in total throughput (table above): its gain is in single-stream decode.
- The draft model's weights, its full-attention layer and its CUDA graphs reduce KV cache capacity.
- At `SPEC_K=3`, a full step takes 17.2 ms, of which the draft model takes 1.8 ms (CUDA-event timing of graph replays); the rest is the target model's verify pass, dominated by its MoE layers.
- vLLM's adaptive verification (`enable_adaptive_verification`) runs and produces correct output with these patches, but on this host its per-step bookkeeping costs more than it saves, so `serve.sh` leaves it off.

## Docker

Prebuilt images contain vLLM with all patches in this repo (including the optional ones), ExLlamaV3 and the decode-once prefill kernel, built from [`docker/Dockerfile`](docker/Dockerfile):

| Image | Platforms |
|---|---|
| [`ghcr.io/meldavy/inkling2-small-sm120-tp3:latest`](https://github.com/meldavy/inkling2-small-sm120-tp3/pkgs/container/inkling2-small-sm120-tp3) | `linux/amd64`, `linux/arm64` |

The host needs an NVIDIA driver with CUDA 13.0 support (580 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. Download the model:

   ```bash
   hf download mel-bw/Inkling-2-Small-EXL3-3.17bpw --local-dir /path/to/Inkling-2-Small-EXL3-3.17bpw
   ```

2. Save [`docker/docker-compose.yml`](docker/docker-compose.yml) and start the server:

   ```bash
   curl -fsSLO https://raw.githubusercontent.com/meldavy/inkling2-small-sm120-tp3/main/docker/docker-compose.yml
   MODEL_DIR=/path/to/Inkling-2-Small-EXL3-3.17bpw docker compose up -d
   docker compose logs -f        # ready when "Application startup complete" appears, about 5 minutes
   ```

3. Send a request:

   ```bash
   curl http://localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "Inkling-2-Small", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

The compose file runs FP8 KV, DSpark at `SPEC_K=3`/`SPEC_K_BATCH=2` and a 1,000,000-token context on 3 GPUs. Every variable in [Launch](#launch) is set through `environment:`; `TP` sets the number of GPUs (and the compose `count`). The same container without compose:

```bash
docker run -d --gpus all --ipc host --shm-size 32g -p 8300:8300 \
  -v /path/to/Inkling-2-Small-EXL3-3.17bpw:/model:ro \
  -e KV_DTYPE=fp8 -e SPEC=1 ghcr.io/meldavy/inkling2-small-sm120-tp3:latest
```

To build the image yourself (about an hour, mostly vLLM), from the repo root:

```bash
docker build -f docker/Dockerfile -t inkling2-small-sm120 .                                    # all patches
docker build -f docker/Dockerfile --build-arg OPTIONAL_PATCHES=0 -t inkling2-small-sm120:base .  # patches 0001-0003 only
```

## Build

On glibc 2.42 or newer, CUDA 13.0's nvcc needs the `math_functions.h` workaround from [Appendix E of the DS4.1 recipe](https://github.com/meldavy/dsv41f-1017-sm120-tp3#appendix-e-glibc-243--cuda-130-workaround).

```bash
export SRC=/path/to/src REPO=/path/to/this/repo
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0a

# vLLM
git clone https://github.com/vllm-project/vllm $SRC/vllm && cd $SRC/vllm
git checkout efe12305f064b9c5301231d66eb42b7c0db5895f
for p in $REPO/patches/vllm/*.diff; do git apply "$p"; done
for p in $REPO/patches/vllm/optional/*.diff; do git apply "$p"; done   # optional, 0004 then 0005
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
CMAKE_CUDA_ARCHITECTURES=120a NVCC_THREADS=4 MAX_JOBS=48 pip install --no-build-isolation -e .

# ExLlamaV3 (built in place, not pip-installed)
git clone https://github.com/turboderp-org/exllamav3 $SRC/exllamav3 && cd $SRC/exllamav3
git checkout a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a
git apply $REPO/patches/exllamav3/*.diff
MAX_JOBS=40 python setup.py build_ext --inplace

# Decode-once prefill kernel
mkdir -p $SRC/xmoe_do && cd $SRC/xmoe_do
cp -r <diffbot repo>/recipe/kernels/xmoe ./xmoe && cp <diffbot repo>/recipe/kernels/xmoe_do/do_gemm.cu .
patch -p1 < do_gemm.diff                                # from the DS4.1 recipe
patch -p1 < $REPO/patches/kernel/*.diff
python -c 'from torch.utils.cpp_extension import load; load(name="ds41_exl3_do", sources=["do_gemm.cu"], extra_include_paths=["xmoe", "xmoe/quant"], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], build_directory="build")'
```

## Launch

```bash
hf download mel-bw/Inkling-2-Small-EXL3-3.17bpw --local-dir /path/to/Inkling-2-Small-EXL3-3.17bpw
MODEL=/path/to/Inkling-2-Small-EXL3-3.17bpw SRC=$SRC $REPO/scripts/serve.sh
```

[`scripts/serve.sh`](scripts/serve.sh) serves an OpenAI-compatible API on port 8300 as `Inkling-2-Small`, with reasoning and tool-call parsing. Startup takes about 5 minutes.

| Variable | Values |
|---|---|
| `KV_DTYPE` | `auto` (BF16, default) or `fp8` |
| `GRAPHS` | `1`: prefill CUDA graphs and the two-microbatch prefill overlap; the default when optional patch 0005 is applied. `0`: eager prefill; the default without 0005, where prefill graphs only cost KV capacity. |
| `SPEC` | `1`: DSpark speculative decoding |
| `SPEC_K` | DSpark drafts verified per step with one request, 1–7, default 3 |
| `SPEC_K_BATCH` | DSpark drafts verified per request with two or more requests, default 2 |
| `IMAGES` | maximum images per prompt; default: vLLM's limit (999). `0` for text only. |
| `PROFILE` | `fast` (default): 4k prefill chunks, 1,000,000-token context. `context`: 2k chunks, 1,048,576 tokens. |
| `UTIL` | GPU memory fraction for vLLM, default 0.98. 0.98 is recommended for a 1M-token context. |
| `MAXLEN` | context window; lower it to fit the KV capacity of the chosen configuration |

## Known limitations

- Prefix-cache hits land only where an earlier prompt ended, so multi-turn chat hits but unrelated prompts sharing a system prompt don't. This is how align mode works for every Mamba-style model.
- The first prefix-cache hit after startup pays a one-time ~3 s kernel compile.

## Credits

- [Thinking Machines](https://huggingface.co/thinkingmachines): Inkling 2 Small.
- [vLLM](https://github.com/vllm-project/vllm): the serving engine, including its DSpark speculator, per-batch-size draft schedule, speculative short-conv metadata, breakable CUDA graphs and multimodal encoder attention. Inkling 2 support by WoosukKwon ([#67325](https://github.com/vllm-project/vllm/pull/67325)); Triton attention fallbacks for Inkling by m4xkushnir ([#51560](https://github.com/vllm-project/vllm/pull/51560)) and TheVerdman ([#55078](https://github.com/vllm-project/vllm/pull/55078)).
- turboderp: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and its EXL3 cooperative MoE kernel.
- diffbot: the decode-once EXL3 MoE kernel ([DeepSeek-V4.1-Flash recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)).
- [Local Inference Lab](https://github.com/local-inference-lab): the KK vLLM fork's TP3 head padding and [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (Estonia, LAVD and hotel-lights benchmarks). JCartu and TheRez for the DSpark draft-count findings on RTX PRO 6000 that `SPEC_K=3` follows.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): kernels and NVLink all-reduce paths used by the vLLM PR.
- NVIDIA [NCCL](https://github.com/NVIDIA/nccl): the LL protocol the host-mapped all-reduce follows.
- DeepSeek's two-batch overlap and vLLM's dual-batch overlap: the idea behind the two-microbatch prefill overlap.

Work is tracked in the Local Inference Lab Discord.
