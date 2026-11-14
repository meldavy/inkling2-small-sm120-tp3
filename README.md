# Inkling-2-Small on 3× RTX PRO 6000 (sm120) at TP3

Patches and a launch script for serving [mel-bw/Inkling-2-Small-EXL3-3.17bpw](https://huggingface.co/mel-bw/Inkling-2-Small-EXL3-3.17bpw) with vLLM on three 96 GB Blackwell workstation cards. Preliminary: work in progress.

| | |
|---|---|
| Prefill | ~5,400 tok/s at 8k prompts, ~5,000 at 32k |
| Decode | ~88–93 tok/s, one stream, no speculative decoding yet |
| Context | 1,000,000 tokens, BF16 KV cache, prefix caching on |

Tested on 3× RTX PRO 6000 Blackwell Max-Q (250 W, PCIe Gen4 x16, no P2P), Ampere Altra (aarch64), CUDA 13.0, Python 3.12.

## Base versions

| Component | Base | Patches |
|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) `efe12305f064b9c5301231d66eb42b7c0db5895f` | `patches/vllm/0001`–`0003`, in order |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` | `patches/exllamav3/` |
| Decode-once MoE kernel | `xmoe_do` from [diffbot's recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), plus [`do_gemm.diff`](https://github.com/meldavy/dsv41f-1017-sm120-tp3/blob/main/patches/kernel/do_gemm.diff) | `patches/kernel/` |
| FlashInfer | `flashinfer-python==0.7.3` from PyPI | none |
| torch / triton / transformers | `2.14.0+cu130` / `3.8.0` / `5.22.2` | none |

## Patches

**vLLM**

| Patch | What it does |
|---|---|
| `0001-pr67325-inkling2-small.diff` | Inkling 2 Small model support, [#67325](https://github.com/vllm-project/vllm/pull/67325) as of `d664cd1e`. |
| `0002-pr55078-pr51560-inkling-triton-fallback.diff` | Triton attention fallback for Inkling on SM8x ([#55078](https://github.com/vllm-project/vllm/pull/55078)) and SM12x ([#51560](https://github.com/vllm-project/vllm/pull/51560)), merged on top of 0001. |
| `0003-inkling2-sm120-exl3-tp3.diff` | Everything below. |

What 0003 changes:

- **SM120 relative attention.** Inkling 2's relative-position attention ships as an FA4 kernel that only runs on SM100. `ops/triton_rel_attention_sm12x.py` is a paged Triton version with split-KV decode, selected in `ops/attention.py` on SM12x.
- **Shared-memory limits.** SM120 has about 100 KB of shared memory per block, against about 228 KB on SM100. `ops/linear.py` caps tile sizes and pipeline stages, disables TMA, and falls back to torch where the CuTe-DSL GEMM is unsupported.
- **Fixed-size conv state.** The PR keeps the short-convolution history per token in the paged cache: 24 KiB per token per layer, which caps context near 65k on this setup. `conv_state.py` replaces it with a fixed per-request state (the last W−1 rows), like vLLM's Mamba models. That makes room for 1M tokens, and prefix caching works through vLLM's `--mamba-cache-mode align`. Wiring is in `sconv.py`, `ops/sconv.py`, `attention.py` and `model.py`.
- **EXL3 experts.** `exl3_moe.py` loads EXL3 routed experts, sliced for tensor parallelism, and runs them on ExLlamaV3 kernels: the cooperative kernel for decode and the decode-once grouped GEMM for prefill. Enabled with `INKLING2_EXL3=1`.
- **All-reduce without P2P.** These GPUs have no peer access, and NCCL's host path tops out near 9.4 GB/s. `ops/host_allreduce.py` reduces decode-size tensors through host-mapped memory. `ops/dma_allreduce.py` reduces prefill tensors of 4 MB or more with a copy-engine reduce-scatter and all-gather through pinned host memory.
- **TP3 at load time.** Eight KV heads don't split three ways. `tp_pad.py`, called from a config hook in `model_executor/models/config.py`, pads attention from 48 query / 8 KV heads to 54 / 9 at load time. The added heads are inert, so outputs are unchanged. The model and DSpark loaders in `model.py` and `dspark.py` pad the tensors as they load.

**ExLlamaV3 and the decode-once kernel.** Both patches add Inkling 2's soft-capped SwiGLU, `4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`, as activation id 5. In ExLlamaV3 that's the cooperative MoE kernel and `hadamard_inner.cuh`; in the decode-once kernel it's `xmoe/quant/hadamard_inner.cuh`.

## Build

On glibc 2.42 or newer, CUDA 13.0's nvcc needs the `math_functions.h` workaround from [Appendix E of the DS4.1 recipe](https://github.com/meldavy/dsv41f-1017-sm120-tp3#appendix-e-glibc-243--cuda-130-workaround).

```bash
export SRC=/path/to/src REPO=/path/to/this/repo
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0a

# vLLM
git clone https://github.com/vllm-project/vllm $SRC/vllm && cd $SRC/vllm
git checkout efe12305f064b9c5301231d66eb42b7c0db5895f
for p in $REPO/patches/vllm/*.diff; do git apply "$p"; done
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
CMAKE_CUDA_ARCHITECTURES=120a NVCC_THREADS=4 MAX_JOBS=48 pip install --no-build-isolation -e .

# ExLlamaV3 (built in place, not pip-installed)
git clone https://github.com/turboderp-org/exllamav3 $SRC/exllamav3 && cd $SRC/exllamav3
git checkout a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a
git apply $REPO/patches/exllamav3/*.diff
MAX_JOBS=40 python setup.py build_ext --inplace

# Decode-once prefill kernel
mkdir -p $SRC/xmoe_do && cd $SRC/xmoe_do
cp -r <diffbot recipe>/kernels/xmoe ./xmoe && cp <diffbot recipe>/kernels/xmoe_do/do_gemm.cu .
patch -p1 < do_gemm.diff                                # from the DS4.1 recipe
patch -p1 < $REPO/patches/kernel/*.diff
python -c 'from torch.utils.cpp_extension import load; load(name="ds41_exl3_do", sources=["do_gemm.cu"], extra_include_paths=["xmoe", "xmoe/quant"], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], build_directory="build")'
```

## Launch

```bash
hf download mel-bw/Inkling-2-Small-EXL3-3.17bpw --local-dir /path/to/Inkling-2-Small-EXL3-3.17bpw
MODEL=/path/to/Inkling-2-Small-EXL3-3.17bpw SRC=$SRC $REPO/scripts/serve.sh
```

[`scripts/serve.sh`](scripts/serve.sh) serves an OpenAI-compatible API on port 8300 as `Inkling-2-Small`, with reasoning and tool-call parsing. Startup takes about 7 minutes. Set `PROFILE=context` for the full 1,048,576-token window, at about 6% slower prefill.

## Known limitations

- No speculative decoding yet: the DSpark drafter loads, but the conv state has no rollback.
- Prefix-cache hits land only where an earlier prompt ended, so multi-turn chat hits but unrelated prompts sharing a system prompt don't. This is how align mode works for every Mamba-style model.
- The first prefix-cache hit after startup pays a one-time ~3 s kernel compile.
- Text only: image inputs are disabled.

## Credits

[Thinking Machines](https://huggingface.co/thinkingmachines) (Inkling 2 Small); vLLM authors of [#67325](https://github.com/vllm-project/vllm/pull/67325) (WoosukKwon), [#51560](https://github.com/vllm-project/vllm/pull/51560) (m4xkushnir) and [#55078](https://github.com/vllm-project/vllm/pull/55078) (TheVerdman); turboderp (ExLlamaV3); diffbot (decode-once kernel). Work is tracked in the Local Inference Lab Discord.
