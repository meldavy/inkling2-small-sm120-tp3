# syntax=docker/dockerfile:1.7
# Inkling-2-Small EXL3 on SM120 (RTX PRO 6000 Blackwell): vLLM with this repo's patches,
# ExLlamaV3 and the decode-once MoE kernel. Build from the repo root:
#   docker build -f docker/Dockerfile -t inkling2-small-sm120 .
# ARCH_LIST also builds SM89 kernels, but Inkling 2's model code (PR 67325) uses PDL, which needs SM90 or newer.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG VLLM_COMMIT=efe12305f064b9c5301231d66eb42b7c0db5895f
ARG EXLLAMAV3_COMMIT=a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a
ARG TORCH_VERSION=2.14.0
ARG ARCH_LIST="8.9 12.0a"
ARG MAX_JOBS=16
ARG OPTIONAL_PATCHES=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    SRC=/opt/inkling2 \
    VENV=/opt/venv \
    TORCH_CUDA_ARCH_LIST=${ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=30G
ENV PATH=${VENV}/bin:${CUDA_HOME}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev git curl ca-certificates build-essential cmake ninja-build patch ccache \
        libdw-dev libelf-dev libnuma-dev libibverbs-dev libssl-dev libzstd-dev zlib1g-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv ${VENV} \
    && pip install --upgrade pip wheel "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
       "setuptools-rust>=1.9.0" "cmake>=3.26.1" ninja jinja2 \
    && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu130

COPY patches /opt/patches

# vLLM at the base commit with patches 0001-0003 (and the optional 0004-0005); ccache persists across builds
RUN --mount=type=cache,target=/root/.cache/ccache git clone https://github.com/vllm-project/vllm ${SRC}/vllm \
    && cd ${SRC}/vllm && git checkout ${VLLM_COMMIT} \
    && for p in /opt/patches/vllm/*.diff; do git apply "$p"; done \
    && if [ "${OPTIONAL_PATCHES}" = "1" ]; then for p in /opt/patches/vllm/optional/*.diff; do git apply "$p"; done; fi \
    && CMAKE_CUDA_ARCHITECTURES="$(echo ${ARCH_LIST} | sed 's/\.//g; s/ /;/g')" NVCC_THREADS=2 MAX_JOBS=${MAX_JOBS} \
       pip install --no-build-isolation -e . \
    && rm -rf ${SRC}/vllm/build

# ExLlamaV3 extension, built in place
RUN --mount=type=cache,target=/root/.cache/ccache git clone https://github.com/turboderp-org/exllamav3 ${SRC}/exllamav3 \
    && cd ${SRC}/exllamav3 && git checkout ${EXLLAMAV3_COMMIT} \
    && git apply /opt/patches/exllamav3/*.diff \
    && MAX_JOBS=${MAX_JOBS} python setup.py build_ext --inplace \
    && rm -rf build

# Decode-once MoE prefill kernel (diffbot's xmoe_do + the DS4.1 recipe's do_gemm.diff + this repo's activation patch)
RUN pip install huggingface_hub \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000', allow_patterns=['recipe/kernels/xmoe/**', 'recipe/kernels/xmoe_do/do_gemm.cu'], local_dir='/tmp/diffbot')" \
    && mkdir -p ${SRC}/xmoe_do && cd ${SRC}/xmoe_do \
    && cp -r /tmp/diffbot/recipe/kernels/xmoe ./xmoe && cp /tmp/diffbot/recipe/kernels/xmoe_do/do_gemm.cu . \
    && curl -fsSL https://raw.githubusercontent.com/meldavy/dsv41f-1017-sm120-tp3/main/patches/kernel/do_gemm.diff | patch -p1 \
    && for p in /opt/patches/kernel/*.diff; do patch -p1 < "$p"; done \
    && python -c 'from torch.utils.cpp_extension import load; load(name="ds41_exl3_do", sources=["do_gemm.cu"], extra_include_paths=["xmoe", "xmoe/quant"], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], build_directory="build")' \
    && rm -rf /tmp/diffbot

COPY scripts/serve.sh /opt/inkling2/serve.sh
RUN chmod +x /opt/inkling2/serve.sh

ENV MODEL=/model PORT=8300
EXPOSE 8300
ENTRYPOINT ["/opt/inkling2/serve.sh"]
