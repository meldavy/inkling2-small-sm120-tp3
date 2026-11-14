#!/bin/bash
# Serve Inkling-2-Small EXL3 on 3x RTX PRO 6000 (sm120) at TP3.
# usage: MODEL=/path/to/Inkling-2-Small-EXL3-3.17bpw SRC=/path/to/src ./serve.sh
# env:   PROFILE=fast    (default) 4k prefill chunks, 1,000,000-token context
#        PROFILE=context 2k prefill chunks, 1,048,576-token context, ~6% slower prefill
#        PORT (8300) UTIL (0.98) MAXSEQS (4) MAXLEN
set -euo pipefail
: "${MODEL:?set MODEL to the checkpoint directory}" "${SRC:?set SRC to the directory holding exllamav3 and xmoe_do}"

# ExLlamaV3 extension (built in-tree) and the decode-once prefill kernel
export PYTHONPATH=$SRC/exllamav3:$SRC/xmoe_do/build${PYTHONPATH:+:$PYTHONPATH}
export INKLING2_EXL3=1 INKLING2_EXL3_DO_DIR=$SRC/xmoe_do/build
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDA_ARCH_LIST=12.0a
# No GPU peer access on this platform: NCCL goes through host memory
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=8

case "${PROFILE:-fast}" in
  fast)    BT=4096; DEF_LEN=1000000 ;;
  context) BT=2048; DEF_LEN=1048576 ;;
  *) echo "unknown PROFILE"; exit 1 ;;
esac
export INKLING2_EXL3_DO_TOKENS=$BT

exec vllm serve "$MODEL" --tensor-parallel-size 3 --host 0.0.0.0 --port "${PORT:-8300}" \
  --served-model-name Inkling-2-Small --max-model-len "${MAXLEN:-$DEF_LEN}" \
  --gpu-memory-utilization "${UTIL:-0.98}" --max-num-seqs "${MAXSEQS:-4}" --max-num-batched-tokens $BT \
  --max-cudagraph-capture-size 4 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 0}' \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
