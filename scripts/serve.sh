#!/bin/bash
# Serve Inkling-2-Small EXL3 on 3x RTX PRO 6000 (sm120) at TP3.
# usage: MODEL=/path/to/Inkling-2-Small-EXL3-3.17bpw SRC=/path/to/src ./serve.sh
# env:   KV_DTYPE=auto     BF16 KV cache (default) | fp8 (e4m3)
#        GRAPHS=1          prefill CUDA graphs + two-part prefill overlap (needs the optional patch)
#                          0: eager prefill (patches only)
#        SPEC=1            DSpark speculative decoding (work in progress)
#        SPEC_K=3          DSpark drafts verified per step (1-7)
#        IMAGES=4          max images per prompt (0: text only)
#        PROFILE=fast      4k prefill chunks, 1,000,000-token default context (default)
#                context   2k prefill chunks, 1,048,576-token default context
#        PORT (8300) UTIL (0.98) MAXSEQS (4) MAXLEN
set -euo pipefail
: "${MODEL:?set MODEL to the checkpoint directory}" "${SRC:?set SRC to the directory holding exllamav3 and xmoe_do}"

# ExLlamaV3 extension (built in-tree) and the decode-once prefill kernel
export PYTHONPATH=$SRC/exllamav3:$SRC/xmoe_do/build${PYTHONPATH:+:$PYTHONPATH}
export INKLING2_EXL3=1 INKLING2_EXL3_DO_DIR=$SRC/xmoe_do/build INKLING2_EXL3_DO_TOKENS=4096
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDA_ARCH_LIST=12.0a
# No GPU peer access on this platform: NCCL goes through host memory
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple NCCL_MIN_NCHANNELS=16 \
  NCCL_MAX_NCHANNELS=16 NCCL_BUFFSIZE=2097152 CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=8

case "${PROFILE:-fast}" in
  fast)    BT=4096; DEF_LEN=1000000 ;;
  context) BT=2048; DEF_LEN=1048576 ;;
  *) echo "unknown PROFILE"; exit 1 ;;
esac

if [ "${GRAPHS:-1}" = "1" ]; then
  export INKLING2_UBATCH_OVERLAP=1
  SIZES=$(python3 -c "print(','.join(str(s) for s in (256, 512, 1024, 2048, 4096) if s <= $BT))")
  CG=(--compilation-config "{\"prefill_cudagraph_capture_token_sizes\": [$SIZES], \"decode_cudagraph_capture_request_sizes\": [1, 2, 4]}")
else
  CG=(--max-cudagraph-capture-size 4)
fi

SP=()
if [ "${SPEC:-0}" = "1" ]; then
  # Verify batches of up to 4 requests x 8 tokens stay on the cooperative MoE kernel
  export INKLING2_COOP_MAX_ROWS=${INKLING2_COOP_MAX_ROWS:-32}
  SP=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${SPEC_K:-3}}")
fi

exec vllm serve "$MODEL" --tensor-parallel-size 3 --host 0.0.0.0 --port "${PORT:-8300}" \
  --served-model-name Inkling-2-Small --max-model-len "${MAXLEN:-$DEF_LEN}" \
  --gpu-memory-utilization "${UTIL:-0.98}" --max-num-seqs "${MAXSEQS:-4}" --max-num-batched-tokens $BT \
  "${CG[@]}" "${SP[@]}" --enable-prefix-caching --mamba-cache-mode align \
  --kv-cache-dtype "${KV_DTYPE:-auto}" \
  --limit-mm-per-prompt "{\"image\": ${IMAGES:-4}}" \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
