#!/bin/bash
# Serve Inkling-2-Small EXL3 on 3x RTX PRO 6000 (sm120) at TP3.
# usage: MODEL=/path/to/Inkling-2-Small-EXL3-3.17bpw SRC=/path/to/src ./serve.sh
# env:   KV_DTYPE=auto     BF16 KV cache (default) | fp8 (e4m3)
#        GRAPHS            1: prefill CUDA graphs and the two-microbatch prefill overlap (default when
#                          optional patch 0005 is applied); 0: eager prefill (default without it)
#        SPEC=1            DSpark speculative decoding (work in progress)
#        SPEC_K=3          DSpark drafts verified per step with one request (1-7)
#        SPEC_K_BATCH=2    DSpark drafts verified per request with 2+ requests
#        IMAGES            max images per prompt (default: vLLM's 999); 0 for text only
#        PROFILE=fast      4k prefill chunks, 1,000,000-token default context (default)
#                context   2k prefill chunks, 1,048,576-token default context
#        TP=3              tensor-parallel size (number of GPUs)
#        UTIL=0.98         GPU memory fraction for vLLM; 0.98 is recommended for a 1M-token context
#        PORT (8300) MAXSEQS (4) MAXLEN
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

# Default GRAPHS from whether optional patch 0005 (the prefill overlap) is applied
if [ -z "${GRAPHS:-}" ]; then
  GRAPHS=$(python3 -c 'import importlib.util as u, os; s = u.find_spec("vllm"); print(int(os.path.exists(os.path.join(os.path.dirname(s.origin), "models", "inkling2", "ubatch.py"))))')
fi
if [ "$GRAPHS" = "1" ]; then
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
  K=${SPEC_K:-3}; KB=${SPEC_K_BATCH:-2}
  # Fewer drafts per request when several requests share a step (verify cost grows per request)
  SCHED=""
  if [ "$KB" -lt "$K" ]; then SCHED=", \"num_speculative_tokens_per_batch_size\": [[1, 1, $K], [2, ${MAXSEQS:-4}, $KB]]"; fi
  SP=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $K$SCHED}")
fi

# Images per prompt: vLLM's default (999) unless IMAGES is set; IMAGES=0 for text only
MM=()
if [ -n "${IMAGES:-}" ]; then MM=(--limit-mm-per-prompt "{\"image\": $IMAGES}"); fi

exec vllm serve "$MODEL" --tensor-parallel-size "${TP:-3}" --host 0.0.0.0 --port "${PORT:-8300}" \
  --served-model-name Inkling-2-Small --max-model-len "${MAXLEN:-$DEF_LEN}" \
  --gpu-memory-utilization "${UTIL:-0.98}" --max-num-seqs "${MAXSEQS:-4}" --max-num-batched-tokens $BT \
  "${CG[@]}" "${SP[@]}" --enable-prefix-caching --mamba-cache-mode align \
  --kv-cache-dtype "${KV_DTYPE:-auto}" \
  "${MM[@]}" \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
