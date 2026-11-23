#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
export CUDA_DEVICE_MAX_CONNECTIONS=8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

inputs=()
if [[ -n ${QUESTIONS_PATH:-} ]]; then inputs+=(--questions "$QUESTIONS_PATH"); fi
if [[ -n ${DATA_PATH:-} ]]; then inputs+=(--data "$DATA_PATH"); fi

exec "${PYTHON:-python}" -B -m training.train_rl \
    --model "${MODEL_PATH:?Set MODEL_PATH to the SFT export}" \
    "${inputs[@]}" \
    --output "${SAVE_PATH:-saves/rl}" \
    --retrieval-url "${RETRIEVAL_URL:-http://127.0.0.1:30920}" "$@"
