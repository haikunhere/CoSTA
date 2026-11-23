#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the exported HF model directory}
GPUS=${GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_IDS <<< "$GPUS"

export CUDA_VISIBLE_DEVICES="$GPUS"
exec "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --host 127.0.0.1 --port "${PORT:-30910}" \
    --dp-size "${#GPU_IDS[@]}" --tp-size 1 --context-length 32768 \
    --dtype bfloat16 --attention-backend fa3 --enable-deterministic-inference \
    --random-seed 42 --mem-fraction-static 0.65 \
    --max-running-requests 96 --cuda-graph-max-bs 64
