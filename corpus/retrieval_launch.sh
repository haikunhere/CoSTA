#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON=${RETRIEVER_PYTHON:-python}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local E5 model directory}
DATA_PATH=${DATA_PATH:-corpus}
GPU=${GPU:-0}
PORT=${PORT:-30920}

if [[ ! $GPU =~ ^[0-9]+$ ]]; then
    echo "GPU must name exactly one numeric GPU" >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m corpus.retrieval_server \
    --model "$MODEL_PATH" --data "$DATA_PATH" --port "$PORT" "$@"
