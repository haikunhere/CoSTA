#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-python}
GPUS=${GPUS:-0,1,2,3}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local base model directory}
DATA_PATH=${DATA_PATH:?Set DATA_PATH to the benchmark SFT JSON}
SAVE_PATH=${SAVE_PATH:-saves/sft}

IFS=, read -r -a GPU_IDS <<< "$GPUS"
args=(--model "$MODEL_PATH" --data "$DATA_PATH" --output "$SAVE_PATH" --gpus "${#GPU_IDS[@]}")

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m training.train_sft "${args[@]}" "$@"
