#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QUESTIONS_PATH=${QUESTIONS_PATH:?Set QUESTIONS_PATH to an external benchmark JSON}
if [[ "$QUESTIONS_PATH" != /* ]]; then
    QUESTIONS_PATH="$PWD/$QUESTIONS_PATH"
fi
cd "$ROOT"

inputs=()
if [[ -n ${DATA_PATH:-} ]]; then inputs+=(--data "$DATA_PATH"); fi
if [[ -n ${DP_SIZE:-} ]]; then inputs+=(--dp-size "$DP_SIZE"); fi

exec "${PYTHON:-python}" -m evaluation.inference \
    --model "${MODEL_PATH:?Set MODEL_PATH to the served HF model directory}" \
    --corpus "${CORPUS:?Set CORPUS to MultiTQ, CronQuestion, or ICEWS}" \
    --questions "$QUESTIONS_PATH" \
    --output "${SAVE_PATH:-saves/inference}" \
    --base-url "${BASE_URL:-http://127.0.0.1:30910}" \
    --retrieval-url "${RETRIEVAL_URL:-http://127.0.0.1:30920}" \
    "${inputs[@]}" "$@"
