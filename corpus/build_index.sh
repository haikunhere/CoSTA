#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=${RETRIEVER_PYTHON:-python}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local E5 model directory}
CORPUS_DIR=${CORPUS_DIR:?Set CORPUS_DIR to the graph corpus directory}

"$PYTHON" -m corpus.build_index "$CORPUS_DIR/corpus.jsonl" \
    "$CORPUS_DIR/e5_mean_Flat.index" --model "$MODEL_PATH"
