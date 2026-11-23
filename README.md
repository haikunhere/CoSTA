# CoSTA: Co-Evolving Self-Play Timeline Agents for Complex Temporal Knowledge Graph Question Answering

This repository provides the SFT, multi-agent reinforcement learning, and inference code for CoSTA on MultiTQ and TimelineKGQA.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/haikunhere/CoSTA.git costa
cd costa
```

### 2. Create the training and inference environment

```bash
conda create -n costa python=3.12 -y
conda activate costa

python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
    python -m pip install flash-attn==2.8.3 --no-build-isolation
```

### 3. Create the retrieval environment

```bash
conda create -n costa-retriever python=3.10 -y
conda activate costa-retriever

conda install pytorch=2.4.0 pytorch-cuda=12.1 faiss-gpu=1.8.0 \
    numpy=1.26.4 fsspec=2026.7.0 \
    -c pytorch -c nvidia -c conda-forge -y

python -m pip install transformers==5.5.0 safetensors==0.7.0 \
    fastapi==0.136.3 uvicorn==0.48.0 pydantic==2.13.4 httpx==0.28.1
```

## Data Preparation

### 1. Obtain the benchmark questions

- [MultiTQ](https://github.com/czy1999/MultiTQ/tree/main/data): extract `Dataset.zip`.
- [TimelineKGQA](https://github.com/PascalSun/TimelineKGQA/tree/main/Datasets): use the CronQuestion and ICEWS Actor question sets.

Use the official training split for RL and the test split for evaluation. Supply each split as a JSON list, retaining the original record fields. For TimelineKGQA, select the corresponding records using the `split` field.

### 2. Build the retrieval indexes

[intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)

```bash
conda activate costa-retriever
MODEL_PATH=/path/to/e5-base-v2 CORPUS_DIR=corpus/CORPUS_DIRECTORY bash corpus/build_index.sh
```

## Training

### 1. SFT

```bash
conda activate costa
MODEL_PATH=/path/to/model DATA_PATH=/path/to/sft.json bash train_sft.sh
```

### 2. Start the retrieval service

```bash
conda activate costa-retriever
MODEL_PATH=/path/to/e5-base-v2 bash corpus/retrieval_launch.sh \
    --corpus CORPUS_NAME
```

For TimelineKGQA RL, use `--corpus CronQuestion ICEWS`.

### 3. RL

```bash
conda activate costa
MODEL_PATH=saves/sft/model bash train_rl.sh \
    --benchmark BENCHMARK --questions /path/to/train.json \
    --recipe /path/to/recipe.yaml
```

For TimelineKGQA, pass the CronQuestion and ICEWS training files to `--questions`, in that order.

## Inference

### 1. Start the retrieval service

```bash
conda activate costa-retriever
MODEL_PATH=/path/to/e5-base-v2 bash corpus/retrieval_launch.sh \
    --corpus CORPUS_NAME
```

To serve both TimelineKGQA corpora, use `--corpus CronQuestion ICEWS`.

### 2. Start the model service

```bash
conda activate costa
MODEL_PATH=saves/rl/checkpoint/actor/huggingface bash evaluation/serve_model.sh
```

### 3. Run inference and evaluation

```bash
conda activate costa
MODEL_PATH=saves/rl/checkpoint/actor/huggingface CORPUS=CORPUS_NAME \
    QUESTIONS_PATH=/path/to/test.json bash evaluation/inference.sh
```

