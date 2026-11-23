import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from conversation import Conversation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def encode_sample(sample, tokenizer):
    messages = sample["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("A demonstration must end with an assistant response")
    conversation = Conversation(tokenizer, messages[:2], sample["tools"])
    for message in messages[2:]:
        role, content = message["role"], message["content"]
        if role == "assistant":
            conversation.append_sample(tokenizer.encode(content, add_special_tokens=False))
        else:
            conversation.observe(content, role=role)
    return dict(input_ids=conversation.ids, loss_mask=conversation.generated)


def prepare_data(args, output, max_length):
    import pyarrow as pa
    from transformers import AutoTokenizer

    samples = json.loads(args.data.read_text())
    if not samples:
        raise ValueError("Empty SFT dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for index, sample in enumerate(samples):
        row = encode_sample(sample, tokenizer)
        if not any(row["loss_mask"]):
            raise ValueError(f"Sample {index} has no supervised text")
        if len(row["input_ids"]) > max_length:
            raise ValueError(f"Sample {index} exceeds max_length={max_length}")
        rows.append(row)
    data = output / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data, compression="zstd")
    print(f"Encoded {len(rows)} SFT samples: {data}", flush=True)
    return data


class SFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if config.pad_mode != "no_padding":
            raise ValueError("SFTDataset requires no_padding packing")
        paths = [parquet_files] if isinstance(parquet_files, (str, Path)) else parquet_files
        self.rows = [row for path in paths
                     for row in pq.read_table(path, columns=["input_ids", "loss_mask"]).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return dict(input_ids=torch.tensor(row["input_ids"], dtype=torch.long),
                    loss_mask=torch.tensor(row["loss_mask"], dtype=torch.long),
                    position_ids=torch.arange(len(row["input_ids"]), dtype=torch.long))


def train(args):
    from omegaconf import OmegaConf

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipe = OmegaConf.load(CONFIG_DIR / "sft.yaml")
    data = prepare_data(args, output, recipe.data.max_length)
    environment = dict(os.environ)
    environment.update(SFT_MODEL=str(args.model.resolve()), SFT_DATA=str(data),
                       SFT_OUTPUT=str(output), VERL_SELECTED_LOGITS="1",
                       VERL_FILE_LOGGER_PATH=str(output / "metrics.jsonl"),
                       CUDA_DEVICE_MAX_CONNECTIONS="8",
                       PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), environment.get("PYTHONPATH")]))
    for name, value in {"OMP_NUM_THREADS": "4",
                        "NCCL_PROTO": "Simple", "NCCL_DEBUG": "WARN"}.items():
        environment.setdefault(name, value)
    os.environ.update(environment)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.gpus}", "-m", "verl.trainer.sft_trainer",
               "--config-path", str(CONFIG_DIR), "--config-name", "sft",
               "hydra.searchpath=[pkg://verl.trainer.config]",
               f"trainer.n_gpus_per_node={args.gpus}"]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    config = OmegaConf.load(output / "hydra/.hydra/config.yaml")
    OmegaConf.resolve(config)
    checkpoints = Path(config.trainer.default_local_dir)
    step = int((checkpoints / "latest_checkpointed_iteration.txt").read_text())
    model = checkpoints / f"global_step_{step}" / "huggingface"
    (output / "model").symlink_to(os.path.relpath(model, output), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("saves/sft"))
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    train(args)


if __name__ == "__main__":
    main()
