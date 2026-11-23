import argparse
import json
import os
from pathlib import Path

from .retrieval_server import QueryEncoder


def iter_texts(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            value = row["contents"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Empty corpus text at row {number}")
            yield value


def iter_batches(values, size):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import faiss

    encoder = QueryEncoder(args.model)
    index = faiss.IndexFlatIP(encoder.model.config.hidden_size)
    for number, batch in enumerate(iter_batches(iter_texts(args.corpus), args.batch_size), 1):
        index.add(encoder.encode(batch, is_query=False))
        if number % 100 == 0:
            print(f"encoded {index.ntotal:,} rows", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    faiss.write_index(index, str(temporary))
    os.replace(temporary, args.output)
    print(f"wrote {index.ntotal:,} vectors to {args.output}")


if __name__ == "__main__":
    main()
