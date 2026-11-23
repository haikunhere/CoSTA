import argparse
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .graph import CORPUS_FILES, DEFAULT_DATA_ROOT, GraphCorpus

SEMANTIC_TOP_K_MAX = 50


class QueryEncoder:
    """Encode text with FP16 masked mean pooling and L2 normalization."""

    def __init__(self, model_path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for E5 retrieval")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).cuda().half().eval()

    def encode(self, texts, is_query=True):
        import torch

        prefix = "query: " if is_query else "passage: "
        inputs = self.tokenizer([prefix + text for text in texts], max_length=256,
                                padding=True, truncation=True, return_tensors="pt")
        inputs = {name: value.cuda() for name, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs, return_dict=True).last_hidden_state
            hidden = hidden.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
            pooled = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
            vectors = torch.nn.functional.normalize(pooled, dim=-1)
        return vectors.float().cpu().numpy()


def load_index(graph, data_root, dimension):
    import faiss
    import json

    path = Path(data_root) / CORPUS_FILES[graph.corpus][1]
    position_ids = []
    seen = set()
    with graph.path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            source_id = record["id"]
            if not isinstance(source_id, str) or source_id in seen:
                raise ValueError(f"Invalid or duplicate source ID: {source_id!r}")
            seen.add(source_id)
            position_ids.append(source_id)
    index = faiss.read_index(str(path))
    if index.ntotal != len(position_ids) or index.d != dimension:
        raise ValueError(f"{graph.corpus}: index row count or vector dimension differs from the corpus/model")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or not isinstance(index, faiss.IndexFlat):
        raise ValueError(f"{graph.corpus}: a Flat inner-product index is required")
    return index, position_ids


class CorpusBackend:
    def __init__(self, corpus, data_root, gpu_resources, dimension, top_k):
        import faiss
        from .graph import Timeline

        self.graph = GraphCorpus(corpus, data_root)
        self.timeline = Timeline(self.graph)
        index, self.position_ids = load_index(self.graph, data_root, dimension)
        self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        self.top_k = top_k

    def public_fact(self, fact):
        row = self.graph.rows[fact]
        if self.graph.corpus == "MultiTQ":
            return row
        lower, upper = self.graph.intervals[fact]
        return {**{key: value for key, value in row.items() if key not in ("start_time", "end_time")},
                "start": self.graph.format(lower), "end": self.graph.format(upper)}

    def semantic(self, vector, top_k=None):
        limit = self.top_k if top_k is None else top_k
        _, positions = self.index.search(vector, min(limit, len(self.position_ids)))
        facts, seen = [], set()
        for position in positions[0]:
            fact = self.timeline.canonical(self.position_ids[int(position)])
            if fact not in seen:
                seen.add(fact)
                facts.append(self.public_fact(fact))
        return {"facts": facts, "truncated": len(self.graph.rows) > len(facts)}


class RetrievalBackend:
    def __init__(self, data_root, model_path, top_k=15, corpora=tuple(CORPUS_FILES)):
        if type(top_k) is not int or not 1 <= top_k <= 15:
            raise ValueError("top_k must be an integer between 1 and 15")
        for corpus in corpora:
            corpus_path, index_path = (Path(data_root) / f for f in CORPUS_FILES[corpus])
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"{corpus}: build the index with python -m corpus.build_index "
                    f"{corpus_path} {index_path} --model <e5-model-directory>"
                )
        import faiss

        self.encoder = QueryEncoder(model_path)
        self.gpu_resources = faiss.StandardGpuResources()
        self.lock = Lock()
        self.backends = {name: CorpusBackend(name, data_root, self.gpu_resources,
                                             self.encoder.model.config.hidden_size, top_k)
                         for name in corpora}

    def semantic(self, corpus, query, top_k=None):
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= SEMANTIC_TOP_K_MAX):
            raise ValueError(f"semantic top_k must be an integer between 1 and {SEMANTIC_TOP_K_MAX}")
        with self.lock:
            return self.backends[corpus].semantic(self.encoder.encode([query]), top_k)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    parameters: dict


def build_app(data_root, model_path, corpora=tuple(CORPUS_FILES)):
    backend = RetrievalBackend(data_root, model_path, corpora=corpora)
    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def invalid_call(request, error):
        return JSONResponse(status_code=400, content={"detail": "Invalid tool call: name and parameters are required; no extra fields"})

    @app.get("/health")
    def health():
        return {"top_k": 15, "semantic_top_k_max": SEMANTIC_TOP_K_MAX,
                "datasets": {name: {"facts": len(store.graph.rows),
                                    "index_rows": len(store.position_ids)}
                             for name, store in backend.backends.items()}}

    @app.post("/{corpus}/search")
    def search(corpus: str, call: ToolCall):
        if corpus not in backend.backends:
            raise HTTPException(status_code=404, detail=f"unknown corpus {corpus!r}")
        try:
            if call.name != "semantic_search":
                raise ValueError("Use semantic_search")
            if "query" not in call.parameters or set(call.parameters) - {"query", "top_k"}:
                raise ValueError("semantic_search requires query and optional host-selected top_k")
            query = call.parameters["query"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("semantic_search requires a non-empty query")
            return backend.semantic(corpus, query, call.parameters.get("top_k"))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corpus", nargs="+", choices=list(CORPUS_FILES), default=list(CORPUS_FILES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30920)
    args = parser.parse_args()
    if len(args.corpus) != len(set(args.corpus)):
        parser.error("Specify each corpus once")
    import socket
    import uvicorn

    with socket.socket() as sock:
        sock.bind((args.host, args.port))
    uvicorn.run(build_app(args.data, args.model, args.corpus), host=args.host, port=args.port,
                access_log=False, workers=1)


if __name__ == "__main__":
    main()
