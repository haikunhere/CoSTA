from contextlib import contextmanager
from dataclasses import dataclass
import threading

import httpx


@dataclass
class Generation:
    output_token_ids: list[int]
    finish_reason: str


class SGLangClient:
    def __init__(self, base_url, data_parallel_size=4, per_endpoint_concurrency=48):
        self.clients = [httpx.Client(
            base_url=base_url.rstrip('/').removesuffix('/v1'), trust_env=False, timeout=600,
            limits=httpx.Limits(max_connections=per_endpoint_concurrency + 4,
                               max_keepalive_connections=per_endpoint_concurrency,
                               keepalive_expiry=3.0)) for _ in range(data_parallel_size)]
        self.ranks = list(range(data_parallel_size)) if data_parallel_size > 1 else [None]
        self.gates = [threading.BoundedSemaphore(per_endpoint_concurrency) for _ in self.clients]
        self.sessions = [0] * len(self.clients)
        self.bindings = {}
        self.cursor = 0
        self.lock = threading.Lock()
        self.thread = threading.local()

    def close(self):
        for client in self.clients:
            client.close()

    @contextmanager
    def sticky_endpoint(self, session_key):
        with self.lock:
            minimum = min(self.sessions)
            candidate = next((self.cursor + offset) % len(self.clients)
                             for offset in range(len(self.clients))
                             if self.sessions[(self.cursor + offset) % len(self.clients)] == minimum)
            index = self.bindings.setdefault(session_key, candidate)
            self.sessions[index] += 1
            self.cursor = (index + 1) % len(self.clients)
        self.thread.index = index
        try:
            yield index
        finally:
            self.thread.index = None
            with self.lock:
                self.sessions[index] -= 1

    def generate_tokens(self, input_ids, *, max_tokens, temperature, top_p, top_k, min_p, seed, stop):
        index = self.thread.index
        payload = dict(input_ids=input_ids, sampling_params=dict(
            max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, sampling_seed=seed, stop=stop, no_stop_trim=True))
        if self.ranks[index] is not None:
            payload['data_parallel_rank'] = self.ranks[index]
        with self.gates[index]:
            response = self.clients[index].post('/generate', json=payload)
            response.raise_for_status()
            data = response.json()
        ids, meta = data.get('output_ids'), data.get('meta_info', {})
        if not isinstance(ids, list) or not all(type(token) is int for token in ids):
            raise RuntimeError('SGLang native response has no output token IDs')
        if meta.get('prompt_tokens') != len(input_ids) or meta.get('completion_tokens') != len(ids):
            raise RuntimeError('SGLang native token counts differ from returned IDs')
        reason = (meta.get('finish_reason') or {}).get('type')
        if reason not in ('stop', 'length'):
            raise RuntimeError(f'SGLang generation interrupted: {meta.get("finish_reason")}')
        return Generation(ids, reason)
