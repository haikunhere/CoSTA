import httpx
import logging
from threading import Lock
from urllib.parse import urlsplit


def retrieval_urls(value):
    if not isinstance(value, str):
        raise ValueError("retrieval_url must contain HTTP URLs separated by commas")
    urls = tuple(url.strip().rstrip('/') for url in value.split(','))
    if not urls or any(not url for url in urls) or len(set(urls)) != len(urls):
        raise ValueError("retrieval URLs must be non-empty and distinct")
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or not parsed.hostname or parsed.path
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("Use an http://host:port retrieval base URL")
        _ = parsed.port
    return urls


class RetrievalClient:
    def __init__(self, base_url, corpus, top_k=15, timeout=30.0):
        if type(top_k) is not int or not 1 <= top_k <= 30:
            raise ValueError("top_k must be an integer between 1 and 30")
        self.corpus = corpus
        self.urls = tuple(f"{url}/{corpus}/search" for url in retrieval_urls(base_url))
        self._next = 0
        self._lock = Lock()
        self.top_k = top_k
        # Close idle connections before Uvicorn's 5 s timeout.
        self.http = httpx.Client(timeout=timeout, limits=httpx.Limits(keepalive_expiry=2.0))

    def close(self):
        self.http.close()

    def reset_rotation(self):
        with self._lock:
            self._next = 0

    def access(self, request):
        if request.get('name') == 'semantic_search' and self.top_k != 15:
            request = {**request, 'parameters': {**request['parameters'], 'top_k': self.top_k}}
        with self._lock:
            url = self.urls[self._next]
            self._next = (self._next + 1) % len(self.urls)
        try:
            try:
                response = self.http.post(url, json=request)
            except httpx.RemoteProtocolError as error:
                logging.getLogger(__name__).warning("Retrying disconnected retrieval at %s: %s", url, error)
                response = self.http.post(url, json=request)
        except httpx.HTTPError as error:
            raise RuntimeError(f"retrieval service unreachable at {url}: {error}") from error
        if response.status_code == 400:
            raise ValueError(response.json().get("detail", "retrieval parameter error"))
        if response.status_code != 200:
            raise RuntimeError(f"retrieval service error: {response.status_code} "
                               f"{response.text[:200]}")
        try:
            result = response.json()
        except ValueError as error:
            raise RuntimeError("retrieval service returned invalid JSON") from error
        if (not isinstance(result, dict) or set(result) != {"facts", "truncated"}
                or not isinstance(result["facts"], list) or len(result["facts"]) > self.top_k
                or type(result["truncated"]) is not bool):
            raise RuntimeError("retrieval service returned an invalid observation")
        return result
