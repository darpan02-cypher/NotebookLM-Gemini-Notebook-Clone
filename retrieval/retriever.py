from __future__ import annotations

import time
from typing import Any, Dict, List

from .embedder import LocalEmbedder, NotebookVectorStore

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
METHODS = ("vector", "rerank")


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: List[str]) -> List[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


class Retriever:
    """Two methods: 'vector' (cosine top-k) and 'rerank' (vector top-N, cross-encoder re-sorted to top-k)."""

    def __init__(self, storage_manager: Any, embedder: LocalEmbedder, reranker: CrossEncoderReranker | None = None):
        self.storage = storage_manager
        self.embedder = embedder
        self.reranker = reranker or CrossEncoderReranker()

    def _store(self, notebook_id: str) -> NotebookVectorStore:
        return NotebookVectorStore(notebook_id, self.storage.get_notebook_dir(notebook_id) / "chroma")

    def retrieve(
        self, notebook_id: str, query: str, method: str = "rerank", top_k: int = 5, candidates: int = 20
    ) -> Dict[str, Any]:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method '{method}'. Use one of {METHODS}.")
        start = time.perf_counter()
        store = self._store(notebook_id)
        query_vec = self.embedder.embed([query])[0]
        hits = store.query(query_vec, n_results=candidates if method == "rerank" else top_k)
        for h in hits:
            h["score"] = 1.0 - h["distance"]  # cosine similarity

        if method == "rerank" and hits:
            scores = self.reranker.score(query, [h["text"] for h in hits])
            for h, s in zip(hits, scores):
                h["score"] = s
            hits.sort(key=lambda h: h["score"], reverse=True)
            hits = hits[:top_k]

        return {"method": method, "hits": hits, "latency_s": time.perf_counter() - start}
