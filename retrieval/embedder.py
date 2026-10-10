from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  #this is a small, fast, and efficient model for generating sentence embeddings. It is part of the Sentence Transformers library and is designed to produce high-quality embeddings for various nlp tasks, such as semantic search, clustering, and classification.


class LocalEmbedder:
    """Wraps a local sentence-transformers model. The model loads lazily on first use.""" #which means that the model is not loaded into memory until it is actually needed, which can save resources and improve startup time.

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()


class NotebookVectorStore:
    """One Chroma collection per notebook, persisted under the notebook's chroma/ folder.""" #This allows for efficient storage and retrieval of vector embeddings associated with each notebook, enabling semantic search and other vector-based operations.

    def __init__(self, notebook_id: str, persist_dir: str | Path):
        import chromadb

        self.notebook_id = notebook_id
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=f"notebook-{notebook_id}", metadata={"hnsw:space": "cosine"}
        )

    def add_many(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Return hits as [{id, text, metadata, distance}], best first."""
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_all(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """All chunks in document order (used for report/quiz context)."""
        res = self.collection.get(include=["documents", "metadatas"])
        rows = [
            {"id": i, "text": d, "metadata": m}
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        rows.sort(key=lambda r: (r["metadata"].get("source_id", ""), r["metadata"].get("chunk_index", 0)))
        return rows[:limit] if limit else rows

    def count(self) -> int:
        return self.collection.count()
