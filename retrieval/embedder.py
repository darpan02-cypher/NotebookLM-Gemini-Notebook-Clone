from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("sentence-transformers is not installed.")
        return self.model.encode(list(texts), convert_to_numpy=False).tolist()

    def add_to_collection(self, notebook_id: str, chunk_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        store = NotebookVectorStore(notebook_id)
        store.add(chunk_id, embedding, {**metadata, "text": text})


class NotebookVectorStore:
    def __init__(self, notebook_id: str, root: str | Path | None = None):
        self.notebook_id = notebook_id
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "notebooks" / notebook_id / "chroma"
        self.root.mkdir(parents=True, exist_ok=True)
        if chromadb is None:
            raise RuntimeError("chromadb is not installed.")
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.collection = self.client.get_or_create_collection(name=f"notebook-{notebook_id}")

    def add(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
