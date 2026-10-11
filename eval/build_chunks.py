"""Ingest eval/corpus/ once through the real pipeline and dump every chunk with a stable ID.

Usage:  python -m eval.build_chunks
Writes: eval/corpus_chunks.json  (refuses to overwrite an existing file)

Chunk ID = "<source file name>#<chunk_index, 3 digits>", e.g. "mercury_gemini.txt#004".
Each entry also stores the sha256 of its text so that later label checks notice a chunking change.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import SourceIngestor  # noqa: E402
from ingestion.source_ingestor import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE  # noqa: E402
from retrieval import LocalEmbedder  # noqa: E402
from retrieval.embedder import NotebookVectorStore  # noqa: E402
from storage import StorageManager  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"
OUT_PATH = ROOT / "eval" / "corpus_chunks.json"


def chunk_id(source_name: str, chunk_index: int) -> str:
    return f"{source_name}#{chunk_index:03d}"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_chunks() -> list[dict]:
    storage = StorageManager(tempfile.mkdtemp())
    embedder = LocalEmbedder()
    nb = storage.create_notebook("eval-chunks")
    ingestor = SourceIngestor(storage, embedder)
    for f in sorted(CORPUS_DIR.glob("*.txt")):
        ingestor.ingest(nb["id"], "txt", f.name, str(f))

    store = NotebookVectorStore(nb["id"], storage.get_notebook_dir(nb["id"]) / "chroma")
    chunks = [
        {
            "id": chunk_id(r["metadata"]["source_name"], r["metadata"]["chunk_index"]),
            "source": r["metadata"]["source_name"],
            "chunk_index": r["metadata"]["chunk_index"],
            "sha256": text_hash(r["text"]),
            "text": r["text"],
        }
        for r in store.get_all()
    ]
    chunks.sort(key=lambda c: (c["source"], c["chunk_index"]))
    return chunks


def main() -> None:
    if OUT_PATH.exists():
        sys.exit(f"{OUT_PATH} already exists; refusing to overwrite. Move it aside first.")
    chunks = build_chunks()
    payload = {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP, "chunks": chunks}
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    per_source: dict[str, int] = {}
    for c in chunks:
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
    print(f"Total chunks: {len(chunks)}")
    for s, n in per_source.items():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
