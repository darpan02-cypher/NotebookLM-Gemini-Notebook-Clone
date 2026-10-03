from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120

# (page/slide label, text). Label is an int for pdf/pptx, "" otherwise.
Segment = Tuple[Any, str]


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            # prefer breaking on a space in the back half of the window
            space = cleaned.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]


class SourceIngestor:
    def __init__(self, storage_manager: Any, embedder: Any | None = None):
        self.storage_manager = storage_manager
        self.embedder = embedder

    # ---- extraction: each returns a list of (page_or_slide, text) segments ----

    def _copy_local_file(self, source_path: str, notebook_id: str, source_id: str) -> Path:
        source_file = Path(source_path)
        target_dir = self.storage_manager.get_notebook_dir(notebook_id) / "raw"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{source_id}_{source_file.name}"
        target_path.write_bytes(source_file.read_bytes())
        return target_path

    def _extract_pdf(self, file_path: Path) -> List[Segment]:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]

    def _extract_pptx(self, file_path: Path) -> List[Segment]:
        from pptx import Presentation

        prs = Presentation(str(file_path))
        segments: List[Segment] = []
        for i, slide in enumerate(prs.slides, start=1):
            parts = [s.text for s in slide.shapes if getattr(s, "has_text_frame", False) and s.text]
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = slide.notes_slide.notes_text_frame.text
                if notes:
                    parts.append(notes)
            segments.append((i, " ".join(parts)))
        return segments

    def _extract_txt(self, file_path: Path) -> List[Segment]:
        return [("", file_path.read_text(encoding="utf-8", errors="ignore"))]

    def _extract_url(self, url: str) -> Tuple[List[Segment], str]:
        import requests
        from bs4 import BeautifulSoup

        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (NotebookClone)"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        root = soup.select_one("main") or soup.select_one("article") or soup.body or soup
        return [("", root.get_text(" ", strip=True))], title

    # ---- pipeline ----

    def ingest(self, notebook_id: str, source_type: str, source_name: str, source_value: str) -> Dict[str, Any]:
        notebook_dir = self.storage_manager.get_notebook_dir(notebook_id)
        source_id = uuid.uuid4().hex
        record: Dict[str, Any] = {
            "id": source_id,
            "type": source_type,
            "name": source_name,
            "path": source_value,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }

        try:
            if source_type == "pdf":
                f = self._copy_local_file(source_value, notebook_id, source_id)
                segments = self._extract_pdf(f)
                record["metadata"].update(source_file=str(f), page_count=len(segments))
            elif source_type == "pptx":
                f = self._copy_local_file(source_value, notebook_id, source_id)
                segments = self._extract_pptx(f)
                record["metadata"].update(source_file=str(f), slide_count=len(segments))
            elif source_type == "txt":
                f = self._copy_local_file(source_value, notebook_id, source_id)
                segments = self._extract_txt(f)
                record["metadata"]["source_file"] = str(f)
            elif source_type == "url":
                segments, title = self._extract_url(source_value)
                record["metadata"]["url"] = source_value
                if title and not source_name:
                    record["name"] = source_name = title
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            full_text = "\n\n".join(t for _, t in segments)
            if not full_text.strip():
                raise ValueError("No extractable text found (scanned/image-only file?).")
            text_path = notebook_dir / "text" / f"{source_id}.txt"
            text_path.write_text(full_text, encoding="utf-8")

            chunks: List[Dict[str, Any]] = []
            for label, seg_text in segments:
                for piece in chunk_text(seg_text):
                    chunks.append({"text": piece, "page_or_slide": label})
            record["metadata"]["chunk_count"] = len(chunks)

            if self.embedder is not None and chunks:
                self._index(notebook_id, record, chunks)

            record["status"] = "ingested"
            self.storage_manager.append_source(notebook_id, record)
            return {
                "source_id": source_id,
                "source_record": record,
                "chunk_count": len(chunks),
                "text_path": str(text_path),
            }
        except Exception as exc:
            record["status"] = "failed"
            record["metadata"]["error"] = str(exc)
            self.storage_manager.append_source(notebook_id, record)
            raise

    def _index(self, notebook_id: str, record: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        from retrieval.embedder import NotebookVectorStore

        store = NotebookVectorStore(notebook_id, self.storage_manager.get_notebook_dir(notebook_id) / "chroma")
        vectors = self.embedder.embed([c["text"] for c in chunks])
        store.add_many(
            ids=[f"{record['id']}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "notebook_id": notebook_id,
                    "source_id": record["id"],
                    "source_name": record["name"],
                    "type": record["type"],
                    "chunk_index": i,
                    "page_or_slide": c["page_or_slide"],
                    "url": record["path"] if record["type"] == "url" else "",
                }
                for i, c in enumerate(chunks)
            ],
        )
