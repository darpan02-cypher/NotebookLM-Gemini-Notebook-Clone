from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from pptx import Presentation
except ImportError:  # pragma: no cover
    Presentation = None


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


class SourceIngestor:
    def __init__(self, storage_manager: Any, embedder: Any | None = None):
        self.storage_manager = storage_manager
        self.embedder = embedder

    def _copy_local_file(self, source_path: str, notebook_id: str, source_id: str) -> Path:
        source_file = Path(source_path)
        target_dir = self.storage_manager.get_notebook_dir(notebook_id) / "raw"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{source_id}_{source_file.name}"
        target_path.write_bytes(source_file.read_bytes())
        return target_path

    def _extract_pdf_text(self, file_path: Path) -> str:
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed.")
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    def _extract_pptx_text(self, file_path: Path) -> str:
        if Presentation is None:
            raise RuntimeError("python-pptx is not installed.")
        prs = Presentation(str(file_path))
        slides: List[str] = []
        for index, slide in enumerate(prs.slides, start=1):
            slide_parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_parts.append(shape.text)
            slides.append(f"Slide {index}: {' '.join(slide_parts)}")
        return "\n\n".join(slides)

    def _extract_txt_text(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _extract_url_text(self, url: str) -> str:
        if requests is None:
            raise RuntimeError("requests is not installed.")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if BeautifulSoup is None:
            return response.text
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in ["main", "article", "body"]:
            element = soup.select_one(selector)
            if element:
                return element.get_text(" ", strip=True)
        return soup.get_text(" ", strip=True)

    def ingest(self, notebook_id: str, source_type: str, source_name: str, source_value: str) -> Dict[str, Any]:
        notebook_dir = self.storage_manager.get_notebook_dir(notebook_id)
        source_id = uuid.uuid4().hex
        source_record: Dict[str, Any] = {
            "id": source_id,
            "type": source_type,
            "name": source_name,
            "path": source_value,
            "status": "processing",
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "metadata": {},
        }

        try:
            if source_type == "pdf":
                source_file = self._copy_local_file(source_value, notebook_id, source_id)
                text = self._extract_pdf_text(source_file)
                source_record["metadata"]["source_file"] = str(source_file)
                source_record["metadata"]["page_count"] = "pdf"
            elif source_type == "pptx":
                source_file = self._copy_local_file(source_value, notebook_id, source_id)
                text = self._extract_pptx_text(source_file)
                source_record["metadata"]["source_file"] = str(source_file)
                source_record["metadata"]["slide_count"] = "pptx"
            elif source_type == "txt":
                source_file = self._copy_local_file(source_value, notebook_id, source_id)
                text = self._extract_txt_text(source_file)
                source_record["metadata"]["source_file"] = str(source_file)
            elif source_type == "url":
                text = self._extract_url_text(source_value)
                source_record["metadata"]["url"] = source_value
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            text_path = notebook_dir / "text" / f"{source_id}_{source_name}.txt"
            text_path.write_text(text, encoding="utf-8")
            chunks = chunk_text(text)
            source_record["metadata"]["chunk_count"] = len(chunks)
            source_record["status"] = "ingested"

            if self.embedder is not None:
                for index, chunk in enumerate(chunks):
                    chunk_id = f"{source_id}-{index}"
                    embeddings = self.embedder.embed([chunk])
                    self.embedder.add_to_collection(
                        notebook_id=notebook_id,
                        chunk_id=chunk_id,
                        text=chunk,
                        embedding=embeddings[0],
                        metadata={
                            "source_id": source_id,
                            "source_name": source_name,
                            "type": source_type,
                            "chunk_index": index,
                            "url": source_value if source_type == "url" else None,
                        },
                    )

            self.storage_manager.append_source(notebook_id, source_record)
            return {
                "source_id": source_id,
                "source_record": source_record,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "text_path": str(text_path),
            }
        except Exception as exc:  # pragma: no cover - surface clear error to caller
            source_record["status"] = "failed"
            source_record["metadata"]["error"] = str(exc)
            self.storage_manager.append_source(notebook_id, source_record)
            raise
