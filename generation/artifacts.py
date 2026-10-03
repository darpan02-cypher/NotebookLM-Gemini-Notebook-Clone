from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from retrieval.embedder import NotebookVectorStore

MAX_CONTEXT_CHARS = 30000

PROMPTS = {
    "report": (
        "Write a well-structured study report in Markdown based ONLY on the source excerpts below. "
        "Include: a title, an executive summary, 3-6 sections with headings and key points, and a short conclusion. "
        "Cite sources inline as [Source name]. Do not invent facts."
    ),
    "quiz": (
        "Write a quiz in Markdown based ONLY on the source excerpts below. Create 8 questions: "
        "5 multiple choice (options A-D) and 3 short answer. Put all questions first under '## Questions', "
        "then a '## Answer Key' section with the correct answer and a one-line explanation for each question. "
        "Do not invent facts."
    ),
}


def _sample_context(chunks: List[Dict[str, Any]], limit: int = MAX_CONTEXT_CHARS) -> str:
    """Take chunks evenly across the notebook so every source is represented within the size cap."""
    total = sum(len(c["text"]) for c in chunks)
    if total > limit and chunks:
        step = total / limit
        chunks = chunks[:: max(1, int(step + 0.999))]
    parts, used = [], 0
    for c in chunks:
        line = f"[{c['metadata'].get('source_name')}] {c['text']}"
        if used + len(line) > limit:
            break
        parts.append(line)
        used += len(line)
    return "\n\n".join(parts)


class ArtifactService:
    def __init__(self, storage_manager: Any, llm: Any):
        self.storage = storage_manager
        self.llm = llm

    def generate(self, notebook_id: str, kind: str) -> Dict[str, Any]:
        if kind not in PROMPTS:
            raise ValueError("Artifact type must be 'report' or 'quiz'.")
        nb_dir = self.storage.get_notebook_dir(notebook_id)
        store = NotebookVectorStore(notebook_id, nb_dir / "chroma")
        chunks = store.get_all()
        if not chunks:
            raise ValueError("This notebook has no indexed sources. Add a source first.")

        markdown = self.llm.generate(f"Source excerpts:\n\n{_sample_context(chunks)}", system=PROMPTS[kind])

        stamp = datetime.now(timezone.utc)
        artifact_id = uuid.uuid4().hex
        filename = f"{kind}_{stamp.strftime('%Y%m%d_%H%M%S')}.md"
        path = nb_dir / "artifacts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

        record = {
            "id": artifact_id,
            "type": kind,
            "filename": filename,
            "path": str(path),
            "created_at": stamp.isoformat(),
        }
        artifacts = self.storage.get_notebook(notebook_id).get("artifacts", [])
        artifacts.append(record)
        self.storage.save_artifacts(notebook_id, artifacts)
        return {**record, "content": markdown}

    def list(self, notebook_id: str) -> List[Dict[str, Any]]:
        return self.storage.get_notebook(notebook_id).get("artifacts", [])

    def read(self, record: Dict[str, Any]) -> str:
        return Path(record["path"]).read_text(encoding="utf-8")
