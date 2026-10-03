from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"


class StorageManager:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.notebooks_root = self.root / "notebooks"
        self.notebooks_root.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_notebook(self, name: str) -> Dict[str, Any]:
        notebook_id = uuid.uuid4().hex
        notebook_dir = self.notebooks_root / notebook_id
        notebook_dir.mkdir(parents=True, exist_ok=True)

        for folder in ["raw", "text", "chroma", "artifacts"]:
            (notebook_dir / folder).mkdir(parents=True, exist_ok=True)

        meta = {
            "id": notebook_id,
            "name": name,
            "created_at": self._now(),
            "updated_at": self._now(),
            "sources": [],
            "artifacts": [],
        }
        (notebook_dir / "info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (notebook_dir / "chats.json").write_text("[]", encoding="utf-8")
        return meta

    def list_notebooks(self) -> List[Dict[str, Any]]:
        notebooks: List[Dict[str, Any]] = []
        for info_path in sorted(self.notebooks_root.glob("*/info.json")):
            try:
                with info_path.open("r", encoding="utf-8") as fh:
                    notebooks.append(json.load(fh))
            except json.JSONDecodeError:
                continue
        return notebooks

    def get_notebook(self, notebook_id: str) -> Dict[str, Any]:
        info_path = self.notebooks_root / notebook_id / "info.json"
        with info_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_notebook_dir(self, notebook_id: str) -> Path:
        return self.notebooks_root / notebook_id

    def update_notebook(self, notebook_id: str, **changes: Any) -> Dict[str, Any]:
        notebook = self.get_notebook(notebook_id)
        notebook.update(changes)
        notebook["updated_at"] = self._now()
        info_path = self.notebooks_root / notebook_id / "info.json"
        info_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        return notebook

    def rename_notebook(self, notebook_id: str, new_name: str) -> Dict[str, Any]:
        return self.update_notebook(notebook_id, name=new_name)

    def delete_notebook(self, notebook_id: str) -> None:
        notebook_dir = self.notebooks_root / notebook_id
        if notebook_dir.exists():
            shutil.rmtree(notebook_dir)

    def save_sources(self, notebook_id: str, source_records: List[Dict[str, Any]]) -> None:
        notebook = self.get_notebook(notebook_id)
        notebook["sources"] = source_records
        notebook["updated_at"] = self._now()
        info_path = self.notebooks_root / notebook_id / "info.json"
        info_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    def save_chats(self, notebook_id: str, messages: List[Dict[str, Any]]) -> None:
        chats_path = self.notebooks_root / notebook_id / "chats.json"
        chats_path.write_text(json.dumps(messages, indent=2), encoding="utf-8")

    def load_chats(self, notebook_id: str) -> List[Dict[str, Any]]:
        chats_path = self.notebooks_root / notebook_id / "chats.json"
        if not chats_path.exists():
            return []
        try:
            with chats_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return []

    def save_artifacts(self, notebook_id: str, artifacts: List[Dict[str, Any]]) -> None:
        notebook = self.get_notebook(notebook_id)
        notebook["artifacts"] = artifacts
        notebook["updated_at"] = self._now()
        info_path = self.notebooks_root / notebook_id / "info.json"
        info_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    def append_source(self, notebook_id: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        notebook = self.get_notebook(notebook_id)
        notebook["sources"].append(source)
        notebook["updated_at"] = self._now()
        info_path = self.notebooks_root / notebook_id / "info.json"
        info_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        return notebook["sources"]
