from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [1] or [2][3]. "
    "If the context does not contain the answer, say you could not find it in the sources. "
    "Do not use outside knowledge."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_citations(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n, h in enumerate(hits, start=1):
        m = h["metadata"]
        out.append(
            {
                "n": n,
                "source_id": m.get("source_id"),
                "source_name": m.get("source_name"),
                "page_or_slide": m.get("page_or_slide", ""),
                "url": m.get("url", ""),
                "chunk_index": m.get("chunk_index"),
                "snippet": h["text"][:300],
            }
        )
    return out


def format_citation(c: Dict[str, Any]) -> str:
    loc = f", p./slide {c['page_or_slide']}" if c.get("page_or_slide") not in ("", None) else ""
    link = f" ({c['url']})" if c.get("url") else ""
    return f"[{c['n']}] {c['source_name']}{loc}{link}, chunk {c['chunk_index']}"


class ChatService:
    def __init__(self, storage_manager: Any, retriever: Any, llm: Any):
        self.storage = storage_manager
        self.retriever = retriever
        self.llm = llm

    def ask(self, notebook_id: str, question: str, method: str = "rerank", top_k: int = 5) -> Dict[str, Any]:
        history = self.storage.load_chats(notebook_id)
        result = self.retriever.retrieve(notebook_id, question, method=method, top_k=top_k)
        hits = result["hits"]
        citations = build_citations(hits)

        if not hits:
            answer = "This notebook has no indexed sources yet. Add a source first."
        else:
            context = "\n\n".join(f"[{c['n']}] ({c['source_name']}) {h['text']}" for c, h in zip(citations, hits))
            recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
            prompt = (
                f"Recent conversation:\n{recent or '(none)'}\n\n"
                f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer with citations:"
            )
            answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        history.append({"id": uuid.uuid4().hex, "role": "user", "content": question, "timestamp": _now(), "citations": []})
        history.append(
            {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": answer,
                "timestamp": _now(),
                "citations": citations,
                "method": method,
                "latency_s": round(result["latency_s"], 3),
            }
        )
        self.storage.save_chats(notebook_id, history)
        return {"answer": answer, "citations": citations, "latency_s": result["latency_s"]}
