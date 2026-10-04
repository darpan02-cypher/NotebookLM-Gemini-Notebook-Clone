"""Compare retrieval methods ('vector' vs 'rerank') on a small question set.

Usage:  python -m eval.run_eval
Writes: eval/results/eval_log.json (full log) and eval/results/eval_summary.md (table).

Automatic quality signals:
  - retrieval_hit: the gold passage text appears in the top-k retrieved chunks
  - rank: 1-based position of the first chunk containing the gold text (0 = missed)
  - keyword_coverage: share of expected keywords present in the generated answer
A 'manual_quality' field is left blank in the log for hand scoring (1-5) before submission.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from generation import ChatService, GroqClient  # noqa: E402
from ingestion import SourceIngestor  # noqa: E402
from retrieval import LocalEmbedder, Retriever  # noqa: E402
from storage import StorageManager  # noqa: E402

TOP_K = 3
METHODS = ["vector", "rerank"]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> None:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    storage = StorageManager(tempfile.mkdtemp())
    embedder = LocalEmbedder()
    retriever = Retriever(storage, embedder)
    chat = ChatService(storage, retriever, GroqClient())

    nb = storage.create_notebook("eval")
    ingestor = SourceIngestor(storage, embedder)
    for f in sorted((ROOT / "eval" / "corpus").glob("*.txt")):
        ingestor.ingest(nb["id"], "txt", f.name, str(f))

    # warm up models so first-call load time doesn't distort latency
    for m in METHODS:
        for _ in range(2):
            retriever.retrieve(nb["id"], "warm up", method=m, top_k=TOP_K)

    log = []
    for item in questions:
        for method in METHODS:
            res = retriever.retrieve(nb["id"], item["q"], method=method, top_k=TOP_K)
            texts = [h["text"] for h in res["hits"]]
            rank = next((i + 1 for i, t in enumerate(texts) if norm(item["gold"]) in norm(t)), 0)
            t0 = time.perf_counter()
            out = chat.ask(nb["id"], item["q"], method=method, top_k=TOP_K)
            gen_s = time.perf_counter() - t0
            ans = out["answer"].lower()
            cov = sum(k in ans for k in item["keywords"]) / len(item["keywords"])
            log.append(
                {
                    "question": item["q"],
                    "method": method,
                    "retrieved_chunks": [
                        {"source": h["metadata"]["source_name"], "chunk_index": h["metadata"]["chunk_index"], "score": round(h["score"], 4), "text": h["text"]}
                        for h in res["hits"]
                    ],
                    "retrieval_latency_s": round(res["latency_s"], 4),
                    "total_latency_s": round(gen_s, 3),
                    "retrieval_hit": rank > 0,
                    "rank": rank,
                    "answer": out["answer"],
                    "keyword_coverage": round(cov, 2),
                    "manual_quality": None,
                }
            )
            print(f"{method:7} hit={rank>0!s:5} rank={rank} cov={cov:.2f} ret={res['latency_s']*1000:6.0f}ms | {item['q']}")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    lines = ["| Method | Hit@%d | Mean rank of hit | MRR | Keyword coverage | Mean retrieval latency (ms) |" % TOP_K, "|---|---|---|---|---|---|"]
    for m in METHODS:
        rows = [r for r in log if r["method"] == m]
        hits = [r for r in rows if r["retrieval_hit"]]
        mrr = sum(1 / r["rank"] for r in hits) / len(rows)
        lines.append(
            f"| {m} | {len(hits)}/{len(rows)} | {sum(r['rank'] for r in hits) / max(1, len(hits)):.2f} | {mrr:.2f} | "
            f"{sum(r['keyword_coverage'] for r in rows) / len(rows):.2f} | {sum(r['retrieval_latency_s'] for r in rows) / len(rows) * 1000:.0f} |"
        )
    (out_dir / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
