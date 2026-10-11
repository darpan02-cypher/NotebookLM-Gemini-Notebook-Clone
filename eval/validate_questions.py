"""Validate eval/questions_v2.json against eval/corpus_chunks.json (and the live chunker).

Usage:  python -m eval.validate_questions
Exits non-zero on any problem. Prints total chunk count and question counts per type.

Checks:
  - schema: id, text, type, gold_chunk_ids (non-empty list), gold_chunk_sha256, reference_answer
  - ids unique; type in {direct, paraphrased, confusable}
  - every gold chunk ID exists in corpus_chunks.json and its stored hash matches
  - re-chunking eval/corpus/ with the current chunker reproduces corpus_chunks.json exactly
    (so a change to chunking or to the corpus text breaks loudly instead of silently)
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.source_ingestor import chunk_text  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"
CHUNKS_PATH = ROOT / "eval" / "corpus_chunks.json"
QUESTIONS_PATH = ROOT / "eval" / "questions_v2.json"
TYPES = ("direct", "paraphrased", "confusable")
REQUIRED = ("id", "text", "type", "gold_chunk_ids", "gold_chunk_sha256", "reference_answer")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_chunking(dumped: list[dict]) -> list[str]:
    """Re-chunk the corpus with the live chunker and compare with the dump."""
    live = {}
    for f in sorted(CORPUS_DIR.glob("*.txt")):
        for i, piece in enumerate(chunk_text(f.read_text(encoding="utf-8", errors="ignore"))):
            live[f"{f.name}#{i:03d}"] = sha256(piece)
    stored = {c["id"]: c["sha256"] for c in dumped}
    errors = []
    for cid in sorted(set(live) | set(stored)):
        if cid not in stored:
            errors.append(f"chunk {cid} appears with the current chunker but is not in corpus_chunks.json")
        elif cid not in live:
            errors.append(f"chunk {cid} is in corpus_chunks.json but the current chunker no longer produces it")
        elif live[cid] != stored[cid]:
            errors.append(f"chunk {cid} text differs between corpus_chunks.json and the current chunker")
    return errors


def validate(questions: list[dict], chunks: list[dict]) -> list[str]:
    errors: list[str] = []
    by_id = {c["id"]: c for c in chunks}
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "<missing id>")
        for field in REQUIRED:
            if field not in q:
                errors.append(f"{qid}: missing field '{field}'")
        if errors and any(e.startswith(f"{qid}: missing") for e in errors):
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q["type"] not in TYPES:
            errors.append(f"{qid}: type '{q['type']}' not in {TYPES}")
        if not q["text"].strip() or not q["reference_answer"].strip():
            errors.append(f"{qid}: empty text or reference_answer")
        gold = q["gold_chunk_ids"]
        if not isinstance(gold, list) or not gold:
            errors.append(f"{qid}: gold_chunk_ids must be a non-empty list")
            continue
        if set(q["gold_chunk_sha256"]) != set(gold):
            errors.append(f"{qid}: gold_chunk_sha256 keys do not match gold_chunk_ids")
        for g in gold:
            if g not in by_id:
                errors.append(f"{qid}: gold chunk '{g}' does not exist")
            elif q["gold_chunk_sha256"].get(g) != by_id[g]["sha256"]:
                errors.append(f"{qid}: hash mismatch for gold chunk '{g}' (chunking or corpus changed)")
    return errors


def main() -> None:
    dump = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = dump["chunks"]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    print(f"Total chunks: {len(chunks)}")
    errors = []
    if len(chunks) < 100:
        errors.append(f"only {len(chunks)} chunks; need at least 100")
    errors += check_chunking(chunks)
    errors += validate(questions, chunks)

    counts = Counter(q.get("type") for q in questions)
    print(f"Questions: {len(questions)}")
    for t in TYPES:
        print(f"  {t}: {counts.get(t, 0)}")
    gold_total = sum(len(q.get("gold_chunk_ids", [])) for q in questions)
    print(f"Gold labels: {gold_total} chunk IDs across {len(questions)} questions")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nOK: schema valid, all gold IDs exist, hashes match, chunking reproduces the dump.")


if __name__ == "__main__":
    main()
