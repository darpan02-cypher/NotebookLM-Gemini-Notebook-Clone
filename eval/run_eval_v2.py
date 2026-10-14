"""v2 eval: vector vs rerank on the harder corpus, 40 questions, gold labelled by chunk ID.

Usage:
  python -m eval.run_eval_v2 --retrieval-only     # no LLM calls
  python -m eval.run_eval_v2                      # retrieval + generation (needs GROQ_API_KEY in env or .env)

Writes to eval/results/v2/:
  per_question.csv, summary.csv, eval_log.json     (both modes)
  generations.json, generation_cache.json          (full mode only)

Ranking: each method returns a ranked top-10 list of chunk IDs ("<file>#<chunk_index>").
  vector = Chroma cosine order; rerank = the 20 vector candidates reordered by the cross-encoder.
Generation uses the first GEN_TOP_K chunks of that same ranking, with the app's system prompt and
prompt layout but without chat history, so questions cannot influence each other.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval.gen_utils import GenerationCache, cache_key, parse_citations, retry_call  # noqa: E402
from eval.metrics import K_MAX, gold_rank, reciprocal_rank, summarize  # noqa: E402
from eval.validate_questions import CHUNKS_PATH, CORPUS_DIR, QUESTIONS_PATH, check_chunking, validate  # noqa: E402
from generation import GroqClient, LLMError  # noqa: E402
from generation.chat import SYSTEM_PROMPT, build_citations  # noqa: E402
from ingestion import SourceIngestor  # noqa: E402
from retrieval import LocalEmbedder, Retriever  # noqa: E402
from storage import StorageManager  # noqa: E402

METHODS = ["vector", "rerank"]
CANDIDATES = 20
GEN_TOP_K = 5  # matches the app default
OUT_DIR = ROOT / "eval" / "results" / "v2"


def chunk_id_of(hit: Dict[str, Any]) -> str:
    m = hit["metadata"]
    return f"{m['source_name']}#{int(m['chunk_index']):03d}"


def load_and_validate() -> tuple[list[dict], list[dict]]:
    dump = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    errors = check_chunking(dump) + validate(questions, dump)
    if errors:
        sys.exit("Question/corpus validation failed:\n  - " + "\n  - ".join(errors))
    return questions, dump


def retrieval_pass(retriever: Retriever, nb_id: str, questions: List[dict]) -> List[Dict[str, Any]]:
    log = []
    for q in questions:
        for method in METHODS:
            res = retriever.retrieve(nb_id, q["text"], method=method, top_k=K_MAX, candidates=CANDIDATES)
            ids = [chunk_id_of(h) for h in res["hits"]]
            log.append(
                {
                    "question_id": q["id"],
                    "type": q["type"],
                    "method": method,
                    "retrieved_chunk_ids": ids,
                    "scores": [round(h["score"], 4) for h in res["hits"]],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "gold_rank": gold_rank(ids, q["gold_chunk_ids"]),
                    "retrieval_latency_s": round(res["latency_s"], 5),
                    "_hits": res["hits"],  # stripped before saving; used for generation
                }
            )
    return log


def summary_rows(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    groups = ["all", "direct", "paraphrased", "confusable"]
    for method in METHODS:
        for g in groups:
            sel = [r for r in log if r["method"] == method and (g == "all" or r["type"] == g)]
            s = summarize([r["gold_rank"] for r in sel], [r["retrieval_latency_s"] for r in sel])
            rows.append({"method": method, "group": g, **s})
    return rows


def print_summary(rows: List[Dict[str, Any]]) -> None:
    head = f"{'method':8} {'group':12} {'n':>3} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR@10':>7} {'meanRank':>9} {'miss':>5} {'lat ms':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        mr = "-" if r["mean_gold_rank"] is None else f"{r['mean_gold_rank']:.2f}"
        print(
            f"{r['method']:8} {r['group']:12} {r['n']:>3} "
            f"{r['hit1_count']:>2}/{r['n']:<4} {r['hit3_count']:>2}/{r['n']:<4} {r['hit5_count']:>2}/{r['n']:<4} "
            f"{r[f'mrr{K_MAX}']:>7.3f} {mr:>9} {r['n_missed']:>5} {r['latency_ms_mean']:>8.1f}"
        )


def write_outputs(log: List[Dict[str, Any]], questions: List[dict], rows: List[Dict[str, Any]], mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by = {(r["question_id"], r["method"]): r for r in log}
    with open(OUT_DIR / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "type", "n_gold", "vector_rank", "rerank_rank", "vector_rr", "rerank_rr"])
        for q in questions:
            v, r = by[(q["id"], "vector")]["gold_rank"], by[(q["id"], "rerank")]["gold_rank"]
            w.writerow(
                [q["id"], q["type"], len(q["gold_chunk_ids"]), "" if v is None else v, "" if r is None else r,
                 f"{reciprocal_rank(v):.4f}", f"{reciprocal_rank(r):.4f}"]
            )
    cols = ["method", "group", "n", "hit1_count", "hit1", "hit3_count", "hit3", "hit5_count", "hit5",
            f"mrr{K_MAX}", "mean_gold_rank", "n_missed", "latency_ms_mean", "latency_ms_median"]
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else (round(r[c], 4) if isinstance(r[c], float) else r[c])) for c in cols})
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in log]
    payload = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS,
        "candidates": CANDIDATES,
        "k_max": K_MAX,
        "records": clean,
    }
    (OUT_DIR / "eval_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_retryable(exc: Exception) -> bool:
    return isinstance(exc, LLMError) and "GROQ_API_KEY is not set" not in str(exc)


def generation_pass(log: List[Dict[str, Any]], questions: List[dict]) -> List[Dict[str, Any]]:
    llm = GroqClient()
    cache = GenerationCache(OUT_DIR / "generation_cache.json")
    qtext = {q["id"]: q for q in questions}
    out = []
    for i, rec in enumerate(log, start=1):
        q = qtext[rec["question_id"]]
        hits = rec["_hits"][:GEN_TOP_K]
        ids = rec["retrieved_chunk_ids"][:GEN_TOP_K]
        citations = build_citations(hits)
        context = "\n\n".join(f"[{c['n']}] ({c['source_name']}) {h['text']}" for c, h in zip(citations, hits))
        prompt = (
            f"Recent conversation:\n(none)\n\n"
            f"Context passages:\n{context}\n\nQuestion: {q['text']}\nAnswer with citations:"
        )
        key = cache_key(llm.model, SYSTEM_PROMPT, prompt)
        entry = cache.get(key)
        cached = entry is not None
        if not cached:
            t0 = time.perf_counter()
            answer = retry_call(lambda: llm.generate(prompt, system=SYSTEM_PROMPT), is_retryable)
            gen_s = time.perf_counter() - t0
            answer = re.sub(r"[【\[]\s*(\d+)\s*[】\]]", r"[\1]", answer)  # same normalisation as ChatService
            cache.put(key, answer, gen_s)
            entry = cache.get(key)
        answer, gen_s = entry["answer"], float(entry["gen_latency_s"])
        cited, invalid = parse_citations(answer, ids)
        out.append(
            {
                "question_id": rec["question_id"],
                "method": rec["method"],
                "context_chunk_ids": ids,
                "answer": answer,
                "cited_chunk_ids": cited,
                "invalid_citation_numbers": invalid,
                "cited_gold": any(c in set(q["gold_chunk_ids"]) for c in cited),
                "context_has_gold": any(c in set(q["gold_chunk_ids"]) for c in ids),
                "generation_latency_s": round(gen_s, 3),
                "end_to_end_latency_s": round(rec["retrieval_latency_s"] + gen_s, 3),
                "from_cache": cached,
                "reference_answer": q["reference_answer"],
            }
        )
        print(f"[{i}/{len(log)}] {rec['question_id']} {rec['method']:7} cached={cached!s:5} gold_in_ctx={out[-1]['context_has_gold']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-only", action="store_true", help="skip all LLM calls")
    args = ap.parse_args()

    if not args.retrieval_only and not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (env or .env). Use --retrieval-only for the no-LLM run.")

    questions, dump = load_and_validate()
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager(tmp)
        embedder = LocalEmbedder()
        retriever = Retriever(storage, embedder)
        nb = storage.create_notebook("eval-v2")
        ingestor = SourceIngestor(storage, embedder)
        for f in sorted(CORPUS_DIR.glob("*.txt")):
            ingestor.ingest(nb["id"], "txt", f.name, str(f))

        for m in METHODS:  # warm up embedder, Chroma and the cross-encoder before timing
            for _ in range(3):
                retriever.retrieve(nb["id"], "warm up query", method=m, top_k=K_MAX, candidates=CANDIDATES)

        log = retrieval_pass(retriever, nb["id"], questions)
        seen = {cid for r in log for cid in r["retrieved_chunk_ids"]}
        known = {c["id"] for c in dump}
        if not seen <= known:
            sys.exit(f"Retrieved chunk IDs missing from corpus_chunks.json: {sorted(seen - known)[:5]}")

    rows = summary_rows(log)
    print(f"\nChunks in corpus: {len(dump)} | questions: {len(questions)} | candidates for rerank: {CANDIDATES}\n")
    print_summary(rows)

    mode = "retrieval-only" if args.retrieval_only else "full"
    write_outputs(log, questions, rows, mode)

    if not args.retrieval_only:
        print(f"\nGenerating answers with top-{GEN_TOP_K} context ...")
        gens = generation_pass(log, questions)
        (OUT_DIR / "generations.json").write_text(json.dumps(gens, indent=2, ensure_ascii=False), encoding="utf-8")
        per = defaultdict(list)
        for g in gens:
            per[g["method"]].append(g)
        print("\nGeneration summary (automatic signals only):")
        for m in METHODS:
            gs = per[m]
            print(
                f"  {m:7} context_has_gold={sum(g['context_has_gold'] for g in gs)}/{len(gs)} "
                f"cited_gold={sum(g['cited_gold'] for g in gs)}/{len(gs)} "
                f"mean_e2e={sum(g['end_to_end_latency_s'] for g in gs) / len(gs):.2f}s"
            )
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
