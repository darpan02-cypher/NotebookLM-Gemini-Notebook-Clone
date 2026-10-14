"""Step 4: judge the generated answers with two independent judges.

  qwen : an LLM judge (Groq, qwen/qwen3.8-27b, temperature 0). Needs GROQ_API_KEY.
  jev  : TypeSafe's JEV structured-decision model. Needs TYPESAFE_API_KEY.
Neither judge is the model that wrote the answers (openai/gpt-oss-120b), and the method name
('vector'/'rerank') is never shown to a judge.

Usage:
  python -m eval.judge --judge qwen|jev|both     # score everything in eval/results/v2/generations.json
  python -m eval.judge --spotcheck               # write the 10-item blind human spot-check (never overwrites)
  python -m eval.judge --agreement               # compare your filled-in spot-check with each judge

Pointwise: faithfulness and correctness, each 1-5 with the rubrics below.
Pairwise : both answers shown as A/B (order randomised with a fixed seed and logged); winner = A, B or tie.
Judge calls are cached in eval/results/v2/judge_cache.json and retried with backoff on rate limits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval.gen_utils import GenerationCache, cache_key, retry_call  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "v2"
CHUNKS_PATH = ROOT / "eval" / "corpus_chunks.json"
QUESTIONS_PATH = ROOT / "eval" / "questions_v2.json"
ORDER_SEED = 4242  # A/B order in pairwise judging
SPOT_SEED = 7  # which 10 items go to the human spot check
SPOT_N = 10
QWEN_MODEL = "qwen/qwen3.8-27b"
# JEV is reached through OpenRouter's System One endpoint (OPENROUTER_API_KEY). The version is pinned for reproducibility.
# Override JEV_URL / JEV_MODEL (and use TYPESAFE_API_KEY) to call TypeSafe directly: https://api.typesafe.ai/v1/systemone, jev-1.13.
JEV_URL = os.getenv("JEV_URL", "https://openrouter.ai/api/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")

FAITHFULNESS_LEVELS = [
    "1: Most claims are not supported by the cited passages, or the answer contradicts them, or it states specific facts while citing nothing.",
    "2: Several claims are unsupported by the cited passages; only a minor part is grounded.",
    "3: About half of the claims are supported by the cited passages; the rest go beyond them.",
    "4: Nearly all claims are supported by the cited passages; at most one minor detail goes beyond them.",
    "5: Every claim is directly supported by the cited passages (or the answer correctly says the sources do not contain the answer).",
]
CORRECTNESS_LEVELS = [
    "1: Wrong, or does not answer the question.",
    "2: Mostly wrong; contains a small correct element.",
    "3: Partly correct: key information is missing or one key fact is wrong.",
    "4: Correct on the key fact(s) but with a minor error or omission compared with the reference answer.",
    "5: Fully correct and consistent with the reference answer.",
]
PAIR_OPTIONS = {
    "A": "Answer A is better: more correct and better supported by the question's reference answer.",
    "B": "Answer B is better: more correct and better supported by the question's reference answer.",
    "tie": "The two answers are equally good (same key facts, or equally wrong).",
}


# ---------------------------------------------------------------- pure helpers (unit-tested)

def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract the last JSON object from a model reply (drops <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    objs = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    if not objs:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return json.loads(objs[-1])


def clamp_score(x: Any) -> int:
    return max(1, min(5, int(round(float(x)))))


def jev_score_to_1_5(score: float, legend: Optional[Dict[str, Any]]) -> float:
    """JEV returns a position along the criteria list. Convert to a 1-5 scale using the legend's first key as base."""
    base = 1
    if legend:
        try:
            base = min(int(k) for k in legend)
        except ValueError:
            pass
    return float(score) - base + 1


def assign_orders(question_ids: List[str], seed: int = ORDER_SEED) -> Dict[str, str]:
    """Which method is shown as 'A' for each question: 'vector' or 'rerank'.

    Balanced (half and half, differing by at most one) and deterministic for a given seed, so position bias
    cannot be confounded with method by an unlucky draw.
    """
    ids = sorted(question_ids)
    labels = ["vector" if i % 2 == 0 else "rerank" for i in range(len(ids))]
    random.Random(seed).shuffle(labels)
    return dict(zip(ids, labels))


def pick_spot_items(item_keys: List[str], n: int = SPOT_N, seed: int = SPOT_SEED) -> List[str]:
    return random.Random(seed).sample(sorted(item_keys), n)


def agreement(human: List[Optional[float]], judge: List[Optional[float]]) -> Dict[str, Any]:
    pairs = [(h, j) for h, j in zip(human, judge) if h is not None and j is not None]
    if not pairs:
        return {"n": 0}
    diffs = [abs(h - round(j)) for h, j in pairs]
    return {
        "n": len(pairs),
        "exact": sum(d == 0 for d in diffs) / len(pairs),
        "within1": sum(d <= 1 for d in diffs) / len(pairs),
        "mean_abs_diff": mean(abs(h - j) for h, j in pairs),
    }


def pointwise_prompt(item: Dict[str, Any]) -> str:
    passages = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(item["cited_passages"], 1)) or "(the answer cited no passages)"
    return (
        f"QUESTION:\n{item['question']}\n\nREFERENCE ANSWER:\n{item['reference_answer']}\n\n"
        f"PASSAGES CITED BY THE ANSWER:\n{passages}\n\nANSWER TO JUDGE:\n{item['answer']}"
    )


def pairwise_prompt(item: Dict[str, Any]) -> str:
    return (
        f"QUESTION:\n{item['question']}\n\nREFERENCE ANSWER:\n{item['reference_answer']}\n\n"
        f"ANSWER A:\n{item['answer_a']}\n\nANSWER B:\n{item['answer_b']}"
    )


# ---------------------------------------------------------------- judges

def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status:
        return status in (408, 409, 429) or status >= 500
    return exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError", "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"}


class QwenJudge:
    name = "qwen"

    def __init__(self, cache: GenerationCache, model: str = QWEN_MODEL):
        from groq import Groq

        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY is not set (env or .env).")
        self.client, self.model, self.cache = Groq(api_key=key), model, cache

    def _ask(self, system: str, prompt: str) -> Dict[str, Any]:
        key = cache_key(self.name + ":" + self.model, system, prompt)
        hit = self.cache.get(key)
        if hit is None:
            def call() -> str:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                )
                return r.choices[0].message.content or ""

            t0 = time.perf_counter()
            raw = retry_call(call, is_retryable)
            self.cache.put(key, raw, time.perf_counter() - t0)
            hit = self.cache.get(key)
        return {"raw": hit["answer"], "parsed": parse_json_object(hit["answer"])}

    def pointwise(self, item: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are a strict evaluator of answers produced by a retrieval-augmented system. Score the ANSWER on two criteria, each 1-5.\n"
            "FAITHFULNESS (is each claim supported by the PASSAGES CITED BY THE ANSWER; ignore outside truth):\n" + "\n".join(FAITHFULNESS_LEVELS) +
            "\nCORRECTNESS (compared with the REFERENCE ANSWER):\n" + "\n".join(CORRECTNESS_LEVELS) +
            '\nReply with ONLY a JSON object: {"reason": "<one short sentence>", "faithfulness": <1-5>, "correctness": <1-5>}'
        )
        r = self._ask(system, pointwise_prompt(item))["parsed"]
        return {"faithfulness": float(clamp_score(r["faithfulness"])), "correctness": float(clamp_score(r["correctness"])), "reason": r.get("reason", "")}

    def pairwise(self, item: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You compare two answers (A and B) to the same question, using the REFERENCE ANSWER as ground truth. "
            "Pick the answer that is more correct and more complete with respect to the reference; if they are equally good or equally wrong, say tie. "
            "Ignore answer length, formatting and position.\n"
            'Reply with ONLY a JSON object: {"reason": "<one short sentence>", "winner": "A" | "B" | "tie"}'
        )
        r = self._ask(system, pairwise_prompt(item))["parsed"]
        w = str(r["winner"]).strip()
        return {"winner": w if w in ("A", "B") else "tie", "reason": r.get("reason", "")}


class JevJudge:
    name = "jev"

    def __init__(self, cache: GenerationCache):
        key = os.getenv("OPENROUTER_API_KEY") if "openrouter" in JEV_URL else os.getenv("TYPESAFE_API_KEY")
        if not key:
            sys.exit("OPENROUTER_API_KEY (or TYPESAFE_API_KEY when JEV_URL points at TypeSafe) is not set (env or .env).")
        self._key, self.cache = key, cache

    def _ask(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        ck = cache_key("jev:" + JEV_MODEL, json.dumps(questions, sort_keys=True), state)
        hit = self.cache.get(ck)
        if hit is None:
            def call() -> str:
                r = requests.post(
                    JEV_URL,
                    headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                    json={"state": state, "model": JEV_MODEL, "questions": questions},
                    timeout=60,
                )
                r.raise_for_status()
                return json.dumps(r.json()["answers"])

            t0 = time.perf_counter()
            raw = retry_call(call, is_retryable)
            self.cache.put(ck, raw, time.perf_counter() - t0)
            hit = self.cache.get(ck)
        return json.loads(hit["answer"])

    def pointwise(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qs = {
            "faithfulness": {
                "type": "score",
                "instructions": "How well is every claim in the ANSWER TO JUDGE supported by the PASSAGES CITED BY THE ANSWER (ignore outside knowledge)",
                "criteria": FAITHFULNESS_LEVELS,
            },
            "correctness": {
                "type": "score",
                "instructions": "How correct is the ANSWER TO JUDGE compared with the REFERENCE ANSWER for the QUESTION",
                "criteria": CORRECTNESS_LEVELS,
            },
        }
        a = self._ask(pointwise_prompt(item), qs)
        f, c = a["faithfulness"], a["correctness"]
        return {
            "faithfulness": jev_score_to_1_5(f["score"], f.get("legend")),
            "correctness": jev_score_to_1_5(c["score"], c.get("legend")),
            "faithfulness_confidence": f.get("confidence"),
            "correctness_confidence": c.get("confidence"),
        }

    def pairwise(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qs = {
            "winner": {
                "type": "choice",
                "instructions": "Which answer to the QUESTION is better, judged against the REFERENCE ANSWER; ignore length and position",
                "criteria": PAIR_OPTIONS,
            }
        }
        a = self._ask(pairwise_prompt(item), qs)["winner"]
        return {"winner": a["choice"], "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}


# ---------------------------------------------------------------- orchestration

def load_inputs():
    gens = json.loads((RESULTS / "generations.json").read_text(encoding="utf-8"))
    chunks = {c["id"]: c["text"] for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]}
    questions = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    return gens, chunks, questions


def run_judge(judge, gens, chunks, questions) -> Dict[str, Any]:
    by = {(g["question_id"], g["method"]): g for g in gens}
    qids = sorted({g["question_id"] for g in gens})
    point, pair = [], []
    for g in gens:
        q = questions[g["question_id"]]
        item = {
            "question": q["text"],
            "reference_answer": q["reference_answer"],
            "cited_passages": [chunks[c] for c in g["cited_chunk_ids"]],
            "answer": g["answer"],
        }
        s = judge.pointwise(item)
        point.append({"question_id": g["question_id"], "method": g["method"], **s})
    orders = assign_orders(qids)
    for qid in qids:
        v, r = by[(qid, "vector")], by[(qid, "rerank")]
        if v["answer"].strip() == r["answer"].strip():
            pair.append({"question_id": qid, "a_is": None, "winner_method": "tie", "identical_answers": True})
            continue
        a_is = orders[qid]
        first, second = (v, r) if a_is == "vector" else (r, v)
        q = questions[qid]
        res = judge.pairwise(
            {"question": q["text"], "reference_answer": q["reference_answer"], "answer_a": first["answer"], "answer_b": second["answer"]}
        )
        other = "rerank" if a_is == "vector" else "vector"
        winner = {"A": a_is, "B": other, "tie": "tie"}[res["winner"]]
        pair.append({"question_id": qid, "a_is": a_is, "judge_winner": res["winner"], "winner_method": winner, "identical_answers": False, **{k: v for k, v in res.items() if k != "winner"}})
    return {"judge": judge.name, "order_seed": ORDER_SEED, "pointwise": point, "pairwise": pair}


def print_and_save_summary() -> None:
    rows = []
    for path in sorted(RESULTS.glob("judgments_*.json")):
        j = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n=== judge: {j['judge']} ===")
        for m in ("vector", "rerank"):
            p = [x for x in j["pointwise"] if x["method"] == m]
            f, c = mean(x["faithfulness"] for x in p), mean(x["correctness"] for x in p)
            print(f"  {m:7} n={len(p)}  faithfulness={f:.2f}  correctness={c:.2f}")
            rows.append({"judge": j["judge"], "view": "pointwise", "method": m, "n": len(p), "faithfulness": round(f, 3), "correctness": round(c, 3)})
        w = [x["winner_method"] for x in j["pairwise"]]
        ident = sum(x["identical_answers"] for x in j["pairwise"])
        print(f"  pairwise: rerank wins {w.count('rerank')}, vector wins {w.count('vector')}, ties {w.count('tie')} (of which {ident} identical answers) / {len(w)}")
        rows.append({"judge": j["judge"], "view": "pairwise", "rerank_wins": w.count("rerank"), "vector_wins": w.count("vector"), "ties": w.count("tie"), "identical_answers": ident, "n": len(w)})
    if rows:
        cols = sorted({k for r in rows for k in r}, key=lambda k: ["judge", "view", "method", "n"].index(k) if k in ("judge", "view", "method", "n") else 9)
        with open(RESULTS / "judge_summary.csv", "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=cols)
            w_.writeheader()
            w_.writerows(rows)


def write_spotcheck() -> None:
    out, key_path = RESULTS / "judge_spotcheck.csv", RESULTS / "judge_spotcheck_key.json"
    if out.exists() or key_path.exists():
        sys.exit(f"{out.name} already exists; refusing to overwrite your scores. Move it aside to regenerate.")
    gens, chunks, questions = load_inputs()
    keys = [f"{g['question_id']}|{g['method']}" for g in gens]
    chosen = pick_spot_items(keys)
    rng = random.Random(SPOT_SEED + 1)
    ids = rng.sample(range(100, 999), len(chosen))
    key, rows = {}, []
    for item_id, k in zip(ids, chosen):
        qid, method = k.split("|")
        g = next(x for x in gens if x["question_id"] == qid and x["method"] == method)
        q = questions[qid]
        key[f"S{item_id}"] = {"question_id": qid, "method": method}
        rows.append(
            {
                "item_id": f"S{item_id}",
                "question": q["text"],
                "reference_answer": q["reference_answer"],
                "cited_passages": " || ".join(chunks[c] for c in g["cited_chunk_ids"]) or "(none cited)",
                "answer": g["answer"],
                "human_faithfulness_1to5": "",
                "human_correctness_1to5": "",
            }
        )
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} items; method hidden, judge scores not shown). Fill the two empty columns (1-5) and run --agreement.")


def run_agreement() -> None:
    rows = list(csv.DictReader(open(RESULTS / "judge_spotcheck.csv", encoding="utf-8")))
    key = json.loads((RESULTS / "judge_spotcheck_key.json").read_text(encoding="utf-8"))
    num = lambda s: float(s) if s.strip() else None  # noqa: E731
    human = {r["item_id"]: (num(r["human_faithfulness_1to5"]), num(r["human_correctness_1to5"])) for r in rows}
    if all(h is None for pair in human.values() for h in pair):
        sys.exit("No human scores found in judge_spotcheck.csv yet.")
    for path in sorted(RESULTS.glob("judgments_*.json")):
        j = json.loads(path.read_text(encoding="utf-8"))
        look = {(x["question_id"], x["method"]): x for x in j["pointwise"]}
        print(f"\n=== agreement with judge: {j['judge']} ===")
        for dim, idx in (("faithfulness", 0), ("correctness", 1)):
            hs = [human[i][idx] for i in human]
            js = [look[(key[i]["question_id"], key[i]["method"])][dim] for i in human]
            a = agreement(hs, js)
            if a["n"]:
                print(f"  {dim:12} n={a['n']}  exact={a['exact']:.0%}  within±1={a['within1']:.0%}  mean|diff|={a['mean_abs_diff']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=["qwen", "jev", "both"])
    ap.add_argument("--spotcheck", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()
    if args.spotcheck:
        return write_spotcheck()
    if args.agreement:
        return run_agreement()
    if not args.judge:
        ap.error("choose --judge, --spotcheck or --agreement")
    gens, chunks, questions = load_inputs()
    cache = GenerationCache(RESULTS / "judge_cache.json")
    names = ["qwen", "jev"] if args.judge == "both" else [args.judge]
    for n in names:
        judge = QwenJudge(cache) if n == "qwen" else JevJudge(cache)
        print(f"Running judge '{n}' on {len(gens)} answers ...")
        res = run_judge(judge, gens, chunks, questions)
        (RESULTS / f"judgments_{n}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print_and_save_summary()


if __name__ == "__main__":
    main()
