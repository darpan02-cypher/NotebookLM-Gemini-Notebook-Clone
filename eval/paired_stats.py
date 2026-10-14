"""Paired comparison of rerank vs vector on the v2 eval (reads eval/results/v2/per_question.csv).

Usage:  python -m eval.paired_stats
Prints wins/losses/ties (reciprocal rank and Hit@1) and paired-bootstrap 95% intervals for the
difference (rerank - vector) in MRR@10 and Hit@1. Saves eval/results/v2/paired_stats.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import hit_at_k, reciprocal_rank  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results" / "v2"
N_RESAMPLES = 10_000
SEED = 20240501


def win_loss_tie(a: Sequence[float], b: Sequence[float], tol: float = 1e-12) -> Dict[str, int]:
    """Count questions where b beats a (win), loses to a (loss), or equals a (tie)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return {"wins": int((d > tol).sum()), "losses": int((d < -tol).sum()), "ties": int((np.abs(d) <= tol).sum())}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = N_RESAMPLES, seed: int = SEED, alpha: float = 0.05
) -> Dict[str, float]:
    """Resample questions with replacement; return mean(b - a) and its percentile interval."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "includes_zero": bool(lo <= 0 <= hi)}


def load_ranks(path: Path) -> Dict[str, List]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    to_rank = lambda s: int(s) if s else None  # noqa: E731
    return {
        "ids": [r["question_id"] for r in rows],
        "types": [r["type"] for r in rows],
        "vector": [to_rank(r["vector_rank"]) for r in rows],
        "rerank": [to_rank(r["rerank_rank"]) for r in rows],
    }


def analyse(ranks: Dict[str, List], subset: str = "all") -> Dict[str, object]:
    sel = [i for i, t in enumerate(ranks["types"]) if subset == "all" or t == subset]
    v = [ranks["vector"][i] for i in sel]
    r = [ranks["rerank"][i] for i in sel]
    rr_v, rr_r = [reciprocal_rank(x) for x in v], [reciprocal_rank(x) for x in r]
    h_v, h_r = [float(hit_at_k(x, 1)) for x in v], [float(hit_at_k(x, 1)) for x in r]
    return {
        "n": len(sel),
        "reciprocal_rank": win_loss_tie(rr_v, rr_r),
        "hit_at_1": win_loss_tie(h_v, h_r),
        "mrr_diff": paired_bootstrap(rr_v, rr_r),
        "hit1_diff": paired_bootstrap(h_v, h_r),
    }


def fmt_ci(c: Dict[str, float]) -> str:
    verdict = "interval INCLUDES 0" if c["includes_zero"] else "interval EXCLUDES 0"
    return f"{c['mean_diff']:+.3f}  95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  -> {verdict}"


def main() -> None:
    ranks = load_ranks(RESULTS_DIR / "per_question.csv")
    result = {"resamples": N_RESAMPLES, "seed": SEED, "comparison": "rerank minus vector"}
    for subset in ("all", "direct", "paraphrased", "confusable"):
        s = analyse(ranks, subset)
        result[subset] = s
        rr, h1 = s["reciprocal_rank"], s["hit_at_1"]
        tag = "ALL (primary)" if subset == "all" else f"{subset} (exploratory)"
        print(f"\n== {tag}, n={s['n']} ==")
        print(f"  reciprocal rank : rerank wins {rr['wins']}, loses {rr['losses']}, ties {rr['ties']}")
        print(f"  Hit@1           : rerank wins {h1['wins']}, loses {h1['losses']}, ties {h1['ties']}")
        print(f"  MRR@10 diff     : {fmt_ci(s['mrr_diff'])}")
        print(f"  Hit@1 diff      : {fmt_ci(s['hit1_diff'])}")
    print(f"\n(paired bootstrap, {N_RESAMPLES} resamples, seed {SEED}; difference = rerank - vector)")
    (RESULTS_DIR / "paired_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
