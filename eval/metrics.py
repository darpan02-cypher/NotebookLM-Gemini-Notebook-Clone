"""Rank-based retrieval metrics. Pure functions, no model or I/O dependencies.

A "rank" is the 1-based position of the first gold chunk in a ranked list of chunk IDs, or None when
no gold chunk appears in the top K_MAX. A question can have several gold chunks; the first one counts.
"""
from __future__ import annotations

from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

K_MAX = 10
HIT_KS = (1, 3, 5)


def gold_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int = K_MAX) -> Optional[int]:
    gold = set(gold_ids)
    for pos, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return pos
    return None


def hit_at_k(rank: Optional[int], k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: Optional[int], k: int = K_MAX) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def summarize(ranks: List[Optional[int]], latencies_s: Optional[List[float]] = None) -> Dict[str, object]:
    """Aggregate one method over a set of questions."""
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    out: Dict[str, object] = {"n": n}
    for k in HIT_KS:
        hits = sum(hit_at_k(r, k) for r in ranks)
        out[f"hit{k}_count"] = hits
        out[f"hit{k}"] = hits / n if n else 0.0
    out[f"mrr{K_MAX}"] = sum(reciprocal_rank(r) for r in ranks) / n if n else 0.0
    out["mean_gold_rank"] = mean(found) if found else None  # over questions where gold is in the top K_MAX
    out["n_missed"] = n - len(found)
    if latencies_s:
        out["latency_ms_mean"] = mean(latencies_s) * 1000
        out["latency_ms_median"] = median(latencies_s) * 1000
    return out
