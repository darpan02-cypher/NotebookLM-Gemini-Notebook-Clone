"""Unit tests for the v2 eval helpers (no models, no network)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gen_utils import GenerationCache, cache_key, parse_citations, retry_call  # noqa: E402
from eval.metrics import gold_rank, hit_at_k, reciprocal_rank, summarize  # noqa: E402

RANKED = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]  # 11 items; only the first 10 count


def test_gold_rank_first_gold_wins():
    assert gold_rank(RANKED, ["a"]) == 1
    assert gold_rank(RANKED, ["c", "e"]) == 3  # several gold chunks: the best one counts
    assert gold_rank(RANKED, ["j"]) == 10


def test_gold_rank_missing_or_beyond_k_is_none():
    assert gold_rank(RANKED, ["zzz"]) is None
    assert gold_rank(RANKED, ["k"]) is None  # position 11 is outside the top 10
    assert gold_rank([], ["a"]) is None


def test_hit_at_k_boundaries():
    assert hit_at_k(1, 1) and hit_at_k(3, 3) and hit_at_k(5, 5)
    assert not hit_at_k(2, 1) and not hit_at_k(4, 3) and not hit_at_k(6, 5)
    assert not hit_at_k(None, 5)


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_summarize_known_values():
    # ranks: 1, 2, 4, 6, miss
    s = summarize([1, 2, 4, 6, None], [0.01, 0.03, 0.02, 0.04, 0.05])
    assert s["n"] == 5
    assert (s["hit1_count"], s["hit3_count"], s["hit5_count"]) == (1, 2, 3)
    assert s["hit1"] == pytest.approx(0.2) and s["hit3"] == pytest.approx(0.4) and s["hit5"] == pytest.approx(0.6)
    assert s["mrr10"] == pytest.approx((1 + 0.5 + 0.25 + 1 / 6 + 0) / 5)
    assert s["mean_gold_rank"] == pytest.approx((1 + 2 + 4 + 6) / 4)  # misses excluded, counted separately
    assert s["n_missed"] == 1
    assert s["latency_ms_mean"] == pytest.approx(30.0)
    assert s["latency_ms_median"] == pytest.approx(30.0)


def test_summarize_all_missed():
    s = summarize([None, None])
    assert s["hit1"] == 0.0 and s["mrr10"] == 0.0 and s["mean_gold_rank"] is None and s["n_missed"] == 2


def test_summarize_perfect():
    s = summarize([1, 1, 1])
    assert s["hit1"] == 1.0 and s["mrr10"] == 1.0 and s["mean_gold_rank"] == 1


def test_parse_citations():
    ids = ["x#000", "y#001", "z#002"]
    cited, invalid = parse_citations("Yes [2]. Also [1][2] and [9], [0].", ids)
    assert cited == ["y#001", "x#000"]
    assert invalid == [9, 0]
    assert parse_citations("no markers", ids) == ([], [])


def test_retry_call_backs_off_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 rate limit")
        return "ok"

    out = retry_call(flaky, lambda e: True, base_delay_s=2.0, sleep=sleeps.append, jitter=lambda: 0.0)
    assert out == "ok" and len(calls) == 3
    assert sleeps == [2.0, 4.0]  # exponential


def test_retry_call_gives_up_and_skips_non_retryable():
    sleeps = []
    with pytest.raises(RuntimeError):
        retry_call(lambda: (_ for _ in ()).throw(RuntimeError("boom")), lambda e: True, retries=2, sleep=sleeps.append, jitter=lambda: 0.0)
    assert len(sleeps) == 2

    sleeps.clear()
    with pytest.raises(ValueError):
        retry_call(lambda: (_ for _ in ()).throw(ValueError("bad")), lambda e: False, sleep=sleeps.append)
    assert sleeps == []


def test_generation_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    key = cache_key("m", "sys", "prompt")
    assert key != cache_key("m", "sys", "prompt2") and key != cache_key("m2", "sys", "prompt")
    c = GenerationCache(path)
    assert c.get(key) is None
    c.put(key, "answer [1]", 1.5)
    again = GenerationCache(path)  # new instance reads from disk
    assert again.get(key) == {"answer": "answer [1]", "gen_latency_s": 1.5}


def test_win_loss_tie_and_bootstrap():
    from eval.paired_stats import paired_bootstrap, win_loss_tie

    assert win_loss_tie([0.5, 1.0, 0.0, 0.25], [1.0, 0.5, 0.0, 0.25]) == {"wins": 1, "losses": 1, "ties": 2}
    same = paired_bootstrap([1.0] * 10, [1.0] * 10)
    assert same["mean_diff"] == 0 and same["includes_zero"]
    better = paired_bootstrap([0.0] * 20, [1.0] * 20)
    assert better["ci_low"] == 1.0 and not better["includes_zero"]
    x, y = [0.0, 1.0, 0.5, 0.2, 0.9], [1.0, 1.0, 0.4, 0.5, 0.9]
    assert paired_bootstrap(x, y) == paired_bootstrap(x, y)  # fixed seed -> reproducible
