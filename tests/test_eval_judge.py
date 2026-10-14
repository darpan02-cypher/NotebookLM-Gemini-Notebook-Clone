"""Unit tests for the pure parts of eval/judge.py (no network)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.judge import (  # noqa: E402
    agreement, assign_orders, clamp_score, jev_score_to_1_5, parse_json_object, pick_spot_items, pointwise_prompt,
)


def test_parse_json_object_handles_think_and_fences():
    reply = '<think>hmm {"faithfulness": 1}</think>\n```json\n{"reason": "ok", "faithfulness": 4, "correctness": 5}\n```'
    assert parse_json_object(reply) == {"reason": "ok", "faithfulness": 4, "correctness": 5}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_clamp_score():
    assert clamp_score(7) == 5 and clamp_score(0) == 1 and clamp_score("3.6") == 4


def test_jev_score_conversion_uses_legend_base():
    assert jev_score_to_1_5(3.0, {"1": "a", "2": "b"}) == 3.0  # 1-based legend
    assert jev_score_to_1_5(2.5, {"0": "a", "1": "b"}) == 3.5  # 0-based legend
    assert jev_score_to_1_5(4.0, None) == 4.0


def test_assign_orders_is_seeded_and_mixed():
    ids = [f"q{i:02d}" for i in range(40)]
    a = assign_orders(ids)
    assert a == assign_orders(list(reversed(ids)))  # independent of input order
    assert set(a.values()) == {"vector", "rerank"}
    assert sum(v == "vector" for v in a.values()) == 20  # balanced
    assert a != assign_orders(ids, seed=1)


def test_pick_spot_items_deterministic_unique():
    keys = [f"q{i}|m" for i in range(80)]
    s = pick_spot_items(keys)
    assert len(s) == 10 and len(set(s)) == 10 and s == pick_spot_items(keys)


def test_agreement_known_values():
    a = agreement([5, 3, 1, None], [5.0, 4.4, 3.0, 2.0])
    assert a["n"] == 3
    assert a["exact"] == pytest.approx(1 / 3)       # 5=5; 3 vs round(4.4)=4; 1 vs 3
    assert a["within1"] == pytest.approx(2 / 3)
    assert a["mean_abs_diff"] == pytest.approx((0 + 1.4 + 2.0) / 3)
    assert agreement([None], [3.0]) == {"n": 0}


def test_prompt_never_contains_method_names():
    p = pointwise_prompt({"question": "Q?", "reference_answer": "R", "cited_passages": ["P1"], "answer": "A [1]"})
    assert "vector" not in p.lower() and "rerank" not in p.lower()
    assert "(the answer cited no passages)" in pointwise_prompt({"question": "Q", "reference_answer": "R", "cited_passages": [], "answer": "A"})
