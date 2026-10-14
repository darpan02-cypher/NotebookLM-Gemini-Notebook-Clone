"""Helpers for the generation half of the eval: citation parsing, retry with backoff, on-disk cache."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def parse_citations(answer: str, chunk_ids: Sequence[str]) -> Tuple[List[str], List[int]]:
    """Map [n] markers in an answer to chunk IDs. Returns (cited chunk IDs in order of first use, invalid n's)."""
    cited: List[str] = []
    invalid: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunk_ids):
            if chunk_ids[n - 1] not in cited:
                cited.append(chunk_ids[n - 1])
        elif n not in invalid:
            invalid.append(n)
    return cited, invalid


def retry_call(
    fn: Callable[[], T],
    is_retryable: Callable[[Exception], bool],
    retries: int = 6,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Call fn(); on a retryable error wait base * 2**attempt (+ up to 25% jitter, capped) and try again."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the caller decides what is retryable
            if attempt >= retries or not is_retryable(exc):
                raise
            delay = min(max_delay_s, base_delay_s * 2**attempt)
            sleep(delay * (1 + 0.25 * jitter()))
            attempt += 1


def cache_key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, system, prompt], ensure_ascii=False).encode("utf-8")).hexdigest()


class GenerationCache:
    """JSON file {key: {"answer": str, "gen_latency_s": float}}, saved after every new entry."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Dict[str, object]] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[Dict[str, object]]:
        return self.data.get(key)

    def put(self, key: str, answer: str, gen_latency_s: float) -> None:
        self.data[key] = {"answer": answer, "gen_latency_s": gen_latency_s}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
