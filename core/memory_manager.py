from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Dict


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


def _tf(text: str) -> Dict[str, float]:
    tokens = _tokenize(text)
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = float(len(tokens) or 1)
    return {k: v / total for k, v in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryManager:
    """Simplest RAG: local jsonl with bag-of-words cosine similarity.
    Convenient for running the flow without external dependencies.
    """

    def __init__(self, persist_path: str, top_k: int = 4) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        if not self.persist_path.exists():
            self.persist_path.touch()

    def add_memory(self, text: str, meta: Dict[str, str] | None = None) -> None:
        record = {"text": text, "meta": meta or {}}
        with self.persist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def search(self, query: str) -> List[str]:
        qvec = _tf(query)
        results: List[tuple[float, str]] = []
        with self.persist_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                score = _cosine(qvec, _tf(text))
                if score > 0:
                    results.append((score, text))
        results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in results[: self.top_k]]

