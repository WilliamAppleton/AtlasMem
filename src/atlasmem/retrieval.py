"""BM25 retrieval over task descriptions only (decoupled from trajectory length)."""

from __future__ import annotations

import math
import re
from collections import Counter

from .store import MemoryBank
from .types import Trajectory

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to in on at for with from by is are be it its this that as into".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a list of documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.tfs = [Counter(d) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.n if self.n else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for doc, tf in zip(self.docs, self.tfs):
            dl = len(doc)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class BM25Retriever:
    """Retrieves the top-k raw trajectories whose task descriptions best match the query."""

    def __init__(self, bank: MemoryBank, k: int = 3, min_score: float = 0.0):
        self.bank = bank
        self.k = k
        self.min_score = min_score
        self._built_version = -1
        self._ids: list[str] = []
        self._index: BM25 | None = None

    def _refresh(self) -> None:
        if self._built_version == self.bank.version and self._index is not None:
            return
        pairs = self.bank.tasks()
        self._ids = [p[0] for p in pairs]
        self._index = BM25([p[1] for p in pairs])
        self._built_version = self.bank.version

    def retrieve(self, task: str, k: int | None = None) -> list[tuple[Trajectory, float]]:
        self._refresh()
        k = self.k if k is None else k
        if not self._ids or k <= 0:
            return []
        assert self._index is not None
        scored = sorted(zip(self._ids, self._index.scores(task)), key=lambda x: -x[1])
        out = []
        for tid, score in scored[:k]:
            if score <= self.min_score:
                break
            traj = self.bank.get(tid)
            if traj is not None:
                out.append((traj, score))
        return out
