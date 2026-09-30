"""JitMemory: the public facade tying store, retriever, curator and judge together.

Write path: keep the raw trajectory, gated only by success (LLM judge or a ground-truth label).
Read path: BM25 top-k over task descriptions -> curator synthesizes an ephemeral briefing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .curator import Curator
from .judge import LLMJudge, Verdict
from .retrieval import BM25Retriever
from .store import MemoryBank
from .types import Briefing, Trajectory


@dataclass
class RecordResult:
    stored: bool
    verdict: Verdict


class JitMemory:
    def __init__(
        self,
        bank: MemoryBank,
        curator: Curator,
        judge: LLMJudge | None = None,
        retriever: BM25Retriever | None = None,
        k: int = 3,
    ):
        self.bank = bank
        self.curator = curator
        self.judge = judge
        self.retriever = retriever or BM25Retriever(bank, k=k)

    def brief(self, task: str, k: int | None = None) -> Briefing:
        hits = self.retriever.retrieve(task, k=k)
        trajs = [t for t, _ in hits]
        payload = self.curator.curate(task, trajs) if trajs else ""
        return Briefing(task=task, payload=payload, retrieved=trajs, scores=[s for _, s in hits])

    def evaluate(self, traj: Trajectory, label: bool | None = None) -> Verdict:
        """Decide whether a trajectory is worth storing.

        Priority: explicit ``label`` > LLM judge > ``traj.success`` from the environment.
        """
        if label is not None:
            return Verdict(success=label, reason="ground-truth label")
        if self.judge is not None:
            return self.judge.judge(traj)
        if traj.success is not None:
            return Verdict(success=traj.success, reason="environment success signal")
        raise ValueError("No label, judge, or traj.success available to gate this trajectory.")

    def record(self, traj: Trajectory, label: bool | None = None) -> RecordResult:
        verdict = self.evaluate(traj, label)
        if verdict.success:
            traj.metadata.setdefault("judge_reason", verdict.reason)
            self.bank.add(traj)
        return RecordResult(stored=verdict.success, verdict=verdict)

    def record_batch(
        self, trajs: Iterable[Trajectory], labels: Iterable[bool | None] | None = None
    ) -> list[RecordResult]:
        """Evaluate every trajectory first, then append, so a batch sees one bank state."""
        trajs = list(trajs)
        labels = list(labels) if labels is not None else [None] * len(trajs)
        verdicts = [self.evaluate(t, lab) for t, lab in zip(trajs, labels)]
        results = []
        for t, v in zip(trajs, verdicts):
            if v.success:
                t.metadata.setdefault("judge_reason", v.reason)
                self.bank.add(t)
            results.append(RecordResult(stored=v.success, verdict=v))
        return results
