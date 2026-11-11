"""JitMemory: the public facade tying store, retriever, curator and judge together.

Write path: keep every raw trajectory along with its verdict (LLM judge or a ground-truth label).
Read path: BM25 top-k over the task descriptions of trajectories that passed the gate -> curator
synthesizes an ephemeral briefing.
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
    stored: bool  # eligible for retrieval; a judged failure is kept in the bank but is not
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
        """Decide whether a trajectory passed, and so whether retrieval may return it.

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
        return self._store(traj, self.evaluate(traj, label))

    def _store(self, traj: Trajectory, verdict: Verdict) -> RecordResult:
        traj.metadata["verdict"] = {
            "success": verdict.success,
            "reason": verdict.reason,
            "evidence": list(verdict.evidence),
            "overruled": verdict.overruled,
        }
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
        return [self._store(t, v) for t, v in zip(trajs, verdicts)]
