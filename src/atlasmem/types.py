"""Core data types: a trajectory is a task description plus the raw observation/action sequence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    observation: str
    action: str


@dataclass
class Trajectory:
    """A raw, unabstracted trajectory xi = (x, o1, a1, ..., on, an)."""

    task: str
    steps: list[Step] = field(default_factory=list)
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        steps = [s if isinstance(s, Step) else Step(**s) for s in data.get("steps", [])]
        kwargs = {k: v for k, v in data.items() if k != "steps"}
        return cls(steps=steps, **kwargs)

    def render(self, max_chars: int | None = None) -> str:
        """Plain-text rendering used by the judge and the curator."""
        lines = [f"Task: {self.task}"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"[{i}] Observation: {s.observation}")
            lines.append(f"[{i}] Action: {s.action}")
        text = "\n".join(lines)
        if max_chars is not None and len(text) > max_chars:
            # Keep the head (task + early context) and the tail (how it finished).
            half = max(max_chars // 2 - 20, 0)
            text = text[:half] + "\n... [truncated] ...\n" + text[-half:]
        return text


@dataclass
class Briefing:
    """The ephemeral, task-conditioned memory payload p_t. Never stored."""

    task: str
    payload: str
    retrieved: list[Trajectory] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
