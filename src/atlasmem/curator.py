"""Read-time curator: synthesizes a task-specific briefing from retrieved raw trajectories.

``build_curator_messages`` is kept as a pure function so the same prompt can be reused
as the policy input when training the curator with GRPO.
"""

from __future__ import annotations

from .llm import LLM, Message
from .types import Trajectory

CURATOR_SYSTEM = """You are a Memory Curator. Your job: synthesize these raw memories into a \
concise, actionable briefing for an agent that is about to attempt a new task.

1. Identify which past experiences are most relevant to the CURRENT task.
2. Extract strategies that worked (action sequences, preconditions, where things were found, \
pitfalls that were overcome).
3. Give specific guidance for THIS task, adapted to its details. Do not copy irrelevant steps.

Ground every instruction in what the memories actually show: use the exact names and \
command phrasing that appear in them. Do not invent facts or physical details that are not \
in the memories; if something the task needs is not covered, say so briefly.

Be concise - the agent has limited context. Output only the briefing."""

SEPARATOR = "=" * 8


def build_curator_messages(
    task: str, trajectories: list[Trajectory], max_chars_per_traj: int = 6000
) -> list[Message]:
    parts = [f"CURRENT TASK: {task}", "", f"RETRIEVED MEMORIES ({len(trajectories)}):"]
    for i, t in enumerate(trajectories, 1):
        parts.append(f"{SEPARATOR} Memory {i} {SEPARATOR}")
        parts.append(t.render(max_chars=max_chars_per_traj))
    parts.append(f"{SEPARATOR} End of memories {SEPARATOR}")
    parts.append("")
    parts.append("Write the briefing for the CURRENT TASK.")
    return [
        {"role": "system", "content": CURATOR_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


class Curator:
    def __init__(self, llm: LLM, max_chars_per_traj: int = 6000):
        self.llm = llm
        self.max_chars_per_traj = max_chars_per_traj

    def curate(self, task: str, trajectories: list[Trajectory]) -> str:
        if not trajectories:
            return ""
        return self.llm.chat(build_curator_messages(task, trajectories, self.max_chars_per_traj))
