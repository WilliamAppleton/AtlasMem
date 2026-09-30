"""Read-time curator: synthesizes a task-specific briefing from retrieved raw trajectories.

``build_curator_messages`` is kept as a pure function so the same prompt can be reused
as the policy input when training the curator with GRPO.
"""

from __future__ import annotations

import re

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
in the memories; if something the task needs is not covered (e.g. where an object is), say it \
is unknown rather than guessing.

The guidance must cover every requirement of the CURRENT task (object, any required state or \
processing, destination), even requirements the memories do not show, and must not include \
steps the CURRENT task does not ask for.

Be concise - the agent has limited context. Use at most {max_words} words of plain bullet \
points (no headings, tables or bold). Lead with the most useful concrete facts, such as where \
things were found and the exact commands that worked.

Write your final answer inside <briefing> and </briefing> tags. Only the text inside the tags \
is shown to the agent."""

SEPARATOR = "=" * 8

_BRIEFING = re.compile(r"<briefing>(.*?)(?:</briefing>|$)", re.DOTALL | re.IGNORECASE)


def build_curator_messages(
    task: str, trajectories: list[Trajectory], max_chars_per_traj: int = 6000, max_words: int = 120
) -> list[Message]:
    parts = [f"CURRENT TASK: {task}", "", f"RETRIEVED MEMORIES ({len(trajectories)}):"]
    for i, t in enumerate(trajectories, 1):
        parts.append(f"{SEPARATOR} Memory {i} {SEPARATOR}")
        parts.append(t.render(max_chars=max_chars_per_traj))
    parts.append(f"{SEPARATOR} End of memories {SEPARATOR}")
    parts.append("")
    parts.append("Write the briefing for the CURRENT TASK.")
    return [
        {"role": "system", "content": CURATOR_SYSTEM.format(max_words=max_words)},
        {"role": "user", "content": "\n".join(parts)},
    ]


def extract_briefing(text: str, max_words: int | None = None) -> str:
    """Keep only the tagged briefing (drops leaked reasoning); hard-cap its length.

    Untagged output is used as-is. The cap is 2x the requested budget, cut at a line boundary
    where possible, so a slightly long briefing survives but a runaway one does not.
    """
    matches = _BRIEFING.findall(text)
    body = next((m.strip() for m in reversed(matches) if m.strip()), text.strip())
    if max_words is None:
        return body
    cap = 2 * max_words
    if len(body.split()) <= cap:
        return body
    kept, count = [], 0
    for line in body.splitlines():
        n = len(line.split())
        if count + n > cap:
            break
        kept.append(line)
        count += n
    if not kept:  # a single enormous line
        return " ".join(body.split()[:cap]) + " ..."
    return "\n".join(kept).rstrip()


class Curator:
    def __init__(self, llm: LLM, max_chars_per_traj: int = 6000, max_words: int = 120):
        self.llm = llm
        self.max_chars_per_traj = max_chars_per_traj
        self.max_words = max_words

    def curate(self, task: str, trajectories: list[Trajectory]) -> str:
        if not trajectories:
            return ""
        messages = build_curator_messages(task, trajectories, self.max_chars_per_traj, self.max_words)
        return extract_briefing(self.llm.chat(messages), self.max_words)
