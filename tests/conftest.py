from __future__ import annotations

import pytest

from atlasmem.envs.kitchen import oracle_actions


class FakeLLM:
    """Returns scripted replies (or a callable's output) and records every call."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = fn
        self.calls: list[list[dict]] = []

    def chat(self, messages, **options):
        self.calls.append(messages)
        if self.fn:
            return self.fn(messages)
        return self.replies.pop(0) if self.replies else ""


def solution(task_goal, layout_seed: int = 0) -> list[str]:
    state, obj, target = task_goal
    desc = f"{state} {obj}" if state else obj
    return oracle_actions(f"put a {desc} in {target}", layout_seed)


@pytest.fixture
def fake_llm():
    return FakeLLM
