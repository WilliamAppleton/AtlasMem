from __future__ import annotations

import pytest

from atlasmem.envs.kitchen import make_layout


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


def solution(task_goal: tuple[str | None, str, str], layout_seed: int = 0) -> list[str]:
    """Optimal action script for a kitchen task."""
    state, obj, target = task_goal
    src = make_layout(layout_seed)[obj]
    acts = [f"go to {src}"]
    if src.split()[0] in ("fridge", "cabinet", "drawer"):
        acts.append(f"open {src}")
    acts.append(f"take {obj} from {src}")
    if state:
        app = {"clean": "sinkbasin 1", "hot": "microwave 1", "cool": "fridge 1"}[state]
        verb = {"clean": "clean", "hot": "heat", "cool": "cool"}[state]
        acts += [f"go to {app}", f"{verb} {obj} with {app}"]
    acts.append(f"go to {target}")
    if target.split()[0] in ("fridge", "cabinet", "drawer"):
        acts.append(f"open {target}")
    acts.append(f"put {obj} in {target}")
    return acts


@pytest.fixture
def fake_llm():
    return FakeLLM
