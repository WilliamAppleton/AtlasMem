"""Frozen executor loop: the briefing is prepended to the executor prompt."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLM, Message, normalize_text
from .memory import JitMemory, RecordResult
from .types import Briefing, Step, Trajectory


class Environment(Protocol):
    """A text environment. ``step`` returns (observation, done, success)."""

    instructions: str
    max_steps: int

    def reset(self, task: str) -> str: ...

    def step(self, action: str) -> tuple[str, bool, bool]: ...


EXECUTOR_SYSTEM = """You are an agent acting in a text environment.
{instructions}

Each turn you receive an observation. Reply with exactly ONE action on a single line and \
nothing else."""

MEMORY_HEADER = "Guidance from your past experience (may be imperfect):\n"


def parse_action(text: str) -> str:
    for line in normalize_text(text).strip().splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        line = re.sub(r"^(action|>)\s*:?\s*", "", line, flags=re.IGNORECASE)
        return line.strip().strip(".").strip('"').strip()
    return ""


class Executor:
    def __init__(self, llm: LLM, history_window: int = 30):
        self.llm = llm
        self.history_window = history_window

    def build_messages(
        self, env: Environment, task: str, payload: str, steps: list[Step], obs: str
    ) -> list[Message]:
        system = EXECUTOR_SYSTEM.format(instructions=env.instructions)
        if payload:
            system = MEMORY_HEADER + payload.strip() + "\n\n" + system
        msgs: list[Message] = [{"role": "system", "content": system}]
        first = True
        for s in steps[-self.history_window :]:
            content = f"Your task: {task}\n\n{s.observation}" if first else s.observation
            msgs.append({"role": "user", "content": content})
            msgs.append({"role": "assistant", "content": s.action})
            first = False
        msgs.append({"role": "user", "content": f"Your task: {task}\n\n{obs}" if first else obs})
        return msgs

    def run(self, env: Environment, task: str, payload: str = "") -> Trajectory:
        obs = env.reset(task)
        steps: list[Step] = []
        success = False
        for _ in range(env.max_steps):
            action = parse_action(self.llm.chat(self.build_messages(env, task, payload, steps, obs)))
            steps.append(Step(observation=obs, action=action))
            obs, done, success = env.step(action)
            if done:
                break
        steps.append(Step(observation=obs, action="[end]"))
        return Trajectory(task=task, steps=steps, success=success)


@dataclass
class EpisodeResult:
    task: str
    success: bool
    n_steps: int
    stored: bool
    briefing: Briefing | None = None
    trajectory: Trajectory | None = None
    record: RecordResult | None = field(default=None, repr=False)


def run_sequence(
    tasks: list[str],
    make_env: Callable[[], Environment],
    executor: Executor,
    memory: JitMemory | None,
    batch_size: int = 1,
    use_ground_truth: bool = False,
    on_episode: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    """Run tasks in batches. Tasks in a batch share one bank state; the bank updates after each batch."""
    results: list[EpisodeResult] = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        briefings = [memory.brief(t) if memory else None for t in batch]
        trajs = [
            executor.run(make_env(), t, b.payload if b else "") for t, b in zip(batch, briefings)
        ]
        if memory:
            labels = [t.success for t in trajs] if use_ground_truth else None
            recs = memory.record_batch(trajs, labels)
        else:
            recs = [None] * len(trajs)
        for t, b, tr, rec in zip(batch, briefings, trajs, recs):
            r = EpisodeResult(
                task=t,
                success=bool(tr.success),
                n_steps=len(tr.steps) - 1,
                stored=bool(rec and rec.stored),
                briefing=b,
                trajectory=tr,
                record=rec,
            )
            results.append(r)
            if on_episode:
                on_episode(r)
    return results
