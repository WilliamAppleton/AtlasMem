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

You will see your task, the history of your previous actions and their results, and the \
current observation. Reply with exactly ONE action on a single line and nothing else: no \
explanation, no numbering, no multiple actions."""

MEMORY_HEADER = "Guidance from your past experience (may be imperfect):\n"

# Chat-template control tokens some models (e.g. gpt-oss) leak into content.
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]*\|>")


def parse_action(text: str) -> str:
    text = _SPECIAL_TOKEN.split(normalize_text(text))[0]
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        line = re.sub(r"^(action|next action|>)\s*:?\s*", "", line, flags=re.IGNORECASE)
        return line.strip().strip(".").strip('"').strip()
    return ""


class Executor:
    """Frozen executor. History is sent as one transcript in a single user turn (ReAct-style),
    which avoids models echoing chat-template artifacts from replayed assistant turns."""

    def __init__(self, llm: LLM, history_window: int = 30):
        self.llm = llm
        self.history_window = history_window

    def build_messages(
        self, env: Environment, task: str, payload: str, steps: list[Step], obs: str
    ) -> list[Message]:
        system = EXECUTOR_SYSTEM.format(instructions=env.instructions)
        if payload:
            system = MEMORY_HEADER + payload.strip() + "\n\n" + system
        parts = [f"Your task: {task}", ""]
        if steps:
            parts.append("History:")
            shown = steps[-self.history_window :]
            offset = len(steps) - len(shown)
            if offset:
                parts.append(f"(... {offset} earlier steps omitted ...)")
            for i, s in enumerate(shown, offset + 1):
                parts.append(f"{i}. Observation: {s.observation}")
                parts.append(f"   Action: {s.action}")
            parts.append("")
        parts.append(f"Current observation: {obs}")
        parts.append("Next action:")
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]

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
