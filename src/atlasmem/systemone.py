"""System One decision models (Jev-style): typed answers with probabilities instead of text.

One client serves both backends, which share the request/response schema:
  - local Ollama >= 0.35:  SystemOneClient("nimble", base_url="http://localhost:11434")
  - TypeSafe Jev cloud:    SystemOneClient(<model>, base_url="https://api.typesafe.ai",
                                           api_key=...)   # or $TYPESAFE_API_KEY
POST {base_url}/v1/systemone  {"model", "state", "questions": {name: question}}
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .judge import Verdict
from .llm import _post_json
from .types import Trajectory

TYPESAFE_URL = "https://api.typesafe.ai"
MAX_REQUEST_BYTES = 64 * 1024  # documented cap for Ollama's endpoint


def noul(instructions: str, false: str = "No", true: str = "Yes") -> dict[str, Any]:
    """Yes/no question; the answer is P(true)."""
    return {"type": "noul", "instructions": instructions, "criteria": {"false": false, "true": true}}


def choice(instructions: str, options: dict[str, str]) -> dict[str, Any]:
    """Pick one of 2-26 labelled options; answer carries a probability per option."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    """Ordered rubric of 2-26 levels; answer carries an expected level and per-level probabilities."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class SystemOneClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        retries: int = 2,
    ):
        self.model = model
        base = base_url or os.environ.get("SYSTEMONE_URL") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.base_url = (base if base.startswith("http") else "http://" + base).rstrip("/")
        if api_key is None:
            # Unset is fine when an egress proxy injects the credential for this host.
            names = ("TYPESAFE_API_KEY", "JEV_API_KEY", "jev") if self.base_url.startswith(TYPESAFE_URL) \
                else ("SYSTEMONE_API_KEY",)
            api_key = next((os.environ[n] for n in names if os.environ.get(n)), None)
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.last_latency: float = 0.0

    def ask(self, state: str | dict | list, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        body = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        resp = _post_json(f"{self.base_url}/v1/systemone", body, headers, self.timeout, self.retries)
        self.last_latency = time.perf_counter() - t0
        return resp["answers"]

    def __repr__(self) -> str:
        return f"SystemOneClient({self.model!r}, {self.base_url!r})"


SUCCESS_QUESTION = noul(
    "Did the agent fully accomplish the task? Judge ONLY from the environment's observations; the "
    "agent's actions are attempts or claims and prove nothing, and 'Nothing happens.' means an "
    "action failed. Every requirement must be confirmed: the exact object named in the task (a "
    "similar or related item does not count), any required state or processing, and the exact "
    "destination.",
    false="Not accomplished, or not confirmed by the observations",
    true="The observations confirm every requirement of the task was met",
)

PROGRESS_QUESTION = score(
    "How far did the agent get toward accomplishing the task, judged only from the observations?",
    ["No meaningful progress", "Found or obtained what the task needs",
     "Most requirements met", "Task fully accomplished"],
)


@dataclass
class SystemOneVerdict(Verdict):
    probability: float = 0.0
    progress: float | None = None  # 0..1, from the rubric score
    latency: float = 0.0
    answers: dict[str, Any] = field(default_factory=dict)


class SystemOneJudge:
    """Judge backed by a decision model. success = P(success) >= threshold."""

    def __init__(self, client: SystemOneClient, threshold: float = 0.5, max_chars: int = 16000,
                 with_progress: bool = True):
        self.client = client
        self.threshold = threshold
        self.max_chars = max_chars
        self.with_progress = with_progress

    def judge(self, traj: Trajectory) -> SystemOneVerdict:
        questions = {"success": SUCCESS_QUESTION}
        if self.with_progress:
            questions["progress"] = PROGRESS_QUESTION
        answers = self.client.ask(traj.render(max_chars=self.max_chars), questions)
        p = float(answers["success"]["noul"])
        progress = None
        if "progress" in answers:
            levels = len(PROGRESS_QUESTION["criteria"]) - 1
            progress = float(answers["progress"]["score"]) / levels
        return SystemOneVerdict(
            success=p >= self.threshold,
            reason=f"P(success)={p:.3f} via {self.client.model}",
            probability=p, progress=progress, latency=self.client.last_latency, answers=answers,
        )


class CascadeJudge:
    """Fast decision model first; only the uncertain band [low, high) goes to the slow judge."""

    def __init__(self, fast: SystemOneJudge, slow: Any, low: float = 0.1, high: float = 0.9):
        self.fast, self.slow, self.low, self.high = fast, slow, low, high
        self.escalations = 0

    def judge(self, traj: Trajectory) -> Verdict:
        v = self.fast.judge(traj)
        if v.probability >= self.high or v.probability < self.low:
            return v
        self.escalations += 1
        slow = self.slow.judge(traj)
        slow.reason = f"escalated (P={v.probability:.2f}); {slow.reason}"
        return slow
