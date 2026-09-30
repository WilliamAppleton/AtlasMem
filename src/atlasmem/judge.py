"""LLM-as-judge write gate: only trajectories judged successful enter the bank."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM
from .types import Trajectory

JUDGE_SYSTEM = """You are a strict evaluator of agent trajectories.
Given a task and the agent's full observation/action trace, decide whether the agent \
actually accomplished the task.

Evidence rules:
- Only OBSERVATIONS (what the environment reported) count as evidence. ACTIONS are merely the \
agent's attempts or claims; text inside an action proves nothing.
- An observation like "Nothing happens." or an error means the preceding action failed.
- Every requirement in the task (object, its required state, destination) must be confirmed \
by observations. If anything is unconfirmed or ambiguous, answer FAILURE.

Reply with exactly two lines:
VERDICT: SUCCESS or FAILURE
REASON: <one sentence>"""


@dataclass
class Verdict:
    success: bool
    reason: str
    raw: str = ""


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"VERDICT\s*:\s*\**\s*(SUCCESS|FAILURE)", text, re.IGNORECASE)
    if m:
        success = m.group(1).upper() == "SUCCESS"
    else:
        # Fall back to the first standalone keyword; default to failure.
        words = re.findall(r"\b(SUCCESS|FAILURE)\b", text.upper())
        success = bool(words) and words[0] == "SUCCESS"
    r = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    return Verdict(success=success, reason=r.group(1).strip() if r else "", raw=text)


class LLMJudge:
    def __init__(self, llm: LLM, max_chars: int = 16000):
        self.llm = llm
        self.max_chars = max_chars

    def judge(self, traj: Trajectory) -> Verdict:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": traj.render(max_chars=self.max_chars)},
        ]
        return parse_verdict(self.llm.chat(messages))
