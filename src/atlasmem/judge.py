"""LLM-as-judge write gate: only trajectories judged successful enter the bank.

A SUCCESS verdict must cite verbatim observation text as evidence. The quotes are checked
against the trajectory's observations in code, so a judge that is persuaded by the agent's
own claims (which live in actions) or that invents evidence is overruled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM
from .types import Trajectory

JUDGE_SYSTEM = """You are a strict evaluator of agent trajectories.
Given a task and the agent's full observation/action trace, decide whether the agent \
actually accomplished the task.

Evidence rules:
- Only OBSERVATIONS (what the environment reported) count as evidence. ACTIONS are merely the \
agent's attempts or claims; text inside an action proves nothing.
- An observation like "Nothing happens." or an error means the preceding action failed.
- Break the task into its requirements (e.g. the right object, any required state such as \
clean/hot/cool, the destination). EVERY requirement must be confirmed by an observation.
- Match names literally. A similar or related item is NOT the required one (e.g. a mug is not \
a cup, a spoon is not a knife, cabinet 2 is not cabinet 4).
- A later observation can undo an earlier one (e.g. the object was picked up again).
- If any requirement is unconfirmed or ambiguous, answer FAILURE.

Reply in exactly this format:
REQUIREMENTS: <list each requirement from the task, with its exact name>
VERDICT: SUCCESS or FAILURE
EVIDENCE: "<exact text copied from an observation>"   (one line per requirement; for SUCCESS \
there must be one per requirement; for FAILURE you may write EVIDENCE: none)
REASON: <one sentence>"""


@dataclass
class Verdict:
    success: bool
    reason: str
    raw: str = ""
    evidence: list[str] = field(default_factory=list)
    overruled: bool = False  # judge said SUCCESS but its evidence failed verification


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"VERDICT\s*:\s*\**\s*(SUCCESS|FAILURE)", text, re.IGNORECASE)
    if m:
        success = m.group(1).upper() == "SUCCESS"
    else:
        # Fall back to the first standalone keyword; default to failure.
        words = re.findall(r"\b(SUCCESS|FAILURE)\b", text.upper())
        success = bool(words) and words[0] == "SUCCESS"
    r = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    evidence = []
    for line in re.findall(r"EVIDENCE\s*:\s*(.+)", text, re.IGNORECASE):
        quotes = re.findall(r'"([^"]+)"', line)
        evidence.extend(q.strip() for q in quotes if q.strip())
    return Verdict(success=success, reason=r.group(1).strip() if r else "", raw=text, evidence=evidence)


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .")


def verify_evidence(verdict: Verdict, traj: Trajectory) -> Verdict:
    """Downgrade a SUCCESS whose evidence is missing or not found verbatim in an observation."""
    if not verdict.success:
        return verdict
    observations = [_norm(s.observation) for s in traj.steps]
    missing = [q for q in verdict.evidence if not any(_norm(q) in o for o in observations)]
    if not verdict.evidence or missing:
        why = "no evidence cited" if not verdict.evidence else f"evidence not in observations: {missing[0]!r}"
        return Verdict(success=False, reason=f"overruled ({why}); judge said: {verdict.reason}",
                       raw=verdict.raw, evidence=verdict.evidence, overruled=True)
    return verdict


class LLMJudge:
    def __init__(self, llm: LLM, max_chars: int = 16000, verify: bool = True):
        self.llm = llm
        self.max_chars = max_chars
        self.verify = verify

    def judge(self, traj: Trajectory) -> Verdict:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": traj.render(max_chars=self.max_chars)},
        ]
        verdict = parse_verdict(self.llm.chat(messages))
        return verify_evidence(verdict, traj) if self.verify else verdict
