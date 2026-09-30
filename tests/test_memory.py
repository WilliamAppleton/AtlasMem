import pytest

from atlasmem import Curator, JitMemory, LLMJudge, MemoryBank, Step, Trajectory, build_curator_messages
from atlasmem.judge import parse_verdict


def T(task, success=None):
    return Trajectory(task=task, steps=[Step("You see a mug.", "take mug from shelf 1")], success=success)


@pytest.mark.parametrize("text,ok", [
    ("VERDICT: SUCCESS\nREASON: done", True),
    ("verdict: failure\nreason: nope", False),
    ("**VERDICT:** SUCCESS", True),
    ("I think this is a SUCCESS overall", True),
    ("unclear", False),
])
def test_parse_verdict(text, ok):
    assert parse_verdict(text).success is ok


def test_curator_prompt_contains_task_and_memories():
    msgs = build_curator_messages("put a mug in shelf 1", [T("put a mug in drawer 1"), T("heat a mug")])
    user = msgs[1]["content"]
    assert "Memory Curator" in msgs[0]["content"]
    assert "CURRENT TASK: put a mug in shelf 1" in user
    assert "Memory 1" in user and "Memory 2" in user and "heat a mug" in user


def test_curator_skips_llm_when_nothing_retrieved(fake_llm):
    llm = fake_llm(["should not be used"])
    assert Curator(llm).curate("t", []) == "" and llm.calls == []


def test_brief_retrieves_then_curates(fake_llm):
    bank = MemoryBank()
    bank.add(T("put a mug in drawer 1"))
    bank.add(T("heat an apple"))
    llm = fake_llm(["Mugs are on shelf 1."])
    b = JitMemory(bank, Curator(llm), k=1).brief("put a clean mug in cabinet 1")
    assert b.payload == "Mugs are on shelf 1."
    assert [t.task for t in b.retrieved] == ["put a mug in drawer 1"]
    assert "put a mug in drawer 1" in llm.calls[0][1]["content"]


def test_brief_empty_bank_no_llm_call(fake_llm):
    llm = fake_llm()
    b = JitMemory(MemoryBank(), Curator(llm)).brief("x")
    assert b.payload == "" and b.retrieved == [] and llm.calls == []


def test_record_gating_priority(fake_llm):
    bank = MemoryBank()
    judge_llm = fake_llm(["VERDICT: FAILURE\nREASON: no", "VERDICT: SUCCESS\nREASON: yes"])
    mem = JitMemory(bank, Curator(fake_llm()), LLMJudge(judge_llm))
    assert mem.record(T("a", success=True)).stored is False  # judge overrides env signal
    assert mem.record(T("b")).stored is True
    assert mem.record(T("c"), label=False).stored is False  # label overrides judge
    assert len(judge_llm.calls) == 2
    assert [t.task for t in bank.all()] == ["b"]
    assert bank.all()[0].metadata["judge_reason"] == "yes"


def test_record_without_judge_uses_env_signal_or_raises(fake_llm):
    mem = JitMemory(MemoryBank(), Curator(fake_llm()))
    assert mem.record(T("a", success=True)).stored
    assert not mem.record(T("b", success=False)).stored
    with pytest.raises(ValueError):
        mem.record(T("c"))


def test_record_batch_evaluates_before_appending(fake_llm):
    bank = MemoryBank()
    seen_sizes = []

    def judge(messages):
        seen_sizes.append(len(bank))
        return "VERDICT: SUCCESS"

    mem = JitMemory(bank, Curator(fake_llm()), LLMJudge(fake_llm(fn=judge)))
    res = mem.record_batch([T("a"), T("b"), T("c")])
    assert all(r.stored for r in res) and len(bank) == 3 and seen_sizes == [0, 0, 0]
