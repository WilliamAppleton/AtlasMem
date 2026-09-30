import re

from atlasmem import Curator, Executor, JitMemory, MemoryBank, run_sequence
from atlasmem.agent import MEMORY_HEADER, parse_action
from atlasmem.envs import KitchenEnv, sample_tasks
from atlasmem.envs.kitchen import TASK_RE

from .conftest import solution


def goal(task):
    return TASK_RE.search(task).groups()


def play(env, task, actions):
    env.reset(task)
    out = None
    for a in actions:
        out = env.step(a)
        if out[1]:
            break
    return out


def test_scripted_solutions_solve_every_sampled_task():
    for task in sample_tasks(40, seed=3):
        obs, done, ok = play(KitchenEnv(max_steps=30), task, solution(goal(task)))
        assert ok and done, task


def test_closed_receptacle_blocks_take():
    env = KitchenEnv()
    env.reset("put a mug in shelf 1")
    env.location = "fridge 1"
    env.contents["fridge 1"].append("mug")
    assert env.step("take mug from fridge 1")[0] == "Nothing happens."
    env.step("open fridge 1")
    assert "pick up" in env.step("take mug from fridge 1")[0]


def test_state_requirement():
    task = "put a clean egg in shelf 1"
    acts = [a for a in solution(goal(task)) if not a.startswith(("clean", "go to sinkbasin"))]
    obs, done, ok = play(KitchenEnv(), task, acts)
    assert not ok


def test_sample_tasks_deterministic_and_parseable():
    a, b = sample_tasks(20, seed=1), sample_tasks(20, seed=1)
    assert a == b and all(TASK_RE.search(t) for t in a)


def test_parse_action():
    assert parse_action("Action: go to fridge 1.\nbecause...") == "go to fridge 1"
    assert parse_action("\n`open drawer 1`") == "open drawer 1"
    assert parse_action("> take mug from shelf 1") == "take mug from shelf 1"
    assert parse_action("") == ""


def scripted_executor_llm(fake_llm, layout_seed=0):
    def fn(messages):
        user0 = next(m["content"] for m in messages if m["role"] == "user")
        task = re.search(r"Your task: (.+)", user0).group(1)
        n_done = sum(1 for m in messages if m["role"] == "assistant")
        acts = solution(goal(task), layout_seed)
        return acts[n_done] if n_done < len(acts) else "look"
    return fake_llm(fn=fn)


def test_executor_solves_and_prepends_payload(fake_llm):
    llm = scripted_executor_llm(fake_llm)
    traj = Executor(llm).run(KitchenEnv(), "put a hot apple in cabinet 1", payload="Apples are in X.")
    assert traj.success
    assert llm.calls[0][0]["content"].startswith(MEMORY_HEADER + "Apples are in X.")
    assert traj.steps[-1].action == "[end]"


def test_run_sequence_batches_share_bank_state(fake_llm):
    bank = MemoryBank()
    retrieved_counts = []

    def curate(messages):
        retrieved_counts.append(messages[1]["content"].count("Memory "))
        return "briefing"

    mem = JitMemory(bank, Curator(fake_llm(fn=curate)), k=5)
    tasks = ["put a mug in shelf 1", "put a mug in drawer 1", "put a mug in cabinet 1",
             "put a mug in countertop 1"]
    results = run_sequence(tasks, KitchenEnv, Executor(scripted_executor_llm(fake_llm)), mem,
                           batch_size=2, use_ground_truth=True)
    assert all(r.success and r.stored for r in results)
    # batch 1 sees an empty bank (no curator call); batch 2 sees the two stored from batch 1
    assert retrieved_counts == [2, 2]
    assert results[0].briefing.payload == "" and results[2].briefing.payload == "briefing"
    assert len(bank) == 4


def test_run_sequence_without_memory(fake_llm):
    res = run_sequence(["put an egg in shelf 1"], KitchenEnv,
                       Executor(scripted_executor_llm(fake_llm)), None)
    assert res[0].success and not res[0].stored and res[0].briefing is None


def test_parse_action_normalizes_typography():
    assert parse_action("go to cabinet 2") == "go to cabinet 2"
