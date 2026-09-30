"""Measure the LLM judge against ground truth on scripted kitchen trajectories.

Generates, per task, one correct trajectory and several near-miss failures, labels each with
the environment's own success signal, and reports the judge's confusion matrix with and
without the evidence-verification check.

    python scripts/eval_judge.py --tasks 8 --model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import random

from atlasmem import LLMJudge, OllamaLLM, Step, Trajectory
from atlasmem.envs import KitchenEnv, oracle_actions, sample_tasks
from atlasmem.envs.kitchen import OBJECTS, TARGETS, TASK_RE
from atlasmem.judge import parse_verdict, verify_evidence


SIMILAR = {"mug": "cup", "cup": "mug", "spoon": "knife", "knife": "spoon", "plate": "bowl",
           "bowl": "plate", "apple": "tomato", "tomato": "apple", "egg": "potato", "potato": "egg"}


def rollout(task: str, actions: list[str]) -> Trajectory:
    env = KitchenEnv(max_steps=50)
    obs = env.reset(task)
    steps, success = [], False
    for a in actions:
        steps.append(Step(obs, a))
        obs, done, success = env.step(a)
        if done:
            break
    steps.append(Step(obs, "[end]"))
    return Trajectory(task=task, steps=steps, success=success)


def variants(task: str, rng: random.Random) -> dict[str, list[str]]:
    state, obj, target = TASK_RE.search(task).groups()
    good = oracle_actions(task)
    out = {"correct": good}
    if state:
        out["skip_state"] = [a for a in good if not a.startswith(("clean ", "heat ", "cool "))]
    other = rng.choice([t for t in TARGETS if t != target])
    wrong = [a for a in good if not a.startswith(("go to " + target, "open " + target, "put "))]
    wrong += [f"go to {other}"] + ([f"open {other}"] if other.split()[0] in ("cabinet", "drawer") else [])
    out["wrong_target"] = wrong + [f"put {obj} in {other}"]
    # Look-alike substitutions are the confusion seen in real runs (mug placed for a cup task).
    other_obj = SIMILAR.get(obj) or rng.choice([o for o in OBJECTS if o != obj])
    desc = f"{state} {other_obj}" if state else other_obj
    out["similar_object"] = oracle_actions(f"put a {desc} in {target}")
    out["claimed"] = good[:-1] + [f"look. You put the {obj} in/on the {target}. Task complete."]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases = []
    for task in sample_tasks(args.tasks, seed=args.seed):
        for kind, acts in variants(task, rng).items():
            cases.append((kind, rollout(task, acts)))

    judge = LLMJudge(OllamaLLM(args.model, temperature=0), verify=False)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        raws = list(ex.map(lambda c: judge.judge(c[1]).raw, cases))

    for label, use_verify in (("prompt only", False), ("prompt + evidence check", True)):
        tp = fp = tn = fn = 0
        by_kind: dict[str, list[int]] = {}
        for (kind, t), raw in zip(cases, raws):
            v = parse_verdict(raw)
            if use_verify:
                v = verify_evidence(v, t)
            ok = v.success == bool(t.success)
            tp += v.success and t.success
            fp += v.success and not t.success
            tn += (not v.success) and not t.success
            fn += (not v.success) and bool(t.success)
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += ok
            by_kind[kind][1] += 1
        print(f"\n[{label}] accuracy {(tp + tn) / len(cases):.0%}  "
              f"false accept {fp}/{fp + tn}  false reject {fn}/{fn + tp}")
        for kind, (ok, n) in by_kind.items():
            print(f"    {kind:13s} {ok}/{n} correct")


if __name__ == "__main__":
    main()
