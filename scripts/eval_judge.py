"""Measure trajectory judges against ground truth on kitchen trajectories.

Cases: per sampled task, one correct trajectory and several near-miss failures (skipped state
step, wrong destination, look-alike object, success claimed only in an action text), each
labelled by the environment itself; plus real executor trajectories from
scripts/data/real_cases.jsonl (includes a mug-for-cup false accept seen in a live run).

Judges (repeat --judge to compare):
    llm:gpt-oss:20b                       LLM judge via Ollama chat (+ evidence check)
    s1:nimble                             System One model on local Ollama (>= 0.35)
    s1:tev1:4b@http://spark1:11434        ... on another host
    s1:<model>@https://api.typesafe.ai    TypeSafe Jev cloud ($TYPESAFE_API_KEY)

    python scripts/eval_judge.py --tasks 10 --noisy --judge llm:gpt-oss:20b --judge s1:nimble
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import time
from pathlib import Path

from atlasmem import LLMJudge, OllamaLLM, Step, SystemOneClient, SystemOneJudge, Trajectory
from atlasmem.envs import KitchenEnv, oracle_actions, sample_tasks
from atlasmem.envs.kitchen import OBJECTS, RECEPTACLES, TARGETS, TASK_RE

SIMILAR = {"mug": "cup", "cup": "mug", "spoon": "knife", "knife": "spoon", "plate": "bowl",
           "bowl": "plate", "apple": "tomato", "tomato": "apple", "egg": "potato", "potato": "egg"}
REAL_CASES = Path(__file__).parent / "data" / "real_cases.jsonl"


def rollout(task: str, actions: list[str]) -> Trajectory:
    env = KitchenEnv(max_steps=80)
    obs = env.reset(task)
    steps, success = [], False
    for a in actions:
        steps.append(Step(obs, a))
        obs, done, success = env.step(a)
        if done:
            break
    steps.append(Step(obs, "[end]"))
    return Trajectory(task=task, steps=steps, success=success)


def add_noise(actions: list[str], rng: random.Random, n: int) -> list[str]:
    """Insert realistic dead ends (looks, detours, invalid commands) that do not change the outcome
    when placed before the first action; later insertions are detours that return to the plan."""
    junk = ["look", "inventory", "examine room", "take it"] + [f"go to {r}" for r in RECEPTACLES]
    out = list(actions)
    for _ in range(n):
        i = rng.randrange(0, max(len(out) - 1, 1))
        a = rng.choice(junk)
        # A detour must return to where the plan expects the agent to be.
        prev_go = next((x for x in reversed(out[:i]) if x.startswith("go to ")), None)
        out[i:i] = [a, prev_go] if a.startswith("go to ") and prev_go else [a]
    return out


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
    other_obj = SIMILAR.get(obj) or rng.choice([o for o in OBJECTS if o != obj])
    desc = f"{state} {other_obj}" if state else other_obj
    out["similar_object"] = oracle_actions(f"put a {desc} in {target}")
    out["claimed"] = good[:-1] + [f"look. You put the {obj} in/on the {target}. Task complete."]
    return out


def build_cases(n_tasks: int, seed: int, noisy: bool, real: bool) -> list[tuple[str, Trajectory]]:
    rng = random.Random(seed)
    cases = []
    for task in sample_tasks(n_tasks, seed=seed):
        for kind, acts in variants(task, rng).items():
            if noisy:
                acts = add_noise(acts, rng, rng.randint(4, 10))
            t = rollout(task, acts)
            cases.append((kind, t))
    if real and REAL_CASES.exists():
        for line in REAL_CASES.read_text().splitlines():
            t = Trajectory.from_dict(json.loads(line))
            cases.append(("real_" + ("success" if t.success else "failure"), t))
    return cases


def make_judge(spec: str):
    kind, _, rest = spec.partition(":")
    model, _, url = rest.partition("@")
    if kind == "llm":
        return LLMJudge(OllamaLLM(model, host=url or None, temperature=0))
    if kind == "s1":
        return SystemOneJudge(SystemOneClient(model, base_url=url or None))
    raise SystemExit(f"unknown judge spec {spec!r} (use llm:<model> or s1:<model>[@url])")


def evaluate(spec: str, cases, workers: int) -> None:
    judge = make_judge(spec)

    def run(case):
        t0 = time.perf_counter()
        try:
            v = judge.judge(case[1])
        except Exception as e:  # report and count as an error, don't abort the whole eval
            return None, time.perf_counter() - t0, str(e)
        return v, time.perf_counter() - t0, None

    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run, cases))

    errors = [e for _, _, e in results if e]
    ok_results = [(c, v, dt) for c, (v, dt, e) in zip(cases, results) if not e]
    tp = fp = tn = fn = 0
    by_kind: dict[str, list] = {}
    for (kind, t), v, _ in ok_results:
        truth = bool(t.success)
        tp += v.success and truth
        fp += v.success and not truth
        tn += (not v.success) and not truth
        fn += (not v.success) and truth
        by_kind.setdefault(kind, []).append((v.success == truth, getattr(v, "probability", None)))
    n = len(ok_results)
    lat = [dt for _, _, dt in ok_results]
    print(f"\n[{spec}]  n={n}  accuracy {(tp + tn) / max(n, 1):.1%}  "
          f"false accept {fp}/{fp + tn}  false reject {fn}/{fn + tp}  "
          f"latency median {statistics.median(lat) if lat else 0:.2f}s")
    if errors:
        print(f"    {len(errors)} errors, e.g. {errors[0][:200]}")
    for kind, rows in by_kind.items():
        probs = [p for _, p in rows if p is not None]
        extra = f"  mean P(success) {statistics.mean(probs):.2f}" if probs else ""
        print(f"    {kind:15s} {sum(ok for ok, _ in rows)}/{len(rows)} correct{extra}")
    # For probabilistic judges: is it miscalibrated (fixable by threshold) or just weak?
    scored = [(v.probability, bool(t.success)) for (_, t), v, _ in ok_results if hasattr(v, "probability")]
    if scored:
        best = max(((sum((p >= th) == y for p, y in scored) / len(scored)), th)
                   for th in [i / 100 for i in range(1, 100)])
        print(f"    best threshold {best[1]:.2f} -> accuracy {best[0]:.1%} (default 0.50)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", action="append", help="judge spec; repeatable (default llm:gpt-oss:20b)")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--noisy", action="store_true", help="pad trajectories with realistic dead ends")
    ap.add_argument("--no-real", action="store_true", help="skip scripts/data/real_cases.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cases = build_cases(args.tasks, args.seed, args.noisy, not args.no_real)
    n_pos = sum(bool(t.success) for _, t in cases)
    print(f"{len(cases)} cases ({n_pos} successes, {len(cases) - n_pos} failures)"
          f"{', noisy' if args.noisy else ''}")
    for spec in args.judge or ["llm:gpt-oss:20b"]:
        evaluate(spec, cases, args.workers)


if __name__ == "__main__":
    main()
