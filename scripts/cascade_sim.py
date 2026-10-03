"""Tune CascadeJudge thresholds: run judges once per case, then simulate every (low, high) band.

Uses the same cases as eval_judge.py. Each fast judge (a probabilistic s1: judge) and the slow
judge run once over every case; the cascade is then simulated offline from the recorded
P(success), verdicts and latencies, so a whole grid of thresholds costs one eval.

    python scripts/cascade_sim.py --tasks 10 --noisy \\
        --fast s1:clef@http://spark1:11434 --slow llm:gpt-oss:20b@http://spark1:11434 \\
        --save runs.json
    python scripts/cascade_sim.py --load runs.json          # re-simulate without calling models

Pick thresholds on one seed, then confirm them on another (--seed 12 --no-real) before
changing CascadeJudge's defaults. A case escalates when low <= P < high, as in CascadeJudge.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
from pathlib import Path

from eval_judge import build_cases, make_judge

LOWS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
HIGHS = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99]


def collect(cases, specs: list[str], workers: int) -> list[dict]:
    rows = [{"kind": k, "truth": bool(t.success)} for k, t in cases]
    for spec in specs:
        judge = make_judge(spec)

        def run(case):
            t0 = time.perf_counter()
            try:
                v = judge.judge(case[1])
            except Exception as e:  # recorded per case; the simulation skips it
                return {"err": str(e)[:200], "dt": time.perf_counter() - t0}
            return {"ok": bool(v.success), "p": getattr(v, "probability", None),
                    "dt": time.perf_counter() - t0}

        # One judge at a time so latencies are not inflated by the other judges' load.
        with cf.ThreadPoolExecutor(workers) as ex:
            for row, r in zip(rows, ex.map(run, cases)):
                row[spec] = r
        print(f"ran {spec}: {sum('err' in r[spec] for r in rows)} errors", flush=True)
    return rows


def alone(rows: list[dict], spec: str) -> None:
    ok = [r for r in rows if "err" not in r[spec]]
    neg = sum(not r["truth"] for r in ok)
    fa = sum(r[spec]["ok"] and not r["truth"] for r in ok)
    fr = sum(not r[spec]["ok"] and r["truth"] for r in ok)
    lat = [r[spec]["dt"] for r in ok]
    print(f"alone  {spec}\n       accuracy {(len(ok) - fa - fr) / len(ok):.1%}  false accept {fa}/{neg}  "
          f"false reject {fr}/{len(ok) - neg}  latency median {statistics.median(lat):.2f}s "
          f"mean {statistics.mean(lat):.2f}s  errors {len(rows) - len(ok)}")


def simulate(rows: list[dict], fast: str, slow: str, low: float, high: float) -> dict:
    neg = pos = fa = fr = esc = 0
    lat, missed = [], {}
    for r in rows:
        f, s = r[fast], r[slow]
        if "err" in f or "err" in s:
            continue
        t = f["dt"]
        if f["p"] >= high:
            verdict = True
        elif f["p"] < low:
            verdict = False
        else:
            esc += 1
            verdict, t = s["ok"], t + s["dt"]
        lat.append(t)
        neg += not r["truth"]
        pos += r["truth"]
        fa += verdict and not r["truth"]
        fr += (not verdict) and r["truth"]
        if verdict != r["truth"]:
            missed[r["kind"]] = missed.get(r["kind"], 0) + 1
    n = len(lat)
    return {"low": low, "high": high, "acc": (n - fa - fr) / n, "fa": fa, "neg": neg, "fr": fr,
            "pos": pos, "esc": esc, "n": n, "mean": statistics.mean(lat), "missed": missed}


def report(rows: list[dict], fast: str, slow: str, top: int) -> None:
    print(f"\n[cascade  fast={fast}  slow={slow}]")
    scored = [(r[fast]["p"], r["truth"], r["kind"]) for r in rows if "err" not in r[fast]]
    fails = sorted(((p, k) for p, y, k in scored if not y), reverse=True)[:3]
    succs = sorted((p, k) for p, y, k in scored if y)[:3]
    print("    highest-scoring failures " + ", ".join(f"{p:.3f} {k}" for p, k in fails))
    print("    lowest-scoring successes " + ", ".join(f"{p:.3f} {k}" for p, k in succs))
    grid = [simulate(rows, fast, slow, lo, hi) for lo in LOWS for hi in HIGHS]
    grid.sort(key=lambda g: (-g["acc"], g["esc"]))
    current = [g for g in grid if (g["low"], g["high"]) == (0.3, 0.95)]
    print("    low   high   accuracy  false acc  false rej  escalated   mean s/case  misses")
    for g in current + [g for g in grid[:top] if g not in current]:
        tag = "  <- CascadeJudge default" if g in current else ""
        print(f"    {g['low']:.2f}  {g['high']:.2f}   {g['acc']:6.1%}    {g['fa']:3d}/{g['neg']:<4d} "
              f"{g['fr']:3d}/{g['pos']:<4d}  {g['esc']:3d} {g['esc'] / g['n']:4.0%}   {g['mean']:6.2f}"
              f"       {g['missed'] or '-'}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="append", help="probabilistic s1: judge spec; repeatable")
    ap.add_argument("--slow", default="llm:gpt-oss:20b", help="judge for the uncertain band")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--noisy", action="store_true", help="pad trajectories with realistic dead ends")
    ap.add_argument("--no-real", action="store_true", help="skip scripts/data/real_cases.jsonl")
    ap.add_argument("--workers", type=int, default=3, help="match the server's parallel slots")
    ap.add_argument("--top", type=int, default=8, help="grid rows to print per fast judge")
    ap.add_argument("--save", type=Path, help="write per-case results as JSON")
    ap.add_argument("--load", type=Path, help="simulate from a --save file instead of running judges")
    args = ap.parse_args()

    if args.load:
        data = json.loads(args.load.read_text())
    else:
        if not args.fast:
            ap.error("--fast is required unless --load is given")
        cases = build_cases(args.tasks, args.seed, args.noisy, not args.no_real)
        print(f"{len(cases)} cases (tasks={args.tasks} seed={args.seed}"
              f"{', noisy' if args.noisy else ''}{', no real' if args.no_real else ''})")
        data = {"fast": args.fast, "slow": args.slow,
                "rows": collect(cases, [args.slow] + args.fast, args.workers)}
        if args.save:
            args.save.write_text(json.dumps(data, indent=1))
    rows = data["rows"]
    print()
    for spec in [data["slow"]] + data["fast"]:
        alone(rows, spec)
    for fast in data["fast"]:
        report(rows, fast, data["slow"], args.top)


if __name__ == "__main__":
    main()
