"""atlasmem command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import Settings, build_memory
from .types import Trajectory


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    for name in ("db", "curator_model", "judge_model", "executor_model", "k", "host"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(s, name, v)
    return s


def cmd_brief(args: argparse.Namespace) -> int:
    mem = build_memory(_settings(args), use_judge=False)
    b = mem.brief(args.task)
    if args.json:
        print(json.dumps({"task": b.task, "payload": b.payload,
                          "retrieved": [{"id": t.id, "task": t.task, "score": s}
                                        for t, s in zip(b.retrieved, b.scores)]}, indent=2))
        return 0
    if not b.retrieved:
        print("(no relevant memories)")
        return 0
    print(f"Retrieved {len(b.retrieved)}:")
    for t, s in zip(b.retrieved, b.scores):
        print(f"  {s:6.2f}  {t.id[:8]}  {t.task}")
    print("\n--- briefing ---\n" + b.payload)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    mem = build_memory(_settings(args), use_judge=not args.label)
    data = json.load(sys.stdin if args.file == "-" else open(args.file))
    items = data if isinstance(data, list) else [data]
    label = None if not args.label else args.label == "success"
    for item in items:
        res = mem.record(Trajectory.from_dict(item), label=label)
        status = "stored" if res.stored else "rejected"
        print(f"{status}: {item.get('task', '')[:70]}  ({res.verdict.reason})")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    mem = build_memory(_settings(args), use_judge=False)
    trajs = mem.bank.all()
    for t in trajs[-args.limit:]:
        print(f"{t.id[:8]}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(t.created_at))}  "
              f"{len(t.steps):3d} steps  {t.task}")
    print(f"({len(trajs)} trajectories in {mem.bank.path})")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    mem = build_memory(_settings(args), use_judge=False)
    matches = [t for t in mem.bank.all() if t.id.startswith(args.id)]
    if len(matches) != 1:
        print(f"{len(matches)} trajectories match {args.id!r}", file=sys.stderr)
        return 1
    print(matches[0].render())
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    mem = build_memory(_settings(args), use_judge=False)
    if args.all:
        mem.bank.clear()
        print("cleared")
        return 0
    matches = [t for t in mem.bank.all() if t.id.startswith(args.id or "")]
    if len(matches) != 1:
        print(f"{len(matches)} trajectories match {args.id!r}", file=sys.stderr)
        return 1
    mem.bank.delete(matches[0].id)
    print(f"deleted {matches[0].id}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .agent import Executor, run_sequence
    from .envs import KitchenEnv, sample_tasks
    from .llm import OllamaLLM

    s = _settings(args)
    if args.db is None:
        s.db = ":memory:"
    mem = None if args.no_memory else build_memory(s, use_judge=not args.ground_truth)
    executor = Executor(OllamaLLM(s.executor_model, host=s.host))
    tasks = sample_tasks(args.episodes, seed=args.seed, layout_seed=args.layout_seed)
    print(f"executor={s.executor_model} curator={s.curator_model} "
          f"judge={'ground-truth' if args.ground_truth else s.judge_model} "
          f"memory={'off' if mem is None else s.db} k={s.k}\n")
    wins = 0

    def show(r):  # noqa: ANN001
        nonlocal wins
        wins += r.success
        n = len(r.briefing.retrieved) if r.briefing else 0
        print(f"{'OK  ' if r.success else 'FAIL'} steps={r.n_steps:2d} mem={n} "
              f"stored={'y' if r.stored else 'n'}  {r.task}")
        if args.verbose and r.briefing and r.briefing.payload:
            print("    briefing: " + r.briefing.payload.replace("\n", "\n              "))

    results = run_sequence(tasks, lambda: KitchenEnv(args.layout_seed, args.max_steps), executor,
                           mem, batch_size=args.batch_size, use_ground_truth=args.ground_truth,
                           on_episode=show)
    steps = [r.n_steps for r in results if r.success]
    print(f"\nsuccess {wins}/{len(results)} = {wins / max(len(results), 1):.0%}"
          + (f", mean steps on success {sum(steps) / len(steps):.1f}" if steps else ""))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .mcp_server import serve

    serve(_settings(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlasmem", description="Just-in-Time Memory for LLM agents")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite path (default $ATLASMEM_DB or ~/.atlasmem/memory.db)")
    common.add_argument("--host", help="Ollama host (default $OLLAMA_HOST or https://ollama.com)")
    common.add_argument("--curator-model")
    common.add_argument("--judge-model")
    common.add_argument("-k", type=int, dest="k", help="trajectories to retrieve")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", parents=[common], help="retrieve + curate a briefing for a task")
    b.add_argument("task")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_brief)

    r = sub.add_parser("record", parents=[common], help="record trajectory JSON (file or '-')")
    r.add_argument("file")
    r.add_argument("--label", choices=["success", "failure"], help="skip the judge, use this label")
    r.set_defaults(func=cmd_record)

    ls = sub.add_parser("list", parents=[common], help="list stored trajectories")
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", parents=[common], help="print a stored trajectory")
    sh.add_argument("id", help="id or unique prefix")
    sh.set_defaults(func=cmd_show)

    d = sub.add_parser("delete", parents=[common], help="delete a trajectory")
    d.add_argument("id", nargs="?")
    d.add_argument("--all", action="store_true")
    d.set_defaults(func=cmd_delete)

    dm = sub.add_parser("demo", parents=[common], help="run the kitchen env with/without memory")
    dm.add_argument("--executor-model")
    dm.add_argument("--episodes", type=int, default=10)
    dm.add_argument("--batch-size", type=int, default=1)
    dm.add_argument("--max-steps", type=int, default=20)
    dm.add_argument("--seed", type=int, default=0)
    dm.add_argument("--layout-seed", type=int, default=0)
    dm.add_argument("--no-memory", action="store_true")
    dm.add_argument("--ground-truth", action="store_true", help="gate writes on env success, not the judge")
    dm.add_argument("-v", "--verbose", action="store_true")
    dm.set_defaults(func=cmd_demo)

    sv = sub.add_parser("serve", parents=[common], help="run the MCP server (stdio)")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
