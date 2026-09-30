"""MCP server exposing AtlasMem to agents (requires the ``mcp`` extra)."""

from __future__ import annotations

from typing import Any

from .config import Settings, build_memory
from .types import Step, Trajectory


def create_server(settings: Settings | None = None):  # noqa: ANN201
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
        except ImportError as e:  # pragma: no cover
            raise SystemExit("MCP support needs: pip install 'atlasmem[mcp]'") from e

    mem = build_memory(settings)
    server = Server("atlasmem")

    @server.tool()
    def get_briefing(task: str, k: int = 3) -> dict[str, Any]:
        """Retrieve past experience relevant to `task` and return a task-specific briefing.

        Call this before starting a task. Returns an empty briefing if nothing relevant is stored.
        """
        b = mem.brief(task, k=k)
        return {"briefing": b.payload,
                "sources": [{"id": t.id, "task": t.task, "score": round(s, 3)}
                            for t, s in zip(b.retrieved, b.scores)]}

    @server.tool()
    def record_trajectory(task: str, steps: list[dict[str, str]], success: bool | None = None) -> dict[str, Any]:
        """Record a finished attempt. `steps` is a list of {"observation": ..., "action": ...}.

        If `success` is omitted, an LLM judge decides; only successful attempts are stored.
        """
        traj = Trajectory(task=task, steps=[Step(s.get("observation", ""), s.get("action", "")) for s in steps])
        res = mem.record(traj, label=success)
        return {"stored": res.stored, "id": traj.id if res.stored else None, "reason": res.verdict.reason}

    @server.tool()
    def memory_stats() -> dict[str, Any]:
        """Number of stored trajectories and the database path."""
        return {"trajectories": len(mem.bank), "db": mem.bank.path}

    return server


def serve(settings: Settings | None = None) -> None:
    create_server(settings).run()
