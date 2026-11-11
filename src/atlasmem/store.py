"""SQLite-backed memory bank of raw trajectories."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .types import Step, Trajectory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    steps TEXT NOT NULL,
    success INTEGER,
    metadata TEXT NOT NULL,
    created_at REAL NOT NULL,
    verdict INTEGER
);
CREATE INDEX IF NOT EXISTS idx_traj_created ON trajectories(created_at);
"""

# Named explicitly: a migrated database and a fresh one must read the same way.
_COLS = "id, task, steps, success, metadata, created_at"

# Rows a verified-only reader may see: everything except a judged failure. A row with no verdict
# was added directly with ``MemoryBank.add`` and stays retrievable, as it always has been.
_NOT_FAILED = "verdict IS NOT 0"


def _verdict_of(traj: Trajectory) -> int | None:
    """The verdict column mirrors ``metadata["verdict"]["success"]``; NULL when never judged."""
    v = traj.metadata.get("verdict")
    if isinstance(v, dict) and v.get("success") is not None:
        return int(bool(v["success"]))
    return None


class MemoryBank:
    """Append-only store of raw trajectories. Use ":memory:" for an in-process bank.

    Two outcome columns sit beside each trajectory: ``success`` is the environment's signal and
    ``verdict`` is the write gate's decision (1 = judged success, 0 = judged failure, NULL = never
    judged). Failures are kept so they can be read later; retrieval leaves them out by default.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._version = 0  # bumped on every write so retrievers can refresh
        self._added_at: dict[str, int] = {}  # id -> version at which this connection added it

    def _migrate(self) -> None:
        """Add the ``verdict`` column to a database created before it existed."""
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(trajectories)")]
        if "verdict" in cols:
            return
        with self._lock:
            self._conn.execute("ALTER TABLE trajectories ADD COLUMN verdict INTEGER")
            # Before this column, the write gate stored successes only and stamped each one with
            # ``judge_reason``. Those rows were judged successful; anything else was never judged.
            rows = self._conn.execute("SELECT id, metadata FROM trajectories").fetchall()
            judged = [(tid,) for tid, meta in rows if "judge_reason" in json.loads(meta)]
            self._conn.executemany("UPDATE trajectories SET verdict = 1 WHERE id = ?", judged)
            self._conn.commit()

    @property
    def version(self) -> int:
        return self._version

    def add(self, traj: Trajectory) -> str:
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO trajectories ({_COLS}, verdict) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    traj.id,
                    traj.task,
                    json.dumps([s.__dict__ for s in traj.steps]),
                    None if traj.success is None else int(traj.success),
                    json.dumps(traj.metadata),
                    traj.created_at,
                    _verdict_of(traj),
                ),
            )
            self._conn.commit()
            self._version += 1
            self._added_at[traj.id] = self._version
        return traj.id

    def get(self, traj_id: str) -> Trajectory | None:
        row = self._conn.execute(
            f"SELECT {_COLS} FROM trajectories WHERE id = ?", (traj_id,)
        ).fetchone()
        return _row_to_traj(row) if row else None

    def all(self) -> list[Trajectory]:
        rows = self._conn.execute(f"SELECT {_COLS} FROM trajectories ORDER BY created_at").fetchall()
        return [_row_to_traj(r) for r in rows]

    def since(self, version: int | None = None, *, time: float | None = None) -> list[Trajectory]:
        """Trajectories added after a point, judged failures included.

        ``since(v)`` returns what this connection added after ``bank.version`` was ``v``, in the
        order added. The version counter starts at 0 each time the database is opened, so use it
        within one process. ``since(time=t)`` returns trajectories with ``created_at > t`` and
        works across processes.
        """
        if (version is None) == (time is None):
            raise ValueError("Pass exactly one of version or time.")
        if time is not None:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM trajectories WHERE created_at > ? ORDER BY created_at",
                (time,),
            ).fetchall()
            return [_row_to_traj(r) for r in rows]
        with self._lock:
            ids = [i for i, v in sorted(self._added_at.items(), key=lambda kv: kv[1]) if v > version]
        return [t for t in map(self.get, ids) if t is not None]

    def tasks(self, verified_only: bool = False) -> list[tuple[str, str]]:
        """(id, task) pairs, cheap enough to index without loading steps.

        ``verified_only`` leaves out trajectories the write gate judged a failure.
        """
        where = f"WHERE {_NOT_FAILED} " if verified_only else ""
        return self._conn.execute(
            f"SELECT id, task FROM trajectories {where}ORDER BY created_at"
        ).fetchall()

    def delete(self, traj_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM trajectories WHERE id = ?", (traj_id,))
            self._conn.commit()
            self._version += 1
            self._added_at.pop(traj_id, None)
        return cur.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM trajectories")
            self._conn.commit()
            self._version += 1
            self._added_at.clear()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _row_to_traj(row: tuple) -> Trajectory:
    tid, task, steps, success, metadata, created_at = row
    return Trajectory(
        id=tid,
        task=task,
        steps=[Step(**s) for s in json.loads(steps)],
        success=None if success is None else bool(success),
        metadata=json.loads(metadata),
        created_at=created_at,
    )
