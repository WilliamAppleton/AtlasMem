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
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traj_created ON trajectories(created_at);
"""


class MemoryBank:
    """Append-only store of raw trajectories. Use ":memory:" for an in-process bank."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._version = 0  # bumped on every write so retrievers can refresh

    @property
    def version(self) -> int:
        return self._version

    def add(self, traj: Trajectory) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO trajectories VALUES (?, ?, ?, ?, ?, ?)",
                (
                    traj.id,
                    traj.task,
                    json.dumps([s.__dict__ for s in traj.steps]),
                    None if traj.success is None else int(traj.success),
                    json.dumps(traj.metadata),
                    traj.created_at,
                ),
            )
            self._conn.commit()
            self._version += 1
        return traj.id

    def get(self, traj_id: str) -> Trajectory | None:
        row = self._conn.execute("SELECT * FROM trajectories WHERE id = ?", (traj_id,)).fetchone()
        return _row_to_traj(row) if row else None

    def all(self) -> list[Trajectory]:
        rows = self._conn.execute("SELECT * FROM trajectories ORDER BY created_at").fetchall()
        return [_row_to_traj(r) for r in rows]

    def tasks(self) -> list[tuple[str, str]]:
        """(id, task) pairs, cheap enough to index without loading steps."""
        return self._conn.execute("SELECT id, task FROM trajectories ORDER BY created_at").fetchall()

    def delete(self, traj_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM trajectories WHERE id = ?", (traj_id,))
            self._conn.commit()
            self._version += 1
        return cur.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM trajectories")
            self._conn.commit()
            self._version += 1

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
