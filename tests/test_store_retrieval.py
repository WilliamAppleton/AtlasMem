import json
import sqlite3

import pytest

from atlasmem import BM25, BM25Retriever, MemoryBank, Step, Trajectory, store
from atlasmem.cli import main
from atlasmem.retrieval import tokenize


def T(task, **kw):
    return Trajectory(task=task, steps=[Step("obs", "act")], **kw)


def test_store_roundtrip_and_persistence(tmp_path):
    db = tmp_path / "m.db"
    bank = MemoryBank(db)
    t = T("put a clean mug in shelf 1", success=True, metadata={"a": 1})
    bank.add(t)
    assert len(bank) == 1
    got = bank.get(t.id)
    assert got.task == t.task and got.steps == t.steps and got.success is True and got.metadata == {"a": 1}
    bank.close()
    bank2 = MemoryBank(db)
    assert [x.id for x in bank2.all()] == [t.id]
    assert bank2.delete(t.id) and len(bank2) == 0
    assert not bank2.delete("nope")


def test_version_bumps_on_write():
    bank = MemoryBank()
    v0 = bank.version
    bank.add(T("x"))
    assert bank.version > v0


def test_tokenize_drops_stopwords():
    assert tokenize("Put a CLEAN mug in the shelf 1") == ["put", "clean", "mug", "shelf", "1"]


def test_bm25_ranks_lexical_match_first():
    idx = BM25(["put a mug in shelf", "heat an apple", "clean the knife"])
    s = idx.scores("put the mug somewhere")
    assert s[0] > s[1] and s[0] > s[2]


def test_retriever_topk_and_refresh():
    bank = MemoryBank()
    r = BM25Retriever(bank, k=2)
    assert r.retrieve("anything") == []
    for task in ["put a mug in shelf 1", "put a hot mug in cabinet 1", "heat an apple", "cool a tomato"]:
        bank.add(T(task))
    hits = r.retrieve("put a clean mug in drawer 1")
    assert len(hits) == 2 and all("mug" in t.task for t, _ in hits)
    bank.add(T("clean a mug and put it in drawer 1"))
    hits = r.retrieve("clean mug drawer", k=1)
    assert hits[0][0].task == "clean a mug and put it in drawer 1"


def test_retriever_skips_zero_score():
    bank = MemoryBank()
    bank.add(T("heat an apple"))
    assert BM25Retriever(bank, k=3).retrieve("zebra xylophone") == []


def test_render_truncates_long_trajectories():
    t = Trajectory(task="t", steps=[Step("o" * 100, "a" * 100) for _ in range(50)])
    out = t.render(max_chars=500)
    assert len(out) <= 520 and "[truncated]" in out and out.startswith("Task: t")


# The trajectories table as it was before the verdict column existed.
OLD_SCHEMA = """
CREATE TABLE trajectories (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    steps TEXT NOT NULL,
    success INTEGER,
    metadata TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX idx_traj_created ON trajectories(created_at);
"""
OLD_TASKS = ["put a mug in shelf 1", "put a hot mug in cabinet 1", "heat an apple", "cool a tomato"]


def old_db(path):
    """An old-schema database: three rows written through the old gate, one added directly."""
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    for i, task in enumerate(OLD_TASKS):
        meta = {"judge_reason": "ok"} if i < 3 else {}
        conn.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?)",
                     (f"old{i}", task, json.dumps([{"observation": "obs", "action": "act"}]),
                      1, json.dumps(meta), float(i)))
    conn.commit()
    conn.close()


def verdicts(path):
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute("SELECT id, verdict FROM trajectories"))
    finally:
        conn.close()


def test_old_schema_db_opens_and_is_migrated_in_place(tmp_path):
    db = tmp_path / "old.db"
    old_db(db)
    bank = MemoryBank(db)
    assert len(bank) == 4 and [t.task for t in bank.all()] == OLD_TASKS
    got = bank.get("old0")
    assert got.steps == [Step("obs", "act")] and got.success is True
    assert got.metadata == {"judge_reason": "ok"} and got.created_at == 0.0
    # rows the old gate stored were judged successes; the directly added one was never judged
    assert verdicts(db) == {"old0": 1, "old1": 1, "old2": 1, "old3": None}
    assert bank.verdicts() == {"old0": True, "old1": True, "old2": True, "old3": None}
    # recorded_at is backfilled from created_at (0.0 .. 3.0)
    assert [t.id for t in bank.since(time=1.0)] == ["old2", "old3"]
    bank.add(T("clean a knife"))
    assert bank.delete("old3") and len(bank) == 4
    bank.close()
    again = MemoryBank(db)  # second open: column already there, nothing to migrate
    assert len(again) == 4 and verdicts(db)["old0"] == 1


def test_old_schema_db_retrieval_unchanged(tmp_path):
    db = tmp_path / "old.db"
    old_db(db)
    fresh = MemoryBank()  # the same rows in a database that never had the old schema
    for i, task in enumerate(OLD_TASKS):
        fresh.add(Trajectory(task=task, steps=[Step("obs", "act")], id=f"old{i}", created_at=float(i)))
    migrated = MemoryBank(db)
    for query in ["put a clean mug in drawer 1", "cool a tomato", "heat something", "zebra"]:
        want = [(t.id, s) for t, s in BM25Retriever(fresh, k=3, verified_only=False).retrieve(query)]
        assert [(t.id, s) for t, s in BM25Retriever(migrated, k=3).retrieve(query)] == want
    hits = BM25Retriever(migrated, k=2).retrieve("put a clean mug in drawer 1")
    assert [t.id for t, _ in hits] == ["old0", "old1"]
    assert [t.id for t, _ in BM25Retriever(migrated).retrieve("cool a tomato")] == ["old3"]


def judged(task, success, **kw):
    return T(task, metadata={"verdict": {"success": success, "reason": "r", "evidence": [],
                                         "overruled": False}}, **kw)


def test_retriever_verified_only_hides_judged_failures():
    bank = MemoryBank()
    good, bad, unjudged = judged("put a mug in shelf 1", True), judged("put a mug in drawer 1", False), \
        T("put a mug in cabinet 1")
    for t in (good, bad, unjudged):
        bank.add(t)
    default = BM25Retriever(bank, k=5)
    assert default.verified_only is True
    assert {t.id for t, _ in default.retrieve("put a mug in drawer 1")} == {good.id, unjudged.id}
    everything = BM25Retriever(bank, k=5, verified_only=False)
    hits = everything.retrieve("put a mug in drawer 1")
    assert hits[0][0].id == bad.id and len(hits) == 3
    assert len(bank) == 3 and len(bank.tasks()) == 3 and len(bank.tasks(verified_only=True)) == 2


def test_failures_do_not_change_scores_of_verified_hits():
    """A stored failure must not shift BM25 statistics for the default retriever."""
    with_failures, without = MemoryBank(), MemoryBank()
    for i, task in enumerate(OLD_TASKS):
        for bank in (with_failures, without):
            bank.add(judged(task, True, id=f"t{i}", created_at=float(i)))
    for i, task in enumerate(["put a mug in shelf 1", "put a mug in a very long list of places"]):
        with_failures.add(judged(task, False, id=f"f{i}", created_at=10.0 + i))
    query = "put a clean mug in shelf 1"
    want = [(t.id, s) for t, s in BM25Retriever(without, k=3).retrieve(query)]
    assert want and [(t.id, s) for t, s in BM25Retriever(with_failures, k=3).retrieve(query)] == want


def test_since_version():
    bank = MemoryBank()
    a = bank.add(T("a"))
    mark = bank.version
    assert bank.since(mark) == []
    b, c = bank.add(judged("b", False)), bank.add(T("c"))
    assert [t.id for t in bank.since(mark)] == [b, c]  # failures included
    assert [t.id for t in bank.since(0)] == [a, b, c]
    bank.delete(b)
    assert [t.id for t in bank.since(mark)] == [c]
    assert bank.since(bank.version) == []


@pytest.fixture
def clock(monkeypatch):
    """Controls the time MemoryBank.add stamps into recorded_at."""
    now = [1000.0]
    monkeypatch.setattr(store, "_now", lambda: now[0])
    return now


def test_since_time_works_across_connections(tmp_path, clock):
    db = tmp_path / "m.db"
    bank = MemoryBank(db)
    for i in range(4):
        clock[0] = 1000.0 + i
        bank.add(T(f"task {i}", id=f"t{i}"))
    bank.close()
    reopened = MemoryBank(db)
    assert [t.id for t in reopened.since(time=1001.0)] == ["t2", "t3"]
    assert len(reopened.since(time=0.0)) == 4 and reopened.since(time=1003.0) == []
    assert reopened.since(0) == []  # versions count this connection's writes only


def test_since_time_uses_when_added_not_when_created(clock):
    """A batch's trajectories are created during the batch and added after it."""
    bank = MemoryBank()
    clock[0] = 1000.0
    bank.add(T("before the cutoff", id="early", created_at=990.0))
    cutoff = 1005.0
    clock[0] = 1010.0
    bank.add(T("ran before the cutoff, added after it", id="late", created_at=995.0))
    bank.add(T("created in the future, added now", id="future", created_at=5000.0))
    assert [t.id for t in bank.since(time=cutoff)] == ["late", "future"]
    assert bank.get("late").created_at == 995.0  # the trajectory's own timestamp is untouched
    clock[0] = 1020.0
    bank.add(bank.get("early"))  # re-adding counts as a new write
    assert [t.id for t in bank.since(time=1015.0)] == ["early"]


def test_db_with_verdict_but_no_recorded_at_is_migrated(tmp_path):
    db = tmp_path / "mid.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA + "ALTER TABLE trajectories ADD COLUMN verdict INTEGER;")
    conn.execute("INSERT INTO trajectories VALUES ('a', 'heat an apple', '[]', NULL, '{}', 7.0, 0)")
    conn.commit()
    conn.close()
    bank = MemoryBank(db)
    assert bank.verdicts() == {"a": False}  # an existing verdict is not overwritten
    assert [t.id for t in bank.since(time=6.0)] == ["a"] and bank.since(time=7.0) == []


def test_cli_list_reads_gate_from_verdict_column(tmp_path, capsys):
    db = tmp_path / "m.db"
    bank = MemoryBank(db)
    bank.add(judged("put a mug in shelf 1", True, id="aaaaaaaa1"))
    bank.add(judged("put a mug in drawer 1", False, id="bbbbbbbb1"))
    bank.add(T("heat an apple", id="cccccccc1"))
    # metadata that disagrees with the column: the column is what list must show
    bank.add(T("cool a tomato", id="dddddddd1", metadata={"verdict": {"success": True}}))
    bank.close()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE trajectories SET verdict = 0 WHERE id = 'dddddddd1'")
    conn.commit()
    conn.close()
    assert main(["list", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line[:1] in "abcd"}
    assert " ok " in rows["aaaaaaaa"] and " fail " in rows["bbbbbbbb"]
    assert " - " in rows["cccccccc"] and " fail " in rows["dddddddd"]
    assert "2 judged failures" in out


def test_since_needs_exactly_one_argument():
    bank = MemoryBank()
    with pytest.raises(ValueError):
        bank.since()
    with pytest.raises(ValueError):
        bank.since(0, time=0.0)
