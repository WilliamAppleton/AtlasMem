from atlasmem import BM25, BM25Retriever, MemoryBank, Step, Trajectory
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
