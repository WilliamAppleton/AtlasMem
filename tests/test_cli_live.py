import json
import os

import pytest

from atlasmem.cli import main


def test_cli_record_list_show_delete(tmp_path, capsys):
    db = str(tmp_path / "m.db")
    f = tmp_path / "t.json"
    f.write_text(json.dumps([
        {"task": "put a mug in shelf 1", "steps": [{"observation": "o", "action": "a"}]},
        {"task": "heat an apple", "steps": []},
    ]))
    assert main(["record", str(f), "--db", db, "--label", "success"]) == 0
    assert main(["list", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "put a mug in shelf 1" in out and "(2 trajectories" in out
    assert main(["brief", "zebra", "--db", db]) == 0
    assert "(no relevant memories)" in capsys.readouterr().out
    assert main(["delete", "--all", "--db", db]) == 0


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ATLASMEM_LIVE"), reason="set ATLASMEM_LIVE=1")
def test_live_ollama_brief_and_judge():
    from atlasmem import Curator, JitMemory, LLMJudge, MemoryBank, OllamaLLM, Step, Trajectory

    model = os.environ.get("ATLASMEM_CURATOR_MODEL", "gpt-oss:20b")
    bank = MemoryBank()
    mem = JitMemory(bank, Curator(OllamaLLM(model)), LLMJudge(OllamaLLM(model, temperature=0)))
    good = Trajectory(task="put a mug in shelf 1", steps=[
        Step("You are in the kitchen.", "go to cabinet 2"),
        Step("The cabinet 2 is closed.", "open cabinet 2"),
        Step("In the cabinet 2, you see a mug.", "take mug from cabinet 2"),
        Step("You pick up the mug from the cabinet 2.", "go to shelf 1"),
        Step("On the shelf 1, you see nothing.", "put mug in shelf 1"),
        Step("You put the mug in/on the shelf 1.", "[end]"),
    ])
    assert mem.record(good).stored
    b = mem.brief("put a clean mug in drawer 1")
    assert b.retrieved and "cabinet 2" in b.payload.lower(), b.payload
