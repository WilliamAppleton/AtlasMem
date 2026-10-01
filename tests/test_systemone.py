import pytest

from atlasmem import CascadeJudge, Step, SystemOneClient, SystemOneJudge, Trajectory
from atlasmem import systemone as so
from atlasmem.judge import Verdict

T = Trajectory(task="put a mug in shelf 1", steps=[Step("You put the mug in/on the shelf 1.", "[end]")])


@pytest.fixture
def fake_post(monkeypatch):
    calls = []

    def install(p_success, score=3.0):
        def post(url, body, headers, timeout, retries):
            calls.append((url, body, headers))
            ans = {"success": {"type": "noul", "noul": p_success}}
            if "progress" in body["questions"]:
                ans["progress"] = {"type": "score", "score": score}
            return {"model": body["model"], "answers": ans, "usage": {}}
        monkeypatch.setattr(so, "_post_json", post)
        return calls
    return install


def test_request_shape_local(fake_post, monkeypatch):
    monkeypatch.delenv("SYSTEMONE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    calls = fake_post(0.97)
    v = SystemOneJudge(SystemOneClient("nimble")).judge(T)
    url, body, headers = calls[0]
    assert url == "http://localhost:11434/v1/systemone" and headers == {}
    assert body["model"] == "nimble" and "Task: put a mug in shelf 1" in body["state"]
    assert body["questions"]["success"]["type"] == "noul"
    assert body["questions"]["progress"]["type"] == "score"
    assert v.success and v.probability == 0.97 and v.progress == 1.0


def test_typesafe_cloud_uses_bearer_key(fake_post, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k123")
    calls = fake_post(0.2)
    v = SystemOneJudge(SystemOneClient("jev", base_url=so.TYPESAFE_URL), with_progress=False).judge(T)
    url, body, headers = calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone" and headers == {"Authorization": "Bearer k123"}
    assert "progress" not in body["questions"] and not v.success and v.progress is None


class SlowJudge:
    def __init__(self):
        self.n = 0

    def judge(self, traj):
        self.n += 1
        return Verdict(success=True, reason="slow says yes")


@pytest.mark.parametrize("p,escalate,ok", [(0.95, False, True), (0.05, False, False), (0.5, True, True)])
def test_cascade_escalates_only_uncertain_band(fake_post, p, escalate, ok):
    fake_post(p)
    slow = SlowJudge()
    c = CascadeJudge(SystemOneJudge(SystemOneClient("tev1", base_url="http://x")), slow, 0.1, 0.9)
    v = c.judge(T)
    assert v.success is ok and slow.n == int(escalate) and c.escalations == int(escalate)


def test_typesafe_key_from_jev_var_or_none(fake_post, monkeypatch):
    for n in ("TYPESAFE_API_KEY", "JEV_API_KEY", "jev"):
        monkeypatch.delenv(n, raising=False)
    calls = fake_post(0.9)
    SystemOneJudge(SystemOneClient("jev", base_url=so.TYPESAFE_URL)).judge(T)
    assert calls[-1][2] == {}  # no key: rely on proxy injection
    monkeypatch.setenv("jev", "abc")
    SystemOneJudge(SystemOneClient("jev", base_url=so.TYPESAFE_URL)).judge(T)
    assert calls[-1][2] == {"Authorization": "Bearer abc"}
