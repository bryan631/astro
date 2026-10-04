from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from astro.agent import Agent
from astro.pointing.coords import Site
from astro.session import Session

EVENING = datetime(2026, 10, 3, 20, 0, tzinfo=timezone(timedelta(hours=-4)))


def session():
    return Session(Site(26.7, -80.1), lambda: (45, 180), clock=lambda: EVENING)


class FakeClient:
    """Scripted responses; records requests."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return self.responses.pop(0)


def text(t):
    return NS(type="text", text=t)


def test_offline_without_client():
    out = Agent(session(), client=None).handle("go to albireo")
    assert out == [{"type": "say", "text": "Let's find Albireo."}]


def test_core_commands_skip_the_llm():
    client = FakeClient([])
    Agent(session(), client=client).handle("stop")
    assert client.requests == []


def test_tool_loop_goto():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="goto",
                                               input={"target": "the ring nebula"})]),
        NS(stop_reason="end_turn", content=[text("Off we go to the Ring Nebula!")]),
    ])
    s = session()
    out = Agent(s, client=client).handle("I'd love to see that smoke ring thing")
    assert out[-1]["text"] == "Off we go to the Ring Nebula!"
    assert s.target == "Ring Nebula"
    tool_result = client.requests[1]["messages"][2]["content"][0]  # user, assistant, results
    assert tool_result["tool_use_id"] == "t1" and "Ring Nebula" in tool_result["content"]
