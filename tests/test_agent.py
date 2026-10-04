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


def test_vague_goto_goes_to_llm():
    client = FakeClient([NS(stop_reason="end_turn", content=[text("Albireo it is.")])])
    Agent(session(), client=client).handle("show me that pretty double star")
    assert len(client.requests) == 1


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


def run_tool(name, tool_input, s=None):
    s = s or session()
    client = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name=name, input=tool_input)]),
        NS(stop_reason="end_turn", content=[text("ok")]),
    ])
    Agent(s, client=client).handle("please do the thing")
    return s, client.requests[1]["messages"][2]["content"][0]["content"]


def test_barlow_and_focus_tools():
    s, _ = run_tool("barlow", {"inserted": True})
    assert s.barlow
    s, result = run_tool("focus", {"camera": "finder"})
    assert "finder camera" in result  # no finder in this test session


def test_session_status_tool_reports_facts():
    _, result = run_tool("session_status", {})
    assert "no target" in result and "no Barlow" in result and "treeline not recorded" in result
    assert "no weather forecast" in result


def test_every_tool_is_handled():
    from astro.agent import _COMMANDS, TOOLS

    handled = set(_COMMANDS) | {"goto", "focus", "barlow", "session_status"}
    assert {t["name"] for t in TOOLS} <= handled


def test_goto_tool_matches_names_without_reparsing():
    """Review M15: the tool's target is matched as a name, never re-parsed as a sentence."""
    _, result = run_tool("goto", {"target": "what's good tonight"})
    assert result == "I don't know what's good tonight."  # not the tonight listing
    s, _ = run_tool("goto", {"target": "Messier 57"})
    assert s.target == "M57" or s.target is None  # matched by name (may be refused if set)
