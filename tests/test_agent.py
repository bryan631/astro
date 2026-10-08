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

    handled = set(_COMMANDS) | {"goto", "describe", "focus", "barlow", "session_status"}
    assert {t["name"] for t in TOOLS} <= handled


class Unreachable:
    """Client whose every call fails like a hotspot with no internet."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kw):
        import httpx

        self.calls += 1
        raise __import__("anthropic").APIConnectionError(request=httpx.Request("POST", "https://x"))


def test_offline_falls_back_once_then_skips_the_llm():
    client = Unreachable()
    a = Agent(session(), client=client)
    a.handle("I'd love to see that smoke ring thing")
    a.handle("anything fun up there?")
    assert client.calls == 1  # second request didn't wait on the network again


def test_tonight_is_answered_offline_even_with_a_key():
    client = FakeClient([])
    out = Agent(session(), client=client).handle("what's good tonight")
    assert client.requests == [] and "is the best" in out[0]["text"]


def test_goto_tool_matches_names_without_reparsing():
    """Review M15: the tool's target is matched as a name, never re-parsed as a sentence."""
    _, result = run_tool("goto", {"target": "what's good tonight"})
    assert result == "I don't know what's good tonight."  # not the tonight listing
    s, _ = run_tool("goto", {"target": "Messier 57"})
    assert s.target == "M57" or s.target is None  # matched by name (may be refused if set)


def test_next_step_stays_offline_with_a_key():
    client = FakeClient([])
    out = Agent(session(), client=client).handle("next step")
    assert client.requests == [] and out[0]["text"] == "Ask me what's good tonight first."


def test_connection_lost_after_a_tool_ran_does_not_redo_it():
    import httpx

    class DropsAfterTool(FakeClient):
        def create(self, **kw):
            if self.requests:
                raise __import__("anthropic").APIConnectionError(
                    request=httpx.Request("POST", "https://x"))
            return super().create(**kw)

    client = DropsAfterTool([NS(stop_reason="tool_use", content=[
        NS(type="tool_use", id="t1", name="goto", input={"target": "the ring nebula"})])])
    s = session()
    calls = []
    s.handle = lambda text: calls.append(text) or []  # the offline fallback must not run
    out = Agent(s, client=client).handle("I'd love to see that smoke ring thing")
    assert s.target == "Ring Nebula" and calls == []
    assert out[-1]["text"].startswith("I lost my connection")


def test_out_of_tool_rounds_ends_with_words():
    from astro.agent import MAX_TOOL_ROUNDS

    tool = NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t", name="session_status",
                                                  input={})])
    client = FakeClient([tool] * MAX_TOOL_ROUNDS + [NS(stop_reason="end_turn",
                                                       content=[text("All set.")])])
    out = Agent(session(), client=client).handle("tell me everything, then more")
    assert out[-1]["text"] == "All set."
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_without_a_key_a_question_says_why_it_cant_help():
    out = Agent(session(), client=None).handle("when will the clouds clear?")
    assert "no Claude API key" in out[0]["text"]


class Failing:
    """A client whose every request raises `error`."""

    def __init__(self, error):
        self.error, self.messages = error, self

    def create(self, **kw):
        raise self.error


def status_error(cls, code, message="nope"):
    import httpx

    response = httpx.Response(code, request=httpx.Request("POST", "https://x"))
    return cls(message, response=response, body=None)


def test_claude_failures_are_explained_and_commands_still_work():
    import anthropic

    cases = [
        (Unreachable(), "can't reach Claude"),
        (Failing(status_error(anthropic.AuthenticationError, 401)), "rejected the API key"),
        (Failing(status_error(anthropic.BadRequestError, 400, "Your credit balance is too low")),
         "out of credit"),
        (Failing(status_error(anthropic.InternalServerError, 529)), "having problems"),
    ]
    for client, why in cases:
        a = Agent(session(), client=client)
        assert why in a.handle("when will the clouds clear?")[0]["text"]
        assert why in a.handle("is it going to rain?")[0]["text"]  # offline window or not
        assert a.handle("go to albireo") == [{"type": "say", "text": "Let's find Albireo."}]
