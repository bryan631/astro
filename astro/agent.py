"""Claude agent: interprets free-form speech, calls session tools, narrates warmly.

Pointing stays deterministic: tools only start/stop guidance; cues come from the guide.
Falls back to the offline grammar when there's no API key or no network.
"""

import os

import anthropic

from astro.intents import parse
from astro.session import Session

MODEL = "claude-haiku-4-5"
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 10

SYSTEM = """You help a 77-year-old amateur astronomer use his telescope by voice.
Replies are spoken aloud: one to three short, warm, plain sentences. No lists, no markdown,
no jargon unless he asks. Use the tools to act; never invent where something is in the sky.
Guidance cues ("push left", "stop") are spoken by the system, not by you."""

TOOLS = [
    {"name": "list_tonight", "description": "Best targets visible tonight, ranked.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "goto", "description": "Start guiding the telescope to a named target "
     "(planet, Moon, or catalog object like 'Ring Nebula' or 'M57').",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}},
                      "required": ["target"]}},
    {"name": "stop", "description": "Stop guiding.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "where_am_i", "description": "What the telescope is pointing at now.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "next", "description": "Go to the next suggestion from tonight's list.",
     "input_schema": {"type": "object", "properties": {}}},
]

# Tool name -> offline command text the session already understands.
_COMMANDS = {"list_tonight": "what's good tonight", "stop": "stop", "where_am_i": "where am i",
             "next": "next"}


class Agent:
    def __init__(self, session: Session, client: anthropic.Anthropic | None = None):
        self.session = session
        self.client = client if client is not None else _default_client()
        self.history: list[dict] = []

    def handle(self, text: str) -> list[dict]:
        """Return messages for the tablet. Core commands never need the network."""
        if self.client is None or parse(text) is not None:
            return self.session.handle(text)
        try:
            return self._run(text)
        except anthropic.APIError:
            return self.session.handle(text)

    def _run(self, text: str) -> list[dict]:
        messages = [*self.history, {"role": "user", "content": text}]
        side_effects: list[dict] = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM,
                                               tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = self._call(block.name, block.input)
                    side_effects += [m for m in out if m["type"] != "say"]
                    said = " ".join(m["text"] for m in out if m["type"] == "say")
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": said or "done"})
            messages.append({"role": "user", "content": results})
        reply = " ".join(b.text for b in resp.content if b.type == "text").strip()
        self.history = [*self.history, {"role": "user", "content": text},
                        {"role": "assistant", "content": reply or "OK."}][-2 * HISTORY_TURNS:]
        return [*side_effects, {"type": "say", "text": reply or "OK."}]

    def _call(self, name: str, args: dict) -> list[dict]:
        if name == "goto":
            return self.session.handle(f"go to {args.get('target', '')}")
        if name in _COMMANDS:
            return self.session.handle(_COMMANDS[name])
        return [{"type": "say", "text": f"Unknown tool {name}."}]


def _default_client() -> anthropic.Anthropic | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic(timeout=15.0, max_retries=1)
