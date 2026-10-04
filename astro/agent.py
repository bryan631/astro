"""Claude agent: interprets free-form speech, calls session tools, narrates warmly.

Pointing stays deterministic: tools only start/stop guidance; cues come from the guide.
Falls back to the offline grammar when there's no API key or no network.
"""

import os

import anthropic

from astro.intents import match_name, parse
from astro.session import Session

MODEL = "claude-haiku-4-5"
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 10

SYSTEM = """You help a 77-year-old amateur astronomer use his telescope by voice.
Replies are spoken aloud: one to three short, warm, plain sentences. No lists, no markdown,
no jargon unless he asks. Use the tools to act; never invent where something is in the sky.
Guidance cues ("push left", "stop") are spoken by the system, not by you.
If he asks what to see, or refers to something suggested earlier, call list_tonight first.
He views on the tablet screen, not through an eyepiece."""

TOOLS = [
    {"name": "list_tonight", "description": "Targets visible tonight: the best one in each "
     "category (planet, moon, nebula, cluster, galaxy, double star) with a short note.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "goto", "description": "Start guiding the telescope to a named target "
     "(planet, Moon, or catalog object like 'Ring Nebula' or 'M57').",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}},
                      "required": ["target"]}},
    {"name": "stop", "description": "Stop guiding.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "where_am_i", "description": "What the telescope is pointing at now.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_location", "description": "Update where the telescope is, using the tablet's "
     "GPS (e.g. after moving to a new place).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "next", "description": "Go to the next suggestion from tonight's list.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "take_picture", "description": "Take a picture of the current target: video for "
     "planets/Moon, a live stack for everything else. Asks for focus first if needed.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "stop_picture", "description": "Stop a picture being taken (keeps what's done).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "focus", "description": "Start the spoken focus helper for the main camera, or "
     "for the finder if camera='finder'. The user says 'done' at the sharpest point.",
     "input_schema": {"type": "object", "properties": {
         "camera": {"type": "string", "enum": ["main", "finder"]}}}},
    {"name": "sync", "description": "Look at the stars through the finder to find exactly "
     "where the telescope points.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "barlow", "description": "Tell the system the Barlow lens was put in or taken out.",
     "input_schema": {"type": "object", "properties": {"inserted": {"type": "boolean"}},
                      "required": ["inserted"]}},
    {"name": "horizon_walk", "description": "Start recording the treeline (the user then says "
     "'mark' at points along it and 'done').", "input_schema": {"type": "object", "properties": {}}},
    {"name": "session_status", "description": "What's going on: aligned or not, target, focus, "
     "Barlow, picture in progress, pictures taken, horizon, clouds.",
     "input_schema": {"type": "object", "properties": {}}},
]

# Tool name -> offline command text the session already understands.
_COMMANDS = {"list_tonight": "what's good tonight", "stop": "stop", "where_am_i": "where am i",
             "next": "next", "set_location": "set location", "take_picture": "take a picture",
             "stop_picture": "stop recording", "sync": "sync",
             "horizon_walk": "start the horizon walk"}


class Agent:
    def __init__(self, session: Session, client: anthropic.Anthropic | None = None):
        self.session = session
        self.client = client if client is not None else _default_client()
        self.history: list[dict] = []

    def handle(self, text: str) -> list[dict]:
        """Return messages for the tablet. Core commands never need the network."""
        if self.client is None or self._offline_understands(text):
            return self.session.handle(text)
        try:
            return self._run(text)
        except anthropic.APIError:
            return self.session.handle(text)

    def _offline_understands(self, text: str) -> bool:
        """Fast path for exact commands; anything vague goes to Claude for context."""
        intent = parse(text)
        if intent is None or intent.name == "tonight":
            return False  # Claude gives a nicer, conversational overview
        if intent.name == "goto":
            return match_name(intent.target or "", self.session.names()) is not None
        return True

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
        if name == "focus":
            return self.session.handle("focus the finder" if args.get("camera") == "finder"
                                       else "focus")
        if name == "barlow":
            return self.session.handle("barlow in" if args.get("inserted") else "barlow out")
        if name == "session_status":
            return [{"type": "say", "text": self.session.status_text()}]
        if name == "list_tonight":
            return [{"type": "say", "text": self.session.tonight_by_category()}]
        if name == "goto":
            return self.session.handle(f"go to {args.get('target', '')}")
        if name in _COMMANDS:
            return self.session.handle(_COMMANDS[name])
        return [{"type": "say", "text": f"Unknown tool {name}."}]


def _default_client() -> anthropic.Anthropic | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic(timeout=15.0, max_retries=1)
