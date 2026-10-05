# astro

A voice-guided push-to telescope helper: plate-solved pointing, spoken guidance from a
tablet, planetary and deep-sky imaging (lucky imaging and live stacking), and a Claude
agent for free-form questions. Runs on a MiniPC at the telescope, with simulators for
development.

- `docs/user-guide.md`: how to use it at the telescope
- `docs/dev-runbook.md`: running, layout, deployment
- `docs/requirements.md`: the requirements this is judged against
- `docs/plan.md`: the original project plan; `CLAUDE.md`: coding rules

## Development

    scripts/setup.sh            # or, by hand:
    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && scripts/install-solver.sh
    ASTRO_SIM=1 .venv/bin/pytest -q
