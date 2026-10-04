# astro
Astrophotography project for telescope control, collimation, alignment, image stacking, AI voice control, and post-processing

## Development

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && scripts/install-solver.sh
    ASTRO_SIM=1 .venv/bin/pytest -q

See `docs/plan.md` for the project plan and `CLAUDE.md` for coding rules.
