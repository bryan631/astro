#!/usr/bin/env bash
# Set up astro on Ubuntu 24.04 (MiniPC) or Debian/ChromeOS Linux (dev). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev,tools]"
scripts/install-solver.sh
[ -f .env ] || echo "# ANTHROPIC_API_KEY=" > .env
ASTRO_SIM=1 .venv/bin/pytest -q
echo "Done. Start with: ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0"
