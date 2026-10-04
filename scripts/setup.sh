#!/usr/bin/env bash
# Set up astro on Ubuntu 24.04 (MiniPC) or Debian/ChromeOS Linux (dev). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev]"
# Plate solver; its Pillow<9 pin is obsolete, so skip its dependency list.
.venv/bin/pip install -q --no-deps "git+https://github.com/smroid/cedar-solve@607e3f8da7db0743bb6a26037e21c40b6befe49a"
[ -f .env ] || echo "# ANTHROPIC_API_KEY=" > .env
ASTRO_SIM=1 .venv/bin/pytest -q
echo "Done. Start with: ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0"
