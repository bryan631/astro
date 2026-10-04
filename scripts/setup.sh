#!/usr/bin/env bash
# Set up astro. Safe to re-run.
#   scripts/setup.sh            dev: Debian/ChromeOS Linux, simulators only
#   scripts/setup.sh --minipc   field box: Ubuntu 24.04 MiniPC, adds voice, Tailscale, services
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip python3-tk
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev,tools]"
scripts/install-solver.sh
[ -f .env ] || echo "# ANTHROPIC_API_KEY=" > .env
ASTRO_SIM=1 .venv/bin/pytest -q

if [ "${1:-}" != "--minipc" ]; then
  echo "Done. Start with: ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0"
  exit 0
fi

scripts/install-voice.sh  # whisper.cpp, Piper, ffmpeg
sudo usermod -aG dialout "$USER"  # the encoder board's serial port
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up  # prints a login link the first time
scripts/tailscale-cert.sh
# The units assume user 'astro' with the repo at /home/astro/astro: rewrite for this install.
for unit in deploy/*.service deploy/*.timer; do
  sed -e "s#/home/astro/astro#$PWD#g" -e "s#^User=astro#User=$USER#" "$unit" \
    | sudo tee "/etc/systemd/system/$(basename "$unit")" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now astro.service astro-cert.timer
cat <<'NOTES'
Done. Still by hand (see docs/plan.md):
  - BIOS: power on after AC loss, so the box starts when the battery is switched on.
  - Phone hotspot: join it once (nmcli dev wifi connect <ssid> password <pw>) so the box
    reconnects in the field; home WiFi stays the first choice.
  - Field check: docs/dev-runbook.md "Field use" (HTTPS with no internet, access token).
NOTES
