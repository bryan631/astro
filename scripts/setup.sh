#!/usr/bin/env bash
# Set up astro. Safe to re-run.
#   scripts/setup.sh            dev: Debian/ChromeOS Linux, simulators only
#   scripts/setup.sh --minipc   field box: Ubuntu 24.04 MiniPC, adds voice, Tailscale, services
set -euo pipefail
case "${1:-}" in
  ""|--minipc) ;;
  *) echo "usage: $0 [--minipc]" >&2; exit 2 ;;
esac
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
sudo usermod -aG dialout,video "$USER"  # the encoder board's serial port
# SVBony cameras (f266:*) need a udev rule for non-root USB access.
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="f266", GROUP="video", MODE="0660"' | sudo tee /etc/udev/rules.d/90-ckusb.rules >/dev/null
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=usb
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh  # prints a login link the first time
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
  - Phone hotspot: join it once so the box reconnects in the field, at a lower priority so
    home WiFi stays the first choice:
      nmcli dev wifi connect <ssid> password <pw> name hotspot
      nmcli con modify hotspot connection.autoconnect-priority -10
  - Field check: docs/dev-runbook.md "Field use" (HTTPS with no internet, access token).
NOTES
