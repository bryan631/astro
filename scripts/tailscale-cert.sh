#!/usr/bin/env bash
# Get a real HTTPS cert for this machine's Tailscale name (MagicDNS + HTTPS must be enabled
# in the Tailscale admin console). The tablet then opens https://<name>.<tailnet>.ts.net:8443
# Re-run monthly (certs last 90 days); the systemd timer in deploy/ does this.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
mkdir -p certs
sudo tailscale cert --cert-file "certs/$NAME.crt" --key-file "certs/$NAME.key" "$NAME"
sudo chown "$(stat -c %U .)" certs/*
echo "Tablet URL: https://$NAME:${ASTRO_PORT:-8443}"
