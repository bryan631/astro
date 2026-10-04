#!/usr/bin/env bash
# Start the astro server. Serves HTTPS when certs exist in certs/ (needed for the tablet mic).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f voice.env ] && set -a && . ./voice.env && set +a
CERT=$(ls certs/*.crt 2>/dev/null | head -1 || true)
TLS=()
[ -n "$CERT" ] && TLS=(--ssl-certfile "$CERT" --ssl-keyfile "${CERT%.crt}.key")
exec .venv/bin/uvicorn astro.server:app --host 0.0.0.0 --port "${ASTRO_PORT:-8443}" "${TLS[@]}"
