#!/usr/bin/env bash
# Copy the Mele's spots (places and treelines) into the SPOTS_TOML repo secret, then rebuild the
# online "Tonight" page. Run on the dev machine after recording a new treeline.
#   scripts/publish-spots.sh [host]     (default host: mele)
set -euo pipefail
host=${1:-mele}
spots=$(timeout 300 tailscale ssh "astro@$host" 'cat ~/astro/data/spots.toml')
grep -q '^\[\[spot\]\]' <<<"$spots" || { echo "no spots on $host yet: record a treeline first" >&2; exit 1; }
gh secret set SPOTS_TOML <<<"$spots"
gh workflow run tonight.yml
echo "Published $(grep -c '^\[\[spot\]\]' <<<"$spots") spot(s); the page updates in about 2 minutes."
