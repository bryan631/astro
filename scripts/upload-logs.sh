#!/usr/bin/env bash
# Daily log upload for remote support (R7): the Mele's JSON logs and a check.sh report go to the
# private repo bryan631/astro-logs, with a deploy key that can write to that repo only. Logs
# hold the site's location and what was said, so the repo must stay private.
#   scripts/upload-logs.sh --setup   once: make the key, then on the dev machine:
#       gh repo deploy-key add <(tailscale ssh astro@mele cat .ssh/astro-logs.pub) \
#         --allow-write -R bryan631/astro-logs -t mele
#   scripts/upload-logs.sh           what the astro-logs timer runs (also fine by hand)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${ASTRO_LOGS_REPO:-git@github.com:bryan631/astro-logs.git}
KEY=~/.ssh/astro-logs
DEST=~/astro-logs
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
if [ "${1:-}" = --setup ]; then
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "astro-logs@$(hostname)" -f "$KEY"
  echo "Add this as a deploy key with write access on bryan631/astro-logs:"
  cat "$KEY.pub"
  exit 0
fi
[ -f "$KEY" ] || { echo "no deploy key: run scripts/upload-logs.sh --setup" >&2; exit 1; }
[ -d "$DEST/.git" ] || timeout 120 git clone -q "$REPO" "$DEST"
git -C "$DEST" remote set-url origin "$REPO"  # the destination is always REPO, even if it changed
timeout 120 git -C "$DEST" pull -q --rebase
out="$DEST/$(hostname)"
mkdir -p "$out"
for f in data/logs/astro-*.jsonl; do  # one file per server run; the newest one is still growing
  [ -e "$f" ] || continue
  gz="$out/$(basename "$f").gz"
  [ "$gz" -nt "$f" ] || gzip -9 -n -c "$f" > "$gz"
done
timeout 60 scripts/check.sh > "$out/check.txt" 2>&1 || true  # exit code = failure count
git -C "$DEST" add -A
git -C "$DEST" diff --cached --quiet \
  || git -C "$DEST" -c user.name=astro -c user.email="astro@$(hostname)" commit -q -m "logs $(date -u +%F)"
timeout 120 git -C "$DEST" push -q
echo "uploaded $(ls "$out"/*.gz 2>/dev/null | wc -l) log file(s) to $REPO"
