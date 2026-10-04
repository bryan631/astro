#!/usr/bin/env bash
# Squash-merge PRs once CI passes on their current head: scripts/dev/merge-pr.sh 15 16 ...
# Updates a branch that is behind main first (branch protection requires it), then waits for
# CI on the new head commit, so it never merges on a stale green result.
set -uo pipefail
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
REQUIRED_CHECK=test  # what branch protection requires; the quick firmware job can finish first
ci_state() {
  gh api "repos/$REPO/commits/$1/check-runs" --jq "[.check_runs[] | {name, c: (.conclusion // \"pending\")}]
    | if any(.c == \"failure\") then \"failure\"
      elif (map(select(.name == \"$REQUIRED_CHECK\" and .c == \"success\")) | length) == 0 then \"pending\"
      elif all(.c == \"success\") then \"success\" else \"pending\" end"
}
merge_one() {
  local n=$1 state sha ci
  for _ in $(seq 6); do
    state=$(gh pr view "$n" --json state,mergeStateStatus -q '.state + " " + .mergeStateStatus')
    case "$state" in
      MERGED*) echo "#$n merged"; return ;;
      *UNKNOWN*) sleep 10; continue ;;  # GitHub recomputing after another merge
      *DIRTY*) echo "#$n has merge conflicts with main: resolve them first"; return 1 ;;
      *BEHIND*)  # update, then wait until the PR head really is the new merge commit
        old=$(gh pr view "$n" --json headRefOid -q .headRefOid) || return 1
        gh api -X PUT "repos/$REPO/pulls/$n/update-branch" >/dev/null || return 1
        updated=
        for _ in $(seq 30); do
          [ "$(gh pr view "$n" --json headRefOid -q .headRefOid)" != "$old" ] && { updated=1; break; }
          sleep 5
        done
        [ -n "$updated" ] || { echo "#$n: branch update never appeared"; return 1; } ;;
    esac
    sha=$(gh pr view "$n" --json headRefOid -q .headRefOid) || return 1
    ci=
    for _ in $(seq 60); do ci=$(ci_state "$sha"); [ "$ci" = success ] || [ "$ci" = failure ] && break; sleep 10; done
    [ "$ci" = success ] || { echo "#$n CI ${ci:-unknown} on $sha"; return 1; }
    # Only merge the commit whose CI we checked; a push meanwhile makes this fail and retry.
    gh pr merge "$n" --squash --delete-branch --match-head-commit "$sha" >/dev/null 2>&1
    [ "$(gh pr view "$n" --json state -q .state)" = MERGED ] && { echo "#$n merged"; return; }
  done
  echo "#$n not merged: $(gh pr view "$n" --json mergeStateStatus -q .mergeStateStatus)"
  return 1
}
for n in "$@"; do merge_one "$n" || exit 1; done
