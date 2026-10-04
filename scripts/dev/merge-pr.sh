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
      *BEHIND*)  # update, then wait until the PR head really is the new merge commit
        old=$(gh pr view "$n" --json headRefOid -q .headRefOid)
        gh api -X PUT "repos/$REPO/pulls/$n/update-branch" >/dev/null
        for _ in $(seq 30); do
          [ "$(gh pr view "$n" --json headRefOid -q .headRefOid)" != "$old" ] && break; sleep 5
        done ;;
    esac
    sha=$(gh pr view "$n" --json headRefOid -q .headRefOid)
    for _ in $(seq 60); do ci=$(ci_state "$sha"); [ "$ci" != pending ] && break; sleep 10; done
    [ "$ci" = failure ] && { echo "#$n CI failed on $sha"; return 1; }
    gh pr merge "$n" --squash --delete-branch >/dev/null 2>&1
    [ "$(gh pr view "$n" --json state -q .state)" = MERGED ] && { echo "#$n merged"; return; }
  done
  echo "#$n not merged: $(gh pr view "$n" --json mergeStateStatus -q .mergeStateStatus)"
  return 1
}
for n in "$@"; do merge_one "$n" || exit 1; done
