#!/usr/bin/env bash
# Request a GitHub Copilot review on a PR: scripts/dev/copilot-review.sh <pr-number>
# (gh's REST reviewer endpoint silently ignores Copilot; the GraphQL bot request works.)
set -euo pipefail
N=$1
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
COPILOT_BOT=BOT_kgDOCnlnWA  # copilot-pull-request-reviewer
PR=$(gh api graphql -f query="{repository(owner:\"${REPO%/*}\",name:\"${REPO#*/}\"){pullRequest(number:$N){id}}}" \
  --jq .data.repository.pullRequest.id)
gh api graphql -f query='mutation($pr:ID!,$bot:ID!){requestReviews(input:{pullRequestId:$pr,botIds:[$bot]}){clientMutationId}}' \
  -f pr="$PR" -f bot="$COPILOT_BOT" >/dev/null
echo "Copilot review requested on #$N"
