# Coding rules for Claude

1. **Clean and concise.** Write simple code a human can review quickly. Avoid overly complex designs and premature abstraction.
2. **Open source first.** Use open source libraries freely. Avoid binaries where possible; closed-source drivers with no open source equivalent are the exception.
3. **Commit often, use modern CI/CD.** Small, focused commits on one working branch. Keep GitHub Actions CI (lint, test, build) green; add CD when there is something to deploy.
4. **Good unit tests, not excessive.** Cover core logic and edge cases that matter. Don't chase 100% coverage or test trivia.
5. **Prototype, then productize.** Get the feature right in a quick prototype first. Only harden (structure, tests, docs, error handling) after the prototype is validated.
6. **Code hygiene nudges.** The user struggles with organization. When a session drifts on a tangent, stop and propose cleanup: commit or stash stray work, remove dead code and scratch files, update README/notes, split branches, sync with `main`.
7. **Few PRs.** This is a one-developer project, not a corporate workflow. Keep committing to one working branch and batch everything (features, fixes, docs, review nits) into one PR, merged when a chunk of work is done or main is needed for a deploy, at most about once a day. Never a PR per task, fix or doc. Request Copilot review once per PR (skip it for docs-only), address its feedback, and merge.
8. **Tag on hardware confirmation.** When the user confirms something works on real hardware (e.g. "ship it!"), create an annotated git tag (e.g. `v0.1.0`) on that commit and push it.
9. **Be token-efficient.** Read only what's needed, keep replies short, avoid re-deriving known facts, and don't spawn agents unless necessary.
