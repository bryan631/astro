# Coding rules for Claude

1. **Clean and concise.** Write simple code a human can review quickly. Avoid overly complex designs and premature abstraction.
2. **Open source first.** Use open source libraries freely. Avoid binaries where possible; closed-source drivers with no open source equivalent are the exception.
3. **Commit often, use modern CI/CD.** Small, focused commits, all on one session branch (`session/<topic>`). Keep GitHub Actions CI (lint, test, build) green; add CD when there is something to deploy.
4. **Good unit tests, not excessive.** Cover core logic and edge cases that matter. Don't chase 100% coverage or test trivia.
5. **Prototype, then productize.** Get the feature right in a quick prototype first. Only harden (structure, tests, docs, error handling) after the prototype is validated.
6. **Code hygiene nudges.** The user struggles with organization. When a session drifts on a tangent, stop and propose cleanup: commit or stash stray work, remove dead code and scratch files, update README/notes, split branches, sync with `main`.
7. **One PR per session, not per change.** Keep working and committing on the session branch; push it at the end if you like, but open a PR only when the session's work is complete and tests pass. Never open a PR per review item or per nit. Then request Copilot code review once, address its feedback, and merge. Docs-only or tiny fixes ride along with the session's PR instead of getting their own.
8. **Tag on hardware confirmation.** When the user confirms something works on real hardware (e.g. "ship it!"), create an annotated git tag (e.g. `v0.1.0`) on that commit and push it.
9. **Be token-efficient.** Read only what's needed, keep replies short, avoid re-deriving known facts, and don't spawn agents unless necessary.
