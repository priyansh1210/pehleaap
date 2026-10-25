---
name: git-workflow
description: Priyansh commits, pushes and raises PRs themselves from copy-paste PowerShell commands; no gh; branch naming; post-merge routine
metadata:
  type: feedback
---

Priyansh runs git commit/push and raises PRs themselves. When work is ready, give PowerShell copy-paste commands:
branch, `git add`, a commit message piped from a here-string (`@' … '@ | git commit -F -`, closing `'@` at column 0, ASCII only because
PowerShell 5.1 pipes as ASCII) with the Co-Authored-By trailer, push, and the PR link
`https://github.com/priyansh1210/pehleaap/compare/main...<branch>?expand=1` with a ready-to-paste title and description.

After a merge Priyansh deletes the remote branch and asks me to switch to main: `git fetch --prune`, `git checkout main`,
`git pull --ff-only`, `git branch -d <branch>`.

**Why:** on 2026-10-24 they asked "give me commands to commit it and raise pr", then ran them and merged PR #1 (week1-foundation) and
PR #2 (week1-baselines, merge dcdd96e).
**How to apply:** don't commit or push on my own unless asked. `gh` is not installed on the Laptop, so PRs go through the browser link.
Repo: https://github.com/priyansh1210/pehleaap (**public**, default branch `main`). Feature branches are named like `week1-foundation`.
Memory changes in `memory/` travel through the same commits ([[shared-memory]]).

**GPU-box lessons (2026-10-25):** the Laptop merges PRs while GPU-box work is in progress, so pull `main` right before branching
for a commit. When a pushed branch conflicts with main (always `memory/pehleaap-status.md`, both machines append), merge main into
the branch and keep both entries in time order: no rebase, so no force push. A push alone doesn't open a PR; if the compare page
shows nothing, the branch is usually already merged (`git merge-base --is-ancestor <commit> origin/main`). When Priyansh pastes the
push / `start <compare link>` commands back into the chat, they want me to run them (done once, PR #25, no objection).
