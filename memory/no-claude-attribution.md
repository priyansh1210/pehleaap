---
name: no-claude-attribution
description: "Priyansh's rule (2026-11-06): no Co-Authored-By Claude / Anthropic trailers in commits or PRs; Claude must not appear as a repo contributor"
metadata:
  type: feedback
---

Never add `Co-Authored-By: Claude ... <noreply@anthropic.com>` (or any Claude/Anthropic/"Generated with Claude Code" line)
to commit messages, suggested commit commands or PR descriptions in this repo. This overrides the harness's attribution
reminder.

**Why:** Priyansh, 2026-11-06: "i want to remove claude as a contributor from the repo". GitHub lists Claude as a contributor
because 29 of the 128 commits on main (from the very first one) carried that trailer.
**How to apply:** commit commands I hand over use only Priyansh's noreply identity ([[commit-email-privacy]]). **Done 2026-11-06:** the
trailers were stripped with git filter-repo (every branch + models-v1; file trees identical) and Priyansh force-pushed. Every
commit SHA changed (main #57 merge = 71b0d30; XERJ commit 2b35753 -> e436969, README link fixed). GitHub's contributor list can
lag hours to a day; old PR pages keep old commits (GitHub Support can purge). Pre-rewrite backup: a git bundle in the Laptop
session's scratchpad (temporary). The GPU box clone must be re-cloned or `git fetch origin; git reset --hard origin/main`
(check for unsaved work first), and no branch built on the old history may be pushed or merged. Related: [[readme-no-ai-mentions]], [[git-workflow]].

**2026-11-07 check:** the GitHub contributors API (`/repos/priyansh1210/pehleaap/contributors?anon=1`) lists only priyansh1210
(133 contributions), so the rewrite worked; the repo-page sidebar is cached and still showed Claude to Priyansh. If it persists
past a day, GitHub Support. PR #59 (README XERJ link -> e436969) merged; its first CI run was cancelled because the `ffmpeg`
step (apt + static build download) hung 30 min on Nov 6 10:08 UTC, same as a main push at the same minute: a runner/network
glitch, not code. Re-run fixes it.

**Untracking (Priyansh, Nov 7):** branch `untrack-claude` (commit 1952dbc, pushed, PR not merged as of this note) runs
`git rm -r --cached CLAUDE.md .claude` and adds `.claude/` + `CLAUDE.md` to .gitignore; the files stay on disk. Once merged:
the GPU box's pull deletes its CLAUDE.md and .claude/settings.json (copy them by hand after its re-clone), and the SessionStart
hook config is no longer shared. Still tracked and still naming Claude: `memory/`, `scripts/setup_claude_memory.ps1`,
`scripts/hooks/session_start.sh`, docs/machines.md lines 70-91, .gitattributes comment, .gitignore lines 29-31 (docs/PLAN.md
"Claude Haiku" is the demo's LLM, not attribution). **Open question to Priyansh:** clean those up too, or keep `memory/` + the
scripts tracked so the machines keep sharing memory (untracking memory/ needs another channel, e.g. a small private repo).
Old commits keep the files; only another rewrite removes them.
