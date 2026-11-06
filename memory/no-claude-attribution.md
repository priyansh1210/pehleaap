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
