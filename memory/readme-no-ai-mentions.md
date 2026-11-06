---
name: readme-no-ai-mentions
description: "Priyansh's rule (2026-11-06): README must be professional and appealing, and never mention Claude, Anthropic, AI agents or AI authorship"
metadata:
  type: feedback
---

The README must read as professional and appealing, and must never mention Claude, Claude Code, Anthropic, "written by an
AI agent" or any AI tooling. The 2026-11-06 polish (PR #57) removed the line about Claude Code's project memory living in git.

**Why:** Priyansh, 2026-11-06: "make sure the readme is very professional and appealing and no mentioning claude, written by
ai agent or anthropic or anything in the readme".
**How to apply:** grep README.md for claude/anthropic/AI before handing over any README change. `CLAUDE.md`, `memory/` and
`.claude/` still exist in the repo (visible when browsing files); the rule covers the README text, not those files. Related:
[[git-workflow]], [[pehleaap-status]].
