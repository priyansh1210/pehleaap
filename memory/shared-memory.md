---
name: shared-memory
description: "How project memory is shared between the two machines: this git-tracked folder, per-machine autoMemoryDirectory, SessionStart git pull"
metadata:
  type: feedback
---

Priyansh requires both machines to share the same project memory, and the GitHub repo is the only link between them (2026-10-24).

Mechanism:
- Project memory lives in `memory/` in the repo (this folder), imported by `CLAUDE.md` via `@memory/MEMORY.md`.
- Each machine points Claude Code's auto memory here once with `powershell -ExecutionPolicy Bypass -File scripts\setup_claude_memory.ps1`,
  which writes `autoMemoryDirectory` (that machine's absolute path) into the gitignored `.claude/settings.local.json`.
  It cannot go in the committed `.claude/settings.json`: Claude Code ignores `autoMemoryDirectory` there, and the repo path differs per machine.
- `.claude/settings.json` (committed) has a SessionStart hook (`scripts/hooks/session_start.sh`) that runs `git pull --ff-only`
  and prints the result into the session.

**Why:** Claude Code's default memory (`~/.claude/projects/<path>/memory/`) is machine-local and keyed by path, so the two machines
would otherwise drift apart.
**How to apply:** write every project memory here, never only in a machine-local folder. Memory reaches the other machine only after
commit + push: include `memory/` changes in the next commit, and when Priyansh is about to switch machines, remind them to commit and push.
If the SessionStart hook reports a failed pull, say memory may be stale. The repo is **public**: no personal data (real names beyond the
GitHub handle and Priyansh's first name, emails, phone numbers), secrets, or recording-participant details in memory. Label
machine-specific facts (Laptop / GPU box); names as in [[naming]].

Related: [[two-machine-setup]], [[git-workflow]].
