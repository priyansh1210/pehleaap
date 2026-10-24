# pehleaap: notes for Claude Code

Turn-taking (end-of-turn, barge-in vs backchannel, hold intent) for Hinglish voice agents on 8 kHz phone lines:
a training recipe plus a benchmark. Plan: `docs/PLAN.md`. Machines and runbooks: `docs/machines.md`.

## Shared memory (both machines)

The repo is the only thing shared between machines (the Dev laptop; the RTX 4000 Ada GPU box until Oct 28, 2026), so project memory lives in git,
in `memory/`. Its index is imported here:

@memory/MEMORY.md

- Save every project memory (status, decisions, results, how the user wants me to work, gotchas) as a file in `memory/`
  with a line in `memory/MEMORY.md`, never only in a machine-local `~/.claude/projects/.../memory/` folder.
- Each machine runs `powershell -ExecutionPolicy Bypass -File scripts\setup_claude_memory.ps1` once. It points auto memory
  at `memory/` through the gitignored `.claude/settings.local.json`, so the built-in memory reads and writes this folder.
- The SessionStart hook pulls from GitHub. If it reports a failed pull, tell the user memory may be stale.
- Memory reaches the other machine only after a commit and push. Include `memory/` changes in the next commit, and remind the
  user to commit and push before switching machines.
- The repo is **public**: no personal data (real names beyond the GitHub handle, emails, phone numbers), secrets, or
  recording-participant details in memory or anywhere else in git. Label machine-specific facts (Dev box / GPU box).

## Working rules

- The user commits, pushes and raises PRs from copy-paste PowerShell commands. Don't commit or push unless asked.
- Develop on the Dev box. Training, batch ASR and full evaluations run on Kaggle (plan §3a); the GPU box only runs the
  baseline eval, until Oct 28, 2026.
- No real customer calls ever; fake PII only; adult, consenting speakers.
- Tests: `.venv\Scripts\python -m pytest` (research stack); `.venv-demo\Scripts\python -m pytest tests\test_pipecat_ext.py` (Pipecat).
- Windows: entry points that print non-ASCII call `sys.stdout.reconfigure(encoding="utf-8")` (pipes default to cp1252).
