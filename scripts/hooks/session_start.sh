#!/usr/bin/env bash
# Claude Code SessionStart hook: bring in the other machine's memory/ notes from GitHub.
# Whatever it prints goes into the session's context. It never fails the session.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
say() { echo "[shared memory] $*"; }

if ! git fetch --quiet origin 2>/dev/null; then
  say "could not reach GitHub (offline?), so memory/ may be stale. Tell the user."
  exit 0
fi

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
  if out=$(git merge --ff-only '@{u}' 2>&1); then
    say "$branch is up to date with its remote ($(printf '%s\n' "$out" | tail -n 1))"
  else
    say "could not fast-forward $branch (local changes or diverged), so memory/ may be stale. Tell the user."
  fi
fi

# Feature branches don't get main's newer memory automatically: point it out.
newer=$(git rev-list --count HEAD..origin/main -- memory CLAUDE.md 2>/dev/null || echo 0)
if [ "${newer:-0}" -gt 0 ]; then
  say "origin/main has $newer memory commit(s) not in $branch. Read them with: git show origin/main:memory/MEMORY.md (or merge main)."
fi
say "$(git status -sb 2>/dev/null | head -n 1)"
exit 0
