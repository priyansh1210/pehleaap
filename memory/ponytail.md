---
name: ponytail
description: "Priyansh's rule (2026-11-03): use the ponytail skill (level full) for all development, and a ponytail-review of each change before handing it over"
metadata:
  type: feedback
---

Use the ponytail skill, level full, for every change in this repo: the smallest complete change, reuse what the repo already
has, no options or abstractions nobody asked for, a small test for new non-trivial logic, `shortcut: <limit>, <when to upgrade>`
comments for known limits, and a reply that ends with what was skipped or not checked. Before each PR, run a ponytail-review
of the diff (correct, safe, holds under load, tested, fast, lean; findings numbered, each with a concrete failing case).

**Why:** Priyansh, 2026-11-03: "go for all the updates one by one make sure you must use ponytail skill I will be reviewing
every thing".
**How to apply:** the skill is not installed in this repo. Its files live in Priyansh's other project, in
`.claude/skills/ponytail*/SKILL.md` (MIT, github.com/DietrichGebert/ponytail); read them there and follow them by hand, or
install the plugin. One update per PR ([[git-workflow]]).
