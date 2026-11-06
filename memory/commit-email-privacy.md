---
name: commit-email-privacy
description: "35 commits expose Priyansh's personal Gmail as author (scraped by XERJ outreach); recommended noreply git email; XERJ email needs no reply, not resume-worthy"
metadata:
  node_type: memory
  type: project
  originSessionId: 7e914bc0-a68b-4459-8501-be82eac6c07c
  modified: 2026-10-25T21:06:54.098Z
---

35 commits in this public repo carry Priyansh's personal Gmail as author email (Laptop-made commits); merge commits use the
GitHub noreply address `147494809+priyansh1210@users.noreply.github.com`. Never write the personal address into git.

**Trigger (2026-10-26):** an outreach email from XERJ (real project, xerj-org/xerj, ~15.5k stars) said commit `2b35753` (now `e436969` after the Nov 6 rewrite) was
"selected" for its Corpus Hub. A bot picked it (the 4 Python files / ~153 lines: `scripts/get_weights.py` + tests + demo), and the
email's real ask is "try XERJ". BSD-2 already allows indexing. Laptop's advice: no reply needed (opt-out only; two one-line drafts
given), don't put it on the resume, don't adopt XERJ because of it.

**Recommended (Priyansh hasn't confirmed doing it):** `git config --global user.email "147494809+priyansh1210@users.noreply.github.com"`
on each machine, plus GitHub Settings -> Emails -> "Keep my email addresses private" and "Block command line pushes that expose my
email". (Superseded Nov 6: history was rewritten, see below.) Ask tomorrow whether it's done; if so, the GPU box needs it too.

**2026-10-26, first noreply commit:** Priyansh chose to have the memory-oct26 commit amended to use the noreply address for author
AND committer (both fields expose the address). The global git config on the Laptop still has the Gmail until Priyansh changes it,
so any commit I run myself sets `--author` plus `GIT_COMMITTER_EMAIL` to the noreply address.

**2026-11-06:** Priyansh asked for XERJ in the README anyway; it is now in the README's "Recognition" section (PR #57),
worded only as "selected for the Corpus Hub". The README commit used the noreply address for author and committer.

**2026-11-06, history rewritten (Priyansh's go-ahead):** git filter-repo with a mailmap put all 35 Gmail commits on the noreply
address, in every branch and the models-v1 tag; Priyansh force-pushed. 0 Gmail left in authors, committers, messages or files.
The Laptop's global git config may still hold the Gmail: keep using `--author` + `GIT_COMMITTER_EMAIL` until Priyansh changes it.
