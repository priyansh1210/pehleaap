"""Speaker-disjoint train/dev/test splits of the recorded sessions (plan §4.5), hash-locked once frozen.

A person may record with several partners, so sessions are grouped by shared speakers (connected
components of the speaker graph) and each group goes to one split whole: no speaker is ever in two
splits. Groups with a real-phone (PSTN) session go to test, since the PSTN set is test-only. The rest
are assigned largest first, each to the split furthest below its target share of recorded minutes
(70/10/20), ties broken towards a balanced gender and device mix.

Frozen splits live in benchmark/splits/{train,dev,test}.txt (session ids only: nothing personal)
with lock.json holding each file's sha256, so any later change is detectable. After the freeze a new
session may only join train, and only if none of its speakers is already in dev or test.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pehleaap.paths import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "benchmark" / "splits"
SPLITS = ("train", "dev", "test")
TARGETS = {"train": 0.7, "dev": 0.1, "test": 0.2}


@dataclass
class Session:
    session_id: str
    speakers: tuple[str, str]
    devices: tuple[str, str]
    minutes: float = 25.0
    genders: tuple[str, str] = ("", "")

    @property
    def pstn(self) -> bool:
        return "pstn" in self.devices


@dataclass
class Group:
    sessions: list[Session] = field(default_factory=list)

    @property
    def minutes(self) -> float:
        return sum(s.minutes for s in self.sessions)

    @property
    def key(self) -> str:  # stable tie-break
        return hashlib.md5(",".join(sorted(s.session_id for s in self.sessions)).encode()).hexdigest()


def speaker_groups(sessions: list[Session]) -> list[Group]:
    """Sessions joined by shared speakers (union-find over speaker ids)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in sessions:
        a, b = (find(x) for x in s.speakers)
        parent[a] = b
    groups: dict[str, Group] = {}
    for s in sessions:
        groups.setdefault(find(s.speakers[0]), Group()).sessions.append(s)
    return list(groups.values())


def _mix(sessions: list[Session], attr: str) -> Counter:
    return Counter(v for s in sessions for v in getattr(s, attr) if v)


def assign(sessions: list[Session]) -> dict[str, str]:
    """session_id -> split."""
    out: dict[str, str] = {}
    totals = {k: 0.0 for k in SPLITS}
    members: dict[str, list[Session]] = {k: [] for k in SPLITS}
    groups = speaker_groups(sessions)
    total = sum(g.minutes for g in groups) or 1.0
    for g in [g for g in groups if any(s.pstn for s in g.sessions)]:  # the PSTN set is test-only
        totals["test"] += g.minutes
        members["test"] += g.sessions
        out.update({s.session_id: "test" for s in g.sessions})
    overall = {attr: _mix(sessions, attr) for attr in ("genders", "devices")}

    def imbalance(split: str, g: Group) -> float:
        """How far the split's gender/device mix would be from the overall mix after adding g (0 = same proportions)."""
        score = 0.0
        for attr, mix in overall.items():
            new = _mix(members[split] + g.sessions, attr)
            n_new, n_all = sum(new.values()) or 1, sum(mix.values()) or 1
            score += sum(abs(new[k] / n_new - mix[k] / n_all) for k in mix)
        return score

    for g in sorted((g for g in groups if not any(s.pstn for s in g.sessions)), key=lambda g: (-g.minutes, g.key)):
        need = {k: TARGETS[k] - totals[k] / total for k in SPLITS}
        best = max(SPLITS, key=lambda k: (round(need[k], 3), -imbalance(k, g)))
        totals[best] += g.minutes
        members[best] += g.sessions
        out.update({s.session_id: best for s in g.sessions})
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(assignment: dict[str, str], directory: Path = SPLITS_DIR, date: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ids = sorted(k for k, v in assignment.items() if v == split)
        (directory / f"{split}.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    lock = {"frozen": date, "sha256": {s: _sha(directory / f"{s}.txt") for s in SPLITS}}
    (directory / "lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def load(directory: Path = SPLITS_DIR) -> dict[str, str] | None:
    """session_id -> split from frozen files, or None if not frozen yet. Raises if a file no longer matches its lock."""
    lock_path = directory / "lock.json"
    if not lock_path.exists():
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    out = {}
    for split in SPLITS:
        path = directory / f"{split}.txt"
        if _sha(path) != lock["sha256"][split]:
            raise ValueError(f"{path} changed since the splits were frozen ({lock['frozen']}): the test set must stay fixed")
        out.update({line.strip(): split for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})
    return out


def add_to_train(new: list[Session], frozen: dict[str, str], sessions_by_id: dict[str, Session], directory: Path = SPLITS_DIR) -> tuple[list[str], list[str]]:
    """Append post-freeze sessions to train if none of their speakers is in dev/test; (added, held_back). Re-locks train only."""
    held = {spk for sid, split in frozen.items() if split != "train" and sid in sessions_by_id for spk in sessions_by_id[sid].speakers}
    added = [s.session_id for s in new if not set(s.speakers) & held]
    held_back = [s.session_id for s in new if set(s.speakers) & held]
    if added:
        path = directory / "train.txt"
        ids = sorted(set(path.read_text(encoding="utf-8").split()) | set(added))
        path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
        lock["sha256"]["train"] = _sha(path)
        lock.setdefault("train_additions", []).extend(added)
        (directory / "lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return added, held_back
