"""Guarded, reversible quarantine for foreign dirt on a dedicated loop worktree.

``foreign_dirty_tree`` is a correct fail-closed default: on an ordinary
checkout, dirty paths may be live human WIP. On the documented deployment — a
dedicated continuous-loop worktree — foreign dirt is almost always leftover
agent WIP or crashed-process droppings, and today it spins the supervisor
forever (FP5).

Double guard (skeptic O6, binding):

1. The playbook refuses unless the operator has placed a
   ``loops/<loop_id>/WORKTREE_DEDICATED`` sentinel whose content matches the
   worktree's resolved git dir — an explicit, per-worktree human opt-in.
2. It refuses during any in-progress merge / rebase / cherry-pick (those are
   healed or hard-blocked elsewhere; stashing mid-operation corrupts state).

The stash is never dropped by any playbook: the receipt records the **stash
commit SHA** (stable identity, unlike ``stash@{0}``) plus the exact restore
command, and the escalation record cross-links it so the next agent can find
the quarantined evidence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealPlanV1,
    HealStepV1,
    HealVerifyV1,
)

PLAYBOOK_ID = "quarantine_dirt/v1"

SENTINEL_FILENAME = "WORKTREE_DEDICATED"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def worktree_sentinel_path(root: Path, loop_id: str) -> Path:
    return Path(root) / "loops" / loop_id / SENTINEL_FILENAME


def sentinel_authorizes(root: Path, loop_id: str, cwd: Path) -> bool:
    """True only when the operator sentinel names this exact worktree."""
    sentinel = worktree_sentinel_path(root, loop_id)
    if not sentinel.is_file():
        return False
    try:
        recorded = sentinel.read_text(encoding="utf-8").strip()
        actual = _git(cwd, "rev-parse", "--absolute-git-dir").stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(recorded) and recorded == actual


def operation_in_progress(cwd: Path) -> bool:
    """Any merge / rebase / cherry-pick / revert in progress — hard refusal."""
    try:
        git_dir = Path(_git(cwd, "rev-parse", "--absolute-git-dir").stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True
    if not git_dir.is_dir():
        return True
    markers = (
        "MERGE_HEAD",
        "REBASE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
    )
    return any((git_dir / name).exists() for name in markers)


class _QuarantineDirtPlaybook:
    playbook_id = PLAYBOOK_ID
    handles = frozenset({"dirty_tree"})

    def matches(self, blocker: dict) -> bool:
        return str(blocker.get("kind") or "") == "foreign_dirty_tree"

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        root = blocker.get("_root")
        loop_id = str(blocker.get("_loop_id") or "")
        if root is None or not loop_id:
            return None
        if not sentinel_authorizes(Path(root), loop_id, cwd):
            return None
        if operation_in_progress(cwd):
            return None
        fingerprint = blocker_fingerprint(
            str(blocker.get("kind") or ""), str(blocker.get("reason") or "")
        )
        message = f"heal-quarantine:{loop_id}:{fingerprint[:12]}"
        return HealPlanV1(
            playbook_id=self.playbook_id,
            blocker_fingerprint=fingerprint,
            blocker_class="dirty_tree",
            steps=(
                HealStepV1(
                    step_id="stash_foreign_dirt",
                    argv=(
                        "git",
                        "stash",
                        "push",
                        "--include-untracked",
                        "-m",
                        message,
                    ),
                    cwd="",
                    timeout_seconds=120,
                    # A stash rewrites only git metadata + removes dirt; any
                    # path the runner sees *newly dirtied* afterwards is a
                    # violation.
                    writes_allowed=(".git/",),
                ),
            ),
            # Healed iff the worktree is clean afterwards. The stash commit
            # SHA is recovered by the supervisor from `git rev-parse stash@{0}`
            # immediately after a healed receipt and cross-linked into the
            # escalation note (see run_autotrain_supervisor).
            verify=HealVerifyV1(
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import subprocess, sys; "
                        "out = subprocess.run(['git', 'status', '--porcelain'], "
                        "capture_output=True, text=True, check=True).stdout; "
                        "sys.exit(0 if not out.strip() else 1)"
                    ),
                ),
                cwd="",
                timeout_seconds=60,
            ),
        )


PLAYBOOK = _QuarantineDirtPlaybook()
