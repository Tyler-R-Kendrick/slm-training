"""Chaos tests: inject a fault into the heal runner, assert the record is honest.

The self-heal loop is the part of the pipeline whose failure mode is *silence*.
For 543 cycles it reported passes while changing nothing, so the property under
test here is never "the heal succeeded" — it is **"whatever happened, the
receipt says so"**. A heal that crashes, times out, writes outside its
allowlist, or repairs nothing must never leave a `healed` receipt behind.

Each test injects one fault at the seam a real failure would arrive through and
asserts the recorded outcome. `run_playbooks` is documented as never raising —
heal-layer bugs must degrade to a receipt, not take down the supervisor — so
"the runner survived" is itself part of every assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch import heal
from slm_training.autoresearch.heal import load_heal_receipts

pytestmark = pytest.mark.chaos

LOOP = "chaos-loop"
CAMPAIGN = "chaos-campaign"


def _blocker(kind: str = "repair_harness", reason: str = "boom") -> dict:
    return {"campaign_id": CAMPAIGN, "index": 0, "kind": kind, "reason": reason}


def _run(tmp_path: Path, playbook) -> tuple:
    return heal.run_playbooks(
        root=tmp_path,
        loop_id=LOOP,
        campaign_id=CAMPAIGN,
        blockers=[{**_blocker(), "_root": tmp_path, "_loop_id": LOOP}],
        cwd=tmp_path,
        playbooks=(playbook,),
    )


class _Playbook:
    """Configurable playbook; each flag injects one fault."""

    playbook_id = "chaos/v1"
    handles = frozenset({"code"})

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def matches(self, blocker: dict) -> bool:
        return True

    def execute(self, blocker: dict, **kwargs: object):
        self.calls += 1
        if self.mode == "crash":
            raise RuntimeError("playbook exploded mid-heal")
        if self.mode == "exit":
            raise SystemExit(9)
        if self.mode == "hang_forever":
            raise TimeoutError("wall clock exceeded")
        raise AssertionError(f"unreachable mode {self.mode}")

    def plan(self, blocker: dict, *, cwd: Path):
        if self.mode == "plan_crash":
            raise RuntimeError("plan construction exploded")
        if self.mode == "declines":
            return None
        raise AssertionError(f"unreachable mode {self.mode}")


@pytest.mark.parametrize("mode", ["crash", "hang_forever"])
def test_a_playbook_that_dies_mid_heal_records_a_failure_not_a_heal(
    tmp_path: Path, mode: str
) -> None:
    receipts = _run(tmp_path, _Playbook(mode=mode))

    assert [r.outcome for r in receipts] == ["step_failed"]
    assert all(r.outcome != "healed" for r in load_heal_receipts(tmp_path, LOOP))


def test_a_crash_is_persisted_so_a_later_pass_is_not_vacuous(tmp_path: Path) -> None:
    """The receipt must survive to disk.

    A pass that only escalated, leaving no receipt, scores as vacuous — which
    is how the loop spent 543 cycles reporting progress it had not made.
    """
    _run(tmp_path, _Playbook(mode="crash"))

    on_disk = load_heal_receipts(tmp_path, LOOP)
    assert len(on_disk) == 1
    assert on_disk[0].outcome == "step_failed"
    assert on_disk[0].note


def test_a_playbook_whose_plan_explodes_is_a_failed_attempt(tmp_path: Path) -> None:
    playbook = _Playbook(mode="plan_crash")
    playbook.execute = None  # not callable: forces the runner down the plan path
    receipts = _run(tmp_path, playbook)

    assert [r.outcome for r in receipts] == ["step_failed"]


def test_a_declining_playbook_is_unhandled_not_healed(tmp_path: Path) -> None:
    playbook = _Playbook(mode="declines")
    playbook.execute = None
    receipts = _run(tmp_path, playbook)

    assert [r.outcome for r in receipts] == ["unhandled"]


def test_no_matching_playbook_still_leaves_a_receipt(tmp_path: Path) -> None:
    """An escalation without a receipt is invisible to the pass classifier."""
    receipts = heal.run_playbooks(
        root=tmp_path,
        loop_id=LOOP,
        campaign_id=CAMPAIGN,
        blockers=[{**_blocker(), "_root": tmp_path, "_loop_id": LOOP}],
        cwd=tmp_path,
        playbooks=(),
    )

    assert [r.outcome for r in receipts] == ["unhandled"]
    assert load_heal_receipts(tmp_path, LOOP)


def test_the_runner_survives_every_injected_fault(tmp_path: Path) -> None:
    """`run_playbooks` degrades to a receipt; it never raises at the supervisor."""
    for mode in ("crash", "exit", "hang_forever"):
        target = tmp_path / mode
        target.mkdir()
        receipts = heal.run_playbooks(
            root=target,
            loop_id=LOOP,
            campaign_id=CAMPAIGN,
            blockers=[{**_blocker(), "_root": target, "_loop_id": LOOP}],
            cwd=target,
            playbooks=(_Playbook(mode=mode),),
        )
        assert receipts, mode
        assert all(r.outcome != "healed" for r in receipts), mode


def test_repeated_identical_faults_exhaust_the_attempt_budget(tmp_path: Path) -> None:
    """A fault that never resolves must stop being retried forever."""
    outcomes = []
    for _ in range(4):
        receipts = heal.run_playbooks(
            root=tmp_path,
            loop_id=LOOP,
            campaign_id=CAMPAIGN,
            blockers=[{**_blocker(), "_root": tmp_path, "_loop_id": LOOP}],
            cwd=tmp_path,
            playbooks=(_Playbook(mode="crash"),),
            max_attempts_per_fingerprint=2,
        )
        outcomes.append(receipts[0].outcome)

    assert "budget_exhausted" in outcomes, outcomes
    assert "healed" not in outcomes
