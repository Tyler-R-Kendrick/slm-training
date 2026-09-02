"""Fail-closed heal primitives: suite-id allocation and verified driver heals."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.autoresearch.heal import load_heal_receipts
from slm_training.autoresearch.heal.escalation import EscalationLedger
from slm_training.autoresearch.heal.fail_closed import (
    allocate_screening_suite_id,
    count_records,
    record_count_probe,
    verify_driver_heal,
)

_PREFIX = "e938_role_safe_all_targets_smoke"


def test_allocate_screening_suite_id_scans_every_version(tmp_path: Path) -> None:
    for name in (f"{_PREFIX}24_v1", f"{_PREFIX}24_v2", f"{_PREFIX}24_v3", f"{_PREFIX}6_v1"):
        (tmp_path / name).mkdir()
    (tmp_path / f"{_PREFIX}24_v9.txt").write_text("not a dir")
    assert allocate_screening_suite_id(tmp_path, 24) == f"{_PREFIX}24_v4"
    assert allocate_screening_suite_id(tmp_path, 6) == f"{_PREFIX}6_v2"
    assert allocate_screening_suite_id(tmp_path, 96) == f"{_PREFIX}96_v1"
    # Gaps are filled, but a published (frozen) id is never handed back.
    (tmp_path / f"{_PREFIX}24_v4").mkdir()
    (tmp_path / f"{_PREFIX}24_v6").mkdir()
    assert allocate_screening_suite_id(tmp_path, 24) == f"{_PREFIX}24_v5"
    for _ in range(3):
        assert not (tmp_path / allocate_screening_suite_id(tmp_path, 24)).exists()


def _jsonl(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(n)))
    return path


def test_verify_driver_heal_failure_is_receipted_and_escalates(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    records = _jsonl(tmp_path / "suite" / "records.jsonl", 24)
    assert count_records(records) == 24
    outcomes = []
    for _ in range(3):
        receipt = verify_driver_heal(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            heal_id="rebuild_screening_eval",
            verify=record_count_probe(records, must_exceed=24),
            cwd=tmp_path,
            counts_before={"smoke_n": 24},
            counts_after={"smoke_n": 24},
            extra_conditions={"must_generate_false": False},
        )
        outcomes.append(receipt.outcome)
    assert outcomes == ["verify_failed"] * 3
    receipts = load_heal_receipts(root, "loop-1")
    assert len(receipts) == 3
    assert all("heal_postcondition_failed" in r.note for r in receipts)
    assert "must_generate_false" in receipts[0].note
    ledger = EscalationLedger.load(root, "loop-1")
    record = next(iter(ledger.records.values()))
    assert record.kind == "heal_postcondition_failed"
    assert record.seen_count == 3
    assert record.status == "escalated"  # BLOCKER_RULES max_attempts=3


def test_verify_driver_heal_success_requires_probe_and_conditions(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    records = _jsonl(tmp_path / "suite" / "records.jsonl", 96)
    healed = verify_driver_heal(
        root=root,
        loop_id="loop-1",
        campaign_id=None,
        heal_id="rebuild_screening_eval",
        verify=record_count_probe(records, must_exceed=24),
        cwd=tmp_path,
        counts_before={"smoke_n": 24},
        counts_after={"smoke_n": 96},
        extra_conditions={"must_generate_false": True},
    )
    assert healed.outcome == "healed"
    assert healed.campaign_id == "unknown"
    assert healed.verify_result is not None and healed.verify_result.returncode == 0
    assert EscalationLedger.load(root, "loop-1").records == {}
    # The probe re-reads disk: a claim of growth with no file growth fails.
    stale = verify_driver_heal(
        root=root,
        loop_id="loop-1",
        campaign_id=None,
        heal_id="rebuild_screening_eval",
        verify=record_count_probe(records, must_exceed=96),
        cwd=tmp_path,
        counts_before={"smoke_n": 96},
        counts_after={"smoke_n": 120},
    )
    assert stale.outcome == "verify_failed"
