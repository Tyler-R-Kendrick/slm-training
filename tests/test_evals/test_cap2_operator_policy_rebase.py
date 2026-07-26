from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.cap2_operator_policy_rebase import (
    Cap2OperatorPolicyRebaseV1,
    DependencyNodeV1,
    InventoryItemV1,
    InventoryStatus,
    IssueRebaseAction,
    IssueRebaseNoteV1,
    build_cap2_operator_policy_rebase,
)

ROOT = Path(__file__).resolve().parents[2]
STAMP = {
    "stamp_schema": "version_stamp/v1",
    "code_commit": "test",
    "code_dirty": True,
    "components": {"evals.cap2_operator_policy_rebase": "v1"},
    "stamped_at": "test",
}
REPRO_NOTE = {
    "attempted": True,
    "reproduced": False,
    "residual_error": "test-only placeholder",
}


def _cap2_report() -> dict:
    return json.loads(
        (
            ROOT
            / "docs/design/dsh3-13-cap2-operator-eval-20260723/report.json"
        ).read_text()
    )


def _build() -> Cap2OperatorPolicyRebaseV1:
    return build_cap2_operator_policy_rebase(
        repo_root=ROOT,
        cap2_report=_cap2_report(),
        reproducibility_note=REPRO_NOTE,
        version_stamp=STAMP,
    )


def test_build_reads_frozen_suite_identity_without_mutating_it() -> None:
    result = _build()
    assert result.pinned_baseline == "5cd5b8b"
    assert result.frozen_suite_identity["suite_version"] == "cap2_operator_v1"
    assert (
        result.frozen_suite_identity["suite_hash"]
        == "16f210786bac7fd5f5edb64d13888c3cc7d634330a81b5065150e7a41fcb1d4d"
    )
    # The frozen report on disk is untouched (adversarial control: never
    # silently mutate the frozen suite).
    on_disk = _cap2_report()
    assert on_disk["suite"]["suite_hash"] == result.frozen_suite_identity["suite_hash"]


def test_inventory_anchors_point_at_real_repo_files_with_matching_digests() -> None:
    result = _build()
    assert len(result.inventory) >= 8
    for item in result.inventory:
        assert item.status is not InventoryStatus.MISSING
        anchor_path = item.code_anchor.split(":", 1)[0]
        assert (ROOT / anchor_path).is_file(), anchor_path
        assert item.source_digest is not None
        assert len(item.source_digest) == 64  # sha256 hex


def test_dependency_map_covers_dsh3_19_through_33_with_no_unknown_alias() -> None:
    result = _build()
    aliases = {node.alias for node in result.dependency_map}
    expected = {f"DSH3-{n}" for n in range(19, 34)}
    assert aliases == expected
    known = aliases | {"DSH3-13", "DSH3-18"}
    for node in result.dependency_map:
        assert set(node.blocked_by) <= known


def test_dsh3_19_and_dsh3_21_are_ready_dsh3_20_is_not() -> None:
    result = _build()
    by_alias = {node.alias: node for node in result.dependency_map}
    assert by_alias["DSH3-19"].ready is True
    assert by_alias["DSH3-19"].blocked_by == ("DSH3-18",)
    assert by_alias["DSH3-21"].ready is True
    assert by_alias["DSH3-21"].status == "in_review"
    # DSH3-20 depends on DSH3-19, which is not yet Done -- not ready even
    # though DSH3-19 has no *unmet* blocker of its own.
    assert by_alias["DSH3-20"].ready is False
    assert by_alias["DSH3-20"].blocked_by == ("DSH3-19",)


def test_dsh3_33_capstone_is_not_ready_until_the_whole_portfolio_lands() -> None:
    result = _build()
    by_alias = {node.alias: node for node in result.dependency_map}
    assert by_alias["DSH3-33"].ready is False
    assert "DSH3-28" in by_alias["DSH3-33"].blocked_by


def test_issue_notes_cover_dsh3_14_15_17_with_valid_actions() -> None:
    result = _build()
    by_alias = {note.alias: note for note in result.issue_notes}
    assert set(by_alias) == {"DSH3-14", "DSH3-15", "DSH3-17"}
    assert by_alias["DSH3-14"].action is IssueRebaseAction.UNCHANGED
    assert by_alias["DSH3-15"].action is IssueRebaseAction.REBASED
    assert by_alias["DSH3-17"].action is IssueRebaseAction.SUPERSEDED
    assert by_alias["DSH3-17"].superseded_by == "DSH3-33"


def test_stop_rule_is_not_triggered_because_no_trained_policy_exists_yet() -> None:
    result = _build()
    assert result.stop_rule_triggered is False
    assert result.stop_rule_reason


def test_to_dict_round_trips_through_json() -> None:
    result = _build()
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["schema"] == "cap2_operator_policy_rebase/v1"
    assert len(payload["dependency_map"]) == 15
    assert len(payload["issue_notes"]) == 3


def test_unsupported_cap2_report_schema_is_rejected() -> None:
    report = _cap2_report()
    report["schema"] = "wrong/v1"
    with pytest.raises(ValueError, match="unsupported frozen CAP2 report"):
        build_cap2_operator_policy_rebase(
            repo_root=ROOT,
            cap2_report=report,
            reproducibility_note=REPRO_NOTE,
            version_stamp=STAMP,
        )


def test_tampered_oracle_gate_pass_is_rejected() -> None:
    report = _cap2_report()
    report["policy_scores"]["oracle"]["gate_pass"] = False
    with pytest.raises(ValueError, match="oracle contract did not pass"):
        build_cap2_operator_policy_rebase(
            repo_root=ROOT,
            cap2_report=report,
            reproducibility_note=REPRO_NOTE,
            version_stamp=STAMP,
        )


def test_dependency_node_rejects_duplicate_blocked_by() -> None:
    with pytest.raises(ValueError, match="duplicate blocked_by"):
        DependencyNodeV1(
            alias="DSH3-99",
            issue_id="SLM-999",
            status="backlog",
            blocked_by=("DSH3-18", "DSH3-18"),
            target_symbols=(),
            ready=True,
        )


def test_dependency_node_ready_must_match_blocker_state() -> None:
    with pytest.raises(ValueError, match="ready="):
        Cap2OperatorPolicyRebaseV1(
            pinned_baseline="5cd5b8b",
            frozen_suite_identity={"suite_hash": "x"},
            reproducibility_note={},
            inventory=(),
            issue_notes=(),
            dependency_map=(
                DependencyNodeV1(
                    alias="DSH3-19",
                    issue_id="SLM-394",
                    status="backlog",
                    blocked_by=("DSH3-18",),
                    target_symbols=(),
                    ready=False,  # wrong: DSH3-18 is external and satisfied
                ),
            ),
            stop_rule_triggered=False,
            stop_rule_reason="reason",
            version_stamp=STAMP,
        )


def test_issue_rebase_note_requires_superseded_by_only_when_superseded() -> None:
    with pytest.raises(ValueError, match="superseded_by"):
        IssueRebaseNoteV1(
            alias="DSH3-14",
            issue_id="SLM-382",
            action=IssueRebaseAction.SUPERSEDED,
            note="missing superseded_by",
        )
    with pytest.raises(ValueError, match="superseded_by"):
        IssueRebaseNoteV1(
            alias="DSH3-14",
            issue_id="SLM-382",
            action=IssueRebaseAction.UNCHANGED,
            note="unexpected superseded_by",
            superseded_by="DSH3-33",
        )


def test_inventory_item_missing_status_cannot_carry_a_digest() -> None:
    with pytest.raises(ValueError, match="cannot carry a source digest"):
        InventoryItemV1(
            symbol_id="does.not.exist",
            status=InventoryStatus.MISSING,
            code_anchor="src/does/not/exist.py",
            note="placeholder",
            source_digest="abc123",
        )


def test_disposition_rejects_unknown_dependency_alias() -> None:
    with pytest.raises(ValueError, match="unknown alias"):
        Cap2OperatorPolicyRebaseV1(
            pinned_baseline="5cd5b8b",
            frozen_suite_identity={"suite_hash": "x"},
            reproducibility_note={},
            inventory=(),
            issue_notes=(),
            dependency_map=(
                DependencyNodeV1(
                    alias="DSH3-19",
                    issue_id="SLM-394",
                    status="backlog",
                    blocked_by=("DSH3-999",),
                    target_symbols=(),
                    ready=False,
                ),
            ),
            stop_rule_triggered=False,
            stop_rule_reason="reason",
            version_stamp=STAMP,
        )
