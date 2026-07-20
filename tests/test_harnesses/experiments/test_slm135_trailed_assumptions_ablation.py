"""Tests for slm_training.harnesses.experiments.slm135_trailed_assumptions_ablation (SLM-135)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm135_trailed_assumptions_ablation import (
    MATRIX_SET,
    MATRIX_VERSION,
    AblationPolicy,
    build_ablation_fixture,
    build_manifest,
    render_markdown,
    run_ablation_search,
    run_fixture_matrix,
)


def test_build_manifest_has_all_arms() -> None:
    manifest = build_manifest()
    ids = {arm.arm_id for arm in manifest.arms}
    assert ids == {"trail", "certified_only", "monotone", "partial"}
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.status == "not_run"
    assert manifest.claim_class == "wiring"


def test_build_manifest_can_omit_unsafe() -> None:
    manifest = build_manifest(include_unsafe=False)
    ids = {arm.arm_id for arm in manifest.arms}
    assert ids == {"trail", "certified_only"}


def test_certified_trail_solves_with_exact_restoration() -> None:
    start = build_ablation_fixture()
    result = run_ablation_search(start, AblationPolicy.CERTIFIED_TRAIL)
    assert result.status == "solved"
    assert result.terminal == (("a2", "c2"),)
    assert not result.false_prune
    assert not result.unknown_violation
    assert len(result.leaked_deductions) == 0
    assert result.restored_fingerprint == start.fingerprint
    assert result.backtracks == 1


def test_certified_trail_solves_with_reverse_order() -> None:
    start = build_ablation_fixture()
    result = run_ablation_search(
        start, AblationPolicy.CERTIFIED_TRAIL, ranker_order=("a2", "a1")
    )
    assert result.status == "solved"
    assert result.terminal == (("a2", "c2"),)
    assert not result.false_prune
    assert len(result.leaked_deductions) == 0


def test_monotone_proposal_false_prunes() -> None:
    start = build_ablation_fixture()
    result = run_ablation_search(
        start, AblationPolicy.MONOTONE_PROPOSAL, ranker_order=("a1", "a2")
    )
    assert result.status == "certified_unsat"
    assert result.false_prune
    assert result.unknown_violation
    assert len(result.leaked_deductions) == 1


def test_partial_retract_false_prunes() -> None:
    start = build_ablation_fixture()
    result = run_ablation_search(
        start, AblationPolicy.PARTIAL_RETRACT, ranker_order=("a1", "a2")
    )
    assert result.status == "certified_unsat"
    assert result.false_prune
    assert result.unknown_violation
    assert len(result.leaked_deductions) == 1


def test_certified_only_no_branch_is_unknown() -> None:
    start = build_ablation_fixture()
    result = run_ablation_search(start, AblationPolicy.CERTIFIED_ONLY_NO_BRANCH)
    assert result.status == "unknown"
    assert not result.false_prune
    assert not result.unknown_violation
    assert len(result.deductions) == 0
    assert len(result.decisions) == 0


def test_run_fixture_matrix(tmp_path: Path) -> None:
    manifest = build_manifest(seeds=(0,))
    report = run_fixture_matrix(
        manifest, run_id="test_slm135", output_dir=tmp_path
    )
    assert report.status == "fixture"
    assert report.verdict == "trail_required"
    assert report.run_id == "test_slm135"
    assert len(report.rows) == 4
    assert report.version_stamp
    assert (tmp_path / "slm135_trailed_assumptions_report.json").exists()


def test_run_fixture_matrix_without_unsafe_is_certified_only_safe(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(seeds=(0,), include_unsafe=False)
    report = run_fixture_matrix(
        manifest, run_id="test_slm135_safe", output_dir=tmp_path
    )
    assert report.status == "fixture"
    assert report.verdict == "certified_only_already_safe"


def test_render_markdown_includes_caveat_and_verdict() -> None:
    manifest = build_manifest(seeds=(0,))
    report = run_fixture_matrix(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-135" in md
    assert "trail_required" in md
    assert "wiring-only evidence" in md
    assert "certified_trail" in md
    assert "monotone_proposal" in md


def test_report_to_dict_has_expected_keys() -> None:
    manifest = build_manifest(seeds=(0,))
    report = run_fixture_matrix(manifest, run_id="rt_test")
    data = report.to_dict()
    assert data["run_id"] == "rt_test"
    assert data["matrix_version"] == MATRIX_VERSION
    assert data["matrix_set"] == MATRIX_SET
    assert "version_stamp" in data
    assert "verdict" in data
