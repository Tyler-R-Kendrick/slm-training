"""Tests for the SLM-186 verified-utility audit harness."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm186_verified_utility import (
    MATRIX_SET,
    MATRIX_VERSION,
    VerifiedUtilityAuditManifestV1,
    VerifiedUtilityAuditReport,
    build_default_weight_manifest,
    build_fixture_candidates,
    render_markdown,
    run_verified_utility_audit,
)


def test_build_default_weight_manifest_is_valid() -> None:
    manifest = build_default_weight_manifest()
    assert manifest.validate() == []
    assert manifest.primary_policy == "scalarized"
    assert "binding_aware_meaningful_v2" in manifest.weights
    assert "binding_aware_meaningful_v2" in manifest.permitted_ranges


def test_build_fixture_candidates_has_expected_scenarios() -> None:
    candidates = build_fixture_candidates(seed=0)
    scenarios = {scenario for _, scenario, _ in candidates}
    assert "pareto_dominant" in scenarios
    assert "pareto_dominated" in scenarios
    assert "abstained" in scenarios
    assert "canary" in scenarios


def test_run_fixture_audit_returns_manifest_and_report() -> None:
    manifest, report = run_verified_utility_audit(mode="fixture", seed=0)
    assert manifest.claim_class == "wiring"
    assert manifest.status == "fixture"
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION


def test_report_has_candidates_and_rankings() -> None:
    _, report = run_verified_utility_audit(mode="fixture", seed=0)
    assert report.candidates
    ids = {rec.candidate_id for rec in report.candidates}
    assert ids == set(report.scalar_ranking)
    assert ids == set(report.lexicographic_ranking)


def test_report_has_version_stamp() -> None:
    _, report = run_verified_utility_audit(mode="fixture", seed=0)
    assert report.version_stamp
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "evals.verified_utility" in components
    assert "harness.experiments.slm186_verified_utility" in components
    assert "evals.scoring" in components


def test_run_fixture_audit_is_deterministic() -> None:
    _, report1 = run_verified_utility_audit(mode="fixture", seed=0)
    _, report2 = run_verified_utility_audit(mode="fixture", seed=0)
    assert [rec.candidate_id for rec in report1.candidates] == [
        rec.candidate_id for rec in report2.candidates
    ]
    assert report1.scalar_ranking == report2.scalar_ranking
    assert report1.lexicographic_ranking == report2.lexicographic_ranking


def test_report_writes_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    _, report = run_verified_utility_audit(
        mode="fixture", output_dir=output_dir, seed=0
    )
    path = output_dir / "verified_utility_report.json"
    assert path.is_file()
    restored = VerifiedUtilityAuditReport.from_dict(
        __import__("json").loads(path.read_text(encoding="utf-8"))
    )
    assert restored.run_id == report.run_id
    assert len(restored.candidates) == len(report.candidates)


def test_manifest_to_dict_roundtrip() -> None:
    manifest = VerifiedUtilityAuditManifestV1(run_id="test")
    data = manifest.to_dict()
    restored = VerifiedUtilityAuditManifestV1.from_dict(data)
    assert restored.run_id == manifest.run_id
    assert restored.claim_class == "wiring"


def test_render_markdown_contains_caveats() -> None:
    _, report = run_verified_utility_audit(mode="fixture", seed=0)
    md = render_markdown(report)
    assert "SLM-186" in md
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "Real eval records" in md or "real eval records" in md
