"""Tests for the SLM-212 constraint-debt routing fixture harness."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm212_debt_routing import (
    ARM_NAMES,
    MATRIX_SET,
    DebtRoutingMatrixManifest,
    build_matrix_manifest,
    build_synthetic_routing_examples,
    render_markdown,
    run_fixture_matrix,
    validate_manifest,
)


def test_build_synthetic_examples_no_torch() -> None:
    examples = build_synthetic_routing_examples(n_examples=50, seed=1)
    assert len(examples) == 50
    for ex in examples:
        assert ex.true_best_route in {"maskgit", "ltr", "asap"}
        assert ex.signal_value >= 0.0
        assert ex.true_best_route in ex.outcome_scores
        assert max(ex.outcome_scores.values()) == ex.outcome_scores[ex.true_best_route]


def test_build_matrix_manifest_runs_all_arms(tmp_path: Path) -> None:
    manifest = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=100, seed=2),
        signal_name="D_legal",
        threshold_high=2.0,
        threshold_low=0.5,
        hysteresis=1,
        budget_mode="equal_verifier_budget",
        run_id="test-run",
        tmp_path=tmp_path,
    )
    assert manifest.matrix_set == MATRIX_SET
    assert {arm.arm_name for arm in manifest.arms} == set(ARM_NAMES)
    assert manifest.n_examples == 100
    assert manifest.lineage.get("calibrator_hash")
    assert manifest.lineage.get("calibration_error") is None

    errors = validate_manifest(manifest)
    assert not errors

    # Oracle should be at least as good as every fixed / learned arm.
    by_name = {arm.arm_name: arm for arm in manifest.arms}
    oracle = by_name["oracle_router_ceiling"]
    for name in ("fixed_maskgit", "fixed_ltr", "fixed_asap", "static_debt_router"):
        assert oracle.accuracy >= by_name[name].accuracy - 1e-6
        assert oracle.mean_regret <= by_name[name].mean_regret + 1e-6

    # Static router should exploit the signal: better than the signal-permuted control.
    static = by_name["static_debt_router"]
    permuted = by_name["signal_permuted_router"]
    assert static.accuracy > permuted.accuracy
    assert static.mean_regret < permuted.mean_regret


def test_render_markdown_contains_caveats(tmp_path: Path) -> None:
    manifest = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=20, seed=3),
        tmp_path=tmp_path,
    )
    md = render_markdown(manifest)
    assert "SLM-212" in md
    assert "No-go for promotion" in md
    assert "wiring / fixture only" in md
    assert manifest.arms[0].arm_name in md


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=20, seed=4),
        tmp_path=tmp_path,
    )
    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    loaded = DebtRoutingMatrixManifest.from_dict(
        __import__("json").loads(path.read_text(encoding="utf-8"))
    )
    assert loaded.n_examples == manifest.n_examples
    assert {arm.arm_name for arm in loaded.arms} == {arm.arm_name for arm in manifest.arms}


def test_run_fixture_matrix_writes_docs(tmp_path: Path) -> None:
    manifest = run_fixture_matrix(
        output_dir=tmp_path,
        n_examples=30,
        write_design_docs=True,
        design_json=tmp_path / "design.json",
        design_md=tmp_path / "design.md",
    )
    assert (tmp_path / "slm212_debt_routing_report.json").is_file()
    assert (tmp_path / "design.json").is_file()
    assert (tmp_path / "design.md").is_file()
    assert manifest.status == "fixture"


def test_validate_manifest_catches_duplicate_arm(tmp_path: Path) -> None:
    manifest = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=20, seed=5),
        tmp_path=tmp_path,
    )
    arms = list(manifest.arms)
    arms[1] = arms[0]  # create duplicate
    data = manifest.to_dict()
    data.pop("arms")
    bad = DebtRoutingMatrixManifest(**data, arms=tuple(arms))
    errors = validate_manifest(bad)
    assert any("duplicate arm" in e for e in errors)


def test_hysteresis_reduces_switches(tmp_path: Path) -> None:
    manifest_h1 = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=100, seed=6),
        hysteresis=1,
        tmp_path=tmp_path / "h1",
    )
    manifest_h3 = build_matrix_manifest(
        build_synthetic_routing_examples(n_examples=100, seed=6),
        hysteresis=3,
        tmp_path=tmp_path / "h3",
    )
    static_h1 = next(a for a in manifest_h1.arms if a.arm_name == "static_debt_router")
    static_h3 = next(a for a in manifest_h3.arms if a.arm_name == "static_debt_router")
    # More hysteresis generally reduces route switching; assert it does not increase.
    h1_entropy = _route_entropy(static_h1.route_counts)
    h3_entropy = _route_entropy(static_h3.route_counts)
    assert h3_entropy <= h1_entropy + 0.1


def _route_entropy(counts: dict[str, int]) -> float:
    import math

    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p:
            entropy -= p * math.log(p)
    return entropy
