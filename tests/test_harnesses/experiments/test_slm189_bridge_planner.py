"""Tests for SLM-189 (FFE2-01) bridge planner fixture harness."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm189_bridge_planner import (
    ARM_NAMES,
    MATRIX_SET,
    MATRIX_VERSION,
    BridgePlannerManifest,
    build_exact_fixture_targets,
    build_synthetic_scale_targets,
    render_markdown,
    run_bridge_planner_fixture,
    validate_manifest,
)


def test_build_exact_fixture_targets_are_valid() -> None:
    targets = build_exact_fixture_targets()
    assert len(targets) >= 5
    for target_id, program, slots in targets:
        assert target_id
        assert program
        assert slots


def test_build_synthetic_scale_targets_are_valid() -> None:
    targets = build_synthetic_scale_targets()
    assert len(targets) >= 3
    for target_id, program, slots in targets:
        assert target_id.startswith("scale_")
        assert program
        assert slots


def test_run_fixture_matrix_small_grid(tmp_path: Path) -> None:
    manifest = run_bridge_planner_fixture(
        output_dir=tmp_path,
        arms=("canonical_greedy", "dependency_dag"),
        source_policies=("minimal",),
        exact_budget=8,
        write_design_docs=True,
        design_json=tmp_path / "design.json",
        design_md=tmp_path / "design.md",
    )
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    assert {arm.arm_name for arm in manifest.arms} == {"canonical_greedy", "dependency_dag"}
    assert manifest.n_cases > 0
    assert manifest.n_reached > 0

    errors = validate_manifest(manifest)
    assert not errors


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = run_bridge_planner_fixture(
        output_dir=tmp_path,
        arms=("canonical_greedy",),
        source_policies=("minimal",),
    )
    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    loaded = BridgePlannerManifest.from_dict(
        __import__("json").loads(path.read_text(encoding="utf-8"))
    )
    assert loaded.n_cases == manifest.n_cases
    assert {arm.arm_name for arm in loaded.arms} == {arm.arm_name for arm in manifest.arms}


def test_render_markdown_contains_caveats(tmp_path: Path) -> None:
    manifest = run_bridge_planner_fixture(
        output_dir=tmp_path,
        arms=("canonical_greedy", "dependency_dag"),
        source_policies=("minimal",),
    )
    md = render_markdown(manifest)
    assert "SLM-189" in md
    assert "No-go for promotion" in md
    assert "wiring / fixture only" in md
    assert manifest.arms[0].arm_name in md
    for caveat in manifest.honest_caveats:
        assert caveat in md


def test_validate_manifest_catches_duplicate_case(tmp_path: Path) -> None:
    manifest = run_bridge_planner_fixture(
        output_dir=tmp_path,
        arms=("canonical_greedy",),
        source_policies=("minimal",),
    )
    cases = list(manifest.cases)
    if cases:
        cases.append(cases[0])
    bad = BridgePlannerManifest(
        schema=manifest.schema,
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        status=manifest.status,
        claim_class=manifest.claim_class,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        arms=manifest.arms,
        cases=tuple(cases),
        n_cases=len(cases),
        n_reached=manifest.n_reached,
        source_policies=manifest.source_policies,
        version_stamp=manifest.version_stamp,
        timestamp=manifest.timestamp,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        honest_caveats=manifest.honest_caveats,
    )
    errors = validate_manifest(bad)
    assert any("duplicate case_id" in e for e in errors)


def test_all_arm_names_present() -> None:
    assert "canonical_greedy" in ARM_NAMES
    assert "exact_shortest" in ARM_NAMES
    assert "random_shortest" in ARM_NAMES
    assert "dependency_dag" in ARM_NAMES
    assert "contract_first" in ARM_NAMES
    assert "source_adaptive" in ARM_NAMES
    assert "solver_guided" in ARM_NAMES
