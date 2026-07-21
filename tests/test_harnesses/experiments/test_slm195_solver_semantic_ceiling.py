"""Tests for SLM-195 (FFE3-04) solver-only semantic ceiling harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.dsl.solver.closure import EnumerativeSupportProvider
from slm_training.dsl.solver.state import DomainValue
from slm_training.harnesses.experiments.slm195_solver_semantic_ceiling import (
    ARM_NAMES,
    CanonicalOrderRanker,
    FixtureTerminalChecker,
    OracleRanker,
    RandomRanker,
    SearchWorkEnergyRanker,
    SolverCeilingManifestV1,
    SolverCeilingReport,
    _adapt_fixture,
    build_default_manifest,
    build_reference_fixture,
    run_astar_arm,
    run_beam_arm,
    run_bfs_arm,
    run_ceiling,
    run_dfs_arm,
)


@pytest.fixture
def fx() -> Any:
    return _adapt_fixture(build_reference_fixture())


@pytest.fixture
def valid_manifest(tmp_path: Path) -> SolverCeilingManifestV1:
    return SolverCeilingManifestV1(
        run_id="slm195_test",
        source_commit="a" * 40,
        dirty_tree_ok=True,
        arms=("canonical_dfs", "oracle_order", "bfs_min_edits"),
        budgets=(10, 100),
        fixture_pack_id="vss4-fixture-word",
        fixture_constraint_version="v1",
    )


def _value_by_letter(values: tuple[DomainValue, ...], letter: str) -> DomainValue:
    for value in values:
        if value.payload.get("letter") == letter:
            return value
    raise ValueError(f"letter {letter!r} not found")


def test_manifest_round_trip(valid_manifest: SolverCeilingManifestV1) -> None:
    data = valid_manifest.to_dict()
    restored = SolverCeilingManifestV1.from_dict(data)
    assert restored == valid_manifest


def test_manifest_json_round_trip(
    valid_manifest: SolverCeilingManifestV1, tmp_path: Path
) -> None:
    path = tmp_path / "manifest.json"
    valid_manifest.write_json(path)
    restored = SolverCeilingManifestV1.load_json(path)
    assert restored == valid_manifest


def test_manifest_unknown_fields_dropped(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "solver_ceiling_manifest/v1",
                "run_id": "future",
                "source_commit": "a" * 40,
                "arms": ["canonical_dfs"],
                "budgets": [10],
                "future_field": "ignored",
            }
        ),
        encoding="utf-8",
    )
    manifest = SolverCeilingManifestV1.load_json(path)
    assert manifest.run_id == "future"
    assert "future_field" not in manifest.to_dict()


def test_invalid_arm_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported arms"):
        SolverCeilingManifestV1(arms=("not_an_arm",), budgets=(10,))


def test_invalid_budget_rejected() -> None:
    with pytest.raises(ValueError, match="budgets must be positive ints"):
        SolverCeilingManifestV1(arms=("canonical_dfs",), budgets=(0,))


def test_build_default_manifest_sets_stamp() -> None:
    manifest = build_default_manifest("slm195_default")
    assert manifest.run_id == "slm195_default"
    assert set(manifest.arms) == set(ARM_NAMES)
    assert manifest.version_stamp.get("stamp_schema") == "version_stamp/v1"
    assert "harness.experiments.slm195_solver_only_semantic_ceiling" in (
        manifest.version_stamp.get("components", {})
    )


def test_rankers_return_permutations(fx: Any) -> None:
    values = fx.state.domain(fx.hole_id).values
    rankers = [
        CanonicalOrderRanker(),
        RandomRanker(123),
        OracleRanker(("a", "a")),
        SearchWorkEnergyRanker(expander=fx.expander),
    ]
    for ranker in rankers:
        ranked = ranker.rank(fx.state, fx.hole_id, values)
        assert set(ranked) == set(values)
        assert len(ranked) == len(values)
        assert len(set(ranked)) == len(ranked)


def test_dfs_canonical_solves(fx: Any) -> None:
    provider = EnumerativeSupportProvider(fx.expander, fx.verifier)
    checker = FixtureTerminalChecker(fx.expander, fx.verifier)
    result = run_dfs_arm(
        "canonical_dfs",
        fx.state,
        fx.hole_id,
        provider,
        checker,
        CanonicalOrderRanker(),
        100,
    )
    assert result.status == "SOLVED_ACCEPTED"
    assert result.terminal_program == "aa"


def test_dfs_oracle_solves(fx: Any) -> None:
    provider = EnumerativeSupportProvider(fx.expander, fx.verifier)
    checker = FixtureTerminalChecker(fx.expander, fx.verifier)
    result = run_dfs_arm(
        "oracle_order",
        fx.state,
        fx.hole_id,
        provider,
        checker,
        OracleRanker(("a", "a")),
        100,
    )
    assert result.status == "SOLVED_ACCEPTED"
    assert result.terminal_program == "aa"


def test_dfs_random_not_false_unsat(fx: Any) -> None:
    provider = EnumerativeSupportProvider(fx.expander, fx.verifier)
    checker = FixtureTerminalChecker(fx.expander, fx.verifier)
    result = run_dfs_arm(
        "random_order",
        fx.state,
        fx.hole_id,
        provider,
        checker,
        RandomRanker(12345),
        100,
    )
    assert result.status != "CERTIFIED_UNSAT"


def test_bfs_astar_beam_solve(fx: Any) -> None:
    provider = EnumerativeSupportProvider(fx.expander, fx.verifier)
    checker = FixtureTerminalChecker(fx.expander, fx.verifier)
    for runner in (run_bfs_arm, run_astar_arm, run_beam_arm):
        result = runner(fx.state, provider, checker, 100)
        assert result.status == "SOLVED_ACCEPTED", f"{runner.__name__} failed"
        assert result.terminal_program == "aa"


def test_terminal_checker_accepts_aa_rejects_bb(fx: Any) -> None:
    checker = FixtureTerminalChecker(fx.expander, fx.verifier)
    values = fx.state.domain(fx.hole_id).values
    a = _value_by_letter(values, "a")
    b = _value_by_letter(values, "b")

    aa_state = fx.state.with_decision(fx.hole_id, a)
    outcome = checker.check(aa_state)
    assert outcome.accepted is True
    assert outcome.source == "aa"

    bb_state = fx.state.with_decision(fx.hole_id, b)
    outcome = checker.check(bb_state)
    assert outcome.accepted is False


def test_report_round_trip(tmp_path: Path) -> None:
    report = SolverCeilingReport(run_id="rt")
    path = tmp_path / "report.json"
    report.write_json(path)
    restored = SolverCeilingReport.load_json(path)
    assert restored.run_id == report.run_id


def test_run_ceiling_writes_design_docs(fx: Any, tmp_path: Path, monkeypatch: Any) -> None:
    manifest = build_default_manifest(
        "slm195_ceiling_test",
        arms=("canonical_dfs", "bfs_min_edits"),
        budgets=(10,),
    )
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "source_commit": "a" * 40, "dirty_tree_ok": True}
    )

    design_dir = tmp_path / "design"

    def fake_write(report: SolverCeilingReport) -> None:
        design_dir.mkdir(parents=True, exist_ok=True)
        date = "20260721"
        (design_dir / f"iter-slm195-solver-only-semantic-ceiling-{date}.json").write_text(
            report.to_json(), encoding="utf-8"
        )
        (design_dir / f"iter-slm195-solver-only-semantic-ceiling-{date}.md").write_text(
            "markdown", encoding="utf-8"
        )

    monkeypatch.setattr(
        "slm_training.harnesses.experiments.slm195_solver_semantic_ceiling._write_design_docs",
        fake_write,
    )
    report = run_ceiling(manifest)
    assert any(r.status == "SOLVED_ACCEPTED" for r in report.rows)
    assert report.disposition == "solver_ceiling_established"
    assert (design_dir / "iter-slm195-solver-only-semantic-ceiling-20260721.json").is_file()
