"""INTEG-04 / SLM-529: opt-in four-axis ledger adapters on harness reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.four_axis_ledger import (
    PROJECTION_FIELD,
    SCHEMA,
    FourAxisLedgerAdapterError,
    attach_four_axis_ledger_projection,
    attach_to_evaluation_report_metadata,
    project_four_axis_ledger,
    validate_harness_four_axis_projection,
)
from slm_training.harnesses.reasoning.revmath.report import build_revmath_report
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
    RevmathResultV1,
    parse_revmath_report,
)
from slm_training.harnesses.train_data.report import build_quality_report
from slm_training.formal.judgment import (
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    ROOT
    / "src/slm_training/resources/harnesses"
    / "integ04_four_axis_ledger_fixtures.v1.json"
)


def _axis(name: str, status: str = "not_claimed", **kwargs):
    if status in ("proved_axis", "refuted_axis"):
        kwargs.setdefault("certificate_sha256", "ab" * 32)
        kwargs.setdefault("theorem_ref", f"Thm.{name}")
        if name == "resource_bounds":
            kwargs.setdefault("bound_ast_id", "bound.finite_search.prefix_tree.v1")
    return AxisCertificateRefV1(axis=name, status=status, **kwargs)


def _analysis(**overrides: str) -> RevmathFourAxisAnalysisV1:
    return RevmathFourAxisAnalysisV1(
        assumption_strength=_axis(
            "assumption_strength",
            overrides.get("assumption_strength", "not_claimed"),
        ),
        computability=_axis(
            "computability",
            overrides.get("computability", "unknown_axis"),
        ),
        resource_bounds=_axis(
            "resource_bounds",
            overrides.get("resource_bounds", "not_claimed"),
        ),
        implementation_refinement=_axis(
            "implementation_refinement",
            overrides.get("implementation_refinement", "empirical_remainder"),
            notes="runtime remainder",
        ),
    )


def _minimal_task_result():
    """Build one frozen task+result pair via existing fixture corpus helper paths."""

    from slm_training.harnesses.reasoning.revmath.schemas import (
        RevmathBaseTheoryV1,
        RevmathBudgetV1,
        RevmathCampaignBindingV1,
        RevmathCorpusIdentityV1,
        RevmathPropositionIdentityV1,
        RevmathTaskV1,
        RevmathVerifierJudgeIdentityV1,
        canonical_hash,
    )
    from slm_training.harness_core.lineage.records import content_sha

    statement = "forall n, n + 0 = n"
    prop = RevmathPropositionIdentityV1(
        proposition_id="p.integ04",
        statement=statement,
        statement_sha256=content_sha(statement),
    )
    corpus = RevmathCorpusIdentityV1(
        corpus_id="corpus.integ04",
        corpus_sha256="11" * 32,
        entry_id="entry.integ04",
    )
    task = RevmathTaskV1(
        task_id="task.integ04",
        task_kind="forward_theorem",
        proposition=prop,
        base_theory=RevmathBaseTheoryV1(
            base_theory_id="RCA0",
            interpretation_status="explicit_interpretation",
        ),
        theorem_direction="forward",
        corpus=corpus,
        budget=RevmathBudgetV1(
            max_wall_seconds=1.0,
            max_tactic_steps=8,
            stopping_rule_ids=("budget",),
        ),
        verifier_judge=RevmathVerifierJudgeIdentityV1(
            verifier_id="v.integ04",
            judge_id="j.integ04",
        ),
        campaign=RevmathCampaignBindingV1(
            campaign_id="camp.integ04",
            experiment_id="exp.integ04",
            campaign_manifest_sha256="22" * 32,
            locked_arm_ids=("arm0",),
        ),
    )
    judgment = SolverJudgmentV1(
        outcome=JudgmentOutcome.UNKNOWN,
        payload=UnknownReason.INCOMPLETE_COVERAGE,
        checked=False,
    )
    analysis = _analysis()
    result = RevmathResultV1(
        result_id="result.integ04",
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=judgment.to_dict(),
        four_axis=analysis,
    )
    assert canonical_hash(task)
    return task, result, analysis


def test_default_off_leaves_report_unchanged() -> None:
    base = {"schema_version": 1, "counts": {"admitted": 0}}
    out = attach_four_axis_ledger_projection(
        base, enabled=False, harness_family="train_data"
    )
    assert out == base
    assert PROJECTION_FIELD not in out


def test_absent_evidence_is_absent_not_failure() -> None:
    proj = project_four_axis_ledger(harness_family="model_build")
    assert proj.logical_strength.presence == "absent"
    assert proj.computability.presence == "absent"
    assert proj.resource_bound.presence == "absent"
    assert proj.refinement_empirical_remainder.presence == "absent"
    assert proj.authority_ceiling is None
    validate_harness_four_axis_projection(proj.to_dict())


def test_cannot_invent_stronger_authority() -> None:
    with pytest.raises(FourAxisLedgerAdapterError, match="cannot invent"):
        project_four_axis_ledger(
            harness_family="train_data",
            analysis=_analysis(),
            claimed_authority_class="checked_semantic",
        )


def test_revmath_report_opt_in_round_trip() -> None:
    task, result, analysis = _minimal_task_result()
    off = build_revmath_report(
        report_id="report.integ04.off",
        tasks=[task],
        results=[result],
    )
    assert off.four_axis_ledger_projection is None
    payload_off = off.model_dump(mode="json")
    assert PROJECTION_FIELD not in payload_off or payload_off[PROJECTION_FIELD] is None
    # Historical-shaped payload without the key still parses.
    raw = {k: v for k, v in payload_off.items() if k != PROJECTION_FIELD}
    parse_revmath_report(raw)

    on = build_revmath_report(
        report_id="report.integ04.on",
        tasks=[task],
        results=[result],
        enable_four_axis_ledger=True,
        four_axis_analysis=analysis,
        formal_authority_class="unchecked",
    )
    assert on.four_axis_ledger_projection is not None
    assert on.four_axis_ledger_projection["schema_version"] == SCHEMA
    assert on.four_axis_ledger_projection["harness_family"] == "reasoning/revmath"
    parse_revmath_report(on.model_dump(mode="json"))


def test_train_data_quality_report_opt_in() -> None:
    off = build_quality_report(
        version="v-integ04",
        profile="strict",
        built_at="2026-08-11T00:00:00Z",
        seed_count=0,
        collected_count=0,
        admitted=[],
        rejections=[],
        source_error_count=0,
        cluster_exposure={},
        per_family=[],
        engines={},
    )
    assert PROJECTION_FIELD not in off

    on = build_quality_report(
        version="v-integ04",
        profile="strict",
        built_at="2026-08-11T00:00:00Z",
        seed_count=0,
        collected_count=0,
        admitted=[],
        rejections=[],
        source_error_count=0,
        cluster_exposure={},
        per_family=[],
        engines={},
        enable_four_axis_ledger=True,
        four_axis_analysis=_analysis(),
        formal_authority_class="unchecked",
    )
    assert PROJECTION_FIELD in on
    assert on[PROJECTION_FIELD]["harness_family"] == "train_data"
    validate_harness_four_axis_projection(on[PROJECTION_FIELD])


def test_model_build_metadata_seam() -> None:
    meta = attach_to_evaluation_report_metadata(
        {"suite": "rico_held"},
        enabled=True,
        analysis=_analysis(computability="not_claimed"),
        computability_class_id="unclassified",
        formal_authority_class="unchecked",
    )
    assert meta[PROJECTION_FIELD]["harness_family"] == "model_build"
    assert meta[PROJECTION_FIELD]["computability_class_id"] == "unclassified"


def test_parity_fixtures_across_families() -> None:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert data["projection_schema"] == SCHEMA
    families = {
        row["harness_family"]
        for row in data["rows"]
        if row["id"].startswith("parity_")
    }
    assert "reasoning/revmath" in families
    assert "train_data" in families
    assert "model_build" in families

    # Shared analysis → equal axis presence across families (parity).
    parity_rows = [row for row in data["rows"] if row["id"].startswith("parity_")]
    presence_sets = []
    for row in parity_rows:
        proj = validate_harness_four_axis_projection(row["projection"])
        presence_sets.append(
            (
                proj.logical_strength.presence,
                proj.computability.presence,
                proj.resource_bound.presence,
                proj.refinement_empirical_remainder.presence,
                proj.authority_ceiling,
            )
        )
        assert PROJECTION_FIELD not in row["report_default_off"]
        assert PROJECTION_FIELD in row["report_enabled"]
        assert row["report_enabled"][PROJECTION_FIELD]["harness_family"] == row[
            "harness_family"
        ]
    assert len(set(presence_sets)) == 1

    absent = next(row for row in data["rows"] if row["id"] == "absent_evidence_train_data")
    ap = validate_harness_four_axis_projection(absent["projection"])
    assert ap.authority_ceiling is None
    assert all(
        slot.presence == "absent"
        for slot in (
            ap.logical_strength,
            ap.computability,
            ap.resource_bound,
            ap.refinement_empirical_remainder,
        )
    )


def test_enabling_projection_does_not_change_solver_judgment() -> None:
    task, result, analysis = _minimal_task_result()
    before = result.solver_judgment().outcome
    _ = build_revmath_report(
        report_id="report.integ04.behavior",
        tasks=[task],
        results=[result],
        enable_four_axis_ledger=True,
        four_axis_analysis=analysis,
    )
    assert result.solver_judgment().outcome is before
    assert before is JudgmentOutcome.UNKNOWN
