"""HARN-05 / SLM-545: reversal generation + bidirectional validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harnesses.reasoning.revmath.plugins import (
    ReversalPlugin,
    default_plugin_registry,
    load_reversal_meta_for_task,
    resolve_plugin,
    reversal_plugin_for_obligation,
)
from slm_training.harnesses.reasoning.revmath.reversal import (
    audit_hidden_stronger,
    evaluate_reversal,
    generate_reversal_obligations,
    materialize_obligation_task,
    parse_reversal_meta,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
    parse_revmath_task,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURES = (
    REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"
)


def _load_pair(stem: str) -> tuple[RevmathTaskV1, object]:
    task = parse_revmath_task(
        json.loads((FIXTURES / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_reversal_meta(
        json.loads((FIXTURES / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def _run_reversal(task: RevmathTaskV1, meta: object):
    def run_obligation(baseline: RevmathTaskV1, obligation):
        obl_task = materialize_obligation_task(baseline, meta, obligation)  # type: ignore[arg-type]
        plugin = reversal_plugin_for_obligation(meta, obligation)  # type: ignore[arg-type]
        registry = default_plugin_registry()
        registry["reversal"] = plugin
        return run_revmath_task(obl_task, hermetic=True, plugin_registry=registry)

    return evaluate_reversal(task, meta, run_obligation=run_obligation)  # type: ignore[arg-type]


def test_generate_obligations_deterministic_both_directions() -> None:
    task, meta = _load_pair("reversal_equivalence")
    first = generate_reversal_obligations(task, meta)  # type: ignore[arg-type]
    second = generate_reversal_obligations(task, meta)  # type: ignore[arg-type]
    assert [o.to_dict() for o in first] == [o.to_dict() for o in second]
    forward, reverse = first
    assert forward.direction == "forward"
    assert reverse.direction == "reverse"
    assert forward.consequent_statement_id == "theorem.T"
    assert reverse.consequent_statement_id == "principle.A"
    assert "principle.A" in forward.antecedent_assumption_ids
    assert "theorem.T" in reverse.antecedent_assumption_ids
    assert forward.interpretation_status == reverse.interpretation_status == (
        task.base_theory.interpretation_status
    )
    assert forward.base_theory_id == reverse.base_theory_id == "RCA0"


def test_positive_equivalence_requires_both_directions() -> None:
    task, meta = _load_pair("reversal_equivalence")
    report = _run_reversal(task, meta)
    assert report.equivalence_label == "equivalent"
    assert report.claims_equivalence is True
    assert report.forward.outcome == "witnessed"
    assert report.reverse.outcome == "witnessed"
    assert report.forward.obligation.coding_id == report.reverse.obligation.coding_id
    assert report.interpretation_status == "explicit_reversal"


def test_one_way_forward_never_labeled_equivalent() -> None:
    task, meta = _load_pair("reversal_one_way_forward")
    report = _run_reversal(task, meta)
    assert report.forward.outcome == "witnessed"
    assert report.reverse.outcome == "refuted"
    assert report.equivalence_label == "one_way_forward"
    assert report.claims_equivalence is False


def test_timeout_records_unknown_not_equivalence() -> None:
    task, meta = _load_pair("reversal_timeout")
    report = _run_reversal(task, meta)
    assert report.forward.outcome == "unknown"
    assert report.reverse.outcome == "unknown"
    assert report.equivalence_label == "unknown"
    assert report.claims_equivalence is False


def test_hidden_stronger_invalidates_equivalence() -> None:
    task, meta = _load_pair("reversal_hidden_stronger")
    report = _run_reversal(task, meta)
    assert report.forward.outcome == "witnessed"
    assert report.reverse.outcome == "invalid"
    assert "lemma.hidden_stronger_principle" in report.reverse.hidden_stronger_assumptions
    assert report.equivalence_label == "invalid"
    assert report.claims_equivalence is False


def test_strength_mismatch_never_equivalent() -> None:
    task, meta = _load_pair("reversal_strength_mismatch")
    report = _run_reversal(task, meta)
    assert report.equivalence_label == "invalid"
    assert report.claims_equivalence is False
    assert any("strength_mismatch" in r for r in (
        report.forward.rejection_reasons + report.reverse.rejection_reasons
    ))


def test_changed_proposition_rejected_at_meta_match() -> None:
    task, meta = _load_pair("reversal_equivalence")
    # Mutate claim digest on meta so assert_meta_matches_task fails closed.
    broken = meta.to_dict()  # type: ignore[union-attr]
    broken["claim_statement_sha256"] = "0" * 64
    broken_meta = parse_reversal_meta(broken)
    with pytest.raises(RevmathSchemaError, match="changed proposition"):
        generate_reversal_obligations(task, broken_meta)


def test_interpretation_status_must_be_explicit() -> None:
    task, meta = _load_pair("reversal_equivalence")
    raw = meta.to_dict()  # type: ignore[union-attr]
    raw["interpretation_status"] = "uninterpreted"
    with pytest.raises(RevmathSchemaError, match="explicit_reversal"):
        parse_reversal_meta(raw)


def test_rca0_task_rejects_uninterpreted_status() -> None:
    task, _meta = _load_pair("reversal_equivalence")
    payload = json.loads(
        (FIXTURES / "reversal_equivalence.task.json").read_text(encoding="utf-8")
    )
    payload["base_theory"]["interpretation_status"] = "uninterpreted"
    with pytest.raises(Exception, match="explicit_reversal|explicit_interpretation"):
        parse_revmath_task(payload)


def test_audit_hidden_stronger_unit() -> None:
    hits = audit_hidden_stronger(
        direction="reverse",
        proof_dependency_ids=("lemma.hidden_stronger_principle", "ax.ISigma1"),
        import_edges={"lemma.hidden_stronger_principle": ("principle.A",)},
        strength_bearing_lemmas=("lemma.hidden_stronger_principle",),
        forbidden_assumption_ids=("principle.A",),
    )
    assert hits == ("lemma.hidden_stronger_principle",)
    assert (
        audit_hidden_stronger(
            direction="reverse",
            proof_dependency_ids=("ax.theorem.T",),
            import_edges={"ax.theorem.T": ("theorem.T",)},
            strength_bearing_lemmas=("ax.theorem.T",),
            forbidden_assumption_ids=("principle.A",),
        )
        == ()
    )


def test_plugin_registered_and_single_direction_run() -> None:
    task, meta = _load_pair("reversal_equivalence")
    plugin = resolve_plugin(task)
    assert isinstance(plugin, ReversalPlugin)
    forward, _reverse = generate_reversal_obligations(task, meta)  # type: ignore[arg-type]
    obl_task = materialize_obligation_task(task, meta, forward)  # type: ignore[arg-type]
    bound = reversal_plugin_for_obligation(meta, forward)  # type: ignore[arg-type]
    registry = default_plugin_registry()
    registry["reversal"] = bound
    record = run_revmath_task(obl_task, hermetic=True, plugin_registry=registry)
    assert record.result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    loaded = load_reversal_meta_for_task(task)
    assert loaded is not None
    assert loaded.to_dict() == meta.to_dict()  # type: ignore[union-attr]


def test_unsupported_scenario_stays_unknown() -> None:
    task, meta = _load_pair("reversal_equivalence")
    raw = meta.to_dict()  # type: ignore[union-attr]
    raw["hermetic_scenario"] = "unsupported"
    raw["oracle_by_direction"] = {"forward": "unsupported", "reverse": "unsupported"}
    unsupported_meta = parse_reversal_meta(raw)
    report = _run_reversal(task, unsupported_meta)
    assert report.equivalence_label == "unknown"
    assert report.forward.outcome == "unknown"
    assert report.reverse.outcome == "unknown"
