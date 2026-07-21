"""Tests for the SLM-231 (SPV0-05) RoleSlot cardinality dead-field
consumption probe."""

from __future__ import annotations

from slm_training.data.progspec.semantic_plan import (
    PlanIdentity,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.harnesses.experiments.slm231_role_cardinality_dead_field import (
    EXPERIMENT_ID,
    MATRIX_SET,
    Slm231Report,
    _derive_sibling_cardinality,
    render_markdown,
    run_fixture_matrix,
)


def _row(report, arm_id):
    return next(r for r in report.rows if r.arm_id == arm_id)


def _plan_with_two_children() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="gold"),
        role_slots=(
            RoleSlot(role_id="role_root", component_family="Stack"),
            RoleSlot(role_id="role_a", component_family="TextContent"),
            RoleSlot(role_id="role_b", component_family="TextContent"),
            RoleSlot(role_id="role_c", component_family="Button"),
        ),
        topology=PlanTopology(
            parent_relation_candidates=(
                {"parent_role_id": "role_root", "child_role_id": "role_a", "relation": "contains"},
                {"parent_role_id": "role_root", "child_role_id": "role_b", "relation": "contains"},
                {"parent_role_id": "role_root", "child_role_id": "role_c", "relation": "contains"},
            )
        ),
    )


def test_derive_sibling_cardinality_counts_same_family_siblings() -> None:
    plan = _plan_with_two_children()
    derived = _derive_sibling_cardinality(plan)
    by_id = {slot.role_id: slot for slot in derived.role_slots}

    # Two TextContent siblings under the same parent -> cardinality (2, 2).
    assert by_id["role_a"].min_cardinality == 2
    assert by_id["role_a"].max_cardinality == 2
    assert by_id["role_b"].min_cardinality == 2
    assert by_id["role_b"].max_cardinality == 2
    # Single Button sibling -> cardinality (1, 1).
    assert by_id["role_c"].min_cardinality == 1
    assert by_id["role_c"].max_cardinality == 1
    # Root has no parent edge -> singleton (1, 1).
    assert by_id["role_root"].min_cardinality == 1
    assert by_id["role_root"].max_cardinality == 1


def test_derive_sibling_cardinality_preserves_other_fields() -> None:
    plan = _plan_with_two_children()
    derived = _derive_sibling_cardinality(plan)
    original_by_id = {slot.role_id: slot for slot in plan.role_slots}
    for slot in derived.role_slots:
        original = original_by_id[slot.role_id]
        assert slot.component_family == original.component_family
        assert slot.role_id == original.role_id
    # Original plan is untouched (frozen model, new object returned).
    assert all(s.min_cardinality is None for s in plan.role_slots)


def test_fixture_runs_both_arms() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == 2
    assert report.gate_hash
    assert report.corpus_size == 19


def test_derivation_populates_non_none_cardinality() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    assert report.derivation_populated_cardinality is True
    with_card = _row(report, "F1_with_cardinality")
    no_card = _row(report, "F0_no_cardinality")
    assert with_card.cardinality_populated is True
    assert no_card.cardinality_populated is False


def test_both_arms_reach_full_seed_validity_matching_slm230_c4() -> None:
    # SLM-230's C4_roles_topology arm reached seed_valid_rate=1.0 and
    # mean_component_coverage=1.0 on this same fixture corpus; both arms
    # here use the same roles+topology substitution and should match.
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    for arm_id in ("F0_no_cardinality", "F1_with_cardinality"):
        row = _row(report, arm_id)
        assert row.seed_valid_rate == 1.0
        assert row.mean_component_coverage == 1.0


def test_no_mismatches_between_arms() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    assert report.mismatches == ()


def test_disposition_confirms_unconsumed() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    assert report.disposition == "cardinality_confirmed_unconsumed"


def test_report_roundtrips_through_dict() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    payload = report.to_dict()
    restored = Slm231Report.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    b = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_fixture_matrix(corpus_size=24, corpus_seed=0)
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | cardinality populated | n |" in text
    assert "No-go for promotion" in text
    assert "F0_no_cardinality" in text
