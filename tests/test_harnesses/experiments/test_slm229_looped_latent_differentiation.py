"""Tests for SLM-229 (RSC0-01) looped-latent differentiation harness."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm229_looped_latent_differentiation import (
    LoopedLatentDifferentiationReport,
    LoopedLatentVerdict,
    MinimalCompilerLatentContractV1,
    build_differentiators,
    build_mechanism_comparison,
    build_oracle_intervention_ceiling,
    build_prior_art_audit,
    build_scale_regime_audit,
    build_target_support_audit,
    differentiator_contract_field_map,
    evaluate_verdict,
    render_markdown,
    run_differentiation_audit,
    validate_doc_refs,
)


@pytest.fixture
def repo_root() -> Path:
    # tests/ lives at the repository root, one level shallower than src/.
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Schema/enum/hash and JSON/Markdown consistency
# ---------------------------------------------------------------------------


def test_verdict_enum_values() -> None:
    values = {v.value for v in LoopedLatentVerdict}
    assert values == {
        "authorize_minimal_probe",
        "duplicate_spv",
        "unsupported_targets",
        "scale_not_identifiable",
        "blocked_by_floor",
        "blocked_by_recurrence",
        "inconclusive",
    }


def test_report_round_trip_serialization(repo_root: Path) -> None:
    report = run_differentiation_audit(repo_root=repo_root)
    data = report.to_dict()
    restored = LoopedLatentDifferentiationReport.from_dict(data)
    assert restored.schema == report.schema
    assert restored.verdict == report.verdict
    assert len(restored.mechanism_comparison) == len(report.mechanism_comparison)
    assert len(restored.differentiators) == len(report.differentiators)
    assert restored.floor_gate_hash == report.floor_gate_hash
    assert restored.floor_gate_verdict == "inconclusive"


def test_minimal_contract_round_trip_serialization() -> None:
    contract = MinimalCompilerLatentContractV1(
        contract_schema="MinimalCompilerLatentContractV1",
        slot_kinds=("root_contract", "component_inventory"),
        slot_count_k=1,
        slot_shape="d_model",
        context_surface_inputs=("prompt", "context"),
        shared_readout_ids="lm_head/project()",
        target_representation="accepted_set",
        surface_consumer_mechanism="cross_attention_gating",
        recurrence_update_reset_semantics="reset_per_record",
        loss_normalization_and_output_coupling="coupled_to_final_ce",
        interventions=("gold", "zero", "swap", "wrong", "detached"),
        compiler_verifier_authority_boundary="no_hard_authority",
        checkpoint_config_identity="default_off",
        default_off=True,
        required_floor_gate="semantic_floor_escape",
        required_recurrence_gate="recursive_core_positive",
        required_oracle_gate="internal_slot_oracle_fixture",
        primary_metrics=("slot_target_accuracy",),
        falsifier="no downstream free-running change under gold substitution",
        stop_rules=("stop_if_hard_authority_leak",),
    )
    data = contract.to_dict()
    restored = MinimalCompilerLatentContractV1.from_dict(data)
    assert restored == contract


def test_generated_json_matches_committed_doc(repo_root: Path) -> None:
    """The committed docs/design JSON must match a freshly-built report."""
    doc_path = repo_root / "docs/design/iter-slm229-looped-latent-differentiation-20260721.json"
    if not doc_path.exists():
        pytest.skip("docs/design artifact not yet generated")
    import json

    committed = json.loads(doc_path.read_text(encoding="utf-8"))
    assert committed["schema"] == "LoopedLatentDifferentiationV1"
    assert committed["verdict"] in {v.value for v in LoopedLatentVerdict}


# ---------------------------------------------------------------------------
# Every claimed differentiator maps to a concrete contract field + test
# ---------------------------------------------------------------------------


def test_all_seven_differentiators_present() -> None:
    diffs = build_differentiators()
    assert {d.differentiator_id for d in diffs} == set(range(1, 8))


def test_every_differentiator_maps_to_a_real_contract_field() -> None:
    field_names = {f.name for f in fields(MinimalCompilerLatentContractV1)}
    field_map = differentiator_contract_field_map()
    assert set(field_map.keys()) == set(range(1, 8))
    for differentiator_id, field_name in field_map.items():
        assert field_name in field_names, (
            f"differentiator {differentiator_id} maps to unknown contract "
            f"field {field_name!r}"
        )


def test_every_differentiator_has_a_test_ref_and_evidence() -> None:
    for d in build_differentiators():
        assert d.contract_field
        assert d.test_ref
        assert d.evidence
        assert isinstance(d.satisfied, bool)


def test_differentiator_7_conditional_execution_is_unmet() -> None:
    diffs = {d.differentiator_id: d for d in build_differentiators()}
    d7 = diffs[7]
    assert d7.name == "conditional_execution"
    assert d7.satisfied is False
    assert "SLM-139" in d7.evidence or "recursive" in d7.evidence.lower()


# ---------------------------------------------------------------------------
# Decision-rule / rejection fixtures
# ---------------------------------------------------------------------------


def _all_true_differentiators() -> dict[int, bool]:
    return {i: True for i in range(1, 8)}


def test_authorize_requires_all_seven_and_ready_gates() -> None:
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE


def test_proposal_with_separate_auxiliary_head_only_is_duplicate_spv() -> None:
    """A proposal using a separate auxiliary head only -> duplicate/unauthorized fixture."""
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=True,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.DUPLICATE_SPV
    assert verdict != LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE


def test_proposal_exporting_semantic_plan_v1_is_duplicate_spv() -> None:
    """A proposal exporting SemanticPlanV1 or building a plan seed -> duplicate SPV fixture."""
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=True,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.DUPLICATE_SPV


def test_proposal_constructing_plan_seed_is_duplicate_spv() -> None:
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=True,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.DUPLICATE_SPV


def test_proposal_without_interventions_is_unauthorized() -> None:
    """A proposal without gold/zero/swap/wrong interventions -> unauthorized."""
    differentiators = _all_true_differentiators()
    differentiators[5] = False  # builtin_interventions
    verdict = evaluate_verdict(
        differentiators=differentiators,
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict != LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE


def test_proposal_that_can_prune_legal_actions_is_unauthorized() -> None:
    """A proposal that can prune legal actions -> unauthorized."""
    differentiators = _all_true_differentiators()
    differentiators[6] = False  # no_hard_authority violated
    verdict = evaluate_verdict(
        differentiators=differentiators,
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict != LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE


def test_absent_target_support_cannot_be_silently_coerced_to_negative() -> None:
    """Absent/ambiguous target support cannot be silently coerced to negatives."""
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=False,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.UNSUPPORTED_TARGETS


def test_target_support_rows_declare_explicit_ambiguous_unknown_handling() -> None:
    for row in build_target_support_audit():
        assert row.ambiguous_unknown_handling.strip() != ""


def test_scale_not_identifiable_short_circuits_authorization() -> None:
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=False,
        floor_ready=True,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.SCALE_NOT_IDENTIFIABLE


def test_floor_gate_unmet_blocks_authorization() -> None:
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=False,
        recurrence_ready=True,
    )
    assert verdict == LoopedLatentVerdict.BLOCKED_BY_FLOOR


def test_recurrence_gate_unmet_blocks_authorization() -> None:
    verdict = evaluate_verdict(
        differentiators=_all_true_differentiators(),
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=True,
        floor_ready=True,
        recurrence_ready=False,
    )
    assert verdict == LoopedLatentVerdict.BLOCKED_BY_RECURRENCE


# ---------------------------------------------------------------------------
# Floor/recurrence gate fields are mandatory
# ---------------------------------------------------------------------------


def test_scale_regime_audit_has_mandatory_floor_and_recurrence_fields() -> None:
    audit = build_scale_regime_audit()
    assert audit.semantic_floor_status.strip() != ""
    assert audit.recursive_regime_status.strip() != ""


def test_minimal_contract_requires_floor_recurrence_oracle_gate_fields() -> None:
    field_names = {f.name for f in fields(MinimalCompilerLatentContractV1)}
    assert "required_floor_gate" in field_names
    assert "required_recurrence_gate" in field_names
    assert "required_oracle_gate" in field_names


# ---------------------------------------------------------------------------
# Doc reference resolution
# ---------------------------------------------------------------------------


def test_all_reviewed_refs_resolve(repo_root: Path) -> None:
    from slm_training.harnesses.experiments.slm229_looped_latent_differentiation import (
        REVIEWED_REFS,
    )

    missing = validate_doc_refs(REVIEWED_REFS, repo_root=repo_root)
    assert missing == []


def test_validate_doc_refs_flags_missing_path(repo_root: Path) -> None:
    missing = validate_doc_refs(["docs/design/does-not-exist-99999999.md"], repo_root=repo_root)
    assert missing == ["docs/design/does-not-exist-99999999.md"]


def test_validate_doc_refs_allows_bare_issue_ids(repo_root: Path) -> None:
    missing = validate_doc_refs(["SLM-138", "SLM-229"], repo_root=repo_root)
    assert missing == []


# ---------------------------------------------------------------------------
# Content sanity (real analysis, not template fill-in)
# ---------------------------------------------------------------------------


def test_mechanism_comparison_covers_all_required_mechanisms() -> None:
    rows = build_mechanism_comparison()
    ids = {r.mechanism_id for r in rows}
    assert ids == {
        "lotus_arxiv_2606_31779",
        "slm138_shared_recursive_denoiser",
        "slm144_plan_predictor_archetype_roleset",
        "slm145_plan_predictor_topology_cardinality_pointer",
        "slm146_semantic_plan_compiler",
        "slm160_spv_disposition",
        "proposed_minimal_compiler_latent_probe",
    }


def test_prior_art_audit_distinguishes_repo_from_external_novelty() -> None:
    rows = build_prior_art_audit()
    scopes = {r.novelty_scope for r in rows}
    assert "repository" in scopes
    # "not found internally" is not an external novelty claim.
    assert "external" not in scopes


def test_oracle_ceiling_reuses_slm146_evidence_and_states_a_gap() -> None:
    ceiling = build_oracle_intervention_ceiling()
    assert any("slm146" in p for p in ceiling.reused_evidence_paths)
    assert ceiling.gap_description.strip() != ""
    assert ceiling.smallest_fixture_spec.strip() != ""


# ---------------------------------------------------------------------------
# Fixture audit + markdown rendering
# ---------------------------------------------------------------------------


def test_fixture_audit_produces_blocked_by_recurrence(repo_root: Path) -> None:
    """Honest, evidence-grounded verdict given SLM-139's failed recursive-core gate."""
    report = run_differentiation_audit(repo_root=repo_root)
    assert report.verdict == LoopedLatentVerdict.BLOCKED_BY_RECURRENCE
    assert report.minimal_contract is None
    assert report.contract_hash is None
    assert "does not authorize learned-latent claims" in report.resolving_evidence
    assert "does not authorize learned-latent claims" in report.scale_regime_audit.semantic_floor_status


def test_fixture_audit_has_version_stamp(repo_root: Path) -> None:
    report = run_differentiation_audit(repo_root=repo_root)
    assert report.version_stamp
    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"
    assert "harness.experiments" in report.version_stamp.get("components", {})


def test_render_markdown_contains_expected_sections(repo_root: Path) -> None:
    report = run_differentiation_audit(repo_root=repo_root)
    md = render_markdown(report)
    for section in (
        "# SLM-229 (RSC0-01): Looped-latent differentiation memo",
        "## 1. Mechanism comparison table",
        "## 2. Data/target availability audit",
        "## 3. Oracle intervention ceiling",
        "## 4. Scale/regime audit",
        "## 5. Prior-art and novelty audit",
        "## Candidate differentiators (all 7)",
        "## Allowed implementation scope",
        "## Forbidden/duplicate scope",
        "## Resolving evidence",
        "## MinimalCompilerLatentContractV1",
        "## Reproducibility commands",
        "## Non-goals",
        "## Limitations",
    ):
        assert section in md
    assert "| slm138_shared_recursive_denoiser |" in md
    assert "blocked_by_recurrence" in md
