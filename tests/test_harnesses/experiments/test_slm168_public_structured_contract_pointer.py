"""Tests for SLM-168 (SDE2-01) public structured contract-index pointer fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm168_public_structured_contract_pointer import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    ContractPointerArm,
    ContractPointerReport,
    build_candidate_set,
    build_cells,
    resolve_disposition,
    run_fixture_campaign,
    score_pointer_decision,
    validate_manifest,
)
from slm_training.models.dynamic_pointer_scorer import (
    DynamicPointerScorer,
    DynamicPointerScorerConfig,
    PointerCandidateSet,
)


def test_build_cells_produces_seven_arms_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == 21
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.arm_id)
    assert len(per_seed) == 3
    for seed, ids in per_seed.items():
        assert len(ids) == 7, f"seed {seed} has {len(ids)} cells"


def test_cells_cover_all_arm_names() -> None:
    cells = build_cells(seeds=(0,))
    seen = {c.arm_name for c in cells}
    assert seen == set(ARM_NAMES)


def test_validate_manifest_accepts_valid_cells() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []


def test_validate_manifest_rejects_duplicate_arm_id() -> None:
    cells = build_cells(seeds=(0,))
    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_invalid_pointer_mode() -> None:
    cells = build_cells(seeds=(0,))
    bad = ContractPointerArm(
        arm_id="bad",
        arm_name="legacy_inventory_in_prompt",
        pointer_mode="invalid",
        candidate_source="structured_contract",
        seed=0,
        d_model=64,
        pointer_hidden_dim=256,
        pointer_heads=4,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid pointer_mode" in e for e in errors)


def test_validate_manifest_rejects_invalid_candidate_source() -> None:
    cells = build_cells(seeds=(0,))
    bad = ContractPointerArm(
        arm_id="bad",
        arm_name="legacy_inventory_in_prompt",
        pointer_mode="legacy_tokens",
        candidate_source="invalid",
        seed=0,
        d_model=64,
        pointer_hidden_dim=256,
        pointer_heads=4,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid candidate_source" in e for e in errors)


def test_build_candidate_set_uses_only_inference_available_fields() -> None:
    prompt = "Build a hero section with title and body."
    candidate_set = build_candidate_set(prompt, source="structured_contract", seed=0)
    assert isinstance(candidate_set, PointerCandidateSet)
    # Candidate provenances are limited to inference-visible sources.
    for candidate in candidate_set.candidates:
        assert candidate.provenance in {
            "request_contract",
            "runtime",
            "authored_prompt",
            "compiler_scope",
        }
        # No evaluator-only gold AST fields appear in display text or stable_id.
        assert "gold" not in candidate.stable_id.lower()
        assert "future" not in candidate.stable_id.lower()


def test_build_candidate_set_authored_only_respects_prompt() -> None:
    prompt_with_title = "Build a hero section with title."
    prompt_without = "Build a hero section."
    set_with = build_candidate_set(prompt_with_title, source="authored_only", seed=0)
    set_without = build_candidate_set(prompt_without, source="authored_only", seed=0)
    assert len(set_with) > len(set_without)


def test_score_pointer_decision_returns_decision() -> None:
    arm = ContractPointerArm(
        arm_id="test",
        arm_name="dynamic_structured_contract",
        pointer_mode="dynamic_head",
        candidate_source="structured_contract",
        seed=0,
        d_model=64,
        pointer_hidden_dim=256,
        pointer_heads=4,
    )
    candidate_set = build_candidate_set(
        "Build a hero section with title.", source="structured_contract", seed=0
    )
    decision = score_pointer_decision(arm, candidate_set, ":hero.title", "state_001")
    assert 0 <= decision.selected_index < len(candidate_set)
    assert decision.gold_index is not None
    assert len(decision.scores) == len(candidate_set)


def test_legacy_tokens_mode_returns_uniform_scores() -> None:
    arm = ContractPointerArm(
        arm_id="test",
        arm_name="legacy_inventory_in_prompt",
        pointer_mode="legacy_tokens",
        candidate_source="inventory_in_prompt",
        seed=0,
        d_model=64,
        pointer_hidden_dim=256,
        pointer_heads=4,
    )
    candidate_set = build_candidate_set(
        "Build a hero section.", source="inventory_in_prompt", seed=0
    )
    decision = score_pointer_decision(arm, candidate_set, ":hero.title", "state_002")
    assert decision.selected_index == 0  # argmax of zeros picks index 0


def test_dynamic_head_beats_legacy_no_inventory() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    dynamic_structured = means["dynamic_structured_contract"]["binding_fidelity"]
    legacy_no_inventory = means["legacy_no_inventory"]["binding_fidelity"]
    assert dynamic_structured > legacy_no_inventory


def test_structured_contract_near_inventory_in_prompt() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    dynamic_structured = means["dynamic_structured_contract"]["binding_fidelity"]
    dynamic_inventory = means["dynamic_inventory_in_prompt"]["binding_fidelity"]
    assert abs(dynamic_structured - dynamic_inventory) < 0.10


def test_disposition_contract_conditioned_or_inventory_equivalent() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    assert report.disposition in {
        "contract_conditioned_pointer_works",
        "inventory_equivalent",
        "inconclusive",
    }


def test_resolve_disposition_pointer_not_better() -> None:
    means = {
        "legacy_inventory_in_prompt": {"binding_fidelity": 0.85},
        "legacy_no_inventory": {"binding_fidelity": 0.45},
        "dynamic_structured_contract": {"binding_fidelity": 0.40},
        "dynamic_authored_only": {"binding_fidelity": 0.30},
        "dynamic_inventory_in_prompt": {"binding_fidelity": 0.80},
        "dynamic_permuted_order": {"binding_fidelity": 0.39},
        "dynamic_hidden_text": {"binding_fidelity": 0.35},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "pointer_not_better_than_legacy_no_inventory"


def test_resolve_disposition_order_sensitive() -> None:
    means = {
        "legacy_inventory_in_prompt": {"binding_fidelity": 0.85},
        "legacy_no_inventory": {"binding_fidelity": 0.45},
        "dynamic_structured_contract": {"binding_fidelity": 0.92},
        "dynamic_authored_only": {"binding_fidelity": 0.55},
        "dynamic_inventory_in_prompt": {"binding_fidelity": 0.88},
        "dynamic_permuted_order": {"binding_fidelity": 0.70},
        "dynamic_hidden_text": {"binding_fidelity": 0.65},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "order_sensitive_pointer"


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,))
    reconstructed = ContractPointerReport.from_dict(report.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.matrix_version == MATRIX_VERSION
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert reconstructed.claim_class == "wiring"
    assert len(reconstructed.rows) == len(report.rows)


def test_report_version_stamp() -> None:
    report = run_fixture_campaign(seeds=(0,))
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm168_public_structured_contract_pointer" in components
    assert "model.twotower" in components


def test_dynamic_pointer_scorer_legacy_tokens_is_no_op() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="legacy_tokens")
    scorer = DynamicPointerScorer(config)
    assert scorer.scorer is None
    candidate_set = build_candidate_set("prompt", source="structured_contract", seed=0)
    import torch

    state_vec = torch.randn(64)
    logits = scorer.forward(state_vec, candidate_set)
    assert torch.allclose(logits, torch.zeros(len(candidate_set)), atol=1e-6)


def test_dynamic_pointer_scorer_hidden_text_lowers_fidelity() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    dynamic_structured = means["dynamic_structured_contract"]["binding_fidelity"]
    dynamic_hidden = means["dynamic_hidden_text"]["binding_fidelity"]
    assert dynamic_hidden < dynamic_structured
