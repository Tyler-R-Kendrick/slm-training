"""LOT0-02 (SLM-249): CompilerReasoningTraceV1 / CompilerReasoningTraceGateV1.

Docs/spec + bounded-probe issue: these tests assert the committed JSON contract
is well-formed, internally consistent, and that its embedded hash is a genuine,
reproducible digest of its own content -- not that an oracle-ceiling experiment
has been run (it has not; see the gate's ``verdict``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "docs" / "design" / "compiler-reasoning-trace-v1.json"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "linear_issue",
    "created_at",
    "version_stamp",
    "parent_authorization_reference",
    "reuse_boundaries",
    "stage_set",
    "typed_step_schema_ref",
    "trace_schema_ref",
    "extractor_ref",
    "visible_trace_representation",
    "k_c_budget_analysis_ref",
    "oracle_ceiling_experiment_plan",
    "causal_latent_use_falsification_test_ref",
    "claim_classes",
    "required_differentiators",
    "comparator_hashes",
    "contract_hash",
    "gate",
}

ALLOWED_VERDICTS = {
    "oracle_ceiling_positive",
    "partial_trace_only",
    "explicit_trace_equivalent_to_existing_plan",
    "no_downstream_ceiling",
    "target_support_insufficient",
    "leakage_or_ambiguity_blocked",
    "inconclusive",
}

REQUIRED_STAGE_KEYS = {"step_kind", "concept", "reused_source"}
EXPECTED_STAGE_ORDER = (
    "intent_contract",
    "component_inventory",
    "topology_skeleton",
    "semantic_edges_roles",
    "binding_scope_references",
    "serialization_plan",
)


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _recompute_contract_hash(contract: dict) -> str:
    core = {k: v for k, v in contract.items() if k not in ("contract_hash", "gate")}
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


@pytest.fixture(scope="module")
def contract() -> dict:
    assert CONTRACT_PATH.exists(), f"missing contract JSON: {CONTRACT_PATH}"
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_schema_round_trip_and_required_top_level_keys(contract: dict) -> None:
    reparsed = json.loads(json.dumps(contract))
    assert reparsed == contract
    missing = REQUIRED_TOP_LEVEL_KEYS - contract.keys()
    assert not missing, f"contract missing required top-level keys: {sorted(missing)}"
    assert contract["schema_version"] == "compiler_reasoning_trace_contract/v1"
    assert contract["linear_issue"] == "SLM-249"


def test_parent_authorization_reference_points_at_lot0_01(contract: dict) -> None:
    ref = contract["parent_authorization_reference"]
    assert ref["linear_issue"] == "SLM-248"
    assert ref["verdict"] == "needs_target_trace_contract"
    assert isinstance(ref["note"], str) and ref["note"].strip()


def test_stage_set_matches_expected_order_and_has_required_fields(contract: dict) -> None:
    rows = contract["stage_set"]
    assert isinstance(rows, list) and len(rows) == 6
    assert tuple(row["step_kind"] for row in rows) == EXPECTED_STAGE_ORDER
    for row in rows:
        missing = REQUIRED_STAGE_KEYS - row.keys()
        assert not missing, f"stage row {row.get('step_kind')!r} missing keys: {missing}"
        for key in REQUIRED_STAGE_KEYS:
            assert isinstance(row[key], str) and row[key].strip()


def test_reuse_boundaries_and_required_differentiators_present(contract: dict) -> None:
    for key in ("reuse_boundaries", "required_differentiators", "claim_classes"):
        values = contract[key]
        assert isinstance(values, list) and values
        assert all(isinstance(v, str) and v.strip() for v in values)


def test_visible_trace_representation_is_lossless_and_separate(contract: dict) -> None:
    rep = contract["visible_trace_representation"]
    assert rep["lossless"] is True
    assert rep["separate_from_final_program"] is True
    assert isinstance(rep["serializer_ref"], str) and rep["serializer_ref"].strip()
    assert isinstance(rep["parser_ref"], str) and rep["parser_ref"].strip()


def test_oracle_ceiling_experiment_plan_is_explicitly_not_run(contract: dict) -> None:
    plan = contract["oracle_ceiling_experiment_plan"]
    assert plan["status"] == "planned_not_run"
    assert isinstance(plan["reason_not_run"], str) and plan["reason_not_run"].strip()
    arms = plan["arms"]
    assert isinstance(arms, list) and len(arms) >= 7
    arm_ids = {arm["arm_id"] for arm in arms}
    expected_arms = {
        "causal_baseline",
        "explicit_gold_trace",
        "shuffled_wrong_trace",
        "partial_trace_by_stage",
        "accepted_alternative_trace",
        "equivalent_length_nuisance_control",
        "existing_plan_baseline",
    }
    assert expected_arms <= arm_ids
    for arm in arms:
        assert isinstance(arm["description"], str) and arm["description"].strip()
    assert isinstance(plan["matching_rules"], list) and len(plan["matching_rules"]) >= 3
    assert isinstance(plan["measurements"], list) and len(plan["measurements"]) >= 3


def test_causal_latent_use_falsification_test_ref_is_defined(contract: dict) -> None:
    ref = contract["causal_latent_use_falsification_test_ref"]
    assert "CausalLatentUseFalsificationSpecV1" in ref
    assert "causal_trace.py" in ref


def test_gate_verdict_is_allowed_and_is_inconclusive_not_a_ceiling_claim(
    contract: dict,
) -> None:
    gate = contract["gate"]
    assert gate["verdict"] in ALLOWED_VERDICTS
    assert isinstance(gate["verdict_rationale"], str) and gate["verdict_rationale"].strip()
    # This issue's honest analysis landed on "inconclusive": the oracle-ceiling
    # experiment is specified but explicitly not run (out of scope per LOT0-01).
    # Pin this literally so a future edit cannot silently upgrade it to a positive
    # ceiling claim without actually running the campaign.
    assert gate["verdict"] == "inconclusive"
    for key in ("k_c_recommendation", "accepted_stage_set_order", "blocked_claims"):
        assert key in gate
    assert gate["accepted_stage_set_order"] == list(EXPECTED_STAGE_ORDER)
    assert gate["k_c_recommendation"]["evidence_class"] == "bounded_probe"
    assert isinstance(gate["blocked_claims"], list) and gate["blocked_claims"]


def test_allowed_lot1_implementation_is_blocked(contract: dict) -> None:
    gate = contract["gate"]
    text = " ".join(gate["allowed_lot1_implementation"]).lower()
    assert "none" in text
    assert "slm-250" in text or "lot1" in text


def test_contract_hash_matches_fresh_recompute(contract: dict) -> None:
    recomputed = _recompute_contract_hash(contract)
    assert len(contract["contract_hash"]) == 64
    assert contract["contract_hash"] == recomputed
    assert contract["gate"]["contract_hash"] == recomputed


def test_comparator_hashes_are_well_formed_path_plus_sha256(contract: dict) -> None:
    rows = contract["comparator_hashes"]
    assert isinstance(rows, list) and len(rows) >= 8
    for row in rows:
        assert set(row.keys()) == {"path", "sha256"}
        assert isinstance(row["path"], str) and row["path"].strip()
        digest = row["sha256"]
        assert isinstance(digest, str) and len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


def test_no_semantic_quality_claim_is_made(contract: dict) -> None:
    """Acceptance criterion: no model or quality claim is made by this issue."""
    blocked_text = " ".join(contract["gate"]["blocked_claims"]).lower()
    assert "no claim that compilerreasoningtracev1 improves" in blocked_text
    assert "no gpu training" in blocked_text or "no gpu" in blocked_text
