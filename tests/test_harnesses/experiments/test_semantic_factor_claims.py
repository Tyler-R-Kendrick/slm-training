"""Unit tests for claim validate/invalidate rules."""

from __future__ import annotations

from slm_training.harnesses.experiments.semantic_factor_claims import (
    CLAIM_SPECS,
    evaluate_claims,
)


def test_claim_specs_cover_runtime_claims() -> None:
    assert len(CLAIM_SPECS) >= 19
    ids = [c.claim_id for c in CLAIM_SPECS]
    assert len(ids) == len(set(ids))
    assert "SFF-C17-runtime-tracked" in ids
    assert "SFF-C18-efficiency-vs-quality" in ids


def _arm_runtime(
    arm_id: str,
    *,
    acc: float,
    apps: int,
    chg: int,
    wall_ms: float,
    kills: list[str] | None = None,
    unknown: int = 1,
) -> dict:
    return {
        "arm_id": arm_id,
        "ranking_accuracy": acc,
        "applications": apps,
        "choice_changes": chg,
        "singleton_skips": 1,
        "unknown_preserved": unknown,
        "kill_criteria_hit": kills or [],
        "wall_ms_mean": wall_ms,
        "wall_ms_p50": wall_ms,
        "wall_ms_p95": wall_ms * 1.2,
        "wall_ms_total": wall_ms * 10,
        "quality_per_ms": (acc / wall_ms) if wall_ms > 0 else None,
        "wall_ms_mean_vs_control": wall_ms / 0.01 if wall_ms else None,
        "delta_accuracy_per_extra_ms": None,
    }


def test_evaluate_claims_singleton_and_promotion() -> None:
    metrics = {
        "claim_class": "fixture",
        "support_preservation_ok": True,
        "membership_roundtrip_ok": True,
        "decode_off_identical_to_control": True,
        "n_ranked_per_arm": 5,
        "math_properties": {
            "column_stochastic_ok": True,
            "soft_token_collision_ok": True,
        },
        "runtime": {"timer": "time.perf_counter", "campaign_wall_s": 0.1},
        "unimplemented": {
            "rl": True,
            "spectral_training": True,
            "production_topology_authority": True,
            "graph_pruning": True,
            "recurrent_semantic_inference": True,
            "faithful_shift_soft_tokens": True,
            "search_r1_policy_training": True,
        },
        "arms": [
            _arm_runtime("control_none", acc=0.0, apps=0, chg=0, wall_ms=0.01),
            _arm_runtime(
                "train_on_decode_on_role_ported",
                acc=0.0,
                apps=5,
                chg=0,
                wall_ms=0.5,
                kills=["applications_without_choice_changes"],
            ),
            _arm_runtime(
                "train_on_decode_on_direct_factors",
                acc=0.4,
                apps=5,
                chg=3,
                wall_ms=0.05,
            ),
            _arm_runtime(
                "exact_typed_zero_parameter",
                acc=0.4,
                apps=5,
                chg=3,
                wall_ms=0.02,
            ),
        ],
        "outcomes": [],
    }
    results = {c.claim_id: c for c in evaluate_claims(metrics)}
    assert results["SFF-C6-role-ported-causal"].verdict == "invalidated"
    assert results["SFF-C8-direct-factors-causal"].verdict == "validated"
    assert results["SFF-C12-apps-without-choice-is-fail"].verdict == "validated"
    assert results["SFF-C16-anti-e237-promotion-bar"].verdict == "invalidated"
    assert results["SFF-C17-runtime-tracked"].verdict == "validated"
    # role_ported is 50× slower than control with no quality gain → efficiency invalidate
    assert results["SFF-C18-efficiency-vs-quality"].verdict == "invalidated"
    assert results["SFF-C19-exact-typed-efficiency"].verdict == "validated"
