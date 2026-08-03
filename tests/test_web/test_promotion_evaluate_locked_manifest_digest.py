"""SLM-431: opt-in locked-manifest digest verification on the HTTP endpoint.

SLM-306 (`docs/design/slm306-locked-manifest-digest-verification.md`) added an
optional ``locked_manifest_path`` to
``slm_training.autoresearch.experiment_campaign.validate_result_claim`` so a
campaign's self-reported ``locked_eval_manifest_sha256`` can be independently
re-derived from real manifest bytes on disk instead of only trusted as a
free-form hex string. SLM-430
(`docs/design/slm430-promotion-cli-locked-manifest-digest-opt-in.md`) wired an
explicit, default-off opt-in into the three real training/CLI promotion
entrypoints (``train()``, ``scripts/resume_climb.py``,
``scripts/run_scaling_ladder.py``) -- but its own writeup only covers those
three; the pure-compute ``POST /api/promotion/evaluate`` endpoint
(`src/slm_training/web/routes.py`, which also calls ``validate_result_claim``
via the Checkpoints gate editor) was never given a way to request the check
either, by default or otherwise.

This module covers the new ``verify_locked_manifest_digest`` field on
``PromotionEvalRequest``: when absent (default ``False``), behavior is
unchanged from before this change -- only self-consistency between the
manifest's and result's declared ``locked_eval_manifest_sha256`` strings is
checked. When explicitly set ``True``, the endpoint threads
``locked_eval_manifest.canonical_manifest_path()`` through to
``validate_result_claim`` as ``locked_manifest_path``, so a declared digest
that does not match the real committed manifest bytes fails closed with
``locked_eval_manifest_digest_unverified_on_disk``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    campaign_manifest_sha256,
    ExperimentCampaignV1,
)
from slm_training.data.locked_eval_manifest import (
    canonical_manifest_path,
    load_locked_manifest_payload,
)
from slm_training.web.routes import PromotionEvalRequest, promotion_evaluate

_REAL_DIGEST = load_locked_manifest_payload(canonical_manifest_path())["manifest_sha256"]
assert isinstance(_REAL_DIGEST, str) and len(_REAL_DIGEST) == 64
_WRONG_DIGEST = hashlib.sha256(b"not-the-real-manifest").hexdigest()


def _manifest_payload(*, locked_eval_manifest_sha256: str) -> dict[str, Any]:
    return {
        "campaign_id": "slm431-http-digest",
        "experiment_id": "e001",
        "hypothesis": "Candidate improves the locked primary endpoint.",
        "decision": "Promote only when every preregistered gate passes.",
        "endpoints": [
            {
                "endpoint_id": "meaning",
                "metric": "binder_reference_f1",
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.01,
            }
        ],
        "arms": [
            {"arm_id": "control", "role": "control", "config_sha256": "c" * 64},
            {
                "arm_id": "mechanism_off",
                "role": "candidate",
                "config_sha256": "a" * 64,
            },
            {"arm_id": "candidate", "role": "candidate", "config_sha256": "d" * 64},
        ],
        "seeds": [7, 11],
        "budget": {"max_experiments": 2, "max_gpu_hours": 0, "max_wall_minutes": 2},
        "stopping_rules": ["Stop after the declared seeds finish."],
        "controls": [
            {
                "control_id": "matched-baseline",
                "description": "Size-matched baseline without the mechanism.",
                "kind": "positive",
            },
            {
                "control_id": "unchanged-baseline",
                "description": "Unchanged baseline must reproduce.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["unchanged-baseline"],
        "mechanism_off_arm_ids": ["mechanism_off"],
        "executable_kill_criteria": [
            "applications_without_choice_changes",
            "quality_decreased_with_choice_changes",
        ],
        "multiplicity_families": [
            {"family_id": "primary", "hypothesis_ids": ["meaning"], "alpha": 0.05}
        ],
        "promotion_gates": [
            {
                "gate_id": "meaning-improves",
                "endpoint_id": "meaning",
                "operator": "ge",
                "threshold": 0.01,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "meaning-regresses",
                "endpoint_id": "meaning",
                "operator": "le",
                "threshold": -0.01,
            }
        ],
        "artifact_requirements": [
            {"kind": kind, "minimum_count": 1}
            for kind in (
                "version_stamp",
                "seed_result",
                "paired_examples",
                "endpoint_result",
                "holm_family",
                "agentevals",
                "agentv",
                "observation_table",
                "analysis_plan",
                "credit_report",
            )
        ],
        "claim_class": "promotion_candidate",
        "locked_eval_manifest_sha256": locked_eval_manifest_sha256,
        "source_commit": "a" * 40,
        "source_dirty": False,
        "author": "test",
        "created_at": "2026-07-27T00:00:00Z",
    }


def _manifest_and_matching_result(
    *, locked_eval_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(locked_eval_manifest_sha256=locked_eval_manifest_sha256)
    )
    manifest_json = manifest.model_dump(mode="json")
    result = {
        "campaign_id": manifest.campaign_id,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "claim_class": manifest.claim_class,
        "locked_eval_manifest_sha256": locked_eval_manifest_sha256,
    }
    return manifest_json, result


def test_verify_flag_defaults_to_false_and_omits_disk_check() -> None:
    """Self-consistent-but-fake digest passes when the flag is omitted."""
    manifest, result = _manifest_and_matching_result(
        locked_eval_manifest_sha256=_WRONG_DIGEST
    )

    response = promotion_evaluate(
        PromotionEvalRequest(campaign_manifest=manifest, campaign_result=result)
    )

    failures = response["checks"]["campaign_governance"]["failures"]
    assert "locked_eval_manifest_digest_unverified_on_disk" not in failures


def test_verify_flag_true_rejects_digest_not_on_disk() -> None:
    """Same self-consistent-but-fake digest fails closed once opted in."""
    manifest, result = _manifest_and_matching_result(
        locked_eval_manifest_sha256=_WRONG_DIGEST
    )

    body = promotion_evaluate(
        PromotionEvalRequest(
            campaign_manifest=manifest,
            campaign_result=result,
            verify_locked_manifest_digest=True,
        )
    )
    failures = body["checks"]["campaign_governance"]["failures"]
    assert "locked_eval_manifest_digest_unverified_on_disk" in failures
    assert body["promotable"] is False


def test_verify_flag_true_accepts_real_committed_digest() -> None:
    """The real, untampered committed manifest digest passes the disk check."""
    manifest, result = _manifest_and_matching_result(
        locked_eval_manifest_sha256=_REAL_DIGEST
    )

    response = promotion_evaluate(
        PromotionEvalRequest(
            campaign_manifest=manifest,
            campaign_result=result,
            verify_locked_manifest_digest=True,
        )
    )

    failures = response["checks"]["campaign_governance"]["failures"]
    assert "locked_eval_manifest_digest_unverified_on_disk" not in failures
    assert "locked_eval_manifest_sha256_mismatch" not in failures
    assert "locked_eval_manifest_sha256_missing" not in failures


def test_promotion_evaluate_charges_parameter_growth() -> None:
    result = promotion_evaluate(
        PromotionEvalRequest(
            baseline_trainable_params=100,
            candidate_trainable_params=200,
        )
    )

    assert result["checks"]["eg_params"]["pass"] is False
    assert "eg_params" in result["failures"]
