"""Deterministic revmath replay bundles (HARN-03).

Composes the canonical ``ReplayBundleV1`` owner — never a profile-local store.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import RevmathResultV1, RevmathTaskV1
from slm_training.models.decode_stats import DecodeIdentityV1
from slm_training.runtime.telemetry.replay_bundle import (
    ReplayBundleV1,
    build_replay_bundle,
    replay_bundle_integrity_ok,
)


def revmath_request_id(task: RevmathTaskV1) -> str:
    return f"revmath:{task.task_id}:{task.identity_digest()[:16]}"


def build_revmath_replay_bundle(
    *,
    task: RevmathTaskV1,
    result: RevmathResultV1,
    capture: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    repair_history_ref: str | None = None,
) -> ReplayBundleV1:
    """Build a deterministic offline-replayable bundle for one task run.

    Identity fields come from the frozen task; capture digests must be exact
    stdout/stderr/result hashes supplied by the runner. The runner never
    mutates proposition or campaign state — those digests are mirrored here.
    """

    checkpoint = content_sha(
        {
            "task_identity_digest": task.identity_digest(),
            "proposition_sha256": task.proposition.statement_sha256,
            "campaign_manifest_sha256": task.campaign.campaign_manifest_sha256,
            "base_theory_id": task.base_theory.base_theory_id,
            "allowed_assumption_ids": list(task.base_theory.allowed_assumption_ids),
        }
    )
    identity = DecodeIdentityV1(
        checkpoint_sha256=checkpoint,
        pack_id=task.corpus.corpus_id,
        evaluator_version="revmath_runner/v1",
        artifact_sha256=task.canonical_hash(),
        code_commit=None,
    )
    # Wall-clock duration is diagnostic only — never identity for replay.
    capture_for_bundle = {
        key: value for key, value in dict(capture).items() if key != "duration_seconds"
    }
    final_evidence = {
        "schema": "revmath_replay_evidence/v1",
        "task_id": task.task_id,
        "task_identity_digest": task.identity_digest(),
        "proposition_id": task.proposition.proposition_id,
        "proposition_sha256": task.proposition.statement_sha256,
        "assumption_ids": list(task.base_theory.allowed_assumption_ids),
        "base_theory_id": task.base_theory.base_theory_id,
        "campaign_manifest_sha256": task.campaign.campaign_manifest_sha256,
        "result_id": result.result_id,
        "judgment": dict(result.judgment),
        "four_axis": result.four_axis.model_dump(mode="json"),
        "proof": (
            None if result.proof is None else result.proof.model_dump(mode="json")
        ),
        "capture": capture_for_bundle,
        "identity_snapshot": dict(identity_snapshot),
        "repair_history_ref": repair_history_ref,
    }
    artifact_state = {
        "task_kind": task.task_kind,
        "theorem_direction": task.theorem_direction,
        "budget": task.budget.model_dump(mode="json"),
        "verifier_judge": task.verifier_judge.model_dump(mode="json"),
    }
    return build_replay_bundle(
        request_id=revmath_request_id(task),
        identity=identity,
        seed=None,
        decoder_choices=(),
        legal_domain_size=1,
        legal_domain_status="singleton",
        legal_domain_digest=task.identity_digest(),
        artifact_state=artifact_state,
        final_evidence=final_evidence,
    )


def assert_revmath_replay_integrity(bundle: ReplayBundleV1) -> None:
    if not replay_bundle_integrity_ok(bundle):
        raise ValueError("revmath replay bundle digest mismatch")
    if not bundle.replayable:
        raise ValueError(
            "revmath replay bundle not replayable: "
            + ", ".join(bundle.non_replayable_reasons)
        )


__all__ = [
    "assert_revmath_replay_integrity",
    "build_revmath_replay_bundle",
    "revmath_request_id",
]
