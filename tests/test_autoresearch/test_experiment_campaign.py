"""Focused contract tests for preregistered autoresearch campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.experiment_campaign import (
    AP001CertificationV1,
    CampaignLockV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    load_ap001_certification,
    select_primary_endpoint,
    validate_result_claim,
)
HEX_40 = "a" * 40
HEX_64 = "b" * 64


def _manifest_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "campaign_id": "ap-036",
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
            {
                "arm_id": "candidate",
                "role": "candidate",
                "config_sha256": "d" * 64,
            },
        ],
        "seeds": [7, 11],
        "budget": {
            "max_experiments": 2,
            "max_gpu_hours": 0,
            "max_wall_minutes": 2,
        },
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
            {
                "family_id": "primary",
                "hypothesis_ids": ["meaning"],
                "alpha": 0.05,
            }
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
        "locked_eval_manifest_sha256": "e" * 64,
        "source_commit": HEX_40,
        "source_dirty": False,
        "author": "test",
        "created_at": "2026-07-23T00:00:00Z",
    }
    payload.update(updates)
    return payload


def test_replay_manifest_requires_content_bound_reason() -> None:
    with pytest.raises(ValidationError, match="must be declared together"):
        ExperimentCampaignV1.model_validate(
            _manifest_payload(replay_of_manifest_sha256=HEX_64)
        )
    replay = ExperimentCampaignV1.model_validate(
        _manifest_payload(
            replay_of_manifest_sha256=HEX_64,
            replay_reason="Current-main successor of an incomplete frozen measurement.",
        )
    )
    assert replay.replay_of_manifest_sha256 == HEX_64


def _manifest(**updates: object) -> ExperimentCampaignV1:
    return ExperimentCampaignV1.model_validate(_manifest_payload(**updates))


def _write_credit_bundle(
    manifest: ExperimentCampaignV1,
    artifact_root: Path,
    *,
    control_arm: str = "control",
    candidate_arm: str = "candidate",
    delta: float = 0.05,
) -> tuple[dict[str, object], dict[str, float], list[dict[str, object]]]:
    """Write observation_table, analysis_plan, credit_report; return digests info."""
    from slm_training.autoresearch.credit_engine import (
        analysis_plan_from_mapping,
        compute_credit_report,
        observation_table_from_mapping,
    )

    primary = next(e for e in manifest.endpoints if e.role == "primary")
    metric = primary.metric
    direction = primary.direction
    examples = ("example-1", "example-2")
    rows: list[dict[str, object]] = []
    for seed in manifest.seeds:
        for ex in examples:
            rows.append(
                {
                    "arm_id": control_arm,
                    "seed": int(seed),
                    "example_id": ex,
                    "metric_id": metric,
                    "value": 0.50,
                    "direction": direction,
                    "split": "search",
                }
            )
            rows.append(
                {
                    "arm_id": candidate_arm,
                    "seed": int(seed),
                    "example_id": ex,
                    "metric_id": metric,
                    "value": 0.50 + delta,
                    "direction": direction,
                    "split": "search",
                }
            )
    table_payload = {
        "schema": "observation_table/v1",
        "version": "v1",
        "campaign_id": manifest.campaign_id,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "locked_eval_manifest_sha256": manifest.locked_eval_manifest_sha256,
        "rows": rows,
    }
    hyp_ids = tuple(
        h
        for family in manifest.multiplicity_families
        for h in family.hypothesis_ids
    )
    plan_payload = {
        "schema": "analysis_plan/v1",
        "version": "v1",
        "primary_metric": metric,
        "primary_endpoint_id": primary.endpoint_id,
        "direction": direction,
        "minimum_effect": float(primary.minimum_effect),
        "control_arm_id": control_arm,
        "candidate_arm_id": candidate_arm,
        "holm_hypothesis_ids": list(hyp_ids),
        "holm_alpha": 0.05,
        "seal_split": "seal",
        "score_once": True,
        "underpowered_n_below": 2,
    }
    table = observation_table_from_mapping(table_payload)
    plan = analysis_plan_from_mapping(plan_payload)
    promo = next(iter(manifest.promotion_gates), None)
    roll = next(iter(manifest.rollback_gates), None)
    report = compute_credit_report(
        table,
        plan,
        promotion_gate_threshold=float(promo.threshold) if promo else None,
        promotion_gate_operator=str(promo.operator) if promo else "ge",
        rollback_gate_threshold=float(roll.threshold) if roll else None,
        rollback_gate_operator=str(roll.operator) if roll else "lt",
    )
    report_payload = report.to_dict()
    # Structural endpoint_values keys must match manifest endpoint ids only.
    endpoints = {
        endpoint.endpoint_id: report.paired_effect for endpoint in manifest.endpoints
    }
    holm = [
        {
            "hypothesis_id": h.hypothesis_id,
            "raw_p_value": h.raw_p_value,
            "rank": h.rank,
            "threshold": h.threshold,
            "adjusted_p_value": h.adjusted_p_value,
            "rejected": h.rejected,
        }
        for h in report.holm_results
    ]
    for kind, payload in (
        ("observation_table", table_payload),
        ("analysis_plan", plan_payload),
        ("credit_report", report_payload),
    ):
        path = artifact_root / f"{kind}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_payload, endpoints, holm


def _complete_result(
    manifest: ExperimentCampaignV1,
    artifact_root: Path,
    **updates: object,
) -> CampaignResultV1:
    artifacts = []
    credit_kinds = {"observation_table", "analysis_plan", "credit_report"}
    need_credit = any(
        r.kind in credit_kinds for r in manifest.artifact_requirements
    ) or manifest.claim_class in {"promotion_candidate", "ship_gate"}
    endpoints_override: dict[str, float] | None = None
    holm_override: list[dict[str, object]] | None = None
    if need_credit:
        _, endpoints_override, holm_override = _write_credit_bundle(
            manifest, artifact_root
        )
    for requirement in manifest.artifact_requirements:
        path = artifact_root / f"{requirement.kind}.json"
        if requirement.kind in credit_kinds and path.is_file():
            pass  # already written with semantic content
        elif not path.is_file():
            path.write_text(json.dumps({"kind": requirement.kind}), encoding="utf-8")
        artifacts.append(
            {
                "kind": requirement.kind,
                "uri": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    # Ensure credit artifacts present even if not in requirements (defense)
    for kind in credit_kinds:
        if need_credit and not any(a["kind"] == kind for a in artifacts):
            path = artifact_root / f"{kind}.json"
            if path.is_file():
                artifacts.append(
                    {
                        "kind": kind,
                        "uri": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    payload: dict[str, object] = {
        "campaign_id": manifest.campaign_id,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "claim_class": manifest.claim_class,
        "locked_eval_manifest_sha256": manifest.locked_eval_manifest_sha256,
        "arm_seed_results": [
            [arm.arm_id, seed] for arm in manifest.arms for seed in manifest.seeds
        ],
        "paired_example_ids": {
            arm.arm_id: ["example-1", "example-2"] for arm in manifest.arms
        },
        "endpoint_values": endpoints_override
        or {endpoint.endpoint_id: 0.02 for endpoint in manifest.endpoints},
        "primary_endpoint_seed_values": [0.02] * len(manifest.seeds),
        "holm_results": holm_override
        or [
            {
                "hypothesis_id": hypothesis,
                "raw_p_value": 0.01,
                "rank": rank,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            }
            for rank, hypothesis in enumerate(
                (
                    hypothesis
                    for family in manifest.multiplicity_families
                    for hypothesis in family.hypothesis_ids
                ),
                start=1,
            )
        ],
        "artifacts": artifacts,
    }
    payload.update(updates)
    return CampaignResultV1.model_validate(payload)


def test_manifest_digest_and_lock_roundtrip() -> None:
    manifest = _manifest()
    digest = campaign_manifest_sha256(manifest)
    lock = CampaignLockV1(
        manifest_sha256=digest,
        manifest=manifest,
        locked_at="2026-07-23T00:01:00Z",
    )

    restored = CampaignLockV1.model_validate_json(lock.model_dump_json())
    assert restored == lock
    assert restored.manifest_sha256 == campaign_manifest_sha256(restored.manifest)


def test_decision_bearing_mutation_changes_digest() -> None:
    manifest = _manifest()
    mutated = ExperimentCampaignV1.model_validate(
        _manifest_payload(decision="Reject unless every preregistered gate passes.")
    )

    assert campaign_manifest_sha256(mutated) != campaign_manifest_sha256(manifest)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("seeds", [7, 7], "seeds must be unique"),
        ("seeds", [True], "seeds must contain only integer identifiers"),
        (
            "endpoints",
            [
                {
                    "endpoint_id": "meaning",
                    "metric": "binder_reference_f1",
                    "role": "primary",
                    "direction": "increase",
                    "minimum_effect": float("nan"),
                }
            ],
            "minimum_effect must be finite",
        ),
    ],
)
def test_manifest_rejects_invalid_seed_and_nonfinite_values(
    field: str, value: object, error: str
) -> None:
    with pytest.raises((TypeError, ValidationError), match=error):
        ExperimentCampaignV1.model_validate(_manifest_payload(**{field: value}))


def test_manifest_rejects_duplicate_identifiers() -> None:
    arms = _manifest_payload()["arms"]
    assert isinstance(arms, list)
    duplicate = [dict(arms[0]), {**dict(arms[1]), "arm_id": "control"}]

    with pytest.raises(ValidationError, match="arm identifiers must be unique"):
        ExperimentCampaignV1.model_validate(_manifest_payload(arms=duplicate))


def _write_certification(
    path: Path, disposition: str, *, valid_digest: bool = True
) -> None:
    artifact = path.parent / "ap-001-evidence.json"
    artifact.write_text(
        json.dumps({"disposition": disposition, "metric": "meaning-v2"}),
        encoding="utf-8",
    )
    envelope = {
        "disposition": disposition,
        "artifact_path": artifact.name,
        "artifact_sha256": (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            if valid_digest
            else HEX_64
        ),
    }
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_ap001_primary_endpoint_requires_verified_certification(
    tmp_path: Path,
) -> None:
    certification_path = tmp_path / "ap-001.json"
    assert load_ap001_certification(certification_path) is None
    assert select_primary_endpoint(None) == "binder_reference_f1"

    _write_certification(certification_path, "certified", valid_digest=False)
    assert load_ap001_certification(certification_path) is None

    _write_certification(certification_path, "revise")
    revise = load_ap001_certification(certification_path)
    assert isinstance(revise, AP001CertificationV1)
    assert select_primary_endpoint(revise) == "binder_reference_f1"

    _write_certification(certification_path, "certified")
    certified = load_ap001_certification(certification_path)
    assert isinstance(certified, AP001CertificationV1)
    assert select_primary_endpoint(certified) == "binding_aware_meaningful_v2"


def test_complete_promotion_candidate_passes(tmp_path: Path) -> None:
    manifest = _manifest()

    assert (
        validate_result_claim(
            manifest,
            _complete_result(manifest, tmp_path),
            artifact_root=tmp_path,
        )
        == ()
    )


def test_promotion_result_requires_the_locked_manifest_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    missing = _complete_result(manifest, tmp_path, locked_eval_manifest_sha256=None)
    wrong = _complete_result(manifest, tmp_path, locked_eval_manifest_sha256="f" * 64)

    assert "locked_eval_manifest_sha256_missing" in validate_result_claim(
        manifest, missing, artifact_root=tmp_path
    )
    assert "locked_eval_manifest_sha256_mismatch" in validate_result_claim(
        manifest, wrong, artifact_root=tmp_path
    )


def test_promotion_candidate_fails_closed_on_incomplete_or_exploratory_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        arm_seed_results=[["control", 7]],
        endpoint_values={"meaning": -0.02},
        artifacts=[],
        exploratory=True,
    )

    failures = set(validate_result_claim(manifest, result, artifact_root=tmp_path))
    # Gate pass/fail is recomputed only from credit; without credit artifacts,
    # caller endpoint_values cannot authorize promotion/rollback outcomes.
    assert {
        "exploratory_result",
        "incomplete_arm_seed_results",
        "missing_artifact:version_stamp",
        "missing_artifact:seed_result",
        "missing_artifact:paired_examples",
        "missing_artifact:endpoint_result",
        "missing_artifact:holm_family",
        "missing_artifact:agentevals",
        "missing_artifact:agentv",
        "missing_credit_artifact:observation_table",
        "missing_credit_artifact:analysis_plan",
        "missing_credit_artifact:credit_report",
        "credit_summary_only_rejected",
    } <= failures
    assert "promotion_gates_not_passed" not in failures or "credit_summary_only_rejected" in failures


def test_promotion_candidate_rejects_empty_pairs_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        arm_seed_results=[
            ["control", 7],
            ["control", 11],
            ["candidate", 7],
            ["candidate", 11],
            ["candidate", 11],
        ],
        paired_example_ids={"control": [], "candidate": []},
        holm_results=[
            {
                "hypothesis_id": "meaning",
                "raw_p_value": 0.01,
                "rank": 1,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            },
            {
                "hypothesis_id": "meaning",
                "raw_p_value": 0.01,
                "rank": 1,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            },
        ],
    )
    failures = set(
        validate_result_claim(manifest, result, artifact_root=tmp_path)
    )
    assert "incomplete_arm_seed_results" in failures
    assert "incomplete_paired_examples" in failures
    assert "incomplete_holm_family" in failures


def test_promotion_result_locked_digest_is_verified_against_real_manifest_bytes(
    tmp_path: Path,
) -> None:
    """SLM-306: a declared locked_eval_manifest_sha256 is checked against disk.

    Without ``locked_manifest_path`` the self-reported digest string alone is
    trusted (unchanged, back-compatible behavior). When a path is supplied,
    the digest must actually be reproducible from real manifest bytes.
    """
    from slm_training.data.locked_eval_manifest import (
        build_locked_manifest,
        write_locked_manifest,
    )
    from slm_training.dsl.schema import load_jsonl

    candidates = load_jsonl("src/slm_training/resources/test_seeds.jsonl")[:4]
    locked = build_locked_manifest(
        candidates, source_records=[], min_locked_records=1, partition_size=1
    )
    manifest_path = tmp_path / "locked_manifest.json"
    digest = write_locked_manifest(manifest_path, locked)

    manifest = _manifest(locked_eval_manifest_sha256=digest)
    result = _complete_result(manifest, tmp_path, locked_eval_manifest_sha256=digest)

    assert (
        validate_result_claim(
            manifest, result, artifact_root=tmp_path, locked_manifest_path=None
        )
        == ()
    )
    assert (
        validate_result_claim(
            manifest,
            result,
            artifact_root=tmp_path,
            locked_manifest_path=manifest_path,
        )
        == ()
    )

    missing_path = tmp_path / "does_not_exist.json"
    failures = validate_result_claim(
        manifest,
        result,
        artifact_root=tmp_path,
        locked_manifest_path=missing_path,
    )
    assert "locked_eval_manifest_digest_unverified_on_disk" in failures

    wrong_digest_manifest = _manifest(locked_eval_manifest_sha256="f" * 64)
    wrong_digest_result = _complete_result(
        wrong_digest_manifest, tmp_path, locked_eval_manifest_sha256="f" * 64
    )
    failures_wrong = validate_result_claim(
        wrong_digest_manifest,
        wrong_digest_result,
        artifact_root=tmp_path,
        locked_manifest_path=manifest_path,
    )
    assert "locked_eval_manifest_digest_unverified_on_disk" in failures_wrong


def test_ship_gate_claim_requires_ship_gates_to_pass(tmp_path: Path) -> None:
    requirements = list(_manifest_payload()["artifact_requirements"])
    requirements.append({"kind": "ship_gates", "minimum_count": 1})
    manifest = _manifest(
        claim_class="ship_gate",
        artifact_requirements=requirements,
    )
    result = _complete_result(manifest, tmp_path, ship_gates_passed=False)

    failures = validate_result_claim(manifest, result, artifact_root=tmp_path)
    assert "ship_gates_not_passed" in failures
