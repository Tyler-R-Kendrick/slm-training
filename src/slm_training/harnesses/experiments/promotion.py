"""Promotion-protocol evaluation (P1c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from slm_training.autoresearch.experiment_campaign import (
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    validate_result_claim,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.promotion_engine import (
    PromotionCriteria,
    check_rank_stability,
)
from slm_training.harness_core.promotion_engine import (
    check_category_regression as _check_category_regression,
)
from slm_training.harness_core.promotion_engine import (
    evaluate_promotion as _evaluate_promotion,
)
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

__all__ = [
    "HARD_CATEGORIES",
    "PromotionCriteria",
    "check_category_regression",
    "check_data_integrity",
    "check_rank_stability",
    "evaluate_promotion",
    "load_campaign_governance",
    "register_promoted_checkpoint",
]

HARD_CATEGORIES = ("binding", "structural", "repair")


def load_campaign_governance(
    *,
    manifest_path: Path,
    result_path: Path,
    store_root: Path,
    artifact_root: Path,
) -> tuple[
    ExperimentCampaignV1,
    CampaignResultV1,
    CampaignStore,
    Path,
]:
    """Load the four explicit inputs required by promotion entrypoints."""
    manifest = ExperimentCampaignV1.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = CampaignResultV1.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    store = CampaignStore(manifest.campaign_id, store_root)
    failures = store.validate_campaign_result(result, artifact_root=artifact_root)
    if failures:
        raise ValueError(
            f"campaign governance validation failed: {', '.join(failures)}"
        )
    return manifest, result, store, artifact_root


def _openui_gate_evaluator(
    suites: dict[str, dict[str, Any]],
    policy: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    return evaluate_ship_gates(suites, thresholds=policy or DEFAULT_SHIP_GATES)


def check_data_integrity(
    train_dir: Path | str,
    test_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Lightweight integrity: train manifest exists + optional leakage scan."""
    from slm_training.data.leakage import find_leakage, load_train_fingerprints
    from slm_training.dsl.schema import load_jsonl

    train_dir = Path(train_dir)
    manifest = train_dir / "manifest.json"
    records_path = train_dir / "records.jsonl"
    failures: list[str] = []
    if not records_path.exists():
        failures.append("missing_train_records")
    if not manifest.exists():
        failures.append("missing_train_manifest")
    leakage_hits = 0
    if test_dir is not None and manifest.exists():
        fps = load_train_fingerprints(manifest)
        suites_root = Path(test_dir) / "suites"
        if suites_root.exists():
            for suite_dir in sorted(suites_root.iterdir()):
                rec_path = suite_dir / "records.jsonl"
                if not rec_path.exists():
                    continue
                for record in load_jsonl(rec_path):
                    hits = find_leakage(record, fps)
                    leakage_hits += len(hits)
        if leakage_hits:
            failures.append(f"leakage_hits:{leakage_hits}")
    return {
        "pass": not failures,
        "failures": failures,
        "leakage_hits": leakage_hits,
    }


def check_category_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """No hard category may regress more than ``tolerance`` relatively."""
    return _check_category_regression(
        baseline,
        candidate,
        categories=HARD_CATEGORIES,
        tolerance=tolerance,
    )


def evaluate_promotion(
    *,
    integrity: dict[str, Any] | None = None,
    baseline_loss_report: dict[str, Any] | None = None,
    candidate_loss_report: dict[str, Any] | None = None,
    rankings: dict[str, list[str]] | None = None,
    eg_time_by_seed: Sequence[float] | None = None,
    ship_suites: dict[str, dict[str, Any]] | None = None,
    criteria: PromotionCriteria | None = None,
    campaign_manifest: ExperimentCampaignV1 | dict[str, Any] | None = None,
    campaign_result: CampaignResultV1 | dict[str, Any] | None = None,
    campaign_store: CampaignStore | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Return ``{promotable, checks, failures}`` mirroring ship-gates shape."""
    result = _evaluate_promotion(
        integrity=integrity,
        baseline_loss_report=baseline_loss_report,
        candidate_loss_report=candidate_loss_report,
        rankings=rankings,
        eg_time_by_seed=eg_time_by_seed,
        ship_suites=ship_suites,
        criteria=criteria,
        hard_categories=HARD_CATEGORIES,
        gate_evaluator=_openui_gate_evaluator,
    )
    if campaign_manifest is None or campaign_result is None:
        governance_failures = ("campaign_governance_missing",)
        manifest_sha = None
    else:
        manifest = (
            campaign_manifest
            if isinstance(campaign_manifest, ExperimentCampaignV1)
            else ExperimentCampaignV1.model_validate(campaign_manifest)
        )
        governed_result = (
            campaign_result
            if isinstance(campaign_result, CampaignResultV1)
            else CampaignResultV1.model_validate(campaign_result)
        )
        governance_failures = validate_result_claim(
            manifest,
            governed_result,
            artifact_root=artifact_root,
        )
        if campaign_store is None or artifact_root is None:
            governance_failures = (*governance_failures, "campaign_store_missing")
        else:
            governance_failures = (
                *governance_failures,
                *campaign_store.validate_campaign_result(
                    governed_result,
                    artifact_root=artifact_root,
                ),
            )
        if governed_result.claim_class not in {"promotion_candidate", "ship_gate"}:
            governance_failures = (
                *governance_failures,
                "claim_class_not_promotable",
            )
        governance_failures = tuple(dict.fromkeys(governance_failures))
        manifest_sha = campaign_manifest_sha256(manifest)
    governance = {
        "pass": not governance_failures,
        "failures": list(governance_failures),
        "manifest_sha256": manifest_sha,
    }
    result.setdefault("checks", {})["campaign_governance"] = governance
    if not governance_failures and result.get("failures") == ["sufficient_evidence"]:
        result["checks"].pop("sufficient_evidence", None)
        result["checks"]["governed_campaign_evidence"] = {"pass": True}
        result["failures"] = []
        result["promotable"] = True
    if governance_failures:
        result["promotable"] = False
        result.setdefault("failures", []).extend(governance_failures)
    return result


def register_promoted_checkpoint(
    checkpoint_dir: Path | str,
    *,
    source: Path | str | None = None,
    meta: dict[str, Any] | None = None,
    promotion_result: dict[str, Any] | None = None,
    campaign_manifest: ExperimentCampaignV1 | dict[str, Any] | None = None,
    campaign_result: CampaignResultV1 | dict[str, Any] | None = None,
    campaign_store: CampaignStore | None = None,
    artifact_root: Path | None = None,
) -> Path:
    """Copy/link the mid-trained anchor to ``promoted.pt`` (P1d)."""
    import shutil

    governance = (
        (promotion_result or {}).get("checks", {}).get("campaign_governance", {})
    )
    manifest = (
        campaign_manifest
        if isinstance(campaign_manifest, ExperimentCampaignV1)
        else (
            ExperimentCampaignV1.model_validate(campaign_manifest)
            if campaign_manifest is not None
            else None
        )
    )
    governed_result = (
        campaign_result
        if isinstance(campaign_result, CampaignResultV1)
        else (
            CampaignResultV1.model_validate(campaign_result)
            if campaign_result is not None
            else None
        )
    )
    independently_verified = (
        manifest is not None
        and governed_result is not None
        and governed_result.claim_class in {"promotion_candidate", "ship_gate"}
        and campaign_store is not None
        and artifact_root is not None
        and not campaign_store.validate_campaign_result(
            governed_result,
            artifact_root=artifact_root,
        )
        and governance.get("manifest_sha256") == campaign_manifest_sha256(manifest)
    )
    if (
        not promotion_result
        or not promotion_result.get("promotable")
        or governance.get("pass") is not True
        or not governance.get("manifest_sha256")
        or not independently_verified
    ):
        raise ValueError(
            "checkpoint registration requires a promotable campaign-governed result"
        )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / "promoted.pt"
    if source is not None:
        source = Path(source)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    meta_path = checkpoint_dir / "promoted.json"
    payload = {
        **(meta or {}),
        "kind": "promoted_anchor",
        "campaign_manifest_sha256": governance["manifest_sha256"],
    }
    meta_path.write_text(
        __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return dest
