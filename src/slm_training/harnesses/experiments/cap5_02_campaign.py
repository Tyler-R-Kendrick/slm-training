"""CAP5-02 (SLM-101) campaign manifest contract.

This module defines a frozen, versioned campaign manifest for the final
end-to-end quality / memory / latency / energy comparison. It is intentionally
plan-only: it records the preregistered arms, hypotheses, matched fields, and
omission reasons without launching expensive training or eval runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

MANIFEST_SCHEMA = "cap5_02_campaign/v1"
CAMPAIGN_ID = "cap5-02"


def _stable_hash(parts: Mapping[str, Any]) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CampaignArm:
    """One preregistered experimental arm in the CAP5-02 campaign.

    An arm corresponds to a surviving architecture/latent/quantization/dynamic-
    compute/sparsity candidate from CAP1-CAP4. Omitted mechanisms are recorded
    as arms with ``eligible=False`` and an ``omission_reason``.
    """

    arm_id: str
    hypothesis_id: str
    mechanism: str
    eligible: bool
    omission_reason: str | None = None
    selection_evidence: str | None = None
    ablation_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "hypothesis_id": self.hypothesis_id,
            "mechanism": self.mechanism,
            "eligible": self.eligible,
            "omission_reason": self.omission_reason,
            "selection_evidence": self.selection_evidence,
            "ablation_of": self.ablation_of,
        }


@dataclass(frozen=True)
class Cap5CampaignManifest:
    """Frozen preregistered campaign manifest for CAP5-02.

    The manifest must be committed before expensive execution. Any change
    requires a new manifest version and explanation.
    """

    campaign_id: str
    manifest_version: str
    schema_version: str
    arms: tuple[CampaignArm, ...]
    primary_metric: str
    comparison_regimes: tuple[str, ...]
    seeds: tuple[int, ...]
    hardware_targets: tuple[str, ...]
    no_test_peeking: bool
    manifest_hash: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "manifest_version": self.manifest_version,
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "comparison_regimes": list(self.comparison_regimes),
            "seeds": list(self.seeds),
            "hardware_targets": list(self.hardware_targets),
            "no_test_peeking": self.no_test_peeking,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }


def build_cap5_campaign_manifest(
    arms: Iterable[CampaignArm],
    *,
    manifest_version: str,
    primary_metric: str = "binding_aware_meaningful_program_rate",
    comparison_regimes: Iterable[str] = (
        "equal_actual_bytes",
        "equal_measured_latency",
        "equal_training_compute",
        "unconstrained_best_quality",
    ),
    seeds: Iterable[int] = (0, 1, 2),
    hardware_targets: Iterable[str] = ("cpu", "cuda"),
    no_test_peeking: bool = True,
    note: str = "",
) -> Cap5CampaignManifest:
    """Build a deterministic CAP5-02 campaign manifest.

    ``manifest_version`` must be bumped whenever the manifest changes after
    results have been inspected.
    """
    arm_list = tuple(arms)
    arm_hashes = [_stable_hash(a.to_dict()) for a in arm_list]
    manifest_hash = _stable_hash(
        {
            "campaign_id": CAMPAIGN_ID,
            "manifest_version": manifest_version,
            "primary_metric": primary_metric,
            "comparison_regimes": list(comparison_regimes),
            "seeds": list(seeds),
            "hardware_targets": list(hardware_targets),
            "no_test_peeking": no_test_peeking,
            "arm_hashes": arm_hashes,
        }
    )
    return Cap5CampaignManifest(
        campaign_id=CAMPAIGN_ID,
        manifest_version=manifest_version,
        schema_version=MANIFEST_SCHEMA,
        arms=arm_list,
        primary_metric=primary_metric,
        comparison_regimes=tuple(comparison_regimes),
        seeds=tuple(seeds),
        hardware_targets=tuple(hardware_targets),
        no_test_peeking=no_test_peeking,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_cap5_campaign_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return validation errors; empty means valid."""
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        errors.append(f"campaign_id must be {CAMPAIGN_ID!r}")
    if not isinstance(manifest.get("manifest_version"), str) or not manifest.get("manifest_version"):
        errors.append("manifest_version must be a non-empty string")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
        return errors
    for idx, arm in enumerate(arms):
        prefix = f"arms[{idx}]"
        if not isinstance(arm, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ("arm_id", "hypothesis_id", "mechanism"):
            if not arm.get(key):
                errors.append(f"{prefix} missing or empty {key!r}")
        if "eligible" not in arm:
            errors.append(f"{prefix} missing 'eligible'")
        elif arm.get("eligible") is False and not arm.get("omission_reason"):
            errors.append(f"{prefix} omitted arm must have omission_reason")
    if not isinstance(manifest.get("primary_metric"), str) or not manifest.get("primary_metric"):
        errors.append("primary_metric must be a non-empty string")
    for key in ("comparison_regimes", "seeds", "hardware_targets"):
        if not isinstance(manifest.get(key), list) or not manifest.get(key):
            errors.append(f"{key} must be a non-empty list")
    if not isinstance(manifest.get("no_test_peeking"), bool):
        errors.append("no_test_peeking must be a boolean")
    return errors
