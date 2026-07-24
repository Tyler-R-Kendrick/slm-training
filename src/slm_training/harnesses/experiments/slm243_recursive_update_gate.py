"""Verdict contract for the SLM-243 recursive-update architecture matrix."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence


class RecursiveUpdateVerdict(str, Enum):
    CURRENT_V1_STABLE = "current_v1_stable"
    DELTA_ONLY_PREFERRED = "delta_only_preferred"
    LAYERSCALE_PREFERRED = "layerscale_preferred"
    GATED_PREFERRED = "gated_preferred"
    TRUE_EMPTY_IDENTITY_REQUIRED = "true_empty_identity_required"
    PRIVATE_NORMS_POSITIVE = "private_norms_positive"
    NO_STABLE_RECURSIVE_REGIME = "no_stable_recursive_regime"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RecursiveUpdateGateV1:
    verdict: str
    maximum_authorized_depth: int
    allowed_slm233_modes: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    selected_variant: str | None
    rationale: str
    evidence_refs: tuple[str, ...]
    schema: str = "RecursiveUpdateGateV1"

    def validate(self) -> None:
        if self.verdict not in {item.value for item in RecursiveUpdateVerdict}:
            raise ValueError(f"unsupported recursive-update verdict: {self.verdict}")
        if self.maximum_authorized_depth not in {0, 1, 2, 4, 6, 8}:
            raise ValueError("maximum_authorized_depth is outside the measured grid")
        if not self.blocked_claims or not self.evidence_refs:
            raise ValueError("blocked claims and evidence references are required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def classify_recursive_update_gate(
    rows: Sequence[dict[str, Any]],
    *,
    depths: Sequence[int],
    seeds: Sequence[int],
) -> RecursiveUpdateGateV1:
    """Apply the preregistered finite/high-depth/paired-seed architecture gate."""
    expected = {
        (variant, depth, seed)
        for variant in (
            "current_v1",
            "delta_only",
            "layerscale",
            "gated_private",
            "current_true_empty",
            "layerscale_private",
        )
        for depth in depths
        for seed in seeds
    }
    observed = {
        (str(row["variant"]), int(row["depth"]), int(row["seed"])) for row in rows
    }
    blocked = (
        "semantic_workspace",
        "checkpoint_promotion",
        "ship_readiness",
        "production_default_change",
    )
    refs = (
        "docs/design/iter-slm282-recurrence-health-20260723.json",
        "docs/design/iter-slm230-recurrence-observability-20260724.json",
        "docs/design/iter-slm231-recurrence-dynamics-20260724.json",
        "docs/design/iter-slm232-latent-state-use-20260724.json",
    )
    if observed != expected:
        return RecursiveUpdateGateV1(
            verdict=RecursiveUpdateVerdict.INCONCLUSIVE.value,
            maximum_authorized_depth=0,
            allowed_slm233_modes=(),
            blocked_claims=blocked,
            selected_variant=None,
            rationale="matrix coverage is incomplete",
            evidence_refs=refs,
        )

    high_depth = max(depths)
    by_key = {
        (str(row["variant"]), int(row["depth"]), int(row["seed"])): row
        for row in rows
    }
    stable: list[str] = []
    for variant in (
        "current_v1",
        "delta_only",
        "layerscale",
        "gated_private",
        "current_true_empty",
        "layerscale_private",
    ):
        candidate = [by_key[(variant, high_depth, seed)] for seed in seeds]
        if all(
            bool(row["all_finite"])
            and math.isfinite(float(row["cross_entropy"]))
            and float(row["maximum_update_ratio"]) <= 2.0
            and float(row["gradient_norm"]) <= 100.0
            for row in candidate
        ):
            stable.append(variant)

    if not stable:
        return RecursiveUpdateGateV1(
            verdict=RecursiveUpdateVerdict.NO_STABLE_RECURSIVE_REGIME.value,
            maximum_authorized_depth=0,
            allowed_slm233_modes=(),
            blocked_claims=blocked,
            selected_variant=None,
            rationale="no variant stayed inside finite high-depth stability bounds",
            evidence_refs=refs,
        )

    baseline = [by_key[("current_v1", high_depth, seed)] for seed in seeds]
    preferred = []
    for variant in stable:
        if variant == "current_v1":
            continue
        candidate = [by_key[(variant, high_depth, seed)] for seed in seeds]
        paired = [
            float(right["maximum_update_ratio"])
            < 0.8 * float(left["maximum_update_ratio"])
            and float(right["cross_entropy"]) <= float(left["cross_entropy"]) + 0.25
            for left, right in zip(baseline, candidate, strict=True)
        ]
        if all(paired):
            preferred.append(variant)

    priority = (
        "layerscale",
        "delta_only",
        "gated_private",
        "current_true_empty",
        "layerscale_private",
    )
    selected = next((name for name in priority if name in preferred), None)
    if selected is None:
        if "current_v1" in stable:
            return RecursiveUpdateGateV1(
                verdict=RecursiveUpdateVerdict.CURRENT_V1_STABLE.value,
                maximum_authorized_depth=high_depth,
                allowed_slm233_modes=("current_v1_diagnostic",),
                blocked_claims=blocked,
                selected_variant="current_v1",
                rationale="current V1 was finite, but no repair won all paired seeds",
                evidence_refs=refs,
            )
        return RecursiveUpdateGateV1(
            verdict=RecursiveUpdateVerdict.INCONCLUSIVE.value,
            maximum_authorized_depth=high_depth,
            allowed_slm233_modes=tuple(f"{name}_diagnostic" for name in stable),
            blocked_claims=blocked,
            selected_variant=None,
            rationale="stable repairs exist, but no repair cleared the paired-seed gate",
            evidence_refs=refs,
        )

    verdict = {
        "delta_only": RecursiveUpdateVerdict.DELTA_ONLY_PREFERRED,
        "layerscale": RecursiveUpdateVerdict.LAYERSCALE_PREFERRED,
        "gated_private": RecursiveUpdateVerdict.GATED_PREFERRED,
        "current_true_empty": RecursiveUpdateVerdict.TRUE_EMPTY_IDENTITY_REQUIRED,
        "layerscale_private": RecursiveUpdateVerdict.PRIVATE_NORMS_POSITIVE,
    }[selected]
    return RecursiveUpdateGateV1(
        verdict=verdict.value,
        maximum_authorized_depth=high_depth,
        allowed_slm233_modes=(f"{selected}_diagnostic",),
        blocked_claims=blocked,
        selected_variant=selected,
        rationale="selected repair improved high-depth update stability on all paired seeds",
        evidence_refs=refs,
    )
