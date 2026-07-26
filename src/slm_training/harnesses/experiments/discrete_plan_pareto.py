"""AP-027 (SLM-322): discrete-plan hybrid quality-latency Pareto contract.

This owner extends the AP-022 (SLM-313) arm/campaign pattern
(``abstract_plan_functional_evidence``) with a refinement-round axis and the
arms AP-027's own issue text names but AP-022 never ran: ``oracle_plan``
(aliasing AP-022's ``gold_semantic_plan``), ``current_auxiliary_heads``,
``learned_plan_binder_precision``, and ``retrieval_baseline``.

AP-022 already produced a locked, complete-matrix disposition for
``learned_abstract_plan``: **ignored_or_collapsed**
(``docs/design/abstract-plan-functional-evidence.md``,
``docs/design/iter-slm313-abstract-plan-functional-evidence-20260726.json``).
Per decode-invariant I14 (a rejected experiment closes an *approach*, never a
*goal*), that verdict permanently retires ``learned_abstract_plan`` as a
promotion candidate here -- it is never re-litigated by this module -- but it
does not retire AP-027's goal. ``learned_plan_binder_precision`` (the plan
connector plus an explicit binder-precision channel, an architecture AP-022
never tested) is the named successor approach and the only arm this module
will ever let ``build_campaign`` mark as ``role="candidate"``.
``KNOWN_COLLAPSED_ARMS`` is enforced structurally: :meth:`ParetoLatencyRow.validate`
raises if a known-collapsed arm is ever marked ``promotion_eligible``,
regardless of what a caller's measured metrics say.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.harnesses.experiments.abstract_plan_functional_evidence import (
    ARMS as AP022_ARMS,
    PATHS,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

# AP-022's locked, ship-closed verdict for this arm (see module docstring).
# Never re-derived here; this module only reads and respects it.
AP022_DISPOSITION: Mapping[str, Any] = {
    "campaign_id": "slm313-abstract-plan-functional-evidence",
    "verdict": "ignored_or_collapsed",
    "doc": "docs/design/abstract-plan-functional-evidence.md",
    "result_json": "docs/design/iter-slm313-abstract-plan-functional-evidence-20260726.json",
}
KNOWN_COLLAPSED_ARMS: frozenset[str] = frozenset({"learned_abstract_plan"})
SUCCESSOR_CANDIDATE_ARM: str = "learned_plan_binder_precision"

# AP-022's arms remain available as matched controls (destructive controls,
# the oracle/`gold_semantic_plan` upper bound, and the now-closed learned arm
# retained only as a latency/diagnostic reference point -- never a candidate).
# `oracle_plan` is a documented alias so AP-027's own issue-text vocabulary
# ("oracle plan") does not silently diverge from AP-022's `gold_semantic_plan`.
ARM_ALIASES: Mapping[str, str] = {"oracle_plan": "gold_semantic_plan"}
NEW_ARMS = (
    "current_auxiliary_heads",
    SUCCESSOR_CANDIDATE_ARM,
    "retrieval_baseline",
)
ARMS: tuple[str, ...] = tuple(AP022_ARMS) + NEW_ARMS
REFINEMENT_ROUNDS: tuple[int, ...] = (1, 2, 4, 8)
PRIMARY_METRICS = ("binding_aware_meaningful_v2", "binder_reference_f1")


def canonical_arm(arm: str) -> str:
    """Resolve an issue-text arm alias to its canonical (measured) arm id."""
    return ARM_ALIASES.get(arm, arm)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha_stamp() -> dict[str, Any]:
    stamp = build_version_stamp(
        "harness.experiments.ap027_discrete_plan_pareto",
        "harness.experiments.slm313_abstract_plan_functional_evidence",
    )
    return {"code_commit": str(stamp["code_commit"]), "code_dirty": bool(stamp["code_dirty"])}


@dataclass(frozen=True)
class ParetoLatencyRow:
    """One (arm, refinement_round, decode path) measurement.

    ``total_latency_ms`` is a wall-clock sum of the other three phase
    latencies within tolerance (see :meth:`validate`); the phase split lets a
    reader attribute cost to planning versus generation versus verification
    rather than reading one opaque number.
    """

    record_id: str
    arm: str
    refinement_round: int
    path: str
    total_latency_ms: float
    plan_latency_ms: float
    generation_latency_ms: float
    verification_latency_ms: float
    peak_memory_proxy: float
    output_tokens: int
    metrics: Mapping[str, float]
    promotion_eligible: bool = False

    def validate(self) -> None:
        arm = canonical_arm(self.arm)
        if arm not in ARMS:
            raise ValueError(f"unknown discrete-plan-pareto arm {self.arm!r}")
        if self.path not in PATHS:
            raise ValueError(f"unknown decode path {self.path!r}")
        if self.refinement_round not in REFINEMENT_ROUNDS:
            raise ValueError(
                f"refinement_round must be one of {REFINEMENT_ROUNDS}, got {self.refinement_round}"
            )
        latencies = (
            self.total_latency_ms,
            self.plan_latency_ms,
            self.generation_latency_ms,
            self.verification_latency_ms,
        )
        if any(value < 0 or not math.isfinite(value) for value in latencies):
            raise ValueError("latencies must be finite and non-negative")
        phase_sum = self.plan_latency_ms + self.generation_latency_ms + self.verification_latency_ms
        if phase_sum - self.total_latency_ms > 1e-6:
            raise ValueError("phase latencies must not exceed the reported total")
        if self.peak_memory_proxy < 0 or not math.isfinite(self.peak_memory_proxy):
            raise ValueError("peak_memory_proxy must be finite and non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("metrics must be finite")
        if self.path == "raw" and self.promotion_eligible:
            raise ValueError("raw decode is diagnostic-only and cannot be promoted")
        # Structural I14 enforcement: AP-022's closed verdict can never be
        # silently overridden by a later measurement claiming eligibility.
        if arm in KNOWN_COLLAPSED_ARMS and self.promotion_eligible:
            raise ValueError(
                f"{arm!r} is closed by AP-022 ({AP022_DISPOSITION['verdict']}); "
                "it can never be marked promotion_eligible here"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(asdict(self))


def pareto_frontier(
    points: Iterable[tuple[str, float, float]],
) -> tuple[str, ...]:
    """Return the ids of points non-dominated on (lower latency, higher quality).

    Each point is ``(point_id, latency_ms, quality)``. A point is dominated
    when another point is at least as good on both axes and strictly better
    on at least one; ties on both axes keep both ids (neither dominates the
    other).
    """
    materialized = list(points)
    keep: list[str] = []
    for point_id, latency, quality in materialized:
        dominated = any(
            other_id != point_id
            and other_latency <= latency
            and other_quality >= quality
            and (other_latency < latency or other_quality > quality)
            for other_id, other_latency, other_quality in materialized
        )
        if not dominated:
            keep.append(point_id)
    return tuple(keep)


def build_campaign() -> ExperimentCampaignV1:
    """Lock AP-027 before any Pareto outcome is observed.

    ``learned_plan_binder_precision`` is the sole ``candidate`` arm: the
    named successor to AP-022's closed ``learned_abstract_plan`` verdict.
    Every other arm -- including ``learned_abstract_plan`` itself -- is a
    matched control, and ``learned_abstract_plan`` is additionally recorded
    as a negative control (per its own closed verdict) rather than left to a
    reader to rediscover.
    """
    stamp = _sha_stamp()
    return ExperimentCampaignV1(
        campaign_id="ap027-discrete-plan-pareto",
        experiment_id="slm322-ap027",
        hypothesis=(
            "A learned discrete plan conditioned through an explicit binder-precision "
            "channel improves the quality-latency Pareto frontier across 1/2/4/8 "
            "refinement rounds versus no-plan and matched destructive controls."
        ),
        decision="Determine whether any discrete-plan arm is non-dominated on quality vs. total p95 latency.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="meaning_v2",
                metric="binding_aware_meaningful_v2",
                role="primary",
                direction="increase",
                minimum_effect=0.0,
            ),
            CampaignEndpointV1(
                endpoint_id="binder_f1",
                metric="binder_reference_f1",
                role="secondary",
                direction="increase",
                minimum_effect=0.0,
            ),
        ),
        arms=tuple(
            CampaignArmV1(
                arm_id=arm,
                role="candidate" if arm == SUCCESSOR_CANDIDATE_ARM else "control",
                config_sha256=_sha({"campaign": "ap027-discrete-plan-pareto", "arm": arm}),
            )
            for arm in ARMS
        ),
        seeds=(322,),
        budget=CampaignBudget(max_experiments=len(ARMS) * len(REFINEMENT_ROUNDS), max_wall_minutes=float(MAX_RUN_MINUTES)),
        stopping_rules=(
            "learned_abstract_plan is closed (ignored_or_collapsed, AP-022/SLM-313); "
            "it is never re-evaluated as a candidate here.",
            "Candidate arms requiring an untrained connector/index "
            "(learned_plan_binder_precision, retrieval_baseline) report unavailable "
            "until that prerequisite artifact exists.",
            "Raw decode is diagnostic-only and cannot determine promotion.",
        ),
        controls=tuple(
            CampaignControlV1(
                control_id=arm,
                description=(
                    f"AP-022 closed verdict ({AP022_DISPOSITION['verdict']}): retained as latency reference only."
                    if arm in KNOWN_COLLAPSED_ARMS
                    else f"Matched discrete-plan-pareto arm: {arm}."
                ),
                kind="negative" if arm in KNOWN_COLLAPSED_ARMS else "quality",
            )
            for arm in ARMS
            if arm != SUCCESSOR_CANDIDATE_ARM
        ),
        negative_controls=tuple(KNOWN_COLLAPSED_ARMS),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="pareto_frontier",
                hypothesis_ids=("meaning_v2", "binder_f1"),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="non_dominated_candidate",
                endpoint_id="meaning_v2",
                operator="gt",
                threshold=0.0,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="dominated_candidate",
                endpoint_id="meaning_v2",
                operator="le",
                threshold=0.0,
            ),
        ),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="paired_examples"),
        ),
        claim_class="wiring",
        source_commit=stamp["code_commit"],
        source_dirty=stamp["code_dirty"],
        author="slm-training",
        created_at="2026-07-26T00:00:00Z",
    )


def summarize_pareto(rows: Iterable[ParetoLatencyRow]) -> dict[str, Any]:
    """Non-dominated (arm, refinement_round) points from measured rows only.

    Arms requiring artifacts this PR does not train/build
    (``learned_plan_binder_precision``, ``retrieval_baseline``, and the
    already-closed ``learned_abstract_plan``/``oracle_plan``) are reported as
    ``arms_pending``/``arms_closed`` rather than silently absent.
    """
    materialized = [row for row in rows]
    for row in materialized:
        row.validate()
    measured_arms = {canonical_arm(row.arm) for row in materialized}
    pending = tuple(
        sorted(
            arm
            for arm in ARMS
            if arm not in measured_arms and arm not in KNOWN_COLLAPSED_ARMS
        )
    )
    closed = tuple(sorted(KNOWN_COLLAPSED_ARMS))
    points = [
        (
            f"{row.arm}@{row.refinement_round}",
            row.total_latency_ms,
            float(row.metrics.get("binding_aware_meaningful_v2", 0.0)),
        )
        for row in materialized
        if row.path == "constrained"
    ]
    return {
        "schema": "discrete_plan_pareto_summary/v1",
        "measured_arms": sorted(measured_arms),
        "arms_pending": pending,
        "arms_closed": closed,
        "ap022_disposition": dict(AP022_DISPOSITION),
        "non_dominated": pareto_frontier(points),
        "claim_class": "wiring",
        "ship_eligible": False,
    }
