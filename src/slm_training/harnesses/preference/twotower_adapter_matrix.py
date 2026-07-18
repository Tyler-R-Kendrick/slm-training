"""LDI2-03 TwoTower adapter-vs-full-update campaign matrix (SLM-126).

Orchestration-only harness for the matched TwoTower exact-state intervention
comparison. It is gated on the LDI2-02 (SLM-125) adapter-subspace diagnostic:
**only an explicit ``authorized`` decision may train.** ``repair_evidence``,
``no_safe_direction``, and ``expired`` all stop the campaign — the guard never
tunes LR/duration to bypass an unsafe direction, and it fails closed when the
diagnostic does not explicitly authorize a bounded direction.

Like the LDI1-03 harness this layer is Torch-free and runs no training itself:
admissible arms with no executable policy + admitted corpus resolve to
``expired`` and never fabricate metrics. The ship-grade run requires GPU, the
parent checkpoint used to mine the V2 events, and an admitted corpus. No quality
claim is made here; this is wiring only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal

__all__ = [
    "AuthorizationDecision",
    "ArmStatus",
    "TwoTowerArm",
    "ArmResult",
    "GuardMetrics",
    "read_authorization",
    "build_arms",
    "exact_signature_guard",
    "run_arm",
    "describe_campaign",
    "PROTECTED_SIGNATURE_METRICS",
]

# The four LDI2-02 diagnostic outcomes. Only ``authorized`` permits training.
AuthorizationDecision = Literal[
    "authorized", "repair_evidence", "no_safe_direction", "expired"
]

# ``expired`` == admissible but no executable policy + admitted corpus here.
ArmStatus = Literal["admitted", "blocked", "expired", "restored_parent", "completed"]
_VALID_STATUS: frozenset[str] = frozenset(
    {"admitted", "blocked", "expired", "restored_parent", "completed"}
)

# Exact-objective-signature held-out metrics the guard protects. A regression in
# any one backtracks/rejects the update — never only an aggregate view.
PROTECTED_SIGNATURE_METRICS: tuple[str, ...] = (
    "held_out_loss",
    "good_mass",
    "bad_mass",
    "margin",
    "locality",
)


def read_authorization(geometry_report: Mapping[str, Any]) -> tuple[AuthorizationDecision, str]:
    """Map a LDI2-02 diagnostic report to an authorization decision.

    Fail-closed: only an explicit ``result.decision == "authorized"`` authorizes
    training. A refused (``not_authorized``) or resultless completed report is
    treated as ``no_safe_direction`` — the safe default that stops the campaign
    rather than training on an unauthorized direction.
    """
    status = geometry_report.get("status")
    if status == "expired":
        return "expired", "diagnostic expired: fix the diagnostic, do not train blindly"
    if status == "not_authorized":
        return (
            "no_safe_direction",
            str(geometry_report.get("reason", "diagnostic refused the request")),
        )
    result = geometry_report.get("result")
    if not isinstance(result, Mapping):
        return "no_safe_direction", "completed diagnostic carried no result to authorize"
    decision = result.get("decision")
    if decision == "authorized":
        return "authorized", str(result.get("reason", "bounded direction authorized"))
    if decision in ("repair_evidence", "no_safe_direction"):
        return decision, str(result.get("reason", str(decision)))
    # Unknown / absent decision -> fail closed.
    return "no_safe_direction", f"unrecognized diagnostic decision {decision!r}; failing closed"


@dataclass(frozen=True)
class TwoTowerArm:
    """One matched arm. ``update_space`` names the actuator; ``rank`` is None for
    the parent (T0) and the full-update control (T1)."""

    arm_id: str
    label: str
    update_space: Literal["parent", "full", "adapter"]
    rank: int | None = None
    reference_tether: bool = False
    role: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "update_space": self.update_space,
            "rank": self.rank,
            "reference_tether": self.reference_tether,
            "role": self.role,
        }


@dataclass(frozen=True)
class GuardMetrics:
    """Held-out exact-signature metrics before/after a proposed update."""

    pre: Mapping[str, float]
    post: Mapping[str, float]


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    status: ArmStatus
    reason: str
    trainable_parameters: int | None = None
    metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(f"invalid arm status: {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "status": self.status,
            "reason": self.reason,
            "trainable_parameters": self.trainable_parameters,
            "metrics": dict(self.metrics) if self.metrics is not None else None,
        }


def build_arms(
    *,
    authorized_rank: int,
    lower_rank: int | None = None,
    higher_rank: int | None = None,
    include_tether_ablation: bool = True,
    campaign: str = "LDI-twotower-adapter",
) -> list[TwoTowerArm]:
    """Build the matched T0–T5 arm set around the authorized rank.

    T0 parent (no update) · T1 full-update control · T2 adapter@authorized ·
    T3 lower-rank · T4 higher-rank · T5 tether ablation. Capacity controls are
    included only when their ranks are supplied/permitted.
    """
    if authorized_rank <= 0:
        raise ValueError("authorized_rank must be positive")
    arms = [
        TwoTowerArm(f"{campaign}/t0", "T0-parent", "parent", role="current-code control"),
        TwoTowerArm(f"{campaign}/t1", "T1-full", "full", role="historical-method control"),
        TwoTowerArm(
            f"{campaign}/t2",
            "T2-adapter",
            "adapter",
            rank=authorized_rank,
            role="primary removable intervention",
        ),
    ]
    if lower_rank is not None:
        arms.append(
            TwoTowerArm(
                f"{campaign}/t3", "T3-lower", "adapter", rank=lower_rank, role="capacity control"
            )
        )
    if higher_rank is not None:
        arms.append(
            TwoTowerArm(
                f"{campaign}/t4", "T4-higher", "adapter", rank=higher_rank, role="capacity control"
            )
        )
    if include_tether_ablation:
        arms.append(
            TwoTowerArm(
                f"{campaign}/t5",
                "T5-tether",
                "adapter",
                rank=authorized_rank,
                reference_tether=True,
                role="locality ablation",
            )
        )
    return arms


def exact_signature_guard(metrics: GuardMetrics, *, tolerance: float = 1e-9) -> tuple[bool, str]:
    """Accept an update only if no protected exact-signature metric regresses.

    ``held_out_loss`` and ``bad_mass`` must not increase; ``good_mass`` and
    ``margin`` must not decrease; ``locality`` (distance from parent) must not
    increase. Any regression rejects the update (caller backtracks/restores).
    """
    pre, post = metrics.pre, metrics.post
    for key in ("held_out_loss", "bad_mass", "locality"):
        if key in pre and key in post and post[key] > pre[key] + tolerance:
            return False, f"{key} regressed ({pre[key]:.6g} -> {post[key]:.6g})"
    for key in ("good_mass", "margin"):
        if key in pre and key in post and post[key] < pre[key] - tolerance:
            return False, f"{key} regressed ({pre[key]:.6g} -> {post[key]:.6g})"
    return True, "all protected signatures held"


def run_arm(
    arm: TwoTowerArm,
    *,
    decision: AuthorizationDecision,
    corpus_admitted: bool,
    policy_factory: Callable[[TwoTowerArm], Any] | None = None,
    train_items: Sequence[Any] = (),
) -> ArmResult:
    """Classify then optionally execute one arm.

    Blocks every trainable arm unless the diagnostic ``authorized`` a direction
    and the corpus is admitted; the T0 parent stays admissible as the control.
    With no executable policy + corpus the admissible arm ``expired`` — never a
    fabricated metric. (Real execution wires the SLM-123 adapter / full-update
    trainer behind ``policy_factory`` on GPU.)
    """
    is_parent = arm.update_space == "parent"
    if not is_parent:
        if decision != "authorized":
            return ArmResult(arm.arm_id, "blocked", f"not authorized: {decision}")
        if not corpus_admitted:
            return ArmResult(arm.arm_id, "blocked", "blocked_by_corpus: corpus not admitted")

    if policy_factory is None or (not is_parent and not train_items):
        return ArmResult(
            arm.arm_id,
            "expired",
            "no executable policy + admitted corpus in this environment "
            "(ship-grade run requires GPU + parent checkpoint + admitted corpus)",
        )
    # Execution path is intentionally deferred to the GPU run; the guarded
    # planner above is what this issue delivers on CPU.
    return ArmResult(arm.arm_id, "expired", "execution deferred to authorized GPU run")


def describe_campaign(
    arms: Sequence[TwoTowerArm], *, decision: AuthorizationDecision
) -> dict[str, Any]:
    """Deterministic, JSON-safe dry-run description of the arm matrix + gate."""
    return {
        "authorization": decision,
        "trainable_permitted": decision == "authorized",
        "arm_count": len(arms),
        "arms": [arm.as_dict() for arm in arms],
        "protected_signature_metrics": list(PROTECTED_SIGNATURE_METRICS),
        "claim": "wiring only; no quality claim; ship-grade run requires GPU + checkpoint",
    }
