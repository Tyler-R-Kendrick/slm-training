"""Default-off STOP supervision over the sanitized operator-policy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
import torch.nn.functional as F

from slm_training.dsl.operators.legal_set import LegalSetCoverage
from slm_training.flow.termination import (
    brier_score,
    expected_calibration_error,
    total_variation,
)
from slm_training.models.operator_policy_objective import (
    ObjectiveLossKind,
    OperatorPolicyObjectiveV1,
    operator_policy_loss,
)

__all__ = [
    "OperatorTerminationMode",
    "OperatorTerminationReportV1",
    "OperatorTerminationTargetV1",
    "operator_termination_losses",
]


class OperatorTerminationMode(str, Enum):
    FACTORED = "factored"
    JOINT = "joint"


@dataclass(frozen=True)
class OperatorTerminationTargetV1:
    """Replay-derived STOP/continue target; diagnostics never reach model input."""

    objective: OperatorPolicyObjectiveV1
    stop: bool
    remaining_distance: int | None
    proof_ids: tuple[str, ...] = ()
    schema: str = "operator_termination_target/v1"

    def __post_init__(self) -> None:
        if self.remaining_distance is not None and self.remaining_distance < 0:
            raise ValueError("remaining_distance must be non-negative")
        if self.objective.coverage is LegalSetCoverage.PARTIAL and self.stop:
            raise ValueError("PARTIAL coverage cannot supervise STOP")

    @property
    def forced_singleton(self) -> bool:
        return (
            self.objective.coverage is LegalSetCoverage.COMPLETE
            and len(self.objective.known_supported_indices) == 1
        )

    def model_input(self) -> dict[str, object]:
        """No distance/proof/target label leakage across the policy boundary."""
        return self.objective.model_input()


def operator_termination_losses(
    action_logits: torch.Tensor,
    stop_logit: torch.Tensor,
    target: OperatorTerminationTargetV1,
    *,
    mode: OperatorTerminationMode,
    mask_forced: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare factored STOP and joint STOP/action without making STOP an action."""
    if stop_logit.numel() != 1:
        raise ValueError("stop_logit must contain exactly one score")
    if mask_forced and target.forced_singleton:
        action_loss = action_logits.sum() * 0.0
    else:
        action_loss, _ = operator_policy_loss(
            action_logits, target.objective, kind=ObjectiveLossKind.KNOWN_SUPPORTED_CE
        )
    stop_target = stop_logit.new_tensor([float(target.stop)])
    stop_loss = F.binary_cross_entropy_with_logits(stop_logit.reshape(1), stop_target)
    if mode is OperatorTerminationMode.FACTORED:
        return action_loss + stop_loss, {"action": action_loss, "stop": stop_loss}
    if mode is not OperatorTerminationMode.JOINT:  # pragma: no cover
        raise ValueError(f"unknown termination mode: {mode}")
    if target.stop:
        known = target.objective.known_supported_indices
        joint_loss = (
            F.softplus(-stop_logit.reshape(()))
            if not known
            else torch.logsumexp(
                torch.cat((action_logits[list(known)], stop_logit.reshape(1))), 0
            )
            - stop_logit.reshape(())
        )
    else:
        known = target.objective.known_supported_indices
        if not known:
            joint_loss = action_logits.sum() * 0.0
        else:
            known_logits = action_logits[list(known)]
            positives = action_logits[list(target.objective.positive_indices)]
            joint_loss = torch.logsumexp(
                torch.cat((known_logits, stop_logit.reshape(1))), 0
            )
            joint_loss = joint_loss - torch.logsumexp(positives, 0)
    return joint_loss, {"action": action_loss, "stop": stop_loss, "joint": joint_loss}


@dataclass(frozen=True)
class OperatorTerminationReportV1:
    """Selection metrics; training loss is intentionally absent."""

    brier: float
    ece: float
    edit_count_tv: float
    premature_rate: float
    late_rate: float
    reached_target_rate: float
    schema: str = "operator_termination_report/v1"

    @classmethod
    def from_predictions(
        cls,
        *,
        stop_probabilities: Sequence[float],
        stop_targets: Sequence[int],
        predicted_edit_counts: dict[str, float],
        exact_edit_counts: dict[str, float],
        premature: Sequence[bool],
        late: Sequence[bool],
        reached_target: Sequence[bool],
    ) -> "OperatorTerminationReportV1":
        n = max(1, len(stop_targets))
        return cls(
            brier=brier_score(stop_probabilities, stop_targets),
            ece=expected_calibration_error(stop_probabilities, stop_targets),
            edit_count_tv=total_variation(predicted_edit_counts, exact_edit_counts),
            premature_rate=sum(premature) / max(1, len(premature)),
            late_rate=sum(late) / max(1, len(late)),
            reached_target_rate=sum(reached_target) / n,
        )
