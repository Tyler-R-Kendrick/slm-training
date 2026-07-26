from __future__ import annotations

import pytest
import torch

from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.models.operator_policy_objective import (
    OperatorPolicyObjectiveV1,
    OperatorPolicyTargetV1,
    TargetUtilityLabel,
)
from slm_training.models.operator_termination import (
    OperatorTerminationMode,
    OperatorTerminationReportV1,
    OperatorTerminationTargetV1,
    operator_termination_losses,
)


def _objective(
    *, coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE, count: int = 2
):
    return OperatorPolicyObjectiveV1(
        coverage=coverage,
        targets=tuple(
            OperatorPolicyTargetV1(
                f"a{index}",
                OperatorSupportVerdict.SUPPORTED,
                TargetUtilityLabel.POSITIVE
                if index == 0
                else TargetUtilityLabel.NEGATIVE,
            )
            for index in range(count)
        ),
    )


def test_stop_is_control_plane_and_never_leaks_distance_or_proofs() -> None:
    target = OperatorTerminationTargetV1(_objective(), False, 3, ("proof",))
    assert target.model_input() == target.objective.model_input()
    assert "distance" not in str(target.model_input())
    with pytest.raises(ValueError, match="PARTIAL"):
        OperatorTerminationTargetV1(
            _objective(coverage=LegalSetCoverage.PARTIAL), True, 0
        )


def test_joint_and_factored_losses_keep_forced_singleton_authority() -> None:
    forced = OperatorTerminationTargetV1(_objective(count=1), False, 1)
    logits = torch.tensor([0.0], requires_grad=True)
    total, losses = operator_termination_losses(
        logits,
        torch.tensor([0.0]),
        forced,
        mode=OperatorTerminationMode.FACTORED,
        mask_forced=True,
    )
    assert losses["action"].item() == 0.0
    assert total.item() > 0.0
    joint, _ = operator_termination_losses(
        logits,
        torch.tensor([0.0]),
        forced,
        mode=OperatorTerminationMode.JOINT,
        mask_forced=True,
    )
    assert torch.isfinite(joint)


def test_terminal_with_legal_actions_and_report_uses_calibration_not_loss() -> None:
    terminal = OperatorTerminationTargetV1(_objective(), True, 0)
    loss, values = operator_termination_losses(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0]),
        terminal,
        mode=OperatorTerminationMode.JOINT,
    )
    assert torch.isfinite(loss) and values["joint"].item() > 0.0
    report = OperatorTerminationReportV1.from_predictions(
        stop_probabilities=(0.9, 0.1),
        stop_targets=(1, 0),
        predicted_edit_counts={"0": 0.5, "1": 0.5},
        exact_edit_counts={"0": 0.5, "1": 0.5},
        premature=(False, False),
        late=(False, False),
        reached_target=(True, True),
    )
    assert report.brier < 0.02 and report.ece < 0.11 and report.edit_count_tv == 0.0
