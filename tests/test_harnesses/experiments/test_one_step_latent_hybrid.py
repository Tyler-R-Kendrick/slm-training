"""Tests for the SLM-336 (AP-035) AP-034 admission gate."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.one_step_latent_hybrid import (
    Ap034GateNotClearedError,
    Ap034PriorArtifacts,
    CURRENT_AP034_ARTIFACTS,
    require_ap034_gate_cleared,
)


def test_current_repo_state_blocks_ap035() -> None:
    """AP-034 has not built its successor artifacts, so AP-035 stays blocked."""
    with pytest.raises(Ap034GateNotClearedError, match="AP-034"):
        require_ap034_gate_cleared(CURRENT_AP034_ARTIFACTS)


def test_gate_clears_once_all_artifacts_exist() -> None:
    require_ap034_gate_cleared(
        Ap034PriorArtifacts(
            prompt_to_plan_manifest_exists=True,
            latent_to_program_decoder_exists=True,
            matched_prior_protocol_exists=True,
        )
    )


@pytest.mark.parametrize(
    "artifacts",
    [
        Ap034PriorArtifacts(False, True, True),
        Ap034PriorArtifacts(True, False, True),
        Ap034PriorArtifacts(True, True, False),
    ],
)
def test_gate_blocks_on_any_missing_artifact(artifacts: Ap034PriorArtifacts) -> None:
    with pytest.raises(Ap034GateNotClearedError):
        require_ap034_gate_cleared(artifacts)
