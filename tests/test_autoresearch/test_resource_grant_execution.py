"""Regression tests for logical grant propagation and enforcement."""

from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.harness_core.activity_contract import ResourceGrant
from scripts.autoresearch_command_cursor import ContinuationGrant


def test_campaign_budget_preserves_explicit_grant_and_legacy_shape() -> None:
    grant = ResourceGrant(
        total_seconds=600,
        interrupt_seconds=60,
        finalization_reserve_seconds=5,
        max_attempts=4,
    )
    budget = CampaignBudget(continuation_grant=grant)
    assert budget.logical_seconds == 600
    assert budget.model_dump(mode="json")["continuation_grant"]["interrupt_seconds"] == 60
    assert "continuation_grant" not in CampaignBudget().model_dump(mode="json")


def test_continuation_grant_carries_the_fenced_execution_limits() -> None:
    grant = ContinuationGrant("release", 600, 4, 60, 5)
    assert grant.total_wall_seconds == 600
    assert grant.interrupt_seconds == 60
    assert grant.finalization_reserve_seconds == 5
