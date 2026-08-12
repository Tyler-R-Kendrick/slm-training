"""RESEARCH-08 campaign harness tests (SLM-540)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.research08_myhill_nerode import (
    CAMPAIGN_ID,
    EXPERIMENT_KEY,
    PRIMARY_METRIC,
    Research08CampaignV1,
    plan_only_preview,
    run_campaign,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="cold_trials"):
        Research08CampaignV1(cold_trials=0)
    with pytest.raises(ValueError, match="fixture"):
        Research08CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Research08CampaignV1(max_wall_minutes=1000.0)


def test_plan_only_preview() -> None:
    preview = plan_only_preview()
    assert preview["experiment_key"] == EXPERIMENT_KEY
    assert preview["campaign_id"] == CAMPAIGN_ID
    assert preview["primary_metric"] == PRIMARY_METRIC
    assert preview["production_authority"] is False
    assert preview["default_off"] is True


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    result = run_campaign(Research08CampaignV1(cold_trials=5), root=tmp_path)
    assert result["experiment_key"] == EXPERIMENT_KEY
    assert result["promotion"] is False
    pilot = result["pilot"]
    assert pilot["enabled"] is True
    assert pilot["zero_semantic_disagreement"] is True
    assert pilot["decision"] in {"accept_research_only", "reject"}
    assert result["manifest_sha256"]
