"""SFF metrics contract: future runs must keep quality + runtime fields."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.anti_e237_semantic_factor_frontier import (
    build_campaign,
    lock_campaign,
    run_semantic_factor_frontier,
)
from slm_training.harnesses.experiments.semantic_factor_metrics import (
    REQUIRED_ARM_RUNTIME_FIELDS,
    REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS,
    SCOREBOARD_KIND,
    ScoreboardMetricsError,
    campaign_declares_runtime_endpoints,
    validate_scoreboard_metrics,
)


def test_campaign_lock_requires_runtime_endpoints() -> None:
    campaign = build_campaign(source_commit="a" * 40, source_dirty=False)
    ids = [e.endpoint_id for e in campaign.endpoints]
    assert campaign_declares_runtime_endpoints(ids)
    assert REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS.issubset(set(ids))
    lock = lock_campaign(campaign)
    assert lock.manifest_sha256


def test_live_run_scoreboard_passes_metrics_contract(tmp_path) -> None:
    payload = run_semantic_factor_frontier(
        seeds=(0,),
        out_dir=tmp_path,
        source_commit="b" * 40,
        source_dirty=False,
    )
    assert payload["kind"] == SCOREBOARD_KIND
    validate_scoreboard_metrics(payload)  # must not raise
    for arm in payload["arms"]:
        for key in REQUIRED_ARM_RUNTIME_FIELDS:
            assert key in arm, key
    resources = payload["config_resources"]
    for key in ("metrics", "scorer_params", "campaign"):
        assert resources[key]["schema"].startswith("sff_")
        assert resources[key]["sha256"]
        assert not str(resources[key]["path"]).startswith("/")


def test_validate_rejects_missing_runtime_fields() -> None:
    good = run_semantic_factor_frontier(
        seeds=(0,),
        source_commit="c" * 40,
        source_dirty=False,
    )
    validate_scoreboard_metrics(good)
    bad = dict(good)
    bad_arms = [dict(a) for a in good["arms"]]
    del bad_arms[0]["wall_ms_mean"]
    del bad_arms[0]["quality_per_ms"]
    bad["arms"] = bad_arms
    with pytest.raises(ScoreboardMetricsError, match="wall_ms_mean"):
        validate_scoreboard_metrics(bad)


def test_validate_rejects_missing_runtime_block() -> None:
    good = run_semantic_factor_frontier(
        seeds=(0,),
        source_commit="d" * 40,
        source_dirty=False,
    )
    bad = dict(good)
    bad.pop("runtime", None)
    with pytest.raises(ScoreboardMetricsError, match="runtime"):
        validate_scoreboard_metrics(bad)


def test_validate_rejects_missing_config_resources() -> None:
    good = run_semantic_factor_frontier(
        seeds=(0,),
        source_commit="e" * 40,
        source_dirty=False,
    )
    bad = dict(good)
    bad.pop("config_resources", None)
    with pytest.raises(ScoreboardMetricsError, match="config_resources"):
        validate_scoreboard_metrics(bad)


def test_validate_rejects_legacy_v1_kind() -> None:
    with pytest.raises(ScoreboardMetricsError, match="predates runtime"):
        validate_scoreboard_metrics(
            {
                "kind": "semantic_factor_frontier_results/v1",
                "arms": [{"arm_id": "x"}],
            }
        )
