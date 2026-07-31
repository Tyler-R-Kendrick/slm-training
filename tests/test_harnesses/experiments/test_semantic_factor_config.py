"""External SFF metrics / scorer_params / campaign schema loaders."""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.experiments.semantic_factor_config import (
    SFF_RESOURCE_DIR,
    SFFConfigError,
    load_sff_campaign,
    load_sff_metrics,
    load_sff_scorer_params,
)
from slm_training.harnesses.experiments.semantic_factor_metric_engine import (
    apply_control_relative_metrics,
    compute_arm_metrics,
)
from slm_training.models.semantic_residual_scorer import SemanticResidualScorerConfig


def test_resources_exist_with_versioned_schemas() -> None:
    for name, schema in (
        ("metrics.v1.json", "sff_metrics/v1"),
        ("scorer_params.v1.json", "sff_scorer_params/v1"),
        ("campaign.v1.json", "sff_campaign/v1"),
    ):
        path = SFF_RESOURCE_DIR / name
        assert path.is_file(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == schema
        assert payload["version"]


def test_load_metrics_and_scorer_params() -> None:
    metrics = load_sff_metrics()
    scorer = load_sff_scorer_params()
    assert "wall_ms_mean" in metrics.required_arm_fields
    assert "quality_per_ms" in metrics.required_arm_fields
    assert "wall_ms_mean" in metrics.required_campaign_runtime_endpoints
    assert scorer.arm("train_on_decode_on_direct_factors")["alpha"] == 1.5
    assert scorer.is_decode_on("exact_typed_zero_parameter")


def test_scorer_config_from_external_params() -> None:
    scorer = load_sff_scorer_params()
    cfg = SemanticResidualScorerConfig.from_external(
        arm=scorer.arm("train_on_decode_on_role_ported"),
        defaults=scorer.defaults,
        seed=7,
    )
    assert cfg.alpha == 1.5
    assert cfg.decode_apply is True
    assert cfg.params is not None
    assert cfg.params["lambda_restart"] == 0.15


def test_metric_engine_adds_metric_from_file_without_harness_edit() -> None:
    metrics = load_sff_metrics()
    rows = [
        {
            "family": "ranked",
            "correct": True,
            "reciprocal_rank": 1.0,
            "applications": 1,
            "choice_changed": True,
            "quality_improving": True,
            "quality_harming": False,
            "singleton_skip": False,
            "incomplete_skip": False,
            "unknown_preserved": False,
            "decision_ms": 2.0,
            "project_ms": 1.0,
            "score_ms": 0.5,
        },
        {
            "family": "ranked",
            "correct": False,
            "reciprocal_rank": 0.5,
            "applications": 1,
            "choice_changed": False,
            "quality_improving": False,
            "quality_harming": False,
            "singleton_skip": False,
            "incomplete_skip": False,
            "unknown_preserved": False,
            "decision_ms": 4.0,
            "project_ms": 1.0,
            "score_ms": 1.0,
        },
    ]
    arm = compute_arm_metrics(
        metrics,
        arm_id="toy",
        role="candidate",
        rows=rows,
        kills=[],
    )
    assert arm["n_ranked"] == 2
    assert arm["ranking_accuracy"] == 0.5
    assert arm["wall_ms_mean"] == 3.0
    assert arm["applications"] == 2
    control = compute_arm_metrics(
        metrics,
        arm_id="control_none",
        role="control",
        rows=[
            {
                **rows[0],
                "correct": False,
                "applications": 0,
                "choice_changed": False,
                "decision_ms": 1.0,
            }
        ],
        kills=[],
    )
    arms = [control, arm]
    apply_control_relative_metrics(metrics, arms, control_arm_id="control_none")
    assert arms[1]["wall_ms_mean_vs_control"] == pytest.approx(3.0)
    assert arms[1]["delta_accuracy_vs_control"] == pytest.approx(0.5)


def test_reject_bad_schema(tmp_path, monkeypatch) -> None:
    load_sff_metrics.cache_clear()
    bad = tmp_path / "metrics.v1.json"
    bad.write_text('{"schema": "nope/v1", "version": "v1"}', encoding="utf-8")
    monkeypatch.setattr(
        "slm_training.harnesses.experiments.semantic_factor_config.SFF_RESOURCE_DIR",
        tmp_path,
    )
    with pytest.raises(SFFConfigError, match="schema"):
        load_sff_metrics("metrics.v1.json")
    load_sff_metrics.cache_clear()


def test_campaign_loads_linked_resources() -> None:
    camp = load_sff_campaign()
    assert camp.metrics.version == "v1"
    assert camp.scorer.version == "v1"
    identity = camp.resource_identity()
    assert "metrics" in identity
    # Durable scoreboards record repo-relative paths, never machine absolutes.
    metrics_path = identity["metrics"]["path"]
    assert not metrics_path.startswith("/")
    assert metrics_path.endswith(
        "resources/experiments/semantic_factor_frontier/metrics.v1.json"
    )
    assert identity["metrics"]["schema"] == "sff_metrics/v1"
    assert identity["scorer_params"]["schema"] == "sff_scorer_params/v1"
    assert identity["campaign"]["schema"] == "sff_campaign/v1"
    assert len(identity["metrics"]["sha256"]) == 64
    assert len(identity["scorer_params"]["sha256"]) == 64
