"""Regression coverage for SLM-407's serving-work denominator."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.operator_systems_benchmark import (
    OperatorServingIdentityV1,
    OperatorServingWorkV1,
    build_operator_serving_report,
)


def _identity(**changes: object) -> OperatorServingIdentityV1:
    values = {
        "target_model": "local-target",
        "target_model_revision": "abc123",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 1,
        "cache_mode": "cold",
        "context_fingerprint": "context-v1",
    }
    values.update(changes)
    return OperatorServingIdentityV1(**values)  # type: ignore[arg-type]


def _row(
    arm: str,
    *,
    identity: OperatorServingIdentityV1 | None = None,
    stratum: str = "small",
    outcome: str = "success",
    quality: float | None = 1.0,
    forwards: int = 4,
    wall_ms: float = 40.0,
    meaningful: bool | None = None,
) -> OperatorServingWorkV1:
    return OperatorServingWorkV1(
        request_id=f"{arm}-{stratum}-{outcome}",
        arm=arm,
        identity=identity or _identity(),
        outcome=outcome,  # type: ignore[arg-type]
        meaningful_parse=(quality is not None and quality > 0) if meaningful is None else meaningful,
        quality_score=quality,
        state_size=2,
        stratum=stratum,
        legal_set_actions=3,
        legal_set_builds=1,
        dry_run_calls=3,
        executor_calls=1,
        validator_calls=1,
        materialization_calls=1,
        cache_hits=0,
        cache_misses=1,
        target_model_forwards=forwards,
        non_target_forward_equivalents=0.5,
        legal_set_ms=2.0,
        dry_run_ms=3.0,
        executor_ms=1.0,
        validator_ms=1.0,
        materialization_ms=1.0,
        model_device_ms=10.0,
        cpu_ms=8.0,
        wall_ms=wall_ms,
        peak_memory_bytes=64,
    )


def test_raw_components_reconcile_and_failures_remain_in_denominator() -> None:
    identity = _identity()
    report = build_operator_serving_report(
        identity=identity,
        rows=(
            _row("full_generation"),
            _row("full_generation", outcome="timeout", wall_ms=99.0),
        ),
        required_arms=("full_generation",),
    )

    summary = report["per_arm"]["full_generation"]
    assert summary["n"] == 2
    assert summary["outcomes"]["timeout"] == 1
    assert summary["primary_numeraire"]["reconciles"] is True
    assert summary["distribution"]["wall_ms_p95"] == 99.0


def test_identity_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="one serving identity"):
        build_operator_serving_report(
            identity=_identity(),
            rows=(_row("full_generation"), _row("typed_operator", identity=_identity(device="cuda"))),
        )


def test_singleton_zero_forward_still_charges_legal_and_executor_work() -> None:
    report = build_operator_serving_report(
        identity=_identity(),
        rows=(
            _row("full_generation"),
            _row("typed_operator", forwards=0, wall_ms=12.0),
        ),
        required_arms=("full_generation", "typed_operator"),
    )

    totals = report["per_arm"]["typed_operator"]["totals"]
    assert totals["target_model_forwards"] == 0
    assert totals["legal_set_builds"] == 1
    assert totals["executor_calls"] == 1
    assert report["comparisons"]["typed_operator"]["efficiency_claim_supported"]


def test_quality_mismatch_and_missing_arms_fail_closed() -> None:
    report = build_operator_serving_report(
        identity=_identity(),
        rows=(
            _row("full_generation", quality=1.0),
            _row(
                "typed_operator",
                quality=1.0,
                meaningful=False,
                forwards=1,
                wall_ms=1.0,
            ),
        ),
    )

    assert report["verdict"] == "unavailable"
    assert report["comparisons"]["typed_operator"]["quality_matched"] is False
    assert report["comparisons"]["typed_operator"]["meaningful_parse_matched"] is False
    assert report["comparisons"]["typed_operator"]["efficiency_claim_supported"] is False


def test_crossover_is_observed_only_not_extrapolated() -> None:
    report = build_operator_serving_report(
        identity=_identity(),
        rows=(
            _row("full_generation", stratum="small", wall_ms=10.0),
            _row("typed_operator", stratum="small", wall_ms=12.0),
            _row("full_generation", stratum="large", wall_ms=30.0),
            _row("typed_operator", stratum="large", wall_ms=20.0),
        ),
        required_arms=("full_generation", "typed_operator"),
    )

    crossover = report["observed_crossover"]["typed_operator"][0]
    assert crossover["crossovers"] == [{"between": ["large", "small"]}]
    assert "threshold" not in crossover
