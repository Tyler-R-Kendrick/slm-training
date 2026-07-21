"""Tests for SLM-212 constraint-debt routing policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.debt_router import (
    CHEAP_ROUTE,
    STRICT_ROUTE,
    CalibratedDebtRouter,
    DebtRoutingPolicy,
    OracleRouter,
    ROUTES,
    build_calibrator_artifact,
    decide_route,
)


class FakeDebtRow:
    """Minimal stand-in for a ConstraintDebtV1 row."""

    def __init__(self, **kwargs: float) -> None:
        self.legal_debt: float | None = kwargs.get("legal_debt")
        self.good_debt: float | None = kwargs.get("good_debt")
        self.legal_mass_deficit: float = kwargs.get("legal_mass_deficit", 0.0)
        self.pre_post_mask_kl: float = kwargs.get("pre_post_mask_kl", 0.0)


def test_fixed_routes_ignore_signal() -> None:
    policy = DebtRoutingPolicy(mode="fixed_ltr")
    route, state = decide_route(999.0, policy)
    assert route == "ltr"
    assert state["reason"] == "fixed_fixed_ltr"

    policy = DebtRoutingPolicy(mode="fixed_asap")
    route, _ = decide_route(999.0, policy)
    assert route == "asap"

    policy = DebtRoutingPolicy(mode="fixed_maskgit")
    route, _ = decide_route(999.0, policy)
    assert route == "maskgit"


def test_off_returns_cheap_route() -> None:
    policy = DebtRoutingPolicy(mode="off")
    route, state = decide_route(999.0, policy)
    assert route == CHEAP_ROUTE
    assert state["reason"] == "routing_off"


def test_threshold_switch() -> None:
    policy = DebtRoutingPolicy(
        mode="debt_router",
        signal="D_legal",
        threshold_high=2.0,
        threshold_low=0.5,
        hysteresis=1,
    )
    route, _ = decide_route(0.1, policy)
    assert route == CHEAP_ROUTE
    route, _ = decide_route(3.0, policy)
    assert route == STRICT_ROUTE
    route, _ = decide_route(0.1, policy)
    assert route == CHEAP_ROUTE


def test_hysteresis_requires_consecutive_steps() -> None:
    policy = DebtRoutingPolicy(
        mode="debt_router",
        threshold_high=2.0,
        threshold_low=0.5,
        hysteresis=3,
    )
    state: dict = {}
    route, state = decide_route(3.0, policy, state=state)
    assert route == CHEAP_ROUTE
    route, state = decide_route(3.0, policy, state=state)
    assert route == CHEAP_ROUTE
    route, state = decide_route(3.0, policy, state=state)
    assert route == STRICT_ROUTE
    # A single in-between reading resets the counter.
    route, state = decide_route(1.0, policy, state=state)
    assert route == STRICT_ROUTE  # still latched
    route, state = decide_route(3.0, policy, state=state)
    assert route == STRICT_ROUTE  # still latched, only one new high step


def test_low_threshold_defaults_to_high_when_unset() -> None:
    policy = DebtRoutingPolicy(mode="debt_router", threshold_high=2.0, threshold_low=None)
    route, _ = decide_route(2.0, policy)
    assert route == STRICT_ROUTE
    route, _ = decide_route(1.999, policy)
    assert route == CHEAP_ROUTE


def test_signal_value_read_from_debt_row() -> None:
    from slm_training.harnesses.model_build.debt_router import _read_signal_value

    row = FakeDebtRow(legal_debt=2.5)
    assert _read_signal_value("D_legal", row) == 2.5
    row = FakeDebtRow(legal_mass_deficit=0.3)
    assert _read_signal_value("legal_mass_deficit", row) == 0.3
    row = FakeDebtRow()
    assert _read_signal_value("D_legal", row) == 0.0


def test_calibrator_artifact_hash_is_stable() -> None:
    artifact = build_calibrator_artifact(threshold_high=1.5, hysteresis=2)
    assert artifact["schema"] == "debt_router_calibrator/v1"
    assert len(artifact["artifact_hash"]) == 64
    artifact2 = build_calibrator_artifact(threshold_high=1.5, hysteresis=2)
    assert artifact["artifact_hash"] == artifact2["artifact_hash"]


def test_calibrated_router_loads_valid_artifact(tmp_path: Path) -> None:
    artifact = build_calibrator_artifact(
        signal="legal_mass_deficit",
        threshold_high=1.0,
        threshold_low=0.2,
        hysteresis=2,
        fallback_policy="fixed_ltr",
    )
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        constraint_debt_routing_mode="debt_router",
        constraint_debt_routing_calibrator_path=path,
    )
    router = CalibratedDebtRouter.from_config(config)
    assert router.calibration_error is None
    assert router.policy.signal == "legal_mass_deficit"
    assert router.policy.threshold_high == 1.0
    assert router.policy.hysteresis == 2
    assert router.policy.mode == "debt_router"
    route, _ = router.decide(1.5, step=0)
    assert route == STRICT_ROUTE


def test_calibrated_router_falls_back_on_hash_mismatch(tmp_path: Path) -> None:
    artifact = build_calibrator_artifact(threshold_high=1.5)
    artifact["artifact_hash"] = "0" * 64
    path = tmp_path / "bad_calibrator.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        constraint_debt_routing_mode="debt_router",
        constraint_debt_routing_calibrator_path=path,
    )
    router = CalibratedDebtRouter.from_config(config)
    assert router.calibration_error == "hash_mismatch"
    # Fallback forces the configured fallback_policy route.
    assert router.policy.mode == "fixed_maskgit"


def test_calibrated_router_falls_back_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        constraint_debt_routing_mode="debt_router",
        constraint_debt_routing_calibrator_path=missing,
    )
    router = CalibratedDebtRouter.from_config(config)
    assert router.calibration_error == "missing_calibrator"
    assert router.policy.mode == "fixed_maskgit"


def test_oracle_router_chooses_best_route() -> None:
    outcomes = {
        "ex_a": {"maskgit": 0.5, "ltr": 0.9, "asap": 0.6},
        "ex_b": {"maskgit": 0.8, "ltr": 0.2, "asap": 0.7},
    }
    oracle = OracleRouter(outcomes)
    route, info = oracle.decide("ex_a")
    assert route == "ltr"
    assert info["reason"] == "oracle_best"
    route, _ = oracle.decide("ex_b")
    assert route == "maskgit"


def test_oracle_router_defaults_when_missing() -> None:
    oracle = OracleRouter({})
    route, info = oracle.decide("ex_unknown")
    assert route in ROUTES
    assert info["reason"] == "oracle_missing"


def test_policy_identity_hash_changes_with_threshold() -> None:
    p1 = DebtRoutingPolicy(mode="debt_router", threshold_high=2.0)
    p2 = DebtRoutingPolicy(mode="debt_router", threshold_high=2.1)
    assert p1.identity_hash() != p2.identity_hash()


def test_policy_validation_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown routing mode"):
        DebtRoutingPolicy(mode="magic")


def test_policy_validation_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError, match="unknown routing signal"):
        DebtRoutingPolicy(signal="D_magic")
