from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm199_legal_edit_flow import run_fixture


def test_exact_oracle_and_production_adapter_contract() -> None:
    report = run_fixture(seeds=(0, 1), train_steps=1, exact_samples=32)
    exact = report["exact_oracle"]
    assert exact["closed"]
    assert exact["illegal_edge_rate_sum"] == 0.0
    assert exact["rate_fit"]["max_abs_error"] < 1e-3
    assert exact["analytic_endpoint_tv"] < 0.01
    assert exact["empirical_endpoint_tv"] < 0.05
    assert exact["event_count_tv"] < 0.05
    assert exact["exact_event_count_distribution"] == {"5": 1.0}
    assert exact["event_count_min"] == 5
    assert exact["event_count_max"] == 5
    production = report["production_adapter"]
    assert production["fidelity"] == "adapted_path_approximation"
    assert not production["unknown_supervised_as_negative"]
    assert all(item["verified_output"] for item in production["samples"])
    assert not report["default_path"]["flow_enabled_by_default"]
    assert report["default_path"]["flow_time_encoding"] == "linear"
    assert not report["checkpoint"]["written"]
    assert report["confirmation"]["status"] == "blocked"


def test_hard_cap_is_enforced() -> None:
    with pytest.raises(ValueError, match="max_wall_minutes"):
        run_fixture(max_wall_minutes=3.1)
    with pytest.raises(ValueError, match="seed"):
        run_fixture(seeds=())
    with pytest.raises(ValueError, match="exact_samples"):
        run_fixture(exact_samples=0)
