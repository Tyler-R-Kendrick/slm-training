"""Integration contracts for the SLM-243 runner."""

import pytest

from scripts.run_slm243_recursive_update_gate import (
    DEPTHS,
    SEEDS,
    VARIANTS,
    _mechanism_fixtures,
)


def test_slm243_matrix_is_six_by_five_by_three() -> None:
    assert len(VARIANTS) * len(DEPTHS) * len(SEEDS) == 90
    assert VARIANTS["current_v1"] == {
        "update_mode": "current_v1",
        "empty_f_mode": "pass_through",
        "norm_mode": "shared",
    }
    assert VARIANTS["current_true_empty"]["update_mode"] == "current_v1"
    assert VARIANTS["current_true_empty"]["empty_f_mode"] == "zero"
    assert VARIANTS["gated_private"]["norm_mode"] == "private"
    assert VARIANTS["layerscale_private"]["norm_mode"] == "private"


def test_slm243_mechanism_fixtures_isolate_empty_f_and_initializers() -> None:
    fixtures = _mechanism_fixtures()
    assert fixtures["historical_empty_f_update_norm"] > 0
    assert fixtures["true_empty_f_update_norm"] == 0
    assert fixtures["true_empty_f_exact_zero"]
    assert fixtures["layerscale_initial_value"] == pytest.approx(1e-3)
    assert fixtures["gated_initial_sigmoid"] < 0.02
    assert fixtures["private_norm_objects_distinct"]
