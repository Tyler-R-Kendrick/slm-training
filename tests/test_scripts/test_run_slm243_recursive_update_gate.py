"""Integration contracts for the SLM-243 runner."""

from scripts.run_slm243_recursive_update_gate import DEPTHS, SEEDS, VARIANTS


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
