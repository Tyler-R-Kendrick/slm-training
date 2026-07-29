from __future__ import annotations

import pytest

from slm_training.runtime.performance_target import (
    build_performance_target,
    parse_parameter_count,
)


def _assumptions() -> list[dict[str, float | str]]:
    return [
        {"key": "peak_compute_gops", "value": 100.0},
        {"key": "memory_bandwidth_gbs", "value": 25.0},
        {"key": "compute_utilization", "value": 0.5},
        {"key": "bandwidth_utilization", "value": 0.5},
        {"key": "ops_per_parameter_forward", "value": 2.0},
        {"key": "bytes_per_parameter_forward", "value": 2.0},
        {"key": "independent_streams", "value": 8.0},
        {"key": "dispatch_step_us", "value": 25.0},
        {"key": "oracle_operation_us", "value": 1.0},
    ]


def test_parameter_count_parses_model_card_units() -> None:
    assert parse_parameter_count("≈135M") == 135_000_000
    assert parse_parameter_count("64,994") == 64_994
    assert parse_parameter_count("—") is None


def test_target_uses_whole_model_and_separates_profiles() -> None:
    target = build_performance_target(
        checkpoint_references=[
            {
                "role": "Latest checkpoint",
                "run_id": "current",
                "parameters": "≈135M",
            }
        ],
        assumptions=_assumptions(),
        output_tokens=256,
        structural_turn_range=(4, 8),
        grammar_inputs={
            "productions": 50,
            "components": 50,
            "structural_tokens": 20,
            "maximum_alternatives": 4,
        },
    )

    single, aggregate = target["profiles"]
    assert target["parameter_count"] == 135_000_000
    assert single["id"] == "single_stream"
    assert aggregate["id"] == "aggregate"
    assert aggregate["floor_tps"] == pytest.approx(single["floor_tps"] * 8)
    assert single["ceiling_tps"] > single["floor_tps"]
    assert target["grammar_inputs"]["oracle_operations_per_step"] == 7
    assert "grammar-description" in target["mismatches"][0]["current"]


def test_target_is_unavailable_without_model_geometry() -> None:
    target = build_performance_target(
        checkpoint_references=[],
        assumptions=_assumptions(),
        output_tokens=256,
        structural_turn_range=(4, 8),
        grammar_inputs={},
    )

    assert target["status"] == "unavailable"
    assert target["profiles"] == []
