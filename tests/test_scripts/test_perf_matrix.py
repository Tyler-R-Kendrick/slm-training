from scripts.run_perf_matrix import _guardrails, experiments


def test_invalid_p0_cannot_promote_candidate() -> None:
    result = _guardrails(
        {"parse_rate": 0.0, "placeholder_fidelity": 0.0, "latency_ms_mean": 100.0},
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 10.0},
    )
    assert result["pass"] is False
    assert "invalid P0" in result["note"]


def test_valid_p0_applies_quality_floor() -> None:
    result = _guardrails(
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 100.0},
        {"parse_rate": 0.96, "placeholder_fidelity": 0.96, "latency_ms_mean": 50.0},
    )
    assert result["pass"] is True
    assert result["speedup_vs_p0"] == 2.0


def test_c5_c8_are_registered_without_running() -> None:
    rows = {row.eid: row for row in experiments()}
    assert set(rows) >= {"C5", "C6", "C7", "C8"}
    assert rows["C6"].grammar_active_symbol_bitsets is True
    assert rows["C8"].grammar_completion_bounds is True
