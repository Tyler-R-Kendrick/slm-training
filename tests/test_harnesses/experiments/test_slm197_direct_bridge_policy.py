from slm_training.harnesses.experiments.slm197_direct_bridge_policy import (
    run_matrix,
)


def test_fixture_matrix_is_matched_and_honestly_blocked() -> None:
    report = run_matrix(seeds=(0,), steps=1)
    assert report["arms"]["D0"]["status"] == "unavailable"
    assert report["arms"]["D1"]["status"] == "unavailable"
    assert report["arms"]["D2"]["status"] == "measured_fixture"
    assert report["arms"]["D3-linear"]["status"] == "measured_fixture"
    assert report["arms"]["D3-fourier"]["status"] == "measured_fixture"
    assert report["matched_controls"]["parameter_count_equal"]
    assert report["confirmation"]["status"] == "blocked"
    assert report["honest_verdict"] == "inconclusive_fixture_only"
    for arm in ("D2", "D3-linear", "D3-fourier", "D4", "D5"):
        run = report["arms"][arm]["runs"][0]
        assert run["evaluation"]["candidate_membership"]["exact"]
        assert run["free_running"]["all_actions_live_rate"] == 1.0
        assert run["artifact_identity"]["param_count"] > 0


def test_fixture_matrix_rejects_over_cap() -> None:
    try:
        run_matrix(seeds=(0,), steps=1, max_wall_minutes=3.1)
    except ValueError as exc:
        assert "max_wall_minutes" in str(exc)
    else:
        raise AssertionError("over-cap matrix must fail")
