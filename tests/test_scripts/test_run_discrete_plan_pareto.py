"""Regression tests for SLM-322/AP-027's bounded discrete-plan Pareto runner.

These tests mock ``run_one`` (real decode is exercised end-to-end by the
harness itself and by the committed screening result under
``docs/design/discrete-plan-pareto.md``) so the suite stays fast and does not
require a checkpoint or torch forward pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.run_discrete_plan_pareto import MEASURED_ARM_LABELS, _phase_latencies, run_screening
from slm_training.harnesses.experiments.discrete_plan_pareto import KNOWN_COLLAPSED_ARMS


def _fake_run_one(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {
        "latency_ms_mean": 100.0,
        "parse_rate": 0.75,
        "placeholder_fidelity": 0.5,
        "tokens_emitted": 40,
        "phase_summary": {
            "denoiser_ms_mean": 60.0,
            "backbone_ms_mean": 10.0,
            "projection_ms_mean": 5.0,
            "finalize_ms_mean": 8.0,
            "dfa_sync_ms_mean": 2.0,
            "stream_check_ms_mean": 1.0,
        },
    }


def test_phase_latencies_never_exceed_the_reported_total() -> None:
    plan_ms, gen_ms, verify_ms = _phase_latencies(_fake_run_one())
    assert plan_ms == 0.0
    assert gen_ms == 75.0
    assert verify_ms == 11.0
    assert plan_ms + gen_ms + verify_ms <= 100.0


def test_run_screening_produces_rows_that_pass_harness_validation() -> None:
    """The runner's own output must satisfy ParetoLatencyRow.validate() --
    the same structural gate that forbids marking a closed arm eligible."""
    with patch("scripts.run_discrete_plan_pareto.run_one", side_effect=_fake_run_one):
        rows = run_screening(
            checkpoint=Path("unused.pt"),
            prompts=[("prompt", None)],
            device="cpu",
            warmup=0,
            out_dir=Path("/tmp/does-not-need-to-exist"),
            rounds=(1, 2),
        )
    assert len(rows) == len(MEASURED_ARM_LABELS) * 2
    assert {row.arm for row in rows} == set(MEASURED_ARM_LABELS)
    assert set(MEASURED_ARM_LABELS).isdisjoint(KNOWN_COLLAPSED_ARMS)
    for row in rows:
        row.validate()  # raises on any structural violation
        assert row.promotion_eligible is False
