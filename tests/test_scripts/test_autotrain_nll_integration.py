"""NLL publication and paired evidence through the canonical continuous driver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_scripts.test_run_autotrain_continuous import (
    _clear_dynamic_thrash_bank_cache as _clear_dynamic_thrash_bank_cache,
    _mod,
    _p3_expectations,
    _p3_nll_pairs,
    _p3_policy,
    _p3_write_gates_insufficient_n,
    _p3_write_nll_records,
    _p3_write_scoreboard,
    _write_eval,
)


def test_run_arm_eval_nll_writes_smoke_eval_nll(tmp_path: Path) -> None:
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash

    run_dir = tmp_path / "runs" / "arm"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps({"suites": {"smoke": {"n": 6, "structural_similarity": 0.1}}}),
        encoding="utf-8",
    )
    out = _mod._run_arm_eval_nll(run_dir, eval_nll=3.25)
    assert out["eval_nll"] == 3.25
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["suites"]["smoke"]["eval_nll"] == 3.25
    assert scoreboard["suites"]["smoke"]["eval_nll_claim_class"] == "diagnostic"
    assert scoreboard["measurement_complete"] is False
    assert scoreboard["suites"]["smoke"]["decoded_probe_complete"] is False
    assert scoreboard["suites"]["smoke"]["eval_nll_definition_hash"] == (
        screening_nll_definition_hash(
            arm_loss_weights={"binder_arity_loss_weight": 7.0}
        )
    )
    metrics = _mod._run_metrics(tmp_path, "arm")
    assert metrics["smoke.eval_nll"] == 3.25
    assert metrics["eval_nll"] == 3.25


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_run_arm_nll_refuses_invalid_evidence_before_publication(tmp_path, value):
    with pytest.raises(ValueError, match="invalid_evidence"):
        _mod._run_arm_eval_nll(tmp_path, eval_nll=value)
    assert not (tmp_path / "scoreboard.json").exists()


def test_attach_screening_eval_nll_skips_without_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "arm"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps({"suites": {"smoke": {"n": 6}}}), encoding="utf-8"
    )
    assert _mod._attach_screening_eval_nll(run_dir) is None
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert "eval_nll" not in scoreboard["suites"]["smoke"]


def test_classify_positive_paired_nll_three_pairs_not_positive(tmp_path: Path) -> None:
    camp = tmp_path / "c601"
    control, candidate = _p3_nll_pairs(3, shift=0.4)
    for arm, recs in (("c601-control", control), ("c601-cand", candidate)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=0.3333333333333333,
            binder_reference_f1=0.6,
            structural_similarity=0.2,
            latency_ms_p50=3000.0,
        )
        _p3_write_scoreboard(run, n=3, eval_nll=sum(recs.values()) / len(recs))
        _p3_write_nll_records(run, recs)
        _p3_write_gates_insufficient_n(run)
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.eval_nll",
        control_id="c601-control",
        candidate_id="c601-cand",
        role="screening",
        policy_path=_p3_policy("policy.v2.json"),
        observed_sd_path=_p3_expectations(tmp_path),
    )
    reasons = result["reasons"]
    assert result["positive"] is False
    assert result["paired_test"]["n_pairs"] == 3
    assert result["paired_test"]["win"] is False
    null = next(r for r in reasons if r.startswith("primary_metric_inconclusive:"))
    assert "n_pairs=3" in null and "paired_inconclusive" in null
    assert "fixture_insufficient_n:quality_probe" not in reasons
    assert not any(r.startswith("primary_metric_win:") for r in reasons)


def test_classify_positive_paired_nll_definition_mismatch_never_pairs(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "c602"
    control, candidate = _p3_nll_pairs(24, shift=0.2)
    for arm, recs, digest in (
        ("c602-control", control, "def-a"),
        ("c602-cand", candidate, "def-b"),
    ):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=0.3333333333333333,
            binder_reference_f1=0.6,
            structural_similarity=0.2,
            latency_ms_p50=3000.0,
        )
        _p3_write_scoreboard(run, n=3, eval_nll=sum(recs.values()) / len(recs))
        _p3_write_nll_records(run, recs, digest=digest)
        _p3_write_gates_insufficient_n(run)
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.eval_nll",
        control_id="c602-control",
        candidate_id="c602-cand",
        role="screening",
        policy_path=_p3_policy("policy.v2.json"),
        observed_sd_path=_p3_expectations(tmp_path),
    )
    assert result["positive"] is False
    assert result["paired_test"] is None
    assert "paired_records_definition_mismatch:c602-control:c602-cand" in result["reasons"]
