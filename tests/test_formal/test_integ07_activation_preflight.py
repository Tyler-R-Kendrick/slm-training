"""INTEG-07 / SLM-561: activation-preflight recall + false-skip gate."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.integ07_activation_preflight import (
    REPLAY_SCHEMA,
    REPORT_SCHEMA,
    assert_benchmark_passed,
    effective_disposition,
    identity_matches,
    load_replay,
    required_adversarial_kinds,
    run_benchmark,
)
from slm_training.formal.mechanism_trigger import NamedCostModel

ROOT = Path(__file__).resolve().parents[2]
REPLAY = (
    ROOT
    / "src/slm_training/resources/formal/integ07_activation_preflight_replay.v1.json"
)
RESULTS = ROOT / "docs/design/integ-07-activation-preflight-recall-results.json"


def test_replay_covers_cohorts_and_adversarial_kinds() -> None:
    replay = load_replay(root=ROOT)
    assert replay["schema"] == REPLAY_SCHEMA
    assert REPLAY.is_file()
    cohorts = {case["cohort"] for case in replay["cases"]}
    assert cohorts == {
        "known_activating",
        "provably_inactive",
        "unknown_fallback",
        "adversarial",
    }
    activating = [c for c in replay["cases"] if c["cohort"] == "known_activating"]
    assert len(activating) >= 4
    assert all(c["expect_disposition"] == "run" for c in activating)
    seen = {
        c.get("adversarial_kind")
        for c in replay["cases"]
        if c["cohort"] == "adversarial"
    }
    assert required_adversarial_kinds().issubset(seen)


def test_identity_mismatch_blocks_effective_skip() -> None:
    ok, _ = identity_matches(
        case={
            "mechanism_id": "singleton_bypass",
            "locked_corpus_id": "locked/a",
            "expected_theorem_ref": "MechanismTrigger.singleton_bypass_output_change_implies_trigger",
            "expected_cost_model_version": 1,
            "cost_model": "NeuralForwardOnlyModel",
        },
        admission_corpus_id="wrong/b",
        admission_theorem_ref="MechanismTrigger.singleton_bypass_output_change_implies_trigger",
        admission_mechanism_id="singleton_bypass",
        cost_model=NamedCostModel("NeuralForwardOnlyModel", 1),
    )
    assert ok is False
    assert effective_disposition(raw_disposition="skip_no_effect", identity_ok=False) == "run"
    assert (
        effective_disposition(raw_disposition="skip_no_effect", identity_ok=True)
        == "skip_no_effect"
    )


def test_benchmark_hits_recall_gate() -> None:
    report = run_benchmark(root=ROOT)
    assert_benchmark_passed(report)
    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True
    assert report["metrics"]["known_activating_recall"] == 1.0
    assert report["metrics"]["false_skip_count"] == 0
    assert report["metrics"]["unknown_fallback_rate"] == 1.0
    assert report["metrics"]["provably_inactive_specificity"] == 1.0
    assert report["metrics"]["compute_saved"] > 0
    assert report["identities"]["revmath_corpus_id"] == "corpus.rm.hermetic.v1"
    assert report["identities"]["revmath_corpus_sha256"]
    assert report["identities"]["trigger_theorem_refs"]


def test_committed_results_match_live_benchmark() -> None:
    report = run_benchmark(root=ROOT)
    assert_benchmark_passed(report)
    committed = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert committed["schema"] == REPORT_SCHEMA
    assert committed["passed"] is True
    assert committed["metrics"]["known_activating_recall"] == 1.0
    assert committed["metrics"]["false_skip_count"] == 0
    assert [item["case_id"] for item in committed["cases"]] == [
        item["case_id"] for item in report["cases"]
    ]
    assert all(item["ok"] for item in committed["cases"])
    assert all(not item["false_skip"] for item in committed["cases"])
