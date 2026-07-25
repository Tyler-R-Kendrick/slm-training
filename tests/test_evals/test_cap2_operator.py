from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.cap2_operator import (
    build_frozen_cap2_suite,
    evaluate_fixture_policies,
    oracle_prediction,
    score_cap2_predictions,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "src/slm_training/resources/evals/cap2_operator_v1.json"
SOURCE = (
    ROOT
    / "src/slm_training/resources/data/eval"
    / "e763_symbol_only_eval_r2_20260722/suites/held_out/records.jsonl"
)
STAMP = {
    "stamp_schema": "version_stamp/v1",
    "code_commit": "test",
    "code_dirty": True,
    "components": {"evals.cap2_operator": "v1"},
    "stamped_at": "test",
}


def _suite(tmp_path: Path) -> dict:
    return build_frozen_cap2_suite(
        manifest_path=MANIFEST,
        source_records_path=SOURCE,
        work_dir=tmp_path,
        version_stamp=STAMP,
    )


def test_frozen_suite_replays_and_covers_explicit_strata(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert suite["suite_hash"] == manifest["suite_hash"]
    assert suite["operator_corpus_fingerprint"] == (
        manifest["operator_corpus_fingerprint"]
    )
    assert len(suite["cases"]) == 20
    assert {case["stratum"] for case in suite["cases"]} == set(
        manifest["required_strata"]
    )
    assert any(
        case["stratum"] == "held_out_composition"
        for case in suite["cases"]
    )
    assert all(
        case["source_record_id"].startswith("held_out_")
        or case["source_record_id"] == "cap2_contract"
        for case in suite["cases"]
    )
    assert suite["nl"] == {
        "available": False,
        "reason": "CERT_CAP1_unavailable",
        "dependency_issue": "SLM-379",
    }
    assert {
        "undo_redo",
        "merge_success",
        "merge_conflict",
        "stale_reference",
        "cap0_retention",
        "cap1_retention",
    } <= set(suite["contract_inventory"])


def test_oracle_passes_and_degenerate_policies_fail(tmp_path: Path) -> None:
    scores = evaluate_fixture_policies(_suite(tmp_path))
    assert scores["oracle"]["gate_pass"] is True
    assert scores["oracle"]["case_successes"] == 20
    for name in ("unchanged", "generic_valid_ast", "constant_operator"):
        assert scores[name]["gate_pass"] is False
        assert scores[name]["case_successes"] < scores[name]["case_count"]
    assert scores["constant_operator"]["case_successes"] == 1


def test_missing_prediction_fails_closed_with_confidence_bounds(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    first = suite["cases"][0]
    score = score_cap2_predictions(
        suite, {first["case_id"]: oracle_prediction(first)}
    )
    assert score["gate_pass"] is False
    assert score["case_successes"] == 1
    assert score["case_interval"]["high"] < 0.4
    assert score["retention_diagnostics"]["cap1"]["status"] == "unavailable"
    assert set(score["telemetry"]) == {
        "active_nodes",
        "compiler_calls",
        "latency_ms",
        "model_calls",
        "node_passes",
        "peak_memory_bytes",
        "remask_phases",
        "verifier_calls",
    }


def test_source_or_suite_drift_invalidates_frozen_contract(
    tmp_path: Path,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["suite_hash"] = "0" * 64
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="suite hash drifted"):
        build_frozen_cap2_suite(
            manifest_path=changed,
            source_records_path=SOURCE,
            work_dir=tmp_path / "run",
            version_stamp=STAMP,
        )


def test_negative_telemetry_is_rejected(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    predictions = {
        case["case_id"]: oracle_prediction(case) for case in suite["cases"]
    }
    predictions[suite["cases"][0]["case_id"]]["telemetry"]["model_calls"] = -1
    with pytest.raises(ValueError, match="negative CAP2 telemetry"):
        score_cap2_predictions(suite, predictions)
