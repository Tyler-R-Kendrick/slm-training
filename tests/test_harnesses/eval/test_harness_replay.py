from pathlib import Path

import pytest

from scripts.replay_harness_failures import main
from slm_training.harnesses.eval.harness_replay import (
    ArchivedFailureV1,
    HarnessProvenanceV1,
    collect_archived_failures,
    replay_failure,
)


def test_replay_preserves_archived_bytes_and_marks_feasibility_only() -> None:
    prediction = "x" * 500
    case = ArchivedFailureV1(
        event_id="archive/smoke/case",
        suite="smoke",
        record_id="case",
        raw_prediction=prediction,
        raw_prediction_sha256="c38c2bf3055c516a98ac5d97f30e7c364e827bc0199e1c3415b794afbe55dcad",
        original_failure="parse",
        provenance=HarnessProvenanceV1(
            source_eval_sha256="0" * 64,
            evaluation_policy={},
            timeout_seconds=1.0,
            canvas_cap=160,
            parser_fallback="none",
            repair_policy="none",
            runtime="archived",
            verifier="production",
        ),
    )
    replay = replay_failure(case)
    assert replay["raw_prediction_preserved"] is True
    assert replay["actual_decode_replayed"] is False
    assert {"truncation_sensitive", "canvas_sensitive"} <= set(replay["classifications"])


def test_collect_archived_failures_uses_stable_bytes(tmp_path) -> None:
    prediction = "root = Stack([])"
    (tmp_path / "eval_smoke.json").write_text(
        __import__("json").dumps({"suite": "smoke", "details": [{"id": "a", "parse_ok": False, "prediction": prediction}]}),
        encoding="utf-8",
    )
    cases = collect_archived_failures(tmp_path)
    assert len(cases) == 1
    assert cases[0].raw_prediction == prediction
    assert cases[0].constrained_id == "unknown_not_captured"


def test_cli_does_not_publish_raw_predictions(tmp_path) -> None:
    prediction = "root = Stack([])"
    (tmp_path / "eval_smoke.json").write_text(
        __import__("json").dumps({"details": [{"id": "a", "parse_ok": False, "prediction": prediction}]}),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    assert main(["--archive-root", str(tmp_path), "--limit", "1", "--output", str(output)]) == 0
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert prediction not in str(payload)
    assert payload["labels_flip_rate"] is None
    assert payload["architecture_claims_blocked"] is None
    assert payload["matrix_status"] == "not_run_missing_original_decoder_traces"
    assert payload["rows"][0]["raw_prediction_id"].startswith("sha256:")


def test_cli_requires_explicit_acknowledgement_for_design_output(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--archive-root",
                str(tmp_path),
                "--output",
                str(Path.cwd() / "docs/design/replay-harness-private.json"),
            ]
        )
