import json
import subprocess
from pathlib import Path

import pytest

from scripts import replay_harness_failures
from scripts.replay_harness_failures import main
from slm_training.harnesses.eval.harness_replay import (
    REPLAY_SUITES,
    ArchivedFailureV1,
    HarnessProvenanceV1,
    collect_archived_failures,
    replay_failure,
    select_suite_stratified_failures,
)


def _case(suite: str, index: int) -> ArchivedFailureV1:
    prediction = f"bad-output-{suite}-{index}"
    return ArchivedFailureV1(
        event_id=f"family-{index % 3}/eval_{suite}.json#{index}",
        suite=suite,
        record_id=f"{suite}-{index}",
        raw_prediction=prediction,
        raw_prediction_sha256=__import__("hashlib").sha256(prediction.encode()).hexdigest(),
        original_failure="parse" if index % 2 else "no_placeholders",
        provenance=HarnessProvenanceV1(
            source_eval_sha256="0" * 64,
            evaluation_policy={"compiler_decode_mode": "tree" if index % 2 else "off"},
            timeout_seconds=1.0,
            canvas_cap=160,
            parser_fallback="none",
            repair_policy="none",
            runtime="archived",
            verifier="production",
        ),
    )


def _write_archive_and_records(tmp_path: Path, *, per_suite: int = 20) -> tuple[Path, Path]:
    archive = tmp_path / "archive"
    records_root = tmp_path / "records"
    archive.mkdir()
    for suite in REPLAY_SUITES:
        details = []
        records = []
        for index in range(per_suite):
            record_id = f"{suite}-{index}"
            details.append({"id": record_id, "parse_ok": False, "prediction": "!!!"})
            records.append(
                {
                    "id": record_id,
                    "prompt": "Render a button.",
                    "openui": 'root = Button(":btn.action")',
                    "placeholders": [":btn.action"],
                    "split": suite,
                    "source": "test",
                }
            )
        (archive / f"eval_{suite}.json").write_text(
            json.dumps({"suite": suite, "details": details}), encoding="utf-8"
        )
        path = records_root / "suites" / suite / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    return archive, records_root


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
        json.dumps({"suite": "smoke", "details": [{"id": "a", "parse_ok": False, "prediction": prediction}]}),
        encoding="utf-8",
    )
    cases = collect_archived_failures(tmp_path)
    assert len(cases) == 1
    assert cases[0].raw_prediction == prediction
    assert cases[0].constrained_id == "unknown_not_captured"


def test_suite_sampler_is_deterministic_and_balanced() -> None:
    cases = [_case(suite, index) for suite in REPLAY_SUITES for index in range(21)]
    selected = select_suite_stratified_failures(cases, suite_quota=20)
    reversed_selected = select_suite_stratified_failures(list(reversed(cases)), suite_quota=20)
    assert len(selected) == 100
    assert [case.event_id for case in selected] == [case.event_id for case in reversed_selected]
    assert {suite: sum(case.suite == suite for case in selected) for suite in REPLAY_SUITES} == {
        suite: 20 for suite in REPLAY_SUITES
    }


def test_suite_sampler_fails_closed_when_a_suite_is_short() -> None:
    cases = [_case(suite, index) for suite in REPLAY_SUITES for index in range(20)]
    with pytest.raises(ValueError, match="rico_held: short 1"):
        select_suite_stratified_failures(cases[:-1], suite_quota=20)


def test_runtime_timeout_is_not_mislabeled_as_decoder_timeout(monkeypatch) -> None:
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["node"], 1.0)

    monkeypatch.setattr(replay_harness_failures, "run_preview_verifier", timed_out)
    assert replay_harness_failures._runtime_result("root = Stack([])", timeout_s=1.0) == {
        "status": "timeout",
        "rendered": None,
    }


def test_cli_replays_redacted_stratified_matrix(tmp_path) -> None:
    archive, records = _write_archive_and_records(tmp_path)
    output = tmp_path / "audit.json"
    assert main(
        [
            "--archive-root",
            str(archive),
            "--eval-data-root",
            str(records),
            "--output",
            str(output),
            "--no-publish-agentv",
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "!!!" not in str(payload)
    assert payload["summary"]["n"] == 100
    assert payload["summary"]["selected_n_by_suite"] == {
        suite: 20 for suite in REPLAY_SUITES
    }
    assert payload["summary"]["decoder_causal_flip_rate"] is None
    assert payload["summary"]["architecture_claims_blocked"] is True
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
