"""Producer attribution and retry attachment for live loss-suite reports."""

import json
from pathlib import Path

import pytest

from scripts.autotrain_nll import campaign_attempt_id
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.autonomous_learning import (
    measurement_bundle as bundle,
)


@pytest.fixture
def loss_attempt(tmp_path: Path):
    store = CampaignStore("live-nll-attempt-repair", tmp_path)
    run_dir = store.root / "runs" / "arm"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "loss_suites.json"
    rows = [
        {"case_id": "case-a", "nll": 1.25, "masked_tokens": 3},
        {"case_id": "case-b", "nll": 1.75, "masked_tokens": 4},
    ]
    report_path.write_text(
        json.dumps({"attempt_id": "producer", "per_record": rows}),
        encoding="utf-8",
    )
    command = [
        "python",
        "-m",
        "scripts.evaluate_loss_suites",
        "--out",
        str(report_path),
    ]
    receipt = {"arm_id": "arm", "commands": [command]}
    return {
        "store": store,
        "run_dir": run_dir,
        "report_path": report_path,
        "rows": rows,
        "command": command,
        "receipt": receipt,
    }


def commit_cursor(case, *, started_attempt="producer", start_after_return=False):
    store = case["store"]
    store.append_event(
        "experiment_attempt_started",
        experiment_id="arm",
        detail={"attempt_id": started_attempt},
    )
    if start_after_return:
        store.append_event(
            "experiment_attempt_returned",
            experiment_id="arm",
            detail={"attempt_id": started_attempt, "exit_code": 0},
        )
    inputs = store.write_artifact(
        "command_cursor_inputs",
        {
            "commands": [case["command"]],
            "experiment": {"experiment_id": "arm"},
        },
    )
    store.append_event(
        "command_cursor_started",
        experiment_id="arm",
        detail={"attempt": 1, "input_digest": inputs.stem},
    )
    payload = {
        "input_digest": inputs.stem,
        "attempt": 1,
        "outcome": {
            "stage_telemetry": [
                {
                    "command": case["command"],
                    "exit_code": 0,
                    "duration_seconds": 1.0,
                    "parsed_output": {
                        "out": str(case["report_path"]),
                        "report_sha256": bundle.file_sha(case["report_path"]),
                        "attempt_id": "producer",
                    },
                }
            ]
        },
    }
    artifact = store.write_artifact("command_cursors", payload)
    store.append_event(
        "command_cursor_committed",
        experiment_id="arm",
        artifact_sha256=artifact.stem,
        detail={"input_digest": inputs.stem},
    )
    if not start_after_return:
        store.append_event(
            "experiment_attempt_returned",
            experiment_id="arm",
            detail={"attempt_id": started_attempt, "exit_code": 0},
        )
    return artifact


def start_retry(case, *, close=False):
    store = case["store"]
    store.append_event(
        "experiment_attempt_started",
        experiment_id="arm",
        detail={"attempt_id": "publication-retry"},
    )
    if close:
        store.append_event(
            "experiment_attempt_returned",
            experiment_id="arm",
            detail={"attempt_id": "publication-retry", "exit_code": 124},
        )


def _arm(case):
    selection = {
        "selected_record_ids": ["case-a", "case-b"],
        "selected_root_ids": ["root-a", "root-b"],
    }
    row_evidence = [
        {**row, "root_id": root}
        for row, root in zip(case["rows"], selection["selected_root_ids"])
    ]
    return {
        "run_dir": case["run_dir"],
        "records": {row["case_id"]: row["nll"] for row in case["rows"]},
        "loss": {
            "categories": {"broad": {"aggregate": {"mean_nll": 1.5}}},
            "selection": selection,
            "per_record": row_evidence,
            "estimator_id": "fixture-estimator",
            "definition": {"fixture": "loss-definition"},
        },
    }


def _patch_finalizer(monkeypatch, case):
    from slm_training.harnesses.experiments.autonomous_learning import (
        measurement_fixture_evidence,
    )

    monkeypatch.setattr(
        bundle,
        "load_contract",
        lambda *_: {"inputs": {"eval_version": "fixture-eval-v1"}},
    )
    monkeypatch.setattr(
        measurement_fixture_evidence, "checked_arm", lambda *_: _arm(case)
    )


def test_valid_cursor_started_during_original_producer_attempt(loss_attempt):
    case = loss_attempt
    cursor = commit_cursor(case)

    evidence = bundle.verified_loss_producer(
        case["store"], case["receipt"], case["report_path"]
    )

    assert evidence["attempt_id"] == "producer"
    assert evidence["loss_report_sha256"] == bundle.file_sha(case["report_path"])
    assert evidence["cursor_artifact_sha256"] == cursor.stem


def test_cursor_started_after_producer_return_is_rejected(loss_attempt):
    case = loss_attempt
    commit_cursor(case, start_after_return=True)

    with pytest.raises(ValueError, match="successful digest-bound producer"):
        bundle.verified_loss_producer(
            case["store"], case["receipt"], case["report_path"]
        )


def test_changed_loss_report_bytes_are_rejected(loss_attempt):
    case = loss_attempt
    commit_cursor(case)
    report = json.loads(case["report_path"].read_text(encoding="utf-8"))
    report["per_record"][0]["nll"] = 99.0
    case["report_path"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="successful digest-bound producer"):
        bundle.verified_loss_producer(
            case["store"], case["receipt"], case["report_path"]
        )


def test_cursor_from_different_producer_attempt_is_rejected(loss_attempt):
    case = loss_attempt
    commit_cursor(case, started_attempt="different-producer")

    with pytest.raises(ValueError, match="successful digest-bound producer"):
        bundle.verified_loss_producer(
            case["store"], case["receipt"], case["report_path"]
        )


def test_closed_retry_cannot_receive_remeasurement(loss_attempt, monkeypatch):
    case = loss_attempt
    commit_cursor(case)
    start_retry(case, close=True)
    _patch_finalizer(monkeypatch, case)

    with pytest.raises(ValueError, match="active campaign attempt"):
        bundle.finalize_bundle_remeasurement(
            case["store"],
            {**case["receipt"], "contract_path": "locked-contract"},
        )


def test_retry_attachment_preserves_producer_binding_and_every_row(
    loss_attempt, monkeypatch
):
    case = loss_attempt
    cursor = commit_cursor(case)
    start_retry(case)
    _patch_finalizer(monkeypatch, case)
    original_loss_bytes = case["report_path"].read_bytes()
    expected_rows = _arm(case)["loss"]["per_record"]

    bundle.finalize_bundle_remeasurement(
        case["store"],
        {**case["receipt"], "contract_path": "locked-contract"},
    )

    attached = json.loads(
        (case["run_dir"] / "eval_nll_records.json").read_text(encoding="utf-8")
    )
    cursor_input_digest = next(
        event["detail"]["input_digest"]
        for event in case["store"].verify_event_chain()
        if event["event_type"] == "command_cursor_committed"
    )
    assert attached["attempt_id"] == "publication-retry"
    assert attached["producer_evidence"] == {
        "attempt_id": "producer",
        "loss_report_sha256": bundle.file_sha(case["report_path"]),
        "cursor_artifact_sha256": cursor.stem,
        "cursor_input_digest": cursor_input_digest,
    }
    assert attached["mean_nll"] == 1.5
    assert attached["n_records"] == 2
    assert attached["records"] == {"case-a": 1.25, "case-b": 1.75}
    assert attached["row_evidence"] == expected_rows
    assert attached["selection"] == _arm(case)["loss"]["selection"]
    assert case["report_path"].read_bytes() == original_loss_bytes
    assert campaign_attempt_id(case["store"], "arm") == "publication-retry"
