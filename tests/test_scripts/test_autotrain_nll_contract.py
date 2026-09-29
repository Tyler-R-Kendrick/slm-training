"""Actual diagnostic producer/consumer contracts, not model evidence."""

import json
import hashlib

import pytest

from tests.casefiles import case_values

from scripts.autotrain_metrics import paired_nll_selection, read_paired_nll
from scripts.autotrain_nll import run_arm_eval_nll


@pytest.mark.parametrize(
    "selection",
    case_values(__file__, "test_malformed_paired_selection_never_crashes_into_a_verdict"),
)
def test_malformed_paired_selection_never_crashes_into_a_verdict(tmp_path, selection):
    for name in ("control", "candidate"):
        run = tmp_path / name
        run.mkdir()
        (run / "eval_nll_records.json").write_text(
            json.dumps(
                {
                    "schema": "eval_nll_records/v1",
                    "definition_hash": "fixture",
                    "records": {"a": 1.0},
                    "selection": selection,
                }
            )
        )
    pairs, counts, reasons = read_paired_nll(
        tmp_path / "control", tmp_path / "candidate"
    )
    assert pairs is None and counts["control_n"] == 1
    assert reasons[0].startswith("measurement_incomplete:")


def test_actual_attachment_is_diagnostic_only_and_pairs_by_locked_identity(tmp_path):
    selection = {
        "selected_record_ids": ["a", "b"],
        "selected_root_ids": ["family-a", "family-b"],
    }
    for name, value in (("control", 2.0), ("candidate", 1.0)):
        run_arm_eval_nll(
            tmp_path / name,
            {
                "eval_nll": value,
                "records": {"a": value, "b": value},
                "selection": selection,
                "definition_hash": "fixture",
                "estimator_id": "fixture-denoising",
                "eval_version": "fixture-v1",
            },
        )
    pairs, counts, reasons = read_paired_nll(
        tmp_path / "control", tmp_path / "candidate"
    )
    assert not reasons and counts["candidate_n"] == 2
    assert pairs["selected_record_ids"] == ["a", "b"]
    assert pairs["root_ids"] == {"a": "family-a", "b": "family-b"}
    board = json.loads((tmp_path / "candidate" / "scoreboard.json").read_text())
    assert board["suites"]["smoke"]["diagnostic_complete"]
    assert not board["measurement_complete"]
    assert not board["suites"]["smoke"]["decoded_probe_complete"]
    run_arm_eval_nll(tmp_path / "candidate", {"eval_nll": 0.5})
    pairs, counts, _ = read_paired_nll(tmp_path / "control", tmp_path / "candidate")
    assert pairs is None and counts["candidate_n"] == 0
    assert (tmp_path / "candidate" / "eval_nll_records.json").is_file()


@pytest.mark.parametrize("key", ["estimator_id", "units", "eval_version"])
def test_different_measurement_contracts_never_pair(key):
    with pytest.raises(ValueError, match="mismatch"):
        paired_nll_selection({key: "old"}, {key: "new"})


@pytest.mark.parametrize("suites", [[], [0], "bad", {"smoke": []}, {"smoke": None}])
def test_malformed_scoreboard_cannot_crash_or_supply_pairs(tmp_path, suites):
    run = tmp_path / "candidate"
    run.mkdir()
    (run / "scoreboard.json").write_text(json.dumps({"suites": suites}))
    pairs, counts, _ = read_paired_nll(tmp_path / "control", run)
    assert pairs is None and counts["candidate_n"] == 0


def test_loss_cli_emits_controller_attempt_and_durable_report_digest(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_loss_suites as cli
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.evals import loss_suites
    from slm_training.models.twotower import TwoTowerModel

    store = CampaignStore("campaign", tmp_path)
    store.append_event("experiment_attempt_started", experiment_id="candidate",
                       detail={"attempt_id": "actual-controller-attempt"})
    monkeypatch.setattr(TwoTowerModel, "from_checkpoint", lambda *a, **k: object())
    monkeypatch.setattr(loss_suites, "evaluate_loss_suites", lambda *a, **k: {
        "aggregate": {"complete": True}, "categories": {}})
    monkeypatch.setattr(loss_suites, "write_loss_suite_report",
                        lambda path, report: path.write_text(json.dumps(report)))
    out = tmp_path / "loss.json"
    assert cli.main(["--checkpoint", "model.pt", "--test-dir", str(tmp_path),
                     "--out", str(out), "--campaign-dir", str(store.root),
                     "--experiment-id", "candidate"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["attempt_id"] == "actual-controller-attempt"
    assert json.loads(out.read_text())["attempt_id"] == emitted["attempt_id"]
    assert emitted["report_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


@pytest.mark.parametrize("extra", case_values(__file__, "test_loss_cli_rejects_unbound_campaign_before_loading_model"))
def test_loss_cli_rejects_unbound_campaign_before_loading_model(extra, monkeypatch):
    from scripts import evaluate_loss_suites as cli
    from slm_training.models.twotower import TwoTowerModel

    def must_not_load(*args, **kwargs):
        pytest.fail("invalid campaign binding reached model loading")

    monkeypatch.setattr(TwoTowerModel, "from_checkpoint", must_not_load)
    with pytest.raises((SystemExit, ValueError)):
        cli.main(["--checkpoint", "model.pt", "--test-dir", "data", *extra])


def test_retry_guard_stays_strict_for_legacy_and_wrong_attempt(tmp_path):
    from scripts.autotrain_search import _loss_report
    from slm_training.autoresearch.storage import CampaignStore

    store = CampaignStore("campaign", tmp_path)
    inputs = {"eval_nll": 1.0, "records": {"a": 1.0},
              "selection": {"selected_record_ids": ["a"], "selected_root_ids": ["root"]},
              "row_evidence": [{"case_id": "a", "nll": 1.0}], "estimator_id": "fixture"}
    run = store.root / "runs/candidate"
    run_arm_eval_nll(run, inputs)
    with pytest.raises(ValueError, match="legacy_report_reused_after_retry"):
        _loss_report(store, "candidate", attempt_id="retry", attempt_count=2)
    run_arm_eval_nll(run, {**inputs, "attempt_id": "producer"})
    with pytest.raises(ValueError, match="attempt_identity_mismatch"):
        _loss_report(store, "candidate", attempt_id="retry", attempt_count=2)


def _legacy_retry_evidence(tmp_path):
    from scripts.autotrain_search import _loss_report
    from slm_training.autoresearch.experiment_campaign import (
        CampaignLockV1,
        ExperimentCampaignV1,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload

    store = CampaignStore("campaign", tmp_path)
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id="campaign", experiment_id="candidate")
    )
    campaign_lock = store.lock_experiment_campaign(manifest)
    assert isinstance(campaign_lock, CampaignLockV1)
    selection = {
        "selected_record_ids": ["case-a"],
        "selected_root_ids": ["root-a"],
        "input_sha256s": ["1" * 64],
        "selection_sha256": "2" * 64,
    }
    pair = {
        "design_digest": "3" * 64,
        "replicate_id": "replicate",
        "arm_ids": ["control", "candidate"],
        "treatment_ids": ["control-treatment", "candidate-treatment"],
        "design": {
            "bindings": {
                "endpoint": {
                    "loss_measurement": {
                        "selection": selection,
                        "version": "suite-v1",
                        "units": "nats_per_masked_token",
                        "estimator_id": "estimator-v1",
                    }
                }
            }
        },
    }
    design_path = store.write_artifact("treatment_designs", pair)
    store.append_event(
        "experiment_design_locked",
        artifact_sha256=design_path.stem,
        detail={"design_digest": pair["design_digest"]},
    )
    manifest_sha = campaign_lock.manifest_sha256

    run = store.root / "runs/candidate"
    run_arm_eval_nll(
        run,
        {
            "eval_nll": 1.0,
            "records": {"case-a": 1.0},
            "selection": selection,
            "row_evidence": [{"case_id": "case-a", "nll": 1.0}],
            "estimator_id": "estimator-v1",
            "eval_version": "suite-v1",
        },
    )
    report_sha = hashlib.sha256((run / "eval_nll_records.json").read_bytes()).hexdigest()
    source_sha = "5" * 64
    checkpoint_sha = "6" * 64
    version_stamp = {"code_commit": "fixture-commit"}
    eval_data_sha = "7" * 64
    scoreboard_path = run / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    smoke = scoreboard["suites"]["smoke"]
    smoke.update(
        checkpoint="checkpoint.pt",
        checkpoint_sha256=checkpoint_sha,
        selection_sha256=selection["selection_sha256"],
        harness_provenance={"source_eval_sha256": source_sha},
    )
    scoreboard.update(version_stamp=version_stamp, eval_data_manifest_sha=eval_data_sha)
    scoreboard_path.write_text(json.dumps(scoreboard))

    command = [
        "python", "-m", "scripts.evaluate_model", "--checkpoint", "checkpoint.pt"
    ]
    cursor_inputs = {
        "schema": "command_cursor/v1",
        "experiment": {"campaign_id": "campaign", "experiment_id": "candidate"},
        "commands": [command],
        "identity": "source-runtime-identity",
        "manifest": manifest_sha,
    }
    input_path = store.write_artifact("command_cursor_inputs", cursor_inputs)
    input_digest = input_path.stem
    store.append_event(
        "command_cursor_locked",
        experiment_id="candidate",
        artifact_sha256=input_digest,
        detail={"input_digest": input_digest},
    )

    failed = {"attempt_id": "failed", "attempt_ordinal": 0,
              "design_digest": pair["design_digest"], "replicate_id": pair["replicate_id"],
              "treatment_id": pair["treatment_ids"][1]}
    current = {**failed, "attempt_id": "retry", "attempt_ordinal": 1}
    store.append_event("experiment_attempt_started", experiment_id="candidate", detail=failed)
    store.append_event("experiment_attempt_returned", experiment_id="candidate",
                       detail={**failed, "exit_code": 124})
    store.append_event("experiment_attempt_started", experiment_id="candidate", detail=current)
    store.append_event("command_cursor_started", experiment_id="candidate",
                       detail={"input_digest": input_digest, "attempt": 1})
    parsed_output = {
        "measurement_complete": True,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha,
        "eval_data_manifest_sha": eval_data_sha,
        "version_stamp": version_stamp,
        "suites": {
            "smoke": {
                "selection_sha256": selection["selection_sha256"],
                "harness_provenance": {"source_eval_sha256": source_sha},
            }
        },
    }
    outcome = {
        "status": "completed", "exit_code": 0, "experiment_id": "candidate",
        "campaign_id": "campaign", "campaign_manifest_sha256": manifest_sha,
        "stage_telemetry": [{"command": command, "parsed_output": parsed_output}],
    }
    cursor = {
        "input_digest": input_digest, "record_type": "committed", "position": 1,
        "spent_seconds": 1.0, "attempt": 1, "outcome": outcome,
    }
    cursor_path = store.write_artifact("command_cursors", cursor)
    store.append_event("command_cursor_committed", experiment_id="candidate",
                       artifact_sha256=cursor_path.stem,
                       detail={"input_digest": input_digest})
    store.append_event("experiment_attempt_returned", experiment_id="candidate",
                       detail={**current, "exit_code": 0})
    return store, pair, scoreboard, report_sha, _loss_report


def test_retry_rebinds_legacy_report_from_exact_committed_attempt(tmp_path):
    store, pair, _, _, loss_report = _legacy_retry_evidence(tmp_path)
    report = loss_report(
        store, "candidate", attempt_id="retry", attempt_count=2, pair=pair
    )
    assert report["selection"]["selection_sha256"] == "2" * 64
    assert report["per_record"] == [{"case_id": "case-a", "nll": 1.0}]


@pytest.mark.parametrize("mismatch", ["checkpoint", "source", "selection", "attempt"])
def test_retry_rejects_legacy_report_when_any_provenance_binding_differs(
    tmp_path, mismatch
):
    store, pair, scoreboard, _, loss_report = _legacy_retry_evidence(tmp_path)
    if mismatch == "checkpoint":
        scoreboard["suites"]["smoke"]["checkpoint_sha256"] = "8" * 64
        (store.root / "runs/candidate/scoreboard.json").write_text(
            json.dumps(scoreboard)
        )
    elif mismatch == "source":
        scoreboard["suites"]["smoke"]["harness_provenance"][
            "source_eval_sha256"
        ] = "8" * 64
        (store.root / "runs/candidate/scoreboard.json").write_text(
            json.dumps(scoreboard)
        )
    elif mismatch == "selection":
        pair["design"]["bindings"]["endpoint"]["loss_measurement"][
            "selection"
        ]["selection_sha256"] = "8" * 64
    else:
        store.append_event(
            "experiment_attempt_returned",
            experiment_id="candidate",
            detail={
                "attempt_id": "retry", "attempt_ordinal": 1,
                "design_digest": pair["design_digest"],
                "replicate_id": pair["replicate_id"],
                "treatment_id": "wrong-treatment", "exit_code": 0,
            },
        )
        with pytest.raises(ValueError, match="legacy_report_reused_after_retry"):
            loss_report(store, "candidate", attempt_id="retry", attempt_count=2,
                        pair=pair)
        return
    with pytest.raises(ValueError, match="legacy_report_reused_after_retry"):
        loss_report(store, "candidate", attempt_id="retry", attempt_count=2,
                    pair=pair)


def test_loss_cli_refuses_attempt_change_before_publication(tmp_path, monkeypatch):
    from scripts import evaluate_loss_suites as cli
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.evals import loss_suites
    from slm_training.models.twotower import TwoTowerModel

    store = CampaignStore("campaign", tmp_path)
    store.append_event("experiment_attempt_started", experiment_id="candidate",
                       detail={"attempt_id": "original"})
    monkeypatch.setattr(TwoTowerModel, "from_checkpoint", lambda *a, **k: object())

    def evaluate(*args, **kwargs):
        store.append_event("experiment_attempt_returned", experiment_id="candidate",
                           detail={"attempt_id": "original", "exit_code": 124})
        store.append_event("experiment_attempt_started", experiment_id="candidate",
                           detail={"attempt_id": "retry"})
        return {"aggregate": {}, "categories": {}}

    monkeypatch.setattr(loss_suites, "evaluate_loss_suites", evaluate)
    out = tmp_path / "loss.json"
    with pytest.raises(ValueError, match="attempt changed"):
        cli.main(["--checkpoint", "model.pt", "--test-dir", str(tmp_path),
                  "--out", str(out), "--campaign-dir", str(store.root),
                  "--experiment-id", "candidate"])
    assert not out.exists()
