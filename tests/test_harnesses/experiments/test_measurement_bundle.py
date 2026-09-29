"""Controller-bundle plumbing tests; tiny bytes are not a trained model claim."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from scripts import autoresearch
from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.experiment_campaign import campaign_manifest_sha256
from slm_training.autoresearch.schemas import CampaignSpec, ExperimentSpec
from slm_training.harness_core.checkpoint_bundle import (
    stage_checkpoint_bundle,
    validate_bundle,
)
from slm_training.harnesses.experiments.autonomous_learning import (
    measurement_bundle as bundle,
)
from slm_training.harnesses.experiments.autonomous_learning.measurement_fixture_plan import (
    supervised_matrix,
)
from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint

pytest_plugins = ("tests.test_harnesses.experiments.measurement_fixture_support",)


def test_fixture_resolves_real_upstream_without_bypassing_canonical_guard(monkeypatch):
    calls = []

    def git(*args, **kwargs):
        calls.append(args)
        value = "upstream" if "origin/main^{commit}" in args else "integration"
        if "upstream^{commit}" in args:
            value = "upstream"
        return SimpleNamespace(stdout=value + "\n", returncode=0)

    monkeypatch.setattr(autoresearch, "_git", git)
    assert bundle.continuous_source_commits("integration") == {
        "upstream_commit": "upstream",
        "integration_commit": "integration",
    }
    assert ("merge-base", "--is-ancestor", "upstream", "integration") in calls
    with pytest.raises(ValueError, match="upstream_commit is stale"):
        autoresearch._validate_continuous_commits("integration", "integration")
    with pytest.raises(ValueError, match="clean tracked worktree"):
        autoresearch._validate_continuous_commits("upstream", "integration")


@pytest.fixture
def contract(prepared, tmp_path, monkeypatch):
    store, previous = prepared
    campaign = CampaignSpec(
        campaign_id=store.campaign_id,
        objective="Diagnostic test plumbing",
        primary_metric=previous["primary"]["metric"],
    )
    store.initialize(campaign)
    experiment = ExperimentSpec.model_validate(
        previous["arms"]["control"]["experiment"]
    )
    manifest = store.load_experiment_campaign(
        "MEA-SIX-RESOLVED-ENDPOINT"
    ).manifest.model_copy(update={"experiment_id": "control"})
    checkpoint = _checkpoint(tmp_path / "source", marker=b"fixture-not-neural")
    root = tmp_path / "bundle-owner"
    digest = stage_checkpoint_bundle(root, checkpoint, {})
    directory, _ = validate_bundle(root, digest)
    source = tmp_path / "train_summary.json"
    source.write_text(
        json.dumps(
            {
                "track": {"trainable_params": 65826},
                "version_stamp": {"code_dirty": True, "code_commit": "a" * 40},
            }
        )
    )
    inputs = copy.deepcopy(previous["inputs"])
    inputs["arms"]["control"].update(
        checkpoint=str(directory / "last.pt"),
        bundle_digest=digest,
        checkpoint_sha256=bundle.file_sha(directory / "last.pt"),
    )
    compiled = compile_commands(campaign, experiment, output_root=store.root.parent)
    commands = bundle.inference_commands(
        compiled, directory / "last.pt", inputs, store.root / "runs/control"
    )
    plan = dict(
        schema="supervised_measurement/v1",
        campaign_id=store.campaign_id,
        role="inference_only_diagnostic",
        new_training=False,
        promotion_allowed=False,
        ship_eligible=False,
        execution_identity="pinned",
        inputs=inputs,
        arms={
            "control": dict(
                experiment=experiment.model_dump(mode="json"),
                manifest_sha256=campaign_manifest_sha256(manifest),
                source_summary=str(source),
                source_summary_sha256=bundle.file_sha(source),
                compiled=compiled,
                commands=commands,
            )
        },
    )
    monkeypatch.setattr(bundle, "selected_identity", lambda rows: inputs["selection"])
    monkeypatch.setattr(bundle, "load_suite_records", lambda *args: [])
    monkeypatch.setattr(
        "scripts.autoresearch_command_cursor.resolved_continuation_grant",
        lambda *args: SimpleNamespace(execution_identity="pinned"),
    )
    path = store.write_artifact("supervised_measurement", plan)
    store.append_event("supervised_measurement_locked", artifact_sha256=path.stem)
    args = SimpleNamespace(
        diagnostic_bundle_plan=path, execute=True, reuse_train_run=None
    )
    return store, experiment, manifest, plan, args


def test_dirty_inference_bundle_does_not_fabricate_clean_training(contract):
    store, experiment, manifest, plan, args = contract
    receipt = bundle.prepare_bundle_remeasurement(args, store, experiment, manifest)
    assert receipt["source_training_version_stamp"]["code_dirty"] is True
    assert receipt["executed"] is False
    assert receipt["training_continued"] is False
    assert receipt["promotion_authority"] is False
    assert receipt["trainable_params"] == 65826
    assert receipt["commands"] == plan["arms"]["control"]["commands"]
    assert [c[2] for c in receipt["commands"]] == [
        "scripts.evaluate_loss_suites",
        "scripts.evaluate_model",
    ]
    loss_command = receipt["commands"][0]
    assert loss_command[loss_command.index("--campaign-dir") + 1] == str(store.root)
    assert loss_command[loss_command.index("--experiment-id") + 1] == "control"
    assert "--ship-gates" in receipt["commands"][1]
    assert "--partial-scoreboard" in receipt["commands"][1]
    assert "--no-unconstrained-fallback" in receipt["commands"][1]


@pytest.mark.parametrize(
    "fault",
    case_values(__file__, "test_remeasurement_binding_fails_closed"),
)
def test_remeasurement_binding_fails_closed(contract, monkeypatch, fault):
    store, experiment, manifest, plan, args = contract
    if fault == "unlocked_edit":
        plan["promotion_allowed"] = True
        args.diagnostic_bundle_plan.write_text(json.dumps(plan))
    elif fault == "wrong_manifest":
        manifest = manifest.model_copy(update={"source_dirty": False})
    elif fault == "wrong_experiment":
        experiment = experiment.model_copy(
            update={
                "hypothesis": "An altered scientific proposition without an input lock"
            }
        )
    elif fault == "missing_sidecar":
        Path(plan["inputs"]["arms"]["control"]["checkpoint"]).with_suffix(
            ".context.tokenizer.json"
        ).unlink()
    elif fault == "summary_changed":
        Path(plan["arms"]["control"]["source_summary"]).write_text("{}")
    elif fault == "wrong_source":
        monkeypatch.setattr(
            "scripts.autoresearch_command_cursor.resolved_continuation_grant",
            lambda *args: SimpleNamespace(execution_identity="successor"),
        )
    else:
        args.reuse_train_run = Path("ordinary-clean-replay")
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.prepare_bundle_remeasurement(args, store, experiment, manifest)


def test_two_arm_screen_reaches_actual_matrix_consumer(prepared):
    store, plan = prepared
    campaign = CampaignSpec(
        campaign_id=store.campaign_id,
        objective="Real matrix consumer contract",
        primary_metric="smoke.eval_nll",
    )
    arms = copy.deepcopy(plan["arms"])
    for name, arm in arms.items():
        arm["experiment"]["hypothesis"] = (
            f"The {name} checkpoint produces a complete bounded observation."
        )
        arm["experiment"]["citations"] = ["fixture-input"]
        arm["experiment"]["knobs"]["lr"] = 0.0003 if name == "control" else 0.0006
    matrix = supervised_matrix(campaign, arms, "fixture-input")
    path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=path.stem)
    exp = matrix.hypotheses[0].experiment
    assert autoresearch._require_hypothesis_matrix(store, campaign, exp) == matrix
    # The shared schema continues to reject undersized search matrices.
    with pytest.raises(ValueError, match="at least five"):
        type(matrix).model_validate(
            {**matrix.model_dump(mode="json"), "matrix_role": "search"}
        )


def test_inference_commands_cannot_execute_supplied_extra_stage(contract):
    store, _, _, plan, _ = contract
    compiled = plan["arms"]["control"]["compiled"] + [["sh", "-c", "not-authorized"]]
    with pytest.raises(ValueError, match="one ordinary"):
        bundle.inference_commands(compiled, Path("last.pt"), plan["inputs"], store.root)


@pytest.fixture
def bound_loss(tmp_path):
    from slm_training.autoresearch.storage import CampaignStore

    store = CampaignStore("attempt-test", tmp_path)
    run = store.root / "runs/candidate"
    run.mkdir(parents=True)
    path = run / "loss_suites.json"
    path.write_text(json.dumps({"attempt_id": "producer", "per_record": [{"nll": 1.0}]}))
    command = ["python", "-m", "scripts.evaluate_loss_suites", "--out", str(path)]
    receipt = {"arm_id": "candidate", "commands": [command]}
    store.append_event("experiment_attempt_started", experiment_id="candidate",
                       detail={"attempt_id": "producer"})
    inputs = store.write_artifact("command_cursor_inputs", {
        "commands": [command], "experiment": {"experiment_id": "candidate"}})
    store.append_event("command_cursor_started", experiment_id="candidate",
                       detail={"attempt": 1, "input_digest": inputs.stem})
    payload = {"input_digest": inputs.stem, "attempt": 1,
               "outcome": {"stage_telemetry": [{
                   "command": command, "exit_code": 0, "duration_seconds": 1.0,
                   "parsed_output": {"out": str(path), "report_sha256": bundle.file_sha(path),
                                     "attempt_id": "producer"}}]}}
    return store, path, receipt, payload


def _commit_loss_and_retry(store, payload):
    artifact = store.write_artifact("command_cursors", payload)
    store.append_event("command_cursor_committed", experiment_id="candidate",
                       artifact_sha256=artifact.stem,
                       detail={"input_digest": payload["input_digest"]})
    store.append_event("experiment_attempt_returned", experiment_id="candidate",
                       detail={"attempt_id": "producer", "exit_code": 124})
    store.append_event("experiment_attempt_started", experiment_id="candidate",
                       detail={"attempt_id": "publication-retry"})
    return artifact


def test_finalizer_preserves_producer_and_binds_retry_for_strict_ingestion(bound_loss, monkeypatch):
    from scripts.autotrain_search import _loss_report
    from slm_training.harnesses.experiments.autonomous_learning import measurement_fixture_evidence

    store, path, receipt, payload = bound_loss
    artifact = _commit_loss_and_retry(store, payload)
    plan = {"inputs": {"eval_version": "fixture-v1"}}
    monkeypatch.setattr(bundle, "load_contract", lambda *a: plan)
    monkeypatch.setattr(measurement_fixture_evidence, "checked_arm", lambda *a: {
        "run_dir": path.parent, "records": {"a": 1.0},
        "loss": {"categories": {"broad": {"aggregate": {"mean_nll": 1.0}}},
                 "selection": {"selected_record_ids": ["a"], "selected_root_ids": ["root"]},
                 "per_record": [{"case_id": "a", "nll": 1.0}], "estimator_id": "fixture",
                 "definition": {"fixture": True}}})
    original = path.read_bytes()
    bundle.finalize_bundle_remeasurement(store, {**receipt, "contract_path": "locked"})
    report = json.loads((path.parent / "eval_nll_records.json").read_text())
    assert report["attempt_id"] == "publication-retry"
    assert report["producer_evidence"]["attempt_id"] == "producer"
    assert report["producer_evidence"]["cursor_artifact_sha256"] == artifact.stem
    assert path.read_bytes() == original
    consumed = _loss_report(store, "candidate", attempt_id="publication-retry", attempt_count=2)
    assert consumed["per_record"] == [{"case_id": "a", "nll": 1.0}]


@pytest.mark.parametrize("fault", case_values(__file__, "test_loss_producer_binding_rejects_unproven_reuse"))
def test_loss_producer_binding_rejects_unproven_reuse(bound_loss, fault):
    store, path, receipt, payload = bound_loss
    stage = payload["outcome"]["stage_telemetry"][0]
    if fault == "missing_digest":
        del stage["parsed_output"]["report_sha256"]
    stage.update({
        "wrong_command": {"command": ["unrelated-evaluation"]},
        "failed_stage": {"exit_code": 124},
        "boolean_exit": {"exit_code": False},
        "interrupted": {"interrupted": True},
        "nonfinite_charge": {"duration_seconds": float("inf")},
    }.get(fault, {}))
    artifact = _commit_loss_and_retry(store, payload)
    loss = json.loads(path.read_text())
    if fault == "missing_attempt":
        del loss["attempt_id"]
    elif fault == "wrong_attempt":
        loss["attempt_id"] = "publication-retry"
    elif fault == "changed_rows":
        loss["per_record"][0]["nll"] = 0.0
    elif fault == "changed_cursor":
        artifact.write_text("{}")
    path.write_text(json.dumps(loss))
    with pytest.raises(ValueError, match="loss"):
        bundle.verified_loss_producer(store, receipt, path)
