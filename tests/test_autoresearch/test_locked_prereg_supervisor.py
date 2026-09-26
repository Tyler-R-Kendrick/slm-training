"""Hermetic planning and closeout proof; no child train/eval process starts."""

import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autoresearch, autotrain_cycle_context as context
from scripts import autotrain_cycle_prepare as prepare
from scripts import autotrain_supervision, autotrain_locked_diagnostic as diagnostic
from scripts import autotrain_cycle_finalize as finalize
from scripts.autoresearch_command_cursor import ContinuationGrant
from slm_training.autoresearch.preflight.compiled_treatment import persist_pair
from slm_training.autoresearch.experiment_campaign import CampaignArmV1, CampaignEndpointV1
from slm_training.autoresearch.schemas import (
    CampaignBudget, ExperimentKnobs,
    ExperimentOutcome, HypothesisCandidate, HypothesisMatrix,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.autoresearch.schemas import (
    AutotrainActionReceiptV1, AutotrainActionV1, AutotrainCycleHandoffV1,
)
from slm_training.autoresearch.storage import autotrain_action_sha256
from slm_training.evals.measurement_identity import content_digest
from slm_training.harness_core.activity_contract import ResourceGrant
from slm_training.harness_core.github_delivery_tree import source_entries, tree_sha
from tests.test_autoresearch.test_harness import campaign, experiment, experiment_campaign, hypothesis_matrix


def _fixture(tmp_path, monkeypatch):
    campaign_id = "fixture-pr-head-diagnostic"
    ids = ["fixture-control", "fixture-candidate"]
    source = tmp_path / "execution"
    from slm_training.harness_core import execution_release
    original = tmp_path / "original"
    original.mkdir()
    (original / "fixture.py").write_text("original bytes\n")
    monkeypatch.setattr(execution_release, "_checkout_provenance", lambda _: {
        "integration_commit": "c" * 40, "upstream_commit": "c" * 40, "code_dirty": False,
    })
    release_manifest = execution_release.prepare_release(
        original, tmp_path / "release", source, tmp_path / "outputs",
    )
    root = tmp_path / "campaigns"
    commit = "c" * 40
    ancestor = tmp_path / "ancestor.pt"
    ancestor.write_bytes(b"frozen ancestor fixture")
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text('{"n":8}')
    eval_manifest = tmp_path / "eval.json"
    eval_manifest.write_text('{"n":6}')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    grant = ResourceGrant(total_seconds=300, max_attempts=4)
    base = campaign().model_dump(mode="json")
    base.update(campaign_id=campaign_id, loop_id=campaign_id, cycle_index=1,
                upstream_commit=commit, integration_commit=commit,
                primary_metric="smoke.eval_nll",
                budget=CampaignBudget(max_experiments=2, continuation_grant=grant).model_dump(mode="json"))
    from slm_training.autoresearch.schemas import CampaignSpec
    spec = CampaignSpec.model_validate(base)
    store = CampaignStore(campaign_id, root)
    store.initialize(spec)
    candidate_fixture = hypothesis_matrix().hypotheses[0]
    experiments = []
    for eid, lr in zip(ids, (0.0003, 0.0006), strict=True):
        knobs = ExperimentKnobs(
            train_version="fixture-train", eval_version="fixture-eval",
            steps=6, max_updates_this_invocation=3, batch_size=2, seed=7301,
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1,
            lr=lr, context_backend="scratch", initialize_from=str(ancestor),
            allow_unconstrained_fallback=False,
        )
        experiments.append(experiment(
            experiment_id=eid, campaign_id=campaign_id,
            hypothesis=f"Fixture hypothesis for {eid} with lr {lr}.", knobs=knobs,
            citations=candidate_fixture.experiment.citations,
        ))
    hashes = [content_digest(e.knobs.model_dump(mode="json")) for e in experiments]
    endpoint = CampaignEndpointV1(
        endpoint_id="primary", metric="smoke.eval_nll", role="primary",
        direction="decrease", minimum_effect=0.02,
    )
    template = experiment_campaign(
            campaign_id=campaign_id, experiment_id=ids[1],
            hypothesis="Matched LR comparison on one frozen diagnostic fixture.",
            decision="Report paired NLL without promotion.",
            endpoints=(endpoint,),
            arms=(CampaignArmV1(arm_id=ids[0], role="control", config_sha256=hashes[0]),
                  CampaignArmV1(arm_id=ids[1], role="candidate", config_sha256=hashes[1])),
            seeds=(7301,), budget=spec.budget, source_commit=commit,
            locked_eval_manifest_sha256=digest(eval_manifest),
        )
    manifests = {}
    for eid in ids:
        manifest = type(template).model_validate({**template.model_dump(mode="json"),
                                                   "experiment_id": eid})
        manifests[eid] = store.lock_experiment_campaign(manifest).manifest_sha256
    novelty = candidate_fixture.novelty
    matrix = HypothesisMatrix(
        matrix_id="fixture-matrix", campaign_id=campaign_id,
        evidence_snapshot_id="fixture-evidence", matrix_role="screen",
        hypotheses=tuple(HypothesisCandidate(
            experiment=e, evidence_uses=candidate_fixture.evidence_uses, novelty=novelty,
        ) for e in experiments),
        recommended_experiment_id=ids[1],
        selection_rationale="Locked two-arm fixture for command compilation",
    )
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    pair = {"arm_ids": ids, "campaign_manifest_id": ids[1],
            "arm_config_sha256s": hashes, "design_digest": "d" * 64,
            "replicate_id": "fixture-replicate", "treatment_ids": ["t0", "t1"]}
    design_path = persist_pair(store, pair)
    commands = {e.experiment_id: autoresearch.compile_commands(spec, e, output_root=root)
                for e in experiments}
    plan = {
        "schema": "locked_pair_preregistration/v1",
        "label": "Other locked pair", "promotion_allowed": False,
        "training_executed": False, "evaluation_executed": False,
        "campaign_id": campaign_id, "campaign_root": str(root),
        "source_path": str(source), "source_commit": commit,
        "source_digest": release_manifest["source_digest"],
        "source_tree": tree_sha(source_entries(source, release_manifest["files"])),
        "logical_updates": 6, "seed": 7301,
        "primary": {"metric": endpoint.metric, "direction": endpoint.direction,
                    "minimum_effect": endpoint.minimum_effect},
        "inputs": {"ancestor": str(ancestor), "ancestor_sha256": digest(ancestor),
                   "data_manifests": {str(train_manifest): digest(train_manifest),
                                      str(eval_manifest): digest(eval_manifest)},
                   "locked_eval_manifest_sha256": digest(eval_manifest),
                   "eval_cases": 6,
                   "selected_record_ids": [f"record-{i}" for i in range(6)],
                   "selected_root_ids": [f"root-{i}" for i in range(6)],
                   "input_sha256s": ["f" * 64 for _ in range(6)],
                   "selection_sha256": "e" * 64},
        "arms": {role: {"run_id": e.experiment_id,
                        "experiment": e.model_dump(mode="json"),
                        "commands": commands[e.experiment_id]}
                 for role, e in zip(("control", "candidate"), experiments, strict=True)},
        "manifest_sha256s": {role: manifests[e.experiment_id]
                             for role, e in zip(("control", "candidate"), experiments, strict=True)},
        "design_sha256": design_path.stem,
    }
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(plan))
    locked = store.write_artifact("science_lab_preregistration", plan)
    store.append_event("science_lab_preregistered", artifact_sha256=locked.stem)
    from slm_training.autoresearch import engine
    monkeypatch.setattr(context, "__file__", str(source / "scripts/autotrain_cycle_context.py"))
    monkeypatch.setattr(engine, "__file__", str(source / "src/slm_training/autoresearch/engine.py"))
    return plan, path, root, store, commands


def _selection(plan, path, root, store):
    return context.locked_preregistration_selection(
        path, Path(plan["source_path"]), root, plan["campaign_id"],
        hashlib.sha256(path.read_bytes()).hexdigest(),
        options={"train_version": "fixture-train", "steps": 6,
                 "primary_metric": "smoke.eval_nll",
                 "continuation_grant": store.load_campaign().budget.continuation_grant.model_dump_json()},
    )


def test_exact_compiled_commands_reach_ordinary_cursor(tmp_path, monkeypatch):
    plan, path, root, store, commands = _fixture(tmp_path, monkeypatch)
    selection = _selection(plan, path, root, store)
    monkeypatch.setattr(prepare, "resolved_continuation_grant", lambda *_: ContinuationGrant("fixture", 300))
    value = prepare.prepare_recorded_cycle(Path(plan["source_path"]), root, SimpleNamespace(), selection)
    ids = value["order"]
    assert [cmd[2] for eid in ids for cmd in commands[eid]] == [
        cmd[2] for eid in ids for cmd in value["arms"][eid]["commands"]]
    assert all(any("scripts.train_model" in arg for cmd in commands[eid] for arg in cmd)
               and any("scripts.evaluate_model" in arg for cmd in commands[eid] for arg in cmd)
               for eid in ids)
    launched = []
    monkeypatch.setattr(autoresearch, "resolved_continuation_grant",
                        lambda *_args, **_kw: ContinuationGrant("fixture", 300, 4))

    def cursor(spec, actual, **kwargs):
        launched.append((spec.experiment_id, actual, kwargs["campaign_manifest_sha256"]))
        return ExperimentOutcome(experiment_id=spec.experiment_id,
                                 campaign_id=spec.campaign_id,
                                 campaign_manifest_sha256=kwargs["campaign_manifest_sha256"],
                                 status="stopped", error="continuation_budget_pending")

    monkeypatch.setattr(autoresearch, "execute_with_continuation", cursor)
    for eid in ids:
        assert autoresearch.cmd_run(SimpleNamespace(
            campaign_id=plan["campaign_id"], root=root,
            experiment=Path(value["by_id"][eid]), campaign_manifest=None,
            execute=True, trackio=False, experiment_wall_seconds=30,
            reuse_train_run=None, reuse_train_manifest=(),
            diagnostic_bundle_plan=None,
        )) == 10
    assert launched == [(eid, commands[eid], value["arms"][eid]["manifest_digest"])
                        for eid in ids]
    assert finalize.stages(value) == ("diagnostic",)


def test_selector_rejects_digest_source_and_arm_changes(tmp_path, monkeypatch):
    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    assert set(_selection(plan, path, root, store).expected_commands) == {
        "fixture-control", "fixture-candidate"}
    with pytest.raises(ValueError, match="digest changed"):
        context.locked_preregistration_selection(
            path, Path(plan["source_path"]), root, plan["campaign_id"], "0" * 64,
            options={})
    changed = copy.deepcopy(plan)
    changed["source_digest"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source differs"):
        _selection(changed, path, root, store)
    for key, value in (("run_id", "changed-arm"), ("manifest", "0" * 64)):
        changed = copy.deepcopy(plan)
        if key == "run_id":
            changed["arms"]["control"]["run_id"] = value
        else:
            changed["manifest_sha256s"]["candidate"] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="preregistered event"):
            _selection(changed, path, root, store)
    for mutate in (
        lambda p: p["inputs"].update(selected_record_ids=[f"wrong-{i}" for i in range(6)]),
        lambda p: p["inputs"].update(eval_cases=5),
        lambda p: p["primary"].update(minimum_effect=0.0),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="preregistered event"):
            _selection(changed, path, root, store)


def test_locked_supervisor_dispatches_typed_repair_before_driver(tmp_path, monkeypatch):
    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    args = SimpleNamespace(loop_id=plan["campaign_id"], root=root, max_cycles=1,
                           stop_after_pass=None, locked_preregistration=path,
                           locked_prereg_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           train_version="fixture-train", steps=6,
                           primary_metric="smoke.eval_nll", continuation_grant="pinned",
                           hard_backoff_seconds=1, soft_backoff_seconds=0,
                           max_heal_attempts=1, no_playbooks=True, exit_on_park=False)
    runtime = SimpleNamespace(store=CampaignStore("runtime", root / "loops" / plan["campaign_id"]),
                              cancel_event=threading.Event())
    blocker = {"kind": "heal_postcondition_failed",
               "blocker_code": "npm_bridge_unavailable",
               "reason": "typed environment repair",
               "affected_activity_id": "fixture-activity"}
    from scripts import autotrain_verification, autotrain_pending, autotrain_repair_activation
    from slm_training.autoresearch.heal import operation_recovery
    monkeypatch.setattr(autotrain_verification, "drain_source_verification", lambda *_a, **_k: None)
    monkeypatch.setattr(autotrain_pending, "drain_driver_pending", lambda *_a, **_k: None)
    monkeypatch.setattr(operation_recovery, "pending_operation_repairs", lambda *_: [])
    monkeypatch.setattr(autotrain_repair_activation, "recover_release", lambda *_a, **_k: None)
    seen = []

    def operation(_runtime, request, **_kwargs):
        seen.append(request)
        if request["operation"] == "inspect":
            return {"campaign_id": plan["campaign_id"],
                    "report": {"hard_pending": [blocker]},
                    "promotion_pending": False, "parked": None}
        return {"any_healed": True}

    assert autotrain_supervision.supervise(args, runtime, {"root": str(root)},
        run_operation=operation, watchdog=lambda **_: None) == 2
    assert [row["operation"] for row in seen] == ["inspect", "repair"]
    assert seen[1]["hard_pending"] == [blocker]
    assert store.load_experiment_campaign("fixture-control").manifest_sha256 == plan["manifest_sha256s"]["control"]
    with pytest.raises(ValueError, match="scientific plan"):
        autotrain_supervision.handle_pending(args, runtime, {},
            {"report": {"hard_pending": [{"kind": "rebuild_data"}]},
             "campaign_id": plan["campaign_id"], "parked": None}, 1,
            lambda *_: None, operation)


def test_locked_report_uses_canonical_receipt_checked_actions(tmp_path, monkeypatch):
    plan, _path, root, store, _ = _fixture(tmp_path, monkeypatch)
    action = AutotrainActionV1(
        kind="repair_harness", owner="improve-openui-harnesses",
        harness_family="experiments", blocker_code="npm_bridge_unavailable",
        reason="AgentV SDK is unavailable", evidence_ids=("campaign:" + plan["campaign_id"],),
        unmet_predicate="AgentV dependency available",
    )
    handoff = AutotrainCycleHandoffV1(
        loop_id=plan["campaign_id"], campaign_id=plan["campaign_id"],
        cycle_index=1, upstream_commit=plan["source_commit"],
        integration_commit=plan["source_commit"], cycle_role="screening",
        cycle_intent="locked_pair_diagnostic", evidence_class="scratch",
        climb_state="harness_failure", ship_state="blocked",
        primary_metric="smoke.eval_nll", actions=(action,),
    )
    from slm_training.autoresearch.campaign_events import publish_cycle_handoff
    publish_cycle_handoff(store, handoff)
    from scripts import run_autotrain_continuous as continuous
    report = continuous.self_heal_unblock_loop(
        cwd=Path(plan["source_path"]), root=root, loop_id=plan["campaign_id"],
        campaign_id=plan["campaign_id"], locked_plan=True,
    )
    assert len(report["hard_pending"]) == 1
    pending = report["hard_pending"][0]
    assert pending["action_sha256"] == autotrain_action_sha256(action)
    assert pending["evidence_ids"] == list(action.evidence_ids)
    assert pending["blocker_class"] == "environment"
    assert diagnostic.require_locked_repair(pending)
    receipts = root / "loops" / plan["campaign_id"] / "action_receipts.jsonl"
    receipts.parent.mkdir(parents=True)
    receipts.write_text(AutotrainActionReceiptV1(
        loop_id=plan["campaign_id"], campaign_id=plan["campaign_id"],
        action_index=0, action_sha256=autotrain_action_sha256(action),
        action_kind="repair_harness", status="completed",
        evidence_uris=("campaign:" + plan["campaign_id"],),
    ).model_dump_json() + "\n")
    # A receipt without independently bound evidence cannot clear the action.
    assert continuous.self_heal_unblock_loop(
        cwd=Path(plan["source_path"]), root=root, loop_id=plan["campaign_id"],
        campaign_id=plan["campaign_id"], locked_plan=True,
    )["hard_pending"] == report["hard_pending"]


def test_locked_report_keeps_real_execution_handoff_pending(tmp_path, monkeypatch):
    plan, _path, root, store, _ = _fixture(tmp_path, monkeypatch)
    from slm_training.autoresearch.campaign_events import publish_cycle_handoff

    actions = tuple(AutotrainActionV1(
        kind=kind, owner="autotrain", reason=f"Pending {kind}",
        evidence_ids=(f"campaign:{plan['campaign_id']}",),
    ) for kind in ("retry_measurement", "next_experiment"))
    publish_cycle_handoff(store, AutotrainCycleHandoffV1(
        loop_id=plan["campaign_id"], campaign_id=plan["campaign_id"],
        cycle_index=1, upstream_commit=plan["source_commit"],
        integration_commit=plan["source_commit"], cycle_role="screening",
        cycle_intent="locked_pair_diagnostic", evidence_class="scratch",
        climb_state="inconclusive", ship_state="blocked",
        primary_metric="smoke.eval_nll", actions=actions,
    ))
    report = diagnostic.locked_prerequisite_report(root, plan["campaign_id"], plan["campaign_id"])
    assert [row["kind"] for row in report["hard_pending"]] == [
        "retry_measurement", "next_experiment"]
    assert [row["index"] for row in report["hard_pending"]] == [0, 1]
    assert [row["action_sha256"] for row in report["hard_pending"]] == [
        autotrain_action_sha256(action) for action in actions]
    assert report["blocker_cleared"] is False
    with pytest.raises(ValueError, match="scientific plan"):
        diagnostic.locked_repair_rows(report["hard_pending"])


def test_run_cycle_locked_copy_never_enters_git_integration(tmp_path, monkeypatch):
    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    from scripts import run_autotrain_continuous as continuous
    from scripts import autotrain_cycle_execution as execution
    monkeypatch.setattr(continuous, "_integrate_origin_main",
                        lambda **_: pytest.fail("locked copy entered Git integration"))
    monkeypatch.setattr(continuous, "_git",
                        lambda *_a, **_k: pytest.fail("locked copy entered Git status"))
    monkeypatch.setattr(prepare, "resolved_continuation_grant",
                        lambda *_: ContinuationGrant("fixture", 300))
    monkeypatch.setattr(execution, "resume_cycle", lambda *_: "fixture-result")
    assert diagnostic.locked_startup_commit(Path(plan["source_path"])) == plan["source_commit"]
    assert continuous.run_cycle(cwd=Path(plan["source_path"]), root=root,
                                loop_id=plan["campaign_id"], train_version="fixture-train",
                                steps=6, objective="fixture", primary_metric="smoke.eval_nll",
                                continuation_grant=store.load_campaign().budget.continuation_grant.model_dump_json(),
                                locked_preregistration=path,
                                locked_prereg_sha256=hashlib.sha256(path.read_bytes()).hexdigest()) == "fixture-result"
    def mutate_prereg(*_args):
        path.write_text(path.read_text() + " ")
        return "fixture-result"

    monkeypatch.setattr(execution, "resume_cycle", mutate_prereg)
    with pytest.raises(ValueError, match="digest changed"):
        continuous.run_cycle(cwd=Path(plan["source_path"]), root=root,
                             loop_id=plan["campaign_id"], train_version="fixture-train",
                             steps=6, objective="fixture", primary_metric="smoke.eval_nll",
                             continuation_grant=store.load_campaign().budget.continuation_grant.model_dump_json(),
                             locked_preregistration=path,
                             locked_prereg_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest())
