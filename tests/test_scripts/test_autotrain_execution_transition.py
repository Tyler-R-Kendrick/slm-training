"""Live issuer, real activity accounting, immutable workload transition fixtures.

Acceptance receipts represent trusted controller issuance; no experiment success
or external verification result is manufactured by these transition tests.
"""
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autotrain_execution_transition as owner
from scripts.autotrain_controller_execution import controller_authority_scope
from scripts.merge_verification_identity import digest, environment_identity
from slm_training.autoresearch.heal.operation_recovery import _successor_plan
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, ResourceGrant, WakeCondition, contract_digest
from slm_training.harness_core.execution_release import prepare_release, runtime_source_identity
from tests.test_autoresearch.test_harness import campaign


@pytest.fixture(scope="module")
def controller():
    root = Path(owner.__file__).resolve().parents[1]
    environment = environment_identity()
    environment.pop("execution_environment_sha256")
    return {"cwd": str(root), "source_digest": runtime_source_identity(root),
            "runtime_digest": digest(environment)}


@pytest.fixture
def chain(tmp_path, controller):
    roots, manifests = [], []
    for index in range(3):
        source = tmp_path / f"source-{index}"
        (source / "scripts").mkdir(parents=True)
        (source / "scripts/autotrain_cursor_reconcile.py").write_text(f"VERSION = {index}\n")
        (source / "science.json").write_text('{"seed": 43, "steps": 6}\n')
        root = tmp_path / f"execution-{index}"
        manifests.append(prepare_release(source, tmp_path / f"release-{index}", root, tmp_path / "outputs"))
        roots.append(root)
    store = CampaignStore("test-campaign", tmp_path / "campaigns")
    store.root.mkdir(parents=True)
    spec = campaign()
    (store.root / "campaign.json").write_text(spec.model_dump_json())
    environment = digest(environment_identity())
    original = {"operation": "driver", "root": str(store.root.parent), "loop_id": "loop",
                "cwd": str(roots[0]), "source_digest": manifests[0]["source_digest"],
                "environment_digest": environment, "seed": 43, "commands": [["train", "--steps", "6"]]}
    value = {"schema_version": "driver_cycle/v1", "campaign_id": store.campaign_id,
             "loop_id": "loop", "cwd": str(roots[0]), "execution_identity": "original-identity",
             "publication_source": {"source_digest": original["source_digest"]},
             "total_seconds": spec.budget.logical_seconds, "initial_spent_seconds": 0,
             "order": ["control"], "arms": {"control": {"commands": original["commands"], "manifest_digest": "a" * 64}}}
    artifact = store.write_artifact("driver_cycle_inputs", value)
    store.append_event("driver_cycle_locked", artifact_sha256=artifact.stem)
    journal = CampaignStore("runtime", store.root.parent / "loops/loop")
    with ActivityRuntime(journal) as runtime:
        runtime.register(ActivitySpec(activity_id="original", family="loop", kind="control",
            source_digest=original["source_digest"], environment_digest=environment,
            input_digest=contract_digest(original), output_namespace="original", capabilities=("local_process", "controller_publication"),
            grant=ResourceGrant(total_seconds=100, max_attempts=5, interrupt_seconds=30,
                                kill_grace_seconds=1, finalization_reserve_seconds=1)))
        plans, request, activity = [], original, "original"
        for index in (1, 2):
            lease = runtime.claim_next(activity_id=activity, capabilities={"local_process", "controller_publication"})
            runtime.finish(lease, outcome=ActivityOutcome.CODE_FAILURE, spent_seconds=2,
                wake=WakeCondition(predicate="verified repair", source="accepted_release", identity_digest=contract_digest(request)))
            journal.append_event("operation_repair_requested", experiment_id=activity, detail={"request": request})
            handoff = {"publication_id": str(index) * 64, "resume_activity_id": activity,
                       "resume_campaign_id": store.campaign_id, "source_digest": manifests[index]["source_digest"],
                       "successor_execution": str(roots[index]), "original_request_digest": contract_digest(request)}
            pointer = {"publication_id": handoff["publication_id"], "execution": str(roots[index]),
                       "runtime_source_digest": handoff["source_digest"]}
            with runtime._transaction():
                journal._replace_durable(journal.root / "source_release_pointer.json", json.dumps(pointer))
                journal.append_event("repair_release_intent", detail={"publication_id": handoff["publication_id"], "pointer": pointer})
                journal.append_event("repair_release_accepted", detail={"publication_id": handoff["publication_id"],
                    "pointer_digest": contract_digest(pointer), "handoff": handoff, "scientific_promotion": False})
                plan = _successor_plan(runtime, handoff, journal.verify_event_chain(), controller)
                runtime.cancel(activity, reason="verified fixture successor")
                runtime.register(ActivitySpec.model_validate(plan["spec"]))
                journal.append_event("operation_successor_planned", experiment_id=activity, detail=plan)
                journal.append_event("operation_successor_activated", experiment_id=activity,
                    detail={"handoff": handoff, "successor_request_digest": contract_digest(plan["request"])})
                journal.append_event("operation_controller_activation_verified", experiment_id=plan["request"]["successor_activity_id"],
                    detail={"plan_digest": contract_digest(plan), "controller_execution": controller})
            plans.append(plan)
            request, activity = plan["request"], plan["request"]["successor_activity_id"]
        yield SimpleNamespace(store=store, runtime=runtime, journal=journal, roots=roots,
                              value=value, plans=plans, controller=controller)


def test_two_hops_preserve_anchor_and_exact_charge(chain):
    f = chain
    before = f.journal.verify_event_chain()
    with controller_authority_scope(f.runtime, f.controller):
        request = owner.verify_locked_execution(f.store, f.roots[-1], f.value)
        assert owner.preregistration_source(f.store, f.roots[-1], {"source_path": str(f.roots[0])}) == f.roots[0]
    assert request["resource_grant"]["total_seconds"] == 96
    assert request["resource_grant"]["max_attempts"] == 3
    assert request["logical_continuation"]["prior_attempts"] == 2
    assert request["logical_continuation"]["prior_charged_seconds"] == 4
    assert request["logical_continuation"]["scientific_replicate_increment"] == 0
    assert request["seed"] == 43 and request["commands"] == f.value["arms"]["control"]["commands"]
    assert f.journal.verify_event_chain() == before


def test_live_issuer_required_and_copied_journal_rejected(chain, tmp_path):
    f = chain
    with pytest.raises(ValueError, match="live_controller"):
        owner.verify_locked_execution(f.store, f.roots[-1], f.value)
    copied = tmp_path / "copied"
    shutil.copytree(f.store.root.parent, copied)
    foreign = CampaignStore(f.store.campaign_id, copied)
    with controller_authority_scope(f.runtime, f.controller), pytest.raises(ValueError, match="live_controller|foreign_journal"):
        owner.verify_locked_execution(foreign, f.roots[-1], f.value)


@pytest.mark.parametrize("fault", ["missing", "fork", "cycle", "grant", "seed", "anchor"])
def test_ancestry_rejects_invalid_edges(chain, fault):
    f = chain
    plans = copy.deepcopy(f.plans)
    value = copy.deepcopy(f.value)
    if fault == "missing":
        plans = plans[1:]
    elif fault == "fork":
        other = copy.deepcopy(plans[0])
        other["request"]["successor_activity_id"] = "fork"
        plans.append(other)
    elif fault == "cycle":
        plans[0]["request"]["logical_continuation"]["predecessor_activity_id"] = plans[-1]["request"]["successor_activity_id"]
    elif fault == "grant":
        plans[-1]["request"]["resource_grant"]["total_seconds"] += 1
    elif fault == "seed":
        plans[-1]["request"]["seed"] = 99
    else:
        value["publication_source"]["source_digest"] = "f" * 64
    endpoint = next(p for p in plans if p["request"]["cwd"] == str(f.roots[-1]))
    with pytest.raises(ValueError):
        owner._verify_ancestry(f.journal, endpoint, plans, f.journal.verify_event_chain(),
                               f.runtime.snapshot(), f.store, value)


def test_cancelled_endpoint_cannot_authorize(chain):
    f = chain
    f.runtime.cancel(f.plans[-1]["request"]["successor_activity_id"], reason="revoked")
    with controller_authority_scope(f.runtime, f.controller), pytest.raises(ValueError, match="missing_or_ambiguous"):
        owner.verify_locked_execution(f.store, f.roots[-1], f.value)


def test_science_change_rejected_even_with_consistent_manifests(chain, tmp_path):
    f = chain
    source = tmp_path / "changed-science"
    shutil.copytree(tmp_path / "source-2", source)
    (source / "science.json").write_text('{"seed":99}')
    target = tmp_path / "altered-execution"
    manifest = prepare_release(source, tmp_path / "altered-release", target, tmp_path / "outputs")
    request = {**f.plans[-1]["request"], "cwd": str(target), "source_digest": manifest["source_digest"]}
    with pytest.raises(ValueError, match="scientific_implementation_changed"):
        owner._unchanged_science(f.plans[0]["request"], request, f.plans[-1]["handoff"])


def test_fresh_delegated_worker_uses_live_issuer(chain):
    f = chain
    request = f.plans[-1]["request"]
    lease = f.runtime.claim_next(activity_id=request["successor_activity_id"],
                                capabilities={"local_process", "controller_publication"})
    program = """
import json, sys
from pathlib import Path
from scripts.autotrain_cycle_lock import load_context
from scripts.autotrain_execution_transition import verify_locked_execution
from slm_training.autoresearch.storage import CampaignStore
from slm_training.autoresearch.runtime.activity_publication import DelegatedPublisher
from slm_training.harness_core.activity_contract import ActivityLease
from slm_training.harness_core.checkpoint_publication import champion_publication_scope
value = json.loads(sys.argv[1])
store = CampaignStore('test-campaign', Path(value['root']))
journal = CampaignStore('runtime', Path(value['root']) / 'loops/loop')
publisher = DelegatedPublisher(journal, value['source_digest'])
with champion_publication_scope(publisher, ActivityLease.model_validate(value['lease']), loop_dir=journal.root.parent):
    request = verify_locked_execution(store, value['cwd'], load_context(store))
    assert request['logical_continuation']['prior_attempts'] == 2
    print('authenticated-two-hop-worker')
"""
    result = f.runtime.run(lease, [sys.executable, "-B", "-c", program, json.dumps({
        **request, "lease": lease.model_dump(mode="json")})], cwd=Path(f.controller["cwd"]))
    assert result.returncode == 0, result.stderr
    assert not result.timed_out and "authenticated-two-hop-worker" in result.stdout


def test_environment_drift_is_not_relabelled(chain, monkeypatch):
    f = chain
    with controller_authority_scope(f.runtime, None):
        monkeypatch.setenv("OMP_NUM_THREADS", "983")
        with pytest.raises(ValueError, match="runtime_changed|environment_mismatch"):
            owner.verify_locked_execution(f.store, f.roots[-1], f.value)


def test_logical_cursor_replays_reservation_without_reset(chain):
    from scripts.autoresearch_command_cursor import CommandCursor
    from scripts.autoresearch_continuation_identity import resolved_continuation_grant
    from slm_training.autoresearch.schemas import ExperimentOutcome
    from tests.test_autoresearch.test_harness import experiment

    f = chain
    spec = experiment(experiment_id="control")
    commands = f.value["arms"]["control"]["commands"]
    total = f.value["total_seconds"]
    with CommandCursor(f.store, spec, commands, "a" * 64, f.value["execution_identity"],
                       total, cwd=f.roots[0]) as cursor:
        cursor.start(12)
        cursor.checkpoint(ExperimentOutcome(experiment_id="control", campaign_id=f.store.campaign_id,
            status="stopped", campaign_manifest_sha256="a" * 64,
            error="continuation_budget_pending"), 0)
        original_digest = cursor.digest
    actual = resolved_continuation_grant(f.roots[-1], total).execution_identity
    with controller_authority_scope(f.runtime, f.controller):
        with CommandCursor(f.store, spec, commands, "a" * 64, actual, total, cwd=f.roots[-1]) as cursor:
            assert cursor.digest == original_digest
            assert cursor.unresolved and cursor.spent == 12 and cursor.attempt == 1
            assert cursor.position == 0 and cursor.outcome.status == "stopped"
        changed = spec.model_copy(update={"knobs": spec.knobs.model_copy(update={"seed": 999})})
        with pytest.raises(ValueError, match="immutable inputs changed"):
            with CommandCursor(f.store, changed, commands, "a" * 64, actual, total, cwd=f.roots[-1]):
                pytest.fail("changed scientific parameters accepted")
