"""Real authority owners and typed consumer propagation; no remote/provider calls."""

import asyncio
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autotrain_source_publication as owner
from scripts.github_source_authority import record_initial_release
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, WakeCondition
from tests.test_harness_core.test_source_authority import initial as initial, _claim, _finish


@pytest.fixture
def subject(initial, tmp_path):
    host, _, connector, calls = initial
    root = tmp_path / "campaigns"
    store = CampaignStore("measured", root)
    source = {"source_digest": host.source_digest, "commit": host.base_ref, "repository": host.repository}
    artifact = store.write_artifact("driver_cycle_inputs", {"campaign_id": store.campaign_id,
                                                          "publication_source": source})
    store.append_event("driver_cycle_locked", artifact_sha256=artifact.stem)
    with ActivityRuntime(CampaignStore("runtime", root / "loops/lab")) as runtime:
        yield store, runtime, host, connector, calls


def publish(subject, branch="main"):
    _, runtime, host, connector, _ = subject
    host.base_branch = branch
    lease = _claim(host, runtime)
    reference = asyncio.run(record_initial_release(runtime, lease, host, connector))
    _finish(runtime, lease)
    return reference


@pytest.mark.parametrize("branch", ["main", "acceptance-only/fault"])
def test_evidence_source_requires_real_main_membership(subject, branch):
    store, _, host, _, _ = subject
    publish(subject, branch)
    if branch == "main":
        proof = owner.require_source_publication(store, "lab", repository=host.repository)
        assert proof["commit"] == host.base_ref and proof["source_digest"] == host.source_digest
    else:
        with pytest.raises(owner.SourcePublicationPrerequisite) as caught:
            owner.require_source_publication(store, "lab", measurement_complete=True)
        from scripts.autotrain_pending import validate_pending
        outcome, _ = validate_pending(caught.value.pending)
        assert outcome == ActivityOutcome.DEPENDENCY
        assert caught.value.pending["diagnostic_measurement_complete"] is True
        assert caught.value.pending["publication_complete"] is False


@pytest.mark.parametrize("mismatch", ["source", "commit", "repository"])
def test_controller_main_cannot_authorize_other_evidence(subject, mismatch):
    store, _, _, _, _ = subject
    publish(subject)
    source = dict(owner.evidence_source(store))
    repository = None
    if mismatch == "repository":
        repository = "another/repository"
    else:
        source["source_digest" if mismatch == "source" else "commit"] = "f" * (64 if mismatch == "source" else 40)
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab", source=source, repository=repository)


def test_missing_authority_dispatches_existing_initial_readback_then_wakes(subject, tmp_path, monkeypatch):
    from slm_training.autoresearch.runtime import operations_reconciliation

    store, runtime, host, connector, calls = subject
    with pytest.raises(owner.SourcePublicationPrerequisite) as caught:
        owner.require_source_publication(store, "lab", measurement_complete=True)
    wake = WakeCondition.model_validate(caught.value.pending["wake"])
    runtime.register(ActivitySpec(activity_id="blocked", family="lab", kind="control",
        source_digest=host.source_digest, environment_digest="a" * 64,
        input_digest="b" * 64, output_namespace="attempts/blocked"))
    lease = runtime.claim_next(activity_id="blocked", capabilities={"local_process"})
    runtime.finish(lease, outcome=ActivityOutcome.DEPENDENCY, outputs={}, spent_seconds=0, wake=wake)
    config = tmp_path / "delivery.json"
    config.write_text(host.model_dump_json())
    common = {"root": str(store.root.parent), "loop_id": "lab", "environment_digest": "a" * 64,
              "delivery_config": str(config), "delivery_config_digest": hashlib.sha256(config.read_bytes()).hexdigest()}
    monkeypatch.setattr(operations_reconciliation, "reader_connector", lambda *_: connector)
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert len(calls) == 4
    assert runtime.snapshot()["blocked"].status == "runnable"
    assert owner.require_source_publication(store, "lab")["ref"] == "refs/heads/main"
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert len(calls) == 4


def test_absent_or_nonmain_grant_never_dispatches(subject, tmp_path, monkeypatch):
    store, runtime, host, _, calls = subject
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab")
    common = {"root": str(store.root.parent), "loop_id": "lab", "environment_digest": "a" * 64}
    owner.drain_source_publications(runtime, common, lambda _: None)
    host.base_branch = "acceptance-only/fault"
    config = tmp_path / "host.json"
    config.write_text(host.model_dump_json())
    common.update(delivery_config=str(config), delivery_config_digest=hashlib.sha256(config.read_bytes()).hexdigest())
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert not calls and not runtime.snapshot()


def test_document_bytes_need_evidence_and_document_main_authority(subject):
    from scripts import run_autotrain_continuous as continuous

    store, _, host, _, _ = subject
    publish(subject)
    cwd = Path(host.verification_plan["initial_release"]["execution"])
    files = {"module.py": "# initial source\n"}
    assert not continuous._committed_document_bundle(cwd, files)
    assert continuous._committed_document_bundle(cwd, files, root=store.root.parent,
                                                  loop_id="lab", campaign_id=store.campaign_id)
    assert not continuous._committed_document_bundle(cwd, {"module.py": "changed"},
        root=store.root.parent, loop_id="lab", campaign_id=store.campaign_id)


def test_cursor_keeps_stage_and_completed_measurement_on_publication_wait(subject, monkeypatch):
    from scripts import autotrain_cycle_execution as execution

    store, _, _, _, _ = subject
    journal = SimpleNamespace(value={"loop_id": "lab", "role": "promotion", "cycle_intent": "promote"},
        state={"phase": "finalizing", "final_index": 0}, store=store)
    settled, stages = [], []
    journal.start = lambda *a: None
    journal.settle = lambda: settled.append(True)
    monkeypatch.setattr(execution, "reconcile_inflight", lambda *a: True)
    monkeypatch.setattr(execution, "_budget_pending", lambda *a: None)
    monkeypatch.setattr(execution, "operation_allowance", lambda *a: 100)
    monkeypatch.setattr(execution, "stages", lambda *a: ("promotion",))
    monkeypatch.setattr(execution, "_outputs", lambda *a: {"measurement": "retained"})
    monkeypatch.setattr(execution, "_run_arm", lambda *a: pytest.fail("completed arm reran"))
    def original_stage(name, *args):
        stages.append(name)
        owner.require_source_publication(store, "lab", measurement_complete=True)
    monkeypatch.setattr(execution, "execute_stage", original_stage)
    continuous = SimpleNamespace(_clear_active_stage=lambda *a: None)
    result = execution.execute_pending(journal, continuous, Path.cwd(), float("inf"))
    assert result["diagnostic_measurement_complete"] is True
    assert result["publication_complete"] is False and settled == [True]
    assert journal.state == {"phase": "finalizing", "final_index": 0}
    publish(subject)
    assert execution.execute_pending(journal, continuous, Path.cwd(), float("inf")) == store.campaign_id
    assert journal.state["phase"] == "completed" and journal.state["final_index"] == 1
    assert journal.state["outputs"] == {"measurement": "retained"}
    assert stages == ["promotion", "promotion"]


def test_worker_and_parent_preserve_promotion_publication_pending(subject, tmp_path, monkeypatch):
    from scripts import autotrain_operation_worker as worker, autotrain_supervisor_operations as operations
    from scripts import autotrain_promotion_chunks as chunks, autotrain_promotion_finalize as finalizer
    from slm_training.harness_core.bounded_process import ProcessOutcome
    from scripts.autotrain_pending import pending_since

    store, runtime, _, _, _ = subject
    pending = owner.SourcePublicationPrerequisite(store, "lab", owner.evidence_source(store), True)
    request = {"operation": "promotion_eval", "cwd": str(tmp_path), "root": str(store.root.parent),
               "loop_id": "lab", "campaign_id": store.campaign_id}
    inp, out = tmp_path / "request.json", tmp_path / "output.json"
    inp.write_text(json.dumps(request))
    monkeypatch.setattr(operations, "validate_operation_identity", lambda *a: None)
    monkeypatch.setattr(operations, "operation_publication_scope", lambda *a: nullcontext())
    monkeypatch.setattr(chunks, "resume_chunks", lambda *a, **k: {"arms": {"candidate": {"status": "complete"}}})
    monkeypatch.setattr(finalizer, "finalization_pending", lambda *a: True)
    def blocked(*a):
        raise pending
    monkeypatch.setattr(finalizer, "finalize_promotion", blocked)
    continuous = SimpleNamespace(_stage_command=None, _promotion_scoreboard_state=None)
    assert worker.operation_main(inp, out, source_identity=None, load_continuous=lambda: continuous,
                                 handle_hard_pending=None, write_family_closures=None) == 0
    process = SimpleNamespace(cancelled=False, progress_stalled=False, timed_out=False,
                              outcome=ProcessOutcome.COMPLETED, returncode=0)
    outcome, payload = operations.interpret_operation_result(process, out, request)
    assert outcome == ActivityOutcome.DEPENDENCY and payload["returncode"] == 10
    assert pending_since(runtime.store, set()) == pending.pending


def test_document_reconciliation_requires_experiment_source_not_controller(subject):
    from slm_training.autoresearch.runtime.operations_reconciliation import reconcile_document_delivery
    from slm_training.autoresearch.action_dependencies import campaign_prerequisites
    from slm_training.autoresearch.schemas import AutotrainActionV1, AutotrainCycleHandoffV1
    from tests.test_autoresearch.test_delivery_reconciliation import connector_fixture

    store, _, host, _, _ = subject
    handoff = AutotrainCycleHandoffV1(loop_id="lab", campaign_id=store.campaign_id,
        cycle_index=1, upstream_commit=host.base_ref, integration_commit=host.base_ref,
        cycle_role="screening", cycle_intent="diagnostic", evidence_class="fixture",
        climb_state="inconclusive", ship_state="blocked", primary_metric="eval_nll",
        actions=(AutotrainActionV1(kind="document", owner="documenting-experiment-results",
            reason="publish measured results", evidence_ids=("fixture",), dependency_scope="delivery"),))
    path = store.root / "cycle_handoff.json"
    path.write_text(handoff.model_dump_json())
    files = {"docs/design/measured-results.md": "# Measured\n", "docs/design/measured-results.json": "{}\n"}
    artifact = store.write_artifact("delivery_documents", {
        "schema": "autotrain_document_materialization/v1", "campaign_id": store.campaign_id,
        "handoff_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "files": files})
    store.append_event("documentation_waiting_delivery", artifact_sha256=artifact.stem)
    _, waits = campaign_prerequisites(store.root.parent, handoff)
    def reconcile():
        return asyncio.run(reconcile_document_delivery(store.root.parent, waits[0],
            {"pr_number": 7, "verified_head_sha": "c" * 40, "merge_sha": "d" * 40},
            repository=host.repository, required_checks=("required",), connector=connector_fixture(files)))
    with pytest.raises(owner.SourcePublicationPrerequisite):
        reconcile()
    assert len(campaign_prerequisites(store.root.parent, handoff)[1]) == 1
    publish(subject)
    result = reconcile()
    proof = json.loads(Path(result["verification_artifact"]).read_text())
    assert proof["source_authority"]["commit"] == host.base_ref
    assert proof["source_authority"]["source_digest"] == host.source_digest
    assert campaign_prerequisites(store.root.parent, handoff) == ((), [])


@pytest.mark.parametrize("repository", [None, "another/repository"])
def test_receipt_cannot_select_expected_repository(subject, repository):
    store, _, _, _, _ = subject
    publish(subject)
    source = dict(owner.evidence_source(store), repository=repository)
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab", source=source)


def test_wrong_configured_repository_cannot_dispatch_or_wake(subject, tmp_path, monkeypatch):
    store, runtime, host, _, calls = subject
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab")
    publish(subject)
    calls.clear()
    host.repository = "another/repository"
    config = tmp_path / "wrong-repository.json"
    config.write_text(host.model_dump_json())
    common = {"root": str(store.root.parent), "loop_id": "lab", "environment_digest": "a" * 64,
              "delivery_config": str(config), "delivery_config_digest": hashlib.sha256(config.read_bytes()).hexdigest()}
    # Retained proof must also match the evidence-pinned repository; wrong config cannot replace it.
    source = dict(owner.evidence_source(store), repository="third/repository")
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab", source=source)
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert calls == []


@pytest.mark.parametrize("expired", [False, True])
def test_interrupted_initial_readback_never_commits_authority(subject, monkeypatch, expired):
    from scripts import github_source_authority
    from slm_training.autoresearch.runtime import operations_reconciliation
    from slm_training.autoresearch.runtime.activity_publication import StaleLease

    _, runtime, host, connector, _ = subject
    monkeypatch.setattr(operations_reconciliation, "reader_connector", lambda *_: connector)
    async def interrupted(runtime, lease, *args):
        if expired:
            monkeypatch.setattr(runtime, "clock", lambda: lease.expires_at + 1)
            raise StaleLease("expired readback")
        raise KeyboardInterrupt
    monkeypatch.setattr(github_source_authority, "record_initial_release", interrupted)
    with pytest.raises(StaleLease if expired else KeyboardInterrupt):
        owner._initial_readback(runtime, {"loop_id": "lab", "environment_digest": "a" * 64}, {}, host)
    assert all(state.status == "running" for state in runtime.snapshot().values())
    assert not any(e["event_type"] == "source_authority_recorded" for e in runtime.store.verify_event_chain())


def test_unresolvable_old_request_cannot_starve_later_source(subject, tmp_path, monkeypatch):
    from slm_training.autoresearch.runtime import operations_reconciliation

    store, runtime, host, connector, calls = subject
    legacy = CampaignStore("legacy", store.root.parent)
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(legacy, "lab")
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(store, "lab")
    config = tmp_path / "fair-host.json"
    config.write_text(host.model_dump_json())
    common = {"root": str(store.root.parent), "loop_id": "lab", "environment_digest": "a" * 64,
              "delivery_config": str(config), "delivery_config_digest": hashlib.sha256(config.read_bytes()).hexdigest()}
    monkeypatch.setattr(operations_reconciliation, "reader_connector", lambda *_: connector)
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert not calls  # Exactly one request serviced; legacy remains unresolved.
    owner.drain_source_publications(runtime, common, lambda _: None)
    assert len(calls) == 4
    assert owner.require_source_publication(store, "lab")["commit"] == host.base_ref
    events = runtime.store.verify_event_chain()
    serviced = [e["detail"]["request_digest"] for e in events if e["event_type"] == "source_publication_serviced"]
    assert len(serviced) == 2 and serviced[0] != serviced[1]
    with pytest.raises(owner.SourcePublicationPrerequisite):
        owner.require_source_publication(legacy, "lab")


def lock_campaign_main_source(initial, root, loop_id, campaign_id):
    host, _, connector, _ = initial
    store = CampaignStore(campaign_id, root)
    lock = store.write_artifact("driver_cycle_inputs", {"campaign_id": campaign_id,
        "publication_source": {"source_digest": host.source_digest, "commit": host.base_ref,
                               "repository": host.repository}})
    store.append_event("driver_cycle_locked", artifact_sha256=lock.stem)
    with ActivityRuntime(CampaignStore("runtime", root / "loops" / loop_id)) as runtime:
        lease = _claim(host, runtime)
        if lease is not None:
            asyncio.run(record_initial_release(runtime, lease, host, connector))
            _finish(runtime, lease)
