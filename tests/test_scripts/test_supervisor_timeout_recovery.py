"""Bounded expiry differs from cancellation; host recovery retains finite accounting."""

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.autotrain_invocation import invocation_scope
from slm_training.autoresearch.heal.operation_timeout import verified_timeout_handoff
from slm_training.autoresearch.heal.operation_recovery import wake_verified_operation
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import (
    ActivitySpec, ActivityOutcome, ResourceGrant, WakeCondition, contract_digest,
)


def test_expiry_signal_requires_host_deadline_and_explicit_cancel_wins(tmp_path):
    store = CampaignStore("runtime", tmp_path)
    with ActivityRuntime(store) as runtime:
        args = SimpleNamespace(_supervisor_started=time.monotonic(), invocation_deadline=time.monotonic() - 1)
        with invocation_scope(runtime, args):
            signal.raise_signal(signal.SIGUSR1)
            assert runtime.invocation_expired and runtime.cancel_event.is_set()
            signal.raise_signal(signal.SIGINT)
            assert not runtime.invocation_expired
        event = store.verify_event_chain()[-1]
        assert event["event_type"] == "supervisor_stop_requested"
        assert event["detail"]["cause"] == "explicit_cancel"


def test_early_expiry_signal_is_not_authenticated_timeout(tmp_path):
    store = CampaignStore("runtime", tmp_path)
    with ActivityRuntime(store) as runtime:
        args = SimpleNamespace(_supervisor_started=time.monotonic(), invocation_deadline=time.monotonic() + 60)
        with invocation_scope(runtime, args):
            signal.raise_signal(signal.SIGUSR1)
            assert not runtime.invocation_expired


def test_controller_child_uses_remaining_outer_budget(tmp_path, monkeypatch):
    store = CampaignStore("runtime", tmp_path)
    with ActivityRuntime(store) as runtime:
        runtime.register(ActivitySpec(activity_id="driver", family="f", kind="control",
            source_digest="a" * 64, environment_digest="b" * 64, input_digest="c" * 64,
            output_namespace="attempts/driver", capabilities=("controller_publication",),
            grant=ResourceGrant(interrupt_seconds=2, kill_grace_seconds=0, total_seconds=10)))
        lease = runtime.claim_next(capabilities={"controller_publication"})
        monkeypatch.setenv("AUTOTRAIN_SUPERVISOR_WORK_DEADLINE", str(time.monotonic() + .3))
        result = runtime.run(lease, [sys.executable, "-I", "-c", "import time;time.sleep(20)"], cwd=tmp_path)
        assert result.timed_out and not result.cancelled and result.duration_seconds < 2


def _produce_cancelled(root, configuration=None, mutation=None):
    """Real exited controller and child; immutable checkpoint survives its cancellation."""
    root = Path(root)
    store = CampaignStore("runtime", root)
    request = dict(operation="driver", cwd=str(root), root=str(root.parent.parent), loop_id=root.name,
                   source_digest="a" * 64, environment_digest="b" * 64, driver_argv=[])
    configuration = configuration or {}
    request = configuration.get("request", request)
    if not configuration:
        root.mkdir(parents=True, exist_ok=True)
        prereg = root / "preregistration.json"
        prereg.write_text(json.dumps(dict(campaign_id="science", campaign_root=request["root"],
            source_path=str(root), source_digest=request["source_digest"])))
        request["driver_argv"] = ["--locked-preregistration", str(prereg),
            "--locked-prereg-sha256", hashlib.sha256(prereg.read_bytes()).hexdigest()]
    activity = configuration.get("activity_id", "driver")
    with ActivityRuntime(store) as runtime:
        spec = ActivitySpec(activity_id=activity, family="f", kind="control",
            source_digest=request["source_digest"], environment_digest=request["environment_digest"],
            input_digest=contract_digest(request), output_namespace="attempts/" + activity,
            capabilities=("local_process", "controller_publication"),
            grant=ResourceGrant.model_validate(configuration["grant"]) if "grant" in configuration else
            ResourceGrant(interrupt_seconds=2, kill_grace_seconds=0, total_seconds=20, max_attempts=4))
        state = runtime.register(spec)
        host = root / ("host-" + activity)
        host.mkdir()
        (host / "preflight.json").write_text(json.dumps(dict(retained_activity=state.model_dump(mode="json"))))
        lease = runtime.claim_next(capabilities={"local_process", "controller_publication"})
        result = runtime.run(lease, [sys.executable, "-I", "-c",
            "from pathlib import Path;Path('checkpoint').write_text('train-once')"], cwd=root)
        assert result.returncode == 0
        runtime.store.append_event("operation_repair_requested", experiment_id=activity, detail={"request": request})
        pending = dict(schema_version="driver_pending/v1", campaign_id="science", input_digest="d" * 64,
                       reason="locked_arm_or_evaluation_yielded", outcome="yielded", measurement_complete=False,
                       wake=dict(predicate="resume cursor", source="driver_cycle_checkpoint", identity_digest="e" * 64))
        pending = configuration.get("pending", pending)
        if not configuration:
            store.append_event("driver_cycle_registered", detail={
                "campaign_id": pending["campaign_id"], "input_digest": pending["input_digest"]})
        if mutation == "foreign_campaign":
            pending = {**pending, "campaign_id": "foreign"}
        artifact = store.write_artifact("driver_pending", pending)
        yielded = store.append_event("driver_yielded", artifact_sha256=artifact.stem,
                                     detail={"pending_digest": "f" * 64 if mutation == "wrong_detail" else artifact.stem})
        state = runtime.finish(lease, outcome=ActivityOutcome.CANCELLED, spent_seconds=3)
        cancelled = store.verify_event_chain()[-1]
        (host / "result.json").write_text(json.dumps(dict(returncode=124, elapsed_seconds=170.1,
            lock_timeout=False, preflight_passed=True, command=["timeout", "170s", str(host)])))
        proof = dict(schema_version="host_timeout_recovery/v1", cause="host_run_timeout",
            journal_root=str(store.root), cancel_event_id=cancelled["event_id"],
            state_digest=contract_digest(state), request_digest=contract_digest(request),
            yield_event_id=yielded["event_id"], legacy_cancel_event_ids=[])
        for key, name in (("host_result", "result.json"), ("host_preflight", "preflight.json")):
            p = host / name
            proof[key] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        (root / "proof.json").write_text(json.dumps(proof))


@pytest.fixture
def cancelled(tmp_path, monkeypatch, request):
    from slm_training.autoresearch.heal import operation_timeout

    monkeypatch.setattr(operation_timeout, "validate_timeout_cursor", lambda *_: None)
    tmp_path = tmp_path / "campaigns/loops/loop"
    repo = Path(__file__).resolve().parents[2]
    bootstrap = ("import sys;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/src'];"
                 "from tests.test_scripts.test_supervisor_timeout_recovery import _produce_cancelled;"
                 "_produce_cancelled(sys.argv[2],mutation=sys.argv[3])")
    subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(repo), str(tmp_path), getattr(request, "param", "")],
                   check=True, timeout=15)
    return tmp_path, json.loads((tmp_path / "proof.json").read_text())


def test_authorized_successor_preserves_charges_and_original_cancel(cancelled, monkeypatch):
    root, proof = cancelled
    store = CampaignStore("runtime", root)
    monkeypatch.chdir(root)
    from slm_training.harness_core import execution_release, controller_execution
    monkeypatch.setattr(execution_release, "runtime_source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(controller_execution, "validate_controller", lambda _: root)
    binding = dict(cwd=str(root), source_digest="a" * 64, runtime_digest="b" * 64)
    with ActivityRuntime(store) as runtime:
        handoff = verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)
        successor = wake_verified_operation(runtime, handoff, cwd=root, controller_execution=binding,
                                             timeout_authorization=(proof, contract_digest(proof)))
        old = runtime.snapshot()["driver"]
        new = runtime.snapshot()[successor["successor_activity_id"]]
        assert old.status == "cancelled" and old.charged_seconds == 3 and old.attempts == 1
        assert new.spec.grant.total_seconds == 17 and new.spec.grant.max_attempts == 3
        assert successor["logical_continuation"]["prior_charged_seconds"] == 3
        assert successor["logical_continuation"]["prior_attempts"] == 1
        # Replayed activation never registers a second grant or charges twice.
        assert wake_verified_operation(runtime, handoff, cwd=root, controller_execution=binding,
            timeout_authorization=(proof, contract_digest(proof))) == successor
        assert len(runtime.snapshot()) == 2
        before = store.verify_event_chain()
        states = {key: value.model_dump() for key, value in runtime.snapshot().items()}
        reference = proof["host_result"]
        aliases = (
            {**proof, "operator_note": "same authorized timeout, reissued proof"},
            {**proof, "host_result": {**reference, "path": str(Path(reference["path"]).parent) + "/./result.json"}},
        )
        for alternate in aliases:
            authorization = (alternate, contract_digest(alternate))
            assert authorization[1] != contract_digest(proof)
            with pytest.raises(ValueError, match="timeout.*(consum|successor|already)"):
                checked = verified_timeout_handoff(runtime, authorization, cwd=root)
                wake_verified_operation(runtime, checked, cwd=root, controller_execution=binding,
                                        timeout_authorization=authorization)
            assert store.verify_event_chain() == before
            assert {key: value.model_dump() for key, value in runtime.snapshot().items()} == states
        lease = runtime.claim_next(capabilities={"local_process", "controller_publication"}, activity_id=new.spec.activity_id)
        result = runtime.run(lease, [sys.executable, "-I", "-c",
            "from pathlib import Path;assert Path('checkpoint').read_text()=='train-once';Path('eval-row').write_text('1')"], cwd=root)
        assert result.returncode == 0
        runtime.finish(lease, outcome=ActivityOutcome.RETRY, spent_seconds=1,
                       wake=WakeCondition.model_validate(handoff["pending"]["wake"]))
        assert runtime.snapshot()[new.spec.activity_id].charged_seconds == 1
        assert (root / "checkpoint").read_text() == "train-once"


def test_changed_receipt_or_copied_journal_rejected(cancelled, tmp_path):
    root, original = cancelled
    with ActivityRuntime(CampaignStore("runtime", root)) as runtime:
        proof = {**original, "journal_root": str(tmp_path / "copy")}
        with pytest.raises(ValueError, match="binding mismatch"):
            verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)
        Path(original["host_result"]["path"]).write_text('{"returncode":0}')
        with pytest.raises(ValueError, match="evidence changed"):
            verified_timeout_handoff(runtime, (original, contract_digest(original)), cwd=root)


def test_explicit_user_cancel_provenance_cannot_be_reclassified(cancelled):
    root, proof = cancelled
    store = CampaignStore("runtime", root)
    cancelled_event = next(e for e in store.verify_event_chain() if e["event_id"] == proof["cancel_event_id"])
    store.append_event("supervisor_stop_requested", detail={
        "epoch": cancelled_event["detail"]["lease"]["epoch"], "cause": "explicit_cancel"})
    with ActivityRuntime(store) as runtime:
        with pytest.raises(ValueError, match="explicit user cancellation"):
            verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)


def test_expiry_after_valid_yield_preserves_pending_but_explicit_cancel_does_not(tmp_path):
    from scripts.autotrain_supervisor_operations import _operation_result_state
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome

    request = dict(operation="driver")
    pending = dict(schema_version="driver_pending/v1", outcome="yielded", reason="bounded chunk",
                   measurement_complete=False, wake=dict(predicate="resume", source="driver_cycle_checkpoint",
                                                        identity_digest="a" * 64))
    path = tmp_path / "result.json"
    path.write_text(json.dumps(dict(schema_version="supervisor_operation/v1", operation="driver",
        request_digest=contract_digest(request), payload=dict(returncode=10, pending=pending))))
    result = BoundedProcessResult(("fixture",), ProcessOutcome.COMPLETED, 0, "", "", 1, cancelled=True)
    for expired, expected in ((True, ActivityOutcome.YIELDED), (False, ActivityOutcome.CANCELLED)):
        outcome, payload, _, _ = _operation_result_state(SimpleNamespace(invocation_expired=expired),
            None, request, contract_digest(request), request, result, path)
        assert outcome == expected
        assert bool(payload) == expired


def test_fresh_supervisor_recovers_timeout_without_repeating_committed_training(tmp_path, monkeypatch):
    from tests.test_autoresearch.test_verified_supervisor_continuation import _setup, _seed_committed_train
    from scripts.autotrain_cycle_context import CycleJournal

    f = _setup(tmp_path, monkeypatch)
    _seed_committed_train(f)
    checkpoint = (f.science / "checkpoint").read_bytes()
    pending = CycleJournal(f.store, f.value).pending("locked_arm_or_evaluation_yielded")
    history = f.root / "loops" / f.args.loop_id
    configuration = dict(request=f.request, pending=pending,
                         grant=f.store.load_campaign().budget.continuation_grant.model_dump(mode="json"))
    config = tmp_path / "historical.json"
    config.write_text(json.dumps(configuration))
    repo = Path(__file__).resolve().parents[2]
    bootstrap = ("import sys,json;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/src'];"
                 "from tests.test_scripts.test_supervisor_timeout_recovery import _produce_cancelled;"
                 "_produce_cancelled(sys.argv[2],json.load(open(sys.argv[3])))")
    subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(repo), str(history), str(config)],
                   check=True, timeout=15, capture_output=True)
    proof_path = history / "proof.json"
    bootstrap = (
        "import runpy,sys\nsys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)]\n"
        "from pathlib import Path\n"
        "from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime\n"
        "run = ActivityRuntime.run\n"
        "def observed(self, lease, *args, **kwargs):\n"
        "    result = run(self, lease, *args, **kwargs)\n"
        "    (self.attempt_dir(lease) / 'fixture-worker-stderr.txt').write_text(result.stderr)\n"
        "    return result\n"
        "ActivityRuntime.run = observed\n"
        "runpy.run_module('scripts.run_autotrain_supervisor',run_name='__main__')"
    )
    argv = [sys.executable, "-I", "-B", "-c", bootstrap, str(f.controller), str(f.controller / "src"),
        "--controller-root", str(f.controller), "--controller-source-digest", f.binding["source_digest"],
        "--controller-runtime-digest", f.binding["runtime_digest"], "--root", str(f.root),
        "--loop-id", f.args.loop_id, "--locked-preregistration", str(f.args.locked_preregistration),
        "--continuation-grant", f.args.continuation_grant, "--train-version", "fixture-train",
        "--steps", "6", "--primary-metric", "smoke.eval_nll", "--stop-after-pass", "1",
        "--timeout-recovery", str(proof_path), "--timeout-recovery-sha256",
        hashlib.sha256(proof_path.read_bytes()).hexdigest()]
    result = subprocess.run(argv, cwd=f.plan["source_path"], capture_output=True, text=True, timeout=100)
    assert result.returncode in (0, 10), result.stdout + result.stderr
    journal = CampaignStore("runtime", history)
    from slm_training.autoresearch.runtime.activity_projection import ActivityProjection

    states = ActivityProjection(journal).read()
    assert states["driver"].status == "cancelled" and states["driver"].charged_seconds == 3
    successors = [s for s in states.values() if s.spec.activity_id.startswith("successor-")]
    assert len(successors) == 1 and successors[0].attempts == 1
    assert successors[0].spec.grant.total_seconds == 1197 and successors[0].spec.grant.max_attempts == 7
    assert (f.science / "trained").read_text() == "x"
    assert (f.science / "checkpoint").read_bytes() == checkpoint
    assert (f.science / "rows").read_text() == "preserved\nrepair-0\n"
    origin = json.loads((f.science / "origin-0.json").read_text())
    assert origin["cwd"] == f.plan["source_path"] and "--resume-run" in origin["argv"]


@pytest.mark.parametrize("cancelled", ["foreign_campaign", "wrong_detail"], indirect=True)
def test_unrelated_or_unbound_yield_cannot_authorize_recovery(cancelled):
    root, proof = cancelled
    with ActivityRuntime(CampaignStore("runtime", root)) as runtime:
        with pytest.raises(ValueError, match="original active driver|yielded artifact changed"):
            verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)


def test_later_explicit_cancel_epoch_also_blocks_recovery(cancelled):
    root, proof = cancelled
    store = CampaignStore("runtime", root)
    store.append_event("supervisor_stop_requested", detail={"epoch": "later", "cause": "explicit_cancel"})
    with ActivityRuntime(store) as runtime:
        with pytest.raises(ValueError, match="explicit user cancellation"):
            verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)


def test_legacy_ambiguous_cancel_requires_exact_operator_acknowledgement(cancelled):
    root, proof = cancelled
    store = CampaignStore("runtime", root)
    with ActivityRuntime(store) as runtime:
        runtime.register(ActivitySpec(activity_id="old-readiness", family="f", kind="verify",
            source_digest="a" * 64, environment_digest="b" * 64, input_digest="c" * 64,
            output_namespace="attempts/old-readiness"))
        runtime.cancel("old-readiness", reason="explicit supervisor stop")
        event = store.verify_event_chain()[-1]
        with pytest.raises(ValueError, match="exact host authorization"):
            verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)
        proof = {**proof, "legacy_cancel_event_ids": [event["event_id"]]}
        assert verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)["resume_activity_id"] == "driver"
        assert runtime.snapshot()["driver"].status == "cancelled"


def test_second_timeout_routes_original_cli_to_latest_authenticated_successor(cancelled, monkeypatch):
    from scripts.autotrain_controller_execution import controller_authority_scope
    from scripts.autotrain_timeout_recovery import retained_timeout_successor
    from slm_training.harness_core import execution_release, controller_execution

    root, proof = cancelled
    store = CampaignStore("runtime", root)
    original = next(e["detail"]["request"] for e in store.verify_event_chain()
                    if e["event_type"] == "operation_repair_requested")
    monkeypatch.chdir(root)
    monkeypatch.setattr(execution_release, "runtime_source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(controller_execution, "validate_controller", lambda _: root)
    binding = dict(cwd=str(root), source_digest="a" * 64, runtime_digest="b" * 64)
    with ActivityRuntime(store) as runtime:
        handoff = verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)
        first = wake_verified_operation(runtime, handoff, cwd=root, controller_execution=binding,
                                       timeout_authorization=(proof, contract_digest(proof)))
    configuration = dict(request=first, pending=handoff["pending"], grant=first["resource_grant"],
                         activity_id=first["successor_activity_id"])
    config = root / "second-timeout.json"
    config.write_text(json.dumps(configuration))
    repo = Path(__file__).resolve().parents[2]
    bootstrap = ("import sys,json;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/src'];"
                 "from tests.test_scripts.test_supervisor_timeout_recovery import _produce_cancelled;"
                 "_produce_cancelled(sys.argv[2],json.load(open(sys.argv[3])))")
    subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(repo), str(root), str(config)],
                   check=True, timeout=15, capture_output=True)
    proof = json.loads((root / "proof.json").read_text())
    with ActivityRuntime(store) as runtime, controller_authority_scope(runtime, None):
        handoff = verified_timeout_handoff(runtime, (proof, contract_digest(proof)), cwd=root)
        latest = wake_verified_operation(runtime, handoff, cwd=root, controller_execution=binding,
                                        timeout_authorization=(proof, contract_digest(proof)))
        before = store.verify_event_chain()
        assert retained_timeout_successor(runtime, original) == latest
        with pytest.raises(ValueError, match="configuration_change_requires_live_host_binding"):
            retained_timeout_successor(runtime, {**original, "repair_config": "/unbound-config.json"})
        assert store.verify_event_chain() == before
        assert latest["logical_continuation"]["prior_attempts"] == 2
        assert latest["logical_continuation"]["prior_charged_seconds"] == 6
        assert latest["resource_grant"]["max_attempts"] == 2
        assert latest["resource_grant"]["total_seconds"] == 14
        assert runtime.snapshot()[first["successor_activity_id"]].status == "cancelled"
        assert (root / "checkpoint").read_text() == "train-once"
        runtime.cancel(latest["successor_activity_id"], reason="explicit user cancellation")
        with pytest.raises(ValueError, match="operation_binding_mismatch"):
            retained_timeout_successor(runtime, original)


@pytest.mark.parametrize("key", ["repair_config", "repair_config_digest", "delivery_config", "delivery_config_digest"])
def test_cancelled_driver_config_change_cannot_allocate_activity_or_grant(cancelled, key):
    from scripts.autotrain_supervisor_operation_runtime import run_operation

    root, _ = cancelled
    store = CampaignStore("runtime", root)
    original = next(e["detail"]["request"] for e in store.verify_event_chain()
                    if e["event_type"] == "operation_repair_requested")
    with ActivityRuntime(store) as runtime:
        before = store.verify_event_chain()
        states = {name: state.model_dump(mode="json") for name, state in runtime.snapshot().items()}
        with pytest.raises(ValueError, match="cancelled driver requires authorized successor"):
            run_operation(runtime, {**original, key: "changed-host-configuration"},
                          sequence=99, log_event=lambda _: None)
        assert store.verify_event_chain() == before
        assert {name: state.model_dump(mode="json") for name, state in runtime.snapshot().items()} == states
