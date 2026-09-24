"""Public worker invocation must exercise the delivered supervisor, not a stub."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.autotrain_supervisor_operations import run_operation
from scripts.merge_verification_evidence import digest, environment_identity
from scripts.run_autotrain_supervisor import _source_identity
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore


def test_real_inspection_uses_supervisor_worker_entrypoint(tmp_path, monkeypatch):
    cwd = Path(__file__).resolve().parents[2]
    root = tmp_path / "campaigns"
    store = CampaignStore("runtime", root / "loops/entrypoint")
    request = {
        "operation": "inspect",
        "cwd": str(cwd),
        "root": str(root),
        "loop_id": "entrypoint",
        "source_digest": _source_identity(cwd),
        "environment_digest": digest(environment_identity()),
    }
    events = []
    with ActivityRuntime(store) as runtime:
        observed = []
        real_run = runtime.run
        commands = []

        def capture(*args, **kwargs):
            commands.append(list(args[1]))
            result = real_run(*args, **kwargs)
            observed.append(result.stderr[-4000:])
            return result

        monkeypatch.setattr(runtime, "run", capture)
        payload = run_operation(runtime, request, sequence=1, log_event=events.append)
        assert payload is not None, (events, observed)
        assert isinstance(payload["report"], dict)
        assert payload["campaign_id"] is None
        assert "promotion_pending" in payload
        command = commands[0]
        request_index = command.index("--operation-request")
        output_index = command.index("--operation-output")
        assert command[request_index + 1] != command[output_index + 1]
        assert Path(command[request_index + 1]).is_file()
        assert Path(command[output_index + 1]).is_file()
        state, = runtime.snapshot().values()
        assert state.status == "succeeded"
        assert state.outputs["result.json"]
        # A reconciled call must reuse the same effect without another attempt.
        assert run_operation(runtime, request, sequence=1, log_event=events.append) == payload
        assert runtime.snapshot()[state.spec.activity_id].attempts == 1


def test_worker_cli_rejects_unpaired_paths_before_execution(tmp_path):
    cwd = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_autotrain_supervisor",
         "--operation-request", str(tmp_path / "absent.json")],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(cwd / "src") + os.pathsep + str(cwd)},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "operation requires both request and output" in result.stderr
    assert "unrecognized arguments" not in result.stderr
    assert not (tmp_path / "absent.json").exists()


def test_delivery_classifies_frozen_source_without_git_and_rejects_drift(tmp_path, monkeypatch):
    from scripts import run_autotrain_continuous as continuous
    from slm_training.harness_core.execution_release import prepare_release

    source = tmp_path / "source"
    source.mkdir()
    (source / "source.py").write_text("value = 1\n")
    execution = tmp_path / "execution"
    root = tmp_path / "outputs"
    prepare_release(source, tmp_path / "release", execution, root)
    (root / "campaign").mkdir()
    monkeypatch.setattr(continuous, "_git", lambda *a, **k: pytest.fail("Git-free release queried Git"))
    arguments = dict(cwd=execution, root=root, loop_id="loop", campaign_id="campaign",
                     primary_metric="smoke.eval_nll", control_id="control", candidate_id="candidate")
    result = continuous._phase_a_delivery(**arguments)
    assert result["has_tracked_delta"] is False
    assert result["measurement_complete"] is False
    assert (root / "campaign/measured-results-continuous.md").is_file()
    (execution / "source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="execution_source_drift"):
        continuous._phase_a_delivery(**arguments)


def test_delivery_config_drift_rejects_before_dispatch(tmp_path):
    from scripts.autotrain_supervision import register_delivery_waits

    config = tmp_path / "delivery.json"
    config.write_text("{}")
    with pytest.raises(ValueError, match="delivery configuration changed"):
        register_delivery_waits(None, {
            "delivery_config": str(config), "delivery_config_digest": "0" * 64,
        }, [], lambda _: None)


def test_repair_release_preserves_only_verified_predecessor_provenance(tmp_path, monkeypatch):
    import json
    from slm_training.harness_core import execution_release as owner

    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n")
    ancestry = {"integration_commit": "a" * 40, "upstream_commit": "b" * 40,
                "code_dirty": False}
    monkeypatch.setattr(owner, "_checkout_provenance", lambda _: dict(ancestry))
    predecessor = tmp_path / "predecessor"
    old = owner.prepare_release(source, tmp_path / "old-release", predecessor,
                                tmp_path / "old-output")
    monkeypatch.setattr(owner, "_checkout_provenance", lambda _: pytest.fail("candidate ancestry queried"))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "code.py").write_text("value = 2\n")
    (candidate / owner.MARKER).write_text(json.dumps({"git_provenance": {"forged": True}}))
    execution = tmp_path / "successor"
    new = owner.prepare_release(candidate, tmp_path / "new-release", execution,
                                tmp_path / "new-output",
                                verified_predecessor=(predecessor, old["source_digest"]))
    assert owner.runtime_git_provenance(execution) == {**ancestry, "code_dirty": True}
    assert new["source_digest"] != old["source_digest"]
    assert owner.runtime_git_provenance(predecessor) == ancestry
    with pytest.raises(ValueError, match="repair_predecessor_source_mismatch"):
        owner.prepare_release(candidate, tmp_path / "wrong-release", tmp_path / "wrong-execution",
                              tmp_path / "wrong-output", verified_predecessor=(predecessor, "0" * 64))
    (predecessor / "code.py").write_text("drift\n")
    with pytest.raises(ValueError, match="execution_source_drift"):
        owner.prepare_release(candidate, tmp_path / "drift-release", tmp_path / "drift-execution",
                              tmp_path / "drift-output", verified_predecessor=(predecessor, old["source_digest"]))


def test_predecessor_is_rechecked_after_successor_copy(tmp_path, monkeypatch):
    from slm_training.harness_core import execution_release as owner

    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n")
    ancestry = {"integration_commit": "a" * 40, "upstream_commit": "b" * 40,
                "code_dirty": True}
    monkeypatch.setattr(owner, "_checkout_provenance", lambda _: ancestry)
    execution = tmp_path / "execution"
    old = owner.prepare_release(source, tmp_path / "release", execution, tmp_path / "output")
    copy = owner.shutil.copyfile

    def copy_and_drift(*args):
        result = copy(*args)
        (execution / "code.py").write_text("changed during copy\n")
        return result

    monkeypatch.setattr(owner.shutil, "copyfile", copy_and_drift)
    with pytest.raises(ValueError, match="execution_source_drift"):
        owner.prepare_release(source, tmp_path / "next-release", tmp_path / "next-execution",
                              tmp_path / "next-output",
                              verified_predecessor=(execution, old["source_digest"]))
    assert not (tmp_path / "next-execution" / owner.MARKER).exists()


def test_repair_delivery_registration_uses_accepted_successor_across_restart(tmp_path, monkeypatch):
    from scripts.autotrain_supervision import register_delivery_waits
    from slm_training.autoresearch.heal import repair_delivery
    from slm_training.autoresearch.runtime import operations_control

    wait = {"kind": "verified_repair_source", "publication_id": "fixture",
            "artifact_sha256": "c" * 64,
            "required_capability": "authorized_github_connector_delivery"}
    common = {"loop_id": "fixture", "source_digest": "a" * 64,
              "environment_digest": "b" * 64}
    resolved, consumed = [], []

    def resolve(store, dependency):
        resolved.append(dependency)
        return {"successor_source_digest": "d" * 64}

    def consume(runtime, activity_id, dependency, host):
        consumed.append(runtime.snapshot()[activity_id].spec)
        return {"activity_id": activity_id, "state": "waiting_capability"}

    monkeypatch.setattr(repair_delivery, "resolve_source_delivery", resolve)
    monkeypatch.setattr(operations_control, "consume_delivery", consume)
    with ActivityRuntime(CampaignStore("runtime", tmp_path)) as runtime:
        first = register_delivery_waits(runtime, common, [wait], lambda _: None)
        again = register_delivery_waits(runtime, {**common, "source_digest": "d" * 64,
            "environment_digest": "e" * 64}, [wait], lambda _: None)
        assert first == again and len(runtime.snapshot()) == 1
    assert resolved == [wait, wait] and consumed[0] == consumed[1]
    assert consumed[0].source_digest == "d" * 64
    assert consumed[0].environment_digest == "b" * 64


def test_driver_import_failure_does_not_disable_supervisor_entrypoint():
    script = '''
import importlib.abc
import sys
class BrokenDriver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "scripts.autotrain_cycle_execution":
            raise ImportError("broken driver fixture")
sys.meta_path.insert(0, BrokenDriver())
from scripts.run_autotrain_supervisor import _build_parser
assert _build_parser().parse_args(["--max-cycles", "1"]).max_cycles == 1
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_operation_repairs_rotate_across_restart(tmp_path):
    from slm_training.autoresearch.heal.operation_recovery import (
        pending_operation_repairs, record_operation_failure,
    )
    from slm_training.harness_core.activity_contract import ActivitySpec, ActivityOutcome, WakeCondition
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome

    store = CampaignStore("runtime", tmp_path)
    with ActivityRuntime(store) as runtime:
        for name in ("first", "second"):
            request = {"operation": "inspect", "loop_id": "loop"}
            runtime.register(ActivitySpec(activity_id=name, family="loop", kind="control",
                source_digest="a" * 64, environment_digest="b" * 64,
                input_digest=digest(name), output_namespace=name))
            lease = runtime.claim_next(activity_id=name, capabilities={"local_process"})
            result = BoundedProcessResult((), ProcessOutcome.COMPLETED, 1, "", "broken", .01)
            record_operation_failure(runtime, lease, request, result, outcome=ActivityOutcome.UNKNOWN_FAILURE)
            runtime.finish(lease, outcome=ActivityOutcome.UNKNOWN_FAILURE, spent_seconds=.01,
                wake=WakeCondition(predicate="repair", source="verification", identity_digest=digest(name)))
        runtime.store.append_event("operation_repair_serviced", experiment_id="first")
    with ActivityRuntime(store) as runtime:
        jobs = pending_operation_repairs(runtime)
        assert [job["hard_pending"][0]["affected_activity_id"] for job in jobs] == ["second", "first"]


@pytest.mark.parametrize("operation", ["repair", "closeout"])
def test_repair_and_closeout_do_not_import_broken_driver(tmp_path, monkeypatch, operation):
    import json
    from contextlib import nullcontext
    from scripts import autotrain_supervisor_operations as owner

    monkeypatch.setattr(owner, "validate_operation_identity", lambda *args: None)
    monkeypatch.setattr(owner, "operation_publication_scope", lambda *args: nullcontext())
    monkeypatch.setattr(owner, "repair_operation", lambda *args, **kwargs: {"any_healed": True})
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({"operation": operation, "cwd": str(tmp_path),
                                  "root": str(tmp_path), "loop_id": "loop"}))
    def broken():
        raise ImportError("continuous driver broken")
    assert owner.operation_main(request, output, source_identity=lambda _: "",
        load_continuous=broken, handle_hard_pending=None, write_family_closures=lambda _: None) == 0
    assert json.loads(output.read_text())["operation"] == operation
