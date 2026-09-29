"""Fresh controller/delegated-worker boundary fixture; no training or provider."""

import json
import hashlib
import os
from pathlib import Path
import sys

from scripts.merge_verification_identity import digest, environment_identity
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.execution_release import runtime_source_identity


def _worker(config, request):
    from scripts import autoresearch
    from scripts.autotrain_controller_execution import journal_authority
    from scripts.autotrain_supervisor_operations import operation_publication_scope, validate_operation_identity
    from slm_training.autoresearch.engine import _stage_environment
    from slm_training.harness_core.bounded_process import run_bounded_process
    from types import SimpleNamespace

    root = Path(config["outputs"])
    validate_operation_identity(request, runtime_source_identity, "fixture launch")
    journal = CampaignStore("runtime", root / "loops" / "fixture")
    _reject_launch_replays(config, request, root)
    _reject_host_configuration_swaps(request)
    _reject_controller_mode_drift(config, request)
    with operation_publication_scope(request, root, "fixture"):
        from scripts.autotrain_source_publication import _REPOSITORY

        assert request.get("delivery_config") is None
        assert _REPOSITORY.get() == "fixture/retained"
        with journal_authority(journal) as state:
            assert state.spec.activity_id == request["lease"]["activity_id"]
        foreign = CampaignStore("runtime", root / "forged")
        try:
            with journal_authority(foreign):
                raise AssertionError("foreign journal acquired delegated authority")
        except ValueError as exc:
            assert str(exc) == "execution_transition_foreign_journal"
        previous = os.environ.get("SLM_FIXTURE_DRIFT")
        os.environ["SLM_FIXTURE_DRIFT"] = "changed"
        try:
            validate_operation_identity(request, runtime_source_identity, "fixture drift")
        except ValueError as exc:
            assert "environment changed" in str(exc)
        else:
            raise AssertionError("environment drift was accepted")
        finally:
            if previous is None:
                os.environ.pop("SLM_FIXTURE_DRIFT")
            else:
                os.environ["SLM_FIXTURE_DRIFT"] = previous
        command = [sys.executable, "-m", "scripts.evaluate_model", str(root)]
        experiment = SimpleNamespace(knobs=SimpleNamespace(context_backend="scratch"))
        result = run_bounded_process(
            command, cwd=Path(request["cwd"]),
            env=_stage_environment(experiment, command, cwd=Path(request["cwd"])),
            interrupt_after_seconds=15, kill_grace_seconds=5,
        )
        assert result.returncode == 0, result.stderr
    validate_operation_identity(request, runtime_source_identity, "fixture return")
    (root / "origins.json").write_text(json.dumps({
        "controller_module": autoresearch.__file__, "controller_ROOT": str(autoresearch.ROOT),
        "scientific": json.loads(result.stdout), "activity": state.spec.activity_id,
    }))



def _reject_controller_mode_drift(config, request):
    from scripts.autotrain_supervisor_operations import validate_operation_identity

    path = Path(config["controller"]) / "scripts/autotrain_controller_execution.py"
    mode = path.stat().st_mode
    original = path.read_bytes()
    try:
        path.chmod(mode ^ 0o111)
        try:
            validate_operation_identity(request, runtime_source_identity, "fixture mode drift")
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("same-content executable-mode controller drift was accepted")
    finally:
        path.chmod(mode)
    assert path.read_bytes() == original


def _reject_launch_replays(config, request, root):
    from scripts.autotrain_controller_execution import operation_controller

    try:
        operation_controller({**request, "controller_launch": config["old_launch"]})
    except ValueError as exc:
        assert "live_request" in str(exc)
    else:
        raise AssertionError("prior lease launch receipt replayed")
    import shutil

    copied = root / "copied"
    shutil.copytree(root / "loops", copied / "loops")
    try:
        operation_controller({**request, "root": str(copied)})
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("copied journal issued launch authority")


def _parent(config):
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant, ActivityOutcome, WakeCondition

    controller = Path(config["controller"])
    environment = environment_identity()
    runtime_environment = {k: v for k, v in environment.items() if k != "execution_environment_sha256"}
    root = Path(config["outputs"])
    request = {"operation": "driver", "root": str(root), "loop_id": "fixture",
               "cwd": str(Path.cwd()), "source_digest": runtime_source_identity(Path.cwd()),
               "environment_digest": digest(environment),
               "controller_execution": {"cwd": str(controller),
                   "source_digest": runtime_source_identity(controller),
                   "runtime_digest": digest(runtime_environment)}}
    from scripts.autotrain_controller_execution import controller_authority_scope, record_launch

    binding = request.pop("controller_execution")
    host = _host_configuration(root, request)
    journal = CampaignStore("runtime", root / "loops" / "fixture")
    with ActivityRuntime(journal) as runtime, controller_authority_scope(runtime, binding, host_configuration=host):
        runtime.register(ActivitySpec(activity_id="origin-proof", family="fixture", kind="control",
            source_digest=request["source_digest"], environment_digest=request["environment_digest"],
            input_digest=digest(request), capabilities=("local_process", "controller_publication"),
            grant=ResourceGrant(), output_namespace="attempts/origin-proof"))
        lease = runtime.claim_next(capabilities={"local_process", "controller_publication"})
        old_launch = record_launch(runtime, request, lease)
        first = runtime.run(lease, [sys.executable, "-I", "-c", "pass"], cwd=Path(request["cwd"]))
        wake = WakeCondition(predicate="fixture child returned", source="fixture", identity_digest=digest(request))
        runtime.finish(lease, outcome=ActivityOutcome.DEPENDENCY, outputs={},
                       spent_seconds=first.duration_seconds, wake=wake)
        assert first.returncode == 0
        runtime.wake(lease.activity_id, evidence=wake)
        lease = runtime.claim_next(capabilities={"local_process", "controller_publication"})
        config["old_launch"] = old_launch
        Path(sys.argv[1]).write_text(json.dumps(config))
        path = root / "request.json"
        path.write_text(json.dumps({**request, "lease": lease.model_dump(mode="json"),
                                   "controller_launch": record_launch(runtime, request, lease)}))
        bootstrap = "import runpy,sys;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        result = runtime.run(lease, [sys.executable, "-I", "-c", bootstrap,
            str(controller), str(controller / "src"), str(Path(__file__).resolve()),
            sys.argv[1], str(path)], cwd=Path(request["cwd"]))
        assert result.returncode == 0, result.stderr
        evidence = (root / "origins.json").read_bytes()
        (runtime.attempt_dir(lease) / "origins.json").write_bytes(evidence)
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
                       outputs={"origins.json": hashlib.sha256(evidence).hexdigest()},
                       spent_seconds=result.duration_seconds)


def _reject_host_configuration_swaps(request):
    from scripts.autotrain_controller_execution import operation_controller

    forged = json.loads(json.dumps(request))
    forged["controller_launch"]["host_configuration"]["repair_config_digest"] = "0" * 64
    path = Path(request["controller_launch"]["host_configuration"]["repair_config"])
    original = path.read_bytes()
    for payload, changed_bytes in ((forged, None), (request, b"{}")):
        try:
            if changed_bytes is not None:
                path.write_bytes(changed_bytes)
            try:
                operation_controller(payload)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("unbound host configuration accepted")
        finally:
            path.write_bytes(original)


def _host_configuration(root, request):
    import time

    repair = root / "recovery-config.json"
    repair.write_text('{"schema_version":"recovery_config/v1","verifier_release":"' + 'a' * 64 + '"}')
    delivery = root / "delivery-config.json"
    delivery.write_text(json.dumps({"command": [sys.executable],
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "repository": "fixture/retained", "base_ref": "a" * 40,
        "source_digest": request["source_digest"], "expires_at": time.time() + 300,
        "authorized": True}))
    return {"repair_config": str(repair), "repair_config_digest": hashlib.sha256(repair.read_bytes()).hexdigest(),
            "delivery_config": str(delivery), "delivery_config_digest": hashlib.sha256(delivery.read_bytes()).hexdigest()}


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    if len(sys.argv) == 3:
        _worker(config, json.loads(Path(sys.argv[2]).read_text()))
    else:
        _parent(config)
