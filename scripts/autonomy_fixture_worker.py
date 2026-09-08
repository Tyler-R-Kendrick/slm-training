"""Disposable controller/workload for the locked finite autonomy acceptance plan.

Fault injection is confined to its own campaign. This is not a model evaluator
or production scheduler; every operation calls the actual activity owners.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from slm_training.autoresearch.runtime.activity_process import prepare_outputs, verify_outputs
from slm_training.autoresearch.runtime.activity_runtime import (
    ActivityRuntime,
    StaleLease,
)
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivityLease,
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
    contract_digest,
)
from slm_training.autoresearch.storage import CampaignStore


def fixture_spec(index: int, plan: dict) -> ActivitySpec:
    fault = plan["faults"][index] if index < len(plan["faults"]) else "complete"
    interrupt = plan.get("repair_workload_interrupt_seconds", 40) if fault == "repair_fixture" else 1 if fault == "hang" else plan.get("ordinary_workload_interrupt_seconds", 2)
    return ActivitySpec(
        activity_id=f"attempt-{index:03d}",
        family=f"fixture-family-{index % 7}",
        kind="repair" if fault == "repair_fixture" else "control",
        source_digest=contract_digest(plan["source_files"]),
        environment_digest=plan["environment_digest"],
        input_digest=contract_digest({"plan": contract_digest(plan), "index": index}),
        output_namespace=f"runs/attempt-{index:03d}",
        grant=ResourceGrant(
            interrupt_seconds=interrupt,
            kill_grace_seconds=0.1,
            total_seconds=max(plan.get("ordinary_total_seconds", 8), interrupt + 6),
            finalization_reserve_seconds=5 if fault == "repair_fixture" else plan.get("ordinary_finalization_reserve_seconds", 5),
            max_attempts=2 if fault == "code_crash" and "repair_fixture" in plan["faults"] else 1,
        ),
    )


def _workload(fault: str) -> str:
    if fault == "code_crash":
        return "raise RuntimeError('injected ordinary code defect')"
    if fault == "missing_output":
        return "pass"
    if fault == "hang":
        return (
            "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); "
            "time.sleep(60)"
        )
    if fault == "malformed_output":
        return "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{')"
    return (
        "import json,sys; from pathlib import Path; i=int(sys.argv[2]); "
        "Path(sys.argv[1]).write_text(json.dumps({'index':i,'value':i*2,'fixture':True}))"
    )


def _outputs(runtime, lease):
    path = runtime.attempt_dir(lease) / "result.json"
    files = (path, path.with_name("repair.json"))
    return path, {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in files if file.exists()}


def _inject_controller_fault(runtime, lease, result, *, fault):
    _, outputs = _outputs(runtime, lease)
    if fault in {"storage_before_verify", "storage_after_verify"}:
        _storage_fault(runtime, lease, outputs, result, fault)
    if fault == "controller_lost":
        os._exit(73)
    if fault == "verified_before_crash":
        prepare_outputs(
            runtime,
            runtime.snapshot()[lease.activity_id],
            lease,
            outputs,
            result.duration_seconds,
        )
        os._exit(74)
    if fault != "stale_fence":
        return False
    runtime.cancel(lease.activity_id, reason="injected owner revocation")
    try:
        runtime.finish(
            lease,
            outcome=ActivityOutcome.SUCCEEDED,
            outputs=outputs,
            spent_seconds=result.duration_seconds,
        )
    except StaleLease:
        return True
    raise AssertionError("stale owner published")


def _storage_fault(runtime, lease, outputs, result, fault):
    """Inject ENOSPC at the real event owner boundary, not physical disk filling."""
    import errno

    original = runtime.store.append_event
    target = "activity_outputs_verified" if fault == "storage_before_verify" else "activity_transition"

    def fail(event_type, **kwargs):
        if event_type == target:
            raise OSError(errno.ENOSPC, "locked fixture storage interruption")
        return original(event_type, **kwargs)

    runtime.store.append_event = fail
    try:
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
                       outputs=outputs, spent_seconds=result.duration_seconds)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
    else:
        raise AssertionError("storage fault did not interrupt publication")
    finally:
        runtime.store.append_event = original
    original("autonomy_fixture_storage_interrupted", experiment_id=lease.activity_id,
             detail={"fault": fault, "boundary": target, "errno": errno.ENOSPC})
    os._exit(75 if fault == "storage_before_verify" else 76)


def _outcome(runtime, lease, result, *, fault):
    path, outputs = _outputs(runtime, lease)
    if fault in ("missing_output", "malformed_output"):
        try:
            runtime.finish(
                lease,
                outcome=ActivityOutcome.SUCCEEDED,
                outputs=outputs,
                spent_seconds=result.duration_seconds,
            )
        except ValueError:
            return ActivityOutcome.UNKNOWN_FAILURE
        raise AssertionError("missing/malformed artifact became success")
    if fault == "hang":
        if not result.timed_out:
            raise AssertionError("watchdog did not interrupt real hang")
        return ActivityOutcome.RETRY
    if fault == "code_crash":
        if result.returncode == 0:
            raise AssertionError("fault did not execute")
        return ActivityOutcome.CODE_FAILURE
    if result.returncode != 0 or result.timed_out:
        raise AssertionError(f"healthy workload failed: {result}")
    index = int(lease.activity_id.removeprefix("attempt-"))
    if json.loads(path.read_text()) != {
        "index": index,
        "value": index * 2,
        "fixture": True,
    }:
        raise AssertionError("original fixture predicate failed")
    return ActivityOutcome.SUCCEEDED


def run_fixture_attempt(store: CampaignStore, plan: dict, index: int) -> int:
    if not 0 <= index < len(plan["faults"]):
        raise ValueError("attempt not in locked fault plan")
    fault = plan["faults"][index]
    with ActivityRuntime(store) as runtime:
        spec = fixture_spec(index - 8 if fault == "repair_replay" else index, plan)
        if fault == "repair_replay":
            _wake_repaired(runtime, index)
        else:
            runtime.register(spec)
        lease = runtime.claim_next(capabilities={"local_process"})
        if lease is None or lease.activity_id != spec.activity_id:
            raise AssertionError("healthy work starved or unrelated activity selected")
        path = runtime.attempt_dir(lease) / "result.json"
        argv, cwd = _command(store, plan, index, path)
        result = runtime.run(lease, argv, cwd=cwd)
        if fault == "repair_replay" and result.returncode == 0:
            path.write_text(result.stdout)
        store.append_event(
            "autonomy_fixture_workload_observed",
            experiment_id=spec.activity_id,
            idempotency_key=f"fixture-observation:{index}",
            detail={
                "index": index,
                "activity_id": spec.activity_id,
                "attempt_id": lease.attempt_id,
                "fault": fault,
                "outcome": result.outcome.value,
                "returncode": result.returncode,
                "seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "attempt_reservation": spec.grant.attempt_seconds,
                "evidence_class": "actual_subprocess",
            },
        )
        _, outputs = _outputs(runtime, lease)
        if _inject_controller_fault(runtime, lease, result, fault=fault):
            return 0
        outcome = _outcome(runtime, lease, result, fault=fault)
        runtime.finish(
            lease,
            outcome=outcome,
            outputs=outputs if outcome == ActivityOutcome.SUCCEEDED else {},
            spent_seconds=result.duration_seconds,
            wake=WakeCondition(
                predicate="original_fixture_predicate_restored",
                source="independent_verifier",
                identity_digest=spec.input_digest,
            )
            if outcome
            in (ActivityOutcome.CODE_FAILURE, ActivityOutcome.UNKNOWN_FAILURE)
            else None,
        )
    return 0


def _command(store, plan, index, output):
    fault = plan["faults"][index]
    cwd = Path(__file__).resolve().parents[1]
    base_index = index - index % 20
    source = store.root / "fixture-sources" / str(base_index)
    if fault == "repair_fixture":
        return [sys.executable, "-m", "scripts.autonomy_fixture_worker", str(source), str(output), str(base_index), str(plan.get("repair_check_seconds", 2))], cwd
    if fault == "repair_replay":
        return ["/usr/bin/python3", "fixture.py", str(base_index)], source / "candidate"
    if fault == "code_crash" and "repair_fixture" in plan["faults"]:
        base = source / "base"
        base.mkdir(parents=True, exist_ok=True)
        program = ("import json,sys\nfactor = 3\n"
                   "if factor != 2:\n print('broken')\n raise SystemExit(1)\n"
                   "i=int(sys.argv[1])\nprint(json.dumps({'index':i,'value':i*factor,'fixture':True}))\n")
        target = base / "fixture.py"
        if target.exists() and target.read_text() != program:
            raise ValueError("frozen fixture source changed")
        target.write_text(program)
        return ["/usr/bin/python3", "fixture.py", str(index)], base
    return [sys.executable, "-c", _workload(fault), str(output), str(index)], cwd


def _wake_repaired(runtime, index):
    repair_id, original_id = f"attempt-{index - 1:03d}", f"attempt-{index - 8:03d}"
    states = runtime.snapshot()
    validate_repair_proof(runtime.store, states[repair_id], index - 8)
    runtime.wake(original_id, evidence=states[original_id].wake)


def validate_repair_proof(store, state, origin_index):
    from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest

    if state.status != "succeeded":
        raise ValueError("repair predicate evidence is not complete")
    event = next(row for row in store.verify_event_chain()
                 if row["event_type"] == "activity_outputs_verified" and row["experiment_id"] == state.spec.activity_id)
    lease = ActivityLease.model_validate(event["detail"]["lease"])
    verify_outputs(store.root, state, lease, state.outputs)
    proof = json.loads((store.root / state.spec.output_namespace / lease.attempt_id / "repair.json").read_text())
    source = store.root / "fixture-sources" / str(origin_index)
    if (proof["origin_index"] != origin_index or not proof["verification"]["accepted"]
            or proof["source_release_authorized"]
            or proof["verification"]["evidence_class"] != "independent_isolated_process"
            or any(manifest_digest(tree_manifest(source / role)) != proof[role + "_digest"] for role in ("base", "candidate"))):
        raise ValueError("fixture repair proof or current predicate identity mismatch")
    return proof


def _repair_fixture(source: Path, output: Path, index: int, check_seconds: float = 2):
    """Fake agent dispatch; actual patch bytes and independent isolated checks.

    Predicate evidence alone is not full source release authorization. No source
    pointer, authoritative repair receipt or promotion is issued by this fixture.
    """
    from slm_training.autoresearch.heal.dispatch import dispatch_repair
    from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, private_snapshot, tree_manifest
    from slm_training.autoresearch.heal.repair_acceptance import check_manifest_digest, proposal_patch_digest
    from slm_training.autoresearch.heal.repair_contracts import RepairBlocker, RepairDispatchResult, RepairGrant, RepairProposal, RepairRequest
    from slm_training.autoresearch.heal.repair_verifier import VerificationCheck, VerificationRequest, verify_candidate

    base, candidate = source / "base", source / "candidate"
    base_digest = manifest_digest(tree_manifest(base))
    expected = json.dumps({"index": index, "value": index * 2, "fixture": True}) + "\n"
    original = VerificationCheck("original", ("/usr/bin/python3", "fixture.py", str(index)), expected)
    checks = (VerificationCheck("regression", ("/usr/bin/python3", "tests/test_repair.py"), "regression\n"),)
    def sha(value):
        return hashlib.sha256(value).hexdigest()
    grant = RepairGrant(grant_id="locked-fake-fixture", provider="fake_fixture", executable=sys.executable,
                        executable_sha256=sha(Path(sys.executable).read_bytes()), expires_at=time.time() + 60,
                        max_attempts=1, total_seconds=40.0, interrupt_seconds=10)
    request = RepairRequest(
        activity_id=f"fixture-repair-{index}", attempt_id="fake-0", campaign_id=f"repair-{index}",
        fence="fixture-controller", parent_event="locked-fixture", grant=grant,
        blocker=RepairBlocker(code="harness_code_failure", blocker_class="code", owner="fixture",
            source_digest=base_digest, environment_digest=sha(sys.version.encode()), input_digest=sha(str(index).encode()),
            reproducer=original.argv, predicate="locked_json_result", needed_capability="fake_fixture_only", evidence=("broken stdout, exit 1",)),
        allowed_paths=("fixture.py", "tests/test_repair.py"), verification_manifest_digest=check_manifest_digest(original, checks),
        project_instructions="Fixture only. Preserve original predicate; no policy or release authority.",
        owner_contract="Patch only disposable fixture.py and add regression; no self acknowledgment.", existing_tests=("original",),
        failure_returncode=1, failure_stdout_sha256=sha(b"broken\n"), failure_stderr_sha256=sha(b""))

    class FakeAgent:
        def capability(self, _request):
            return None

        def execute(self, request, **kwargs):
            private_snapshot(base, candidate)
            (candidate / "fixture.py").write_text((base / "fixture.py").read_text().replace("factor = 3", "factor = 2"))
            (candidate / "tests").mkdir()
            (candidate / "tests/test_repair.py").write_text(
                f"import contextlib,io,json,runpy,sys\nsys.argv=['fixture.py','{index}']\nout=io.StringIO()\n"
                f"with contextlib.redirect_stdout(out): runpy.run_path('fixture.py',run_name='__main__')\n"
                f"assert json.loads(out.getvalue())['value']=={index * 2}\nprint('regression')\n")
            proposal = RepairProposal(request_digest=request.digest(), tree_digest=manifest_digest(tree_manifest(candidate)),
                patch_digest=proposal_patch_digest(base, candidate), root_cause="injected factor check crash", regression_test="tests/test_repair.py",
                reproduction_artifacts=(request.failure_stdout_sha256,), classification="implementation")
            return RepairDispatchResult(status="waiting_verification", request_digest=request.digest(), reason="fake_fixture_patch", proposal=proposal)

    result = dispatch_repair(request, executor=FakeAgent(), journal=CampaignStore(request.campaign_id, source / "journal"), fence_valid=lambda value: value == request.fence)
    spec = VerificationRequest(request.digest(), request.blocker.fingerprint(), base_digest, result.proposal.tree_digest,
        request.blocker.environment_digest, sha(Path(__file__).read_bytes()), grant.digest(), request.fence,
        request.allowed_paths, original, checks, 1, request.failure_stdout_sha256, request.failure_stderr_sha256, timeout_seconds=check_seconds)
    evidence = verify_candidate(spec, base=base, candidate=candidate)
    output.with_name("repair.json").write_text(json.dumps({"origin_index": index, "agent_class": "deterministic_fake_adapter",
        "base_digest": base_digest, "candidate_digest": result.proposal.tree_digest, "verification": asdict(evidence), "source_release_authorized": False}))
    if not evidence.accepted:
        raise AssertionError(f"original fixture predicate not restored: {evidence.reason}")
    output.write_text(json.dumps({"index": index + 7, "value": (index + 7) * 2, "fixture": True}))


if __name__ == "__main__":
    _repair_fixture(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
