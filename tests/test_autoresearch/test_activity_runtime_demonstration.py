"""Finite preregistered operational fixture: 100 real subprocess attempts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
    contract_digest,
)
from slm_training.autoresearch.storage import CampaignStore


def _spec(index, source_digest=None):
    if source_digest is None:
        from scripts.verify_autonomy import _plan

        source_digest = contract_digest({
            "runtime_owners": _plan(Path(__file__).resolve().parents[2])["source_files"],
            "fixture_test": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
    return ActivitySpec(
        activity_id=f"fixture-{index}",
        family=f"family-{index % 7 if isinstance(index, int) else index}",
        kind="eval",
        source_digest=source_digest,
        environment_digest=hashlib.sha256(sys.version.encode()).hexdigest(),
        input_digest=hashlib.sha256(str(index).encode()).hexdigest(),
        output_namespace=f"runs/fixture-{index}",
        grant=ResourceGrant(
            interrupt_seconds=5,
            kill_grace_seconds=0.1,
            finalization_reserve_seconds=3,
            total_seconds=20,
        ),
    )


def _fault(index):
    if index % 20 == 18:
        return "lost_terminal"
    if index % 20 == 19:
        return "verified_before_crash"
    if index % 10 == 0:
        return "code_failure"
    if index % 10 == 1:
        return "missing_output"
    return "complete"


def _finish_workload(runtime, lease, result, outputs, fault, item, monkeypatch):
    if fault == "missing_output":
        with pytest.raises(ValueError):
            runtime.finish(
                lease,
                outcome=ActivityOutcome.SUCCEEDED,
                outputs=outputs,
                spent_seconds=result.duration_seconds,
            )
        runtime.finish(
            lease, outcome=ActivityOutcome.RETRY, spent_seconds=result.duration_seconds
        )
    elif fault == "code_failure":
        assert result.returncode == 1 and not outputs
        runtime.finish(
            lease,
            outcome=ActivityOutcome.CODE_FAILURE,
            spent_seconds=result.duration_seconds,
            wake=WakeCondition(
                predicate="original_reproducer_passes",
                source="independent_verifier",
                identity_digest=contract_digest(item),
            ),
        )
    elif fault == "verified_before_crash":
        store = runtime.store
        append = store.append_event

        def crash(kind, **kwargs):
            if (
                kind == "activity_transition"
                and kwargs["detail"]["operation"] == "finish"
            ):
                raise OSError("injected terminal publication failure")
            return append(kind, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(store, "append_event", crash)
            with pytest.raises(OSError):
                runtime.finish(
                    lease,
                    outcome=ActivityOutcome.SUCCEEDED,
                    outputs=outputs,
                    spent_seconds=result.duration_seconds,
                )
    else:
        runtime.finish(
            lease,
            outcome=ActivityOutcome.SUCCEEDED,
            outputs=outputs,
            spent_seconds=result.duration_seconds,
        )


def test_one_hundred_real_attempts_resume_and_scoped_wait(tmp_path, monkeypatch):
    evidence = Path(
        os.environ.get("SLM_RUNTIME_EVIDENCE_ROOT", str(tmp_path))
    ).resolve()
    store = CampaignStore("runtime-finite-100", evidence)
    source_digest = _spec(0).source_digest
    plan = {
        "schema": "runtime_fault_fixture/v3",
        "successor_reason": "v2_remembered_HEAD_and_healthy_exit_relabelled_code_failure",
        "source_digest": source_digest,
        "per_activity_resource_grant": _spec(0, source_digest).grant.model_dump(mode="json"),
        "attempt_count": 100,
        "faults": [_fault(i) for i in range(100)],
        "controller_contexts": 11,
        "expect_successes": 75,
        "claim_class": "operational_fixture_only",
    }
    store.write_artifact("runtime_fixture_plan", plan)
    receipts = []
    for batch in range(5):
        with ExitStack() as stack:
            runtime = stack.enter_context(ActivityRuntime(store))
            if batch == 0:
                remote = _spec("remote", source_digest).model_copy(
                    update={"capabilities": ("unavailable_remote_authority",)}
                )
                runtime.register(remote)
            for index in range(batch * 20, batch * 20 + 20):
                item = _spec(index, source_digest)
                runtime.register(item)
                lease = runtime.claim_next(
                    capabilities={"local_process"}, activity_id=item.activity_id
                )
                assert lease is not None
                path = runtime.attempt_dir(lease) / "result.json"
                fault = plan["faults"][index]
                code = (
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text('{\"fixture\":true}')"
                )
                if fault == "missing_output":
                    code = "pass"
                elif fault == "code_failure":
                    code = "raise RuntimeError('injected actual fixture code defect')"
                result = runtime.run(
                    lease, [sys.executable, "-c", code, str(path)], cwd=tmp_path
                )
                assert result.returncode == (1 if fault == "code_failure" else 0) and not result.timed_out
                outputs = (
                    {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
                    if path.exists()
                    else {}
                )
                receipts.append(
                    {
                        "index": index,
                        "attempt_id": lease.attempt_id,
                        "fault": fault,
                        "exit_code": result.returncode,
                        "seconds": result.duration_seconds,
                    }
                )
                if fault == "lost_terminal":
                    runtime.__exit__(None, None, None)
                    runtime = stack.enter_context(ActivityRuntime(store))
                    continue
                _finish_workload(
                    runtime, lease, result, outputs, fault, item, monkeypatch
                )
    with ActivityRuntime(store) as runtime:
        states = runtime.snapshot()
        assert len(receipts) == 100 and len({r["attempt_id"] for r in receipts}) == 100
        assert sum(s.status == "succeeded" for s in states.values()) == 75
        assert states["fixture-remote"].status == "waiting_capability"
        assert sum(s.attempts for s in states.values()) == 100
        expected_charge = sum(
            _spec(r["index"], source_digest).grant.attempt_seconds
            if r["fault"] == "lost_terminal"
            else r["seconds"]
            for r in receipts
        )
        charged = sum(s.charged_seconds for s in states.values())
        assert charged == pytest.approx(expected_charge)
        summary = {
            "schema": "runtime_fixture_evidence/v1",
            "plan_sha256": contract_digest(plan),
            "source_digest": source_digest,
            "python": sys.version,
            "platform": platform.platform(),
            "attempts": receipts,
            "actual_workload_attempts": 100,
            "controller_context_restarts": 10,
            "complete_operational_artifacts": 75,
            "charged_seconds": charged,
            "actual_seconds": sum(r["seconds"] for r in receipts),
            "states": {k: s.status for k, s in states.items()},
            "evidence_class": "real_subprocess_with_injected_controller_faults",
            "model_trials": 0,
            "scientific_promotions": 0,
            "real_agent_repairs": 0,
        }
        assert _spec(0).source_digest == source_digest, "fixture source changed during execution"
        store.write_artifact("runtime_fixture_evidence", summary)
        (evidence / "runtime-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


def test_finite_fixture_missing_isolation_is_durable_capability_wait(tmp_path, monkeypatch, capsys):
    from scripts.verify_autonomy import main
    from slm_training.autoresearch.heal.isolation import IsolationCapability

    monkeypatch.setattr("slm_training.autoresearch.heal.isolation.probe_isolation",
                        lambda: IsolationCapability(False, "bubblewrap", "fixture_namespace_denied", "/usr/bin/bwrap"))
    assert main(["--root", str(tmp_path), "--batch-size", "1"]) == 10
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "waiting_capability" and not result["complete"]
    events = CampaignStore("autonomy-adversary-finite-100", tmp_path).verify_event_chain()
    assert any(row["event_type"] == "autonomy_fixture_capability_wait" for row in events)
    assert not any(row["event_type"] == "autonomy_fixture_workload_observed" for row in events)


def test_successor_resource_contract_preserves_historical_timeout_plan():
    from scripts.autonomy_fixture_worker import fixture_spec
    from scripts.verify_autonomy import _plan

    root = Path(__file__).resolve().parents[2]
    old, current = _plan(root, version=3), _plan(root)
    assert old["schema_version"] == "autonomy_adversary_plan/v3"
    assert fixture_spec(26, old).grant.interrupt_seconds == 2
    assert old["logical_worker_seconds_grant"] == 4175
    assert current["faults"] == old["faults"]
    assert _plan(root, version=4)["logical_worker_seconds_grant"] == 5600
    assert current["logical_worker_seconds_grant"] == 5725
    assert current["repair_check_seconds"] == 10
    assert fixture_spec(26, current).grant.interrupt_seconds == 10
    assert fixture_spec(26, current).grant.finalization_reserve_seconds == 10
    assert fixture_spec(23, current).grant.interrupt_seconds == 1
    assert max(current["controller_interrupt_seconds"]) == 90
