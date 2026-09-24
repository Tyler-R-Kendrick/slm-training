"""Real reducer/journal boundaries; the repair process result is simulated."""

import json
from pathlib import Path

import pytest

from scripts.autotrain_supervisor_operations import run_operation
from scripts.merge_verification_evidence import digest
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id
from slm_training.harness_core.activity_contract import ResourceGrant, WakeCondition
from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome


@pytest.mark.parametrize("configured", [True, False])
def test_verifier_dependency_is_not_a_failed_repair_attempt(tmp_path, monkeypatch, configured):
    identity = "c" * 64
    grant = ResourceGrant().model_dump(mode="json") if configured else None
    dependency = {
        "schema_version": "repair_verification_dependency/v1",
        "activity_id": source_verification_activity_id(identity, grant) if configured else None,
        "verification_identity": identity if configured else None,
        "grant": grant,
        "wake": {"predicate": "complete_current_source_verification" if configured else "source_verifier_configured",
                 "source": "source_verification_completed" if configured else "controller_policy",
                 "identity_digest": identity},
    }
    request = {"cwd": str(tmp_path), "root": str(tmp_path / "data"), "loop_id": "fixture",
               "operation": "repair", "source_digest": "a" * 64, "environment_digest": "b" * 64}
    journal = CampaignStore("runtime", tmp_path / "events")
    with ActivityRuntime(journal) as runtime:
        launches = []

        def run(lease, argv, **kwargs):
            launches.append(lease.attempt_id)
            assert Path(kwargs["cwd"]).resolve() == Path(
                run_operation.__code__.co_filename
            ).resolve().parents[1]
            controller = Path(kwargs["cwd"]).resolve()
            assert argv[3:5] == [str(controller), str(controller / "src")]
            execution = json.loads(Path(argv[-3]).read_text())
            Path(argv[-1]).write_text(json.dumps({"schema_version": "supervisor_operation/v1",
                "request_digest": digest(execution), "operation": "repair",
                "payload": {"agent_repairs": [{"status": "waiting_verification",
                                              "verification_dependency": dependency}]}}))
            return BoundedProcessResult(tuple(argv), ProcessOutcome.COMPLETED, 0, "", "", .01)

        monkeypatch.setattr(runtime, "run", run)
        run_operation(runtime, request, sequence=1, log_event=lambda _: None)
        state = next(iter(runtime.snapshot().values()))
        assert state.status == ("waiting_dependency" if configured else "waiting_capability")
        assert state.wake == WakeCondition.model_validate(dependency["wake"])
        events = journal.verify_event_chain()
        assert sum(e["event_type"] == "source_verification_requested" for e in events) == 1
        assert not any(e["event_type"] == "operation_failed" for e in events)
        from slm_training.autoresearch.heal.operation_recovery import pending_operation_repairs

        assert pending_operation_repairs(runtime) == []
        assert run_operation(runtime, request, sequence=2, log_event=lambda _: None) is None
        assert len(launches) == 1
