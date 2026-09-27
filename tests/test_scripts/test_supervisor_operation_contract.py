"""Activity and output contracts for supervised operation workers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_autotrain_supervisor as _mod
from tests.casefiles import case_values
from tests.test_scripts.controller_fixtures import bind_test_controller


@pytest.mark.parametrize("process_state,driver_code,expected", case_values(Path(__file__).with_name("test_run_autotrain_supervisor.py"), "test_operation_output_cannot_override_failed_execution"))
def test_operation_output_cannot_override_failed_execution(tmp_path, process_state, driver_code, expected):
    import json
    from scripts.autotrain_supervisor_operations import interpret_operation_result
    from scripts.merge_verification_evidence import digest
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome

    request = {"operation": "driver"}
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"schema_version": "supervisor_operation/v1",
        "request_digest": digest(request), "operation": "driver",
        "payload": {"returncode": driver_code, "campaign_id": "fixture"}}))
    result = BoundedProcessResult((), ProcessOutcome(process_state), 0, "", "", .01,
        interrupted=process_state == "interrupted", killed=process_state == "killed")
    outcome, payload = interpret_operation_result(result, output, request)
    assert outcome.value == expected
    assert (payload is not None) == (expected == "succeeded")


@pytest.mark.parametrize("missing_output", [False, True])
@pytest.mark.parametrize("process_state", ["completed", "interrupted", "killed"])
def test_supervisor_operation_uses_real_activity_contract(tmp_path, monkeypatch, missing_output, process_state):
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome
    from types import SimpleNamespace
    from scripts import merge_verification_evidence as evidence
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore

    request = {"cwd": str(tmp_path), "root": str(tmp_path / "data"),
               "loop_id": "fixture", "operation": "inspect",
               "source_digest": "a" * 64, "environment_digest": evidence.digest({"fixture_environment": True})}
    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": True})
    controller = bind_test_controller(monkeypatch, {"fixture_environment": True})
    request["controller_execution"] = controller
    monkeypatch.setattr(_mod, "_load_continuous", lambda: SimpleNamespace(
        self_heal_unblock_loop=lambda **_: {"hard_pending": []},
        _check_regime_parked=lambda **_: None,
        _latest_cycle=lambda *_: (0, None)))
    journal = CampaignStore("runtime", tmp_path / "data" / "loops" / "fixture")
    with ActivityRuntime(journal) as runtime:
        launches = []

        def run(lease, argv, **kwargs):
            launches.append(lease.attempt_id)
            if not missing_output:
                from scripts import autotrain_supervisor_operations as operations
                from slm_training.harness_core.checkpoint_publication import champion_publication_scope
                # In-process fixture retains real fencing; separate child test proves delegation.
                monkeypatch.setattr(operations, "operation_publication_scope", lambda req, root, loop:
                                    champion_publication_scope(runtime, lease, loop_dir=root / "loops" / loop))
                _mod._operation_main(Path(argv[-3]), Path(argv[-1]))
            return BoundedProcessResult(tuple(argv), ProcessOutcome(process_state), 0, "", "", .01)

        monkeypatch.setattr(runtime, "run", run)
        result = _mod._run_operation(runtime, request, sequence=1, log_event=lambda _: None)
        states = list(runtime.snapshot().values())
        failed = missing_output or process_state != "completed"
        assert len(states) == 1
        assert states[0].status == ("waiting_repair" if failed else "succeeded")
        assert (result is None) == failed
        from slm_training.autoresearch.heal.operation_recovery import pending_operation_repairs
        assert len(pending_operation_repairs(runtime)) == int(failed)
        repeated = _mod._run_operation(runtime, request,
            sequence=2 if missing_output else 1, log_event=lambda _: None)
        assert repeated == result
        assert len(launches) == 1  # No fresh budget or repeated committed effect.
        assert len(runtime.snapshot()) == 1


@pytest.mark.parametrize("stale_handoff", [False, True])
@pytest.mark.parametrize("driver_code", [0, 2])
def test_operation_driver_zero_without_handoff_is_failure(tmp_path, monkeypatch, stale_handoff, driver_code):
    import json
    from types import SimpleNamespace
    from scripts import merge_verification_evidence as evidence
    from contextlib import nullcontext
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.harness_core.activity_contract import ActivitySpec
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core import checkpoint_publication as champion_publication

    # This unit check isolates handoff validation; real delegation has its own
    # process tests and is not established by this mocked context manager.
    monkeypatch.setattr(champion_publication, "champion_publication_scope", lambda *a, **k: nullcontext())
    runtime_root = tmp_path / "data"
    with ActivityRuntime(
        CampaignStore("runtime", runtime_root / "loops" / "fixture")
    ) as runtime:
        runtime.register(ActivitySpec(activity_id="driver", family="fixture", kind="control",
            source_digest="a" * 64, environment_digest="b" * 64, input_digest="c" * 64,
            output_namespace="attempts/driver"))
        lease = runtime.claim_next(capabilities={"local_process"})

    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": True})
    controller = bind_test_controller(monkeypatch, {"fixture_environment": True})
    monkeypatch.setattr(_mod, "_load_continuous", lambda: SimpleNamespace(
        main=lambda _: driver_code, _latest_cycle=lambda *_: (1, "missing")))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"cwd": str(tmp_path), "root": str(runtime_root),
        "source_digest": "a" * 64, "operation": "driver", "loop_id": "fixture",
        "environment_digest": evidence.digest({"fixture_environment": True}),
        "driver_argv": [], "controller_execution": controller,
        "lease": lease.model_dump(mode="json")}))
    if stale_handoff:
        (runtime_root / "missing").mkdir(parents=True)
        (runtime_root / "missing" / "cycle_handoff.json").write_text("{}")
    expected = "same-campaign completion lacks one current retirement" if stale_handoff else "without its required handoff"
    if driver_code:
        expected = "driver operation returned unsuccessful status"
    with pytest.raises(ValueError, match=expected):
        _mod._operation_main(request, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()
