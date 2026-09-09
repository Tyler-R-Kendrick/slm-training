"""Production supervisor calls; workload/publication delegation explicitly mocked."""

import hashlib
import json
from contextlib import nullcontext

import pytest

from tests.casefiles import case_values

from scripts.autotrain_supervisor_operations import repair_operation, run_operation
from scripts.merge_verification_evidence import digest
from scripts import autotrain_controller_repair as controller
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import ActivitySpec
from slm_training.harness_core.execution_release import prepare_release
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
)
from tests.test_autoresearch.test_operation_recovery import diagnosed

SHA = "a" * 64


@pytest.mark.parametrize(
    ("status", "expected"),
    case_values(__file__, "test_real_operation_keeps_typed_wait_not_success"),
)
def test_real_operation_keeps_typed_wait_not_success(
    tmp_path, monkeypatch, status, expected
):
    calls = []

    def workload(runtime, lease, argv, **kwargs):
        calls.append(lease)
        directory = runtime.attempt_dir(lease)
        request = json.loads((directory / "request.json").read_text())
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "supervisor_operation/v1",
                    "operation": "repair",
                    "request_digest": digest(request),
                    "payload": {
                        "agent_repairs": [{"status": status, "reason": "fixture"}]
                    },
                }
            )
        )
        return BoundedProcessResult(
            tuple(argv), ProcessOutcome.COMPLETED, 0, "", "", 0.01
        )

    monkeypatch.setattr(ActivityRuntime, "run", workload)
    request = {
        "operation": "repair",
        "loop_id": "fixture",
        "cwd": str(tmp_path),
        "source_digest": SHA,
        "environment_digest": SHA,
    }
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "store")) as runtime:
        run_operation(runtime, request, sequence=1, log_event=lambda _: None)
        state = next(iter(runtime.snapshot().values()))
        assert state.status == expected
        assert state.spec.kind == "control"
        assert "controller_publication" in state.spec.capabilities
        assert (
            run_operation(runtime, request, sequence=2, log_event=lambda _: None)
            is None
        )
        assert len(calls) == 1 and len(runtime.snapshot()) == 1


def test_script_repair_calls_pinned_source_and_publication_seam(tmp_path, monkeypatch):
    context, config, _ = diagnosed.__wrapped__(tmp_path, monkeypatch)
    original, execution = tmp_path / "immutable", tmp_path / "execution"
    marker = prepare_release(context.source, original, execution, context.root)
    config_path = tmp_path / "repair-config.json"
    config_path.write_text(config.model_dump_json())
    calls = []

    class ExplicitMockDelegation:
        def __init__(self, store, source):
            self.store = store

        def publication(self, lease):
            return nullcontext()

    def dispatched(pending, supplied, **kwargs):
        assert supplied.source == original
        assert supplied.source_digest == manifest_digest(tree_manifest(original))
        assert callable(kwargs["publish_verified"])
        assert kwargs["fence_valid"](supplied.fence)
        calls.append(pending)
        return {"status": "verified", "release_handoff": {"fixture": True}}

    monkeypatch.setattr(controller, "DelegatedPublisher", ExplicitMockDelegation)
    monkeypatch.setattr(controller, "dispatch_hard_pending", dispatched)
    with ActivityRuntime(CampaignStore("runtime", context.root)) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="repair",
                family="fixture",
                kind="control",
                source_digest=marker["source_digest"],
                environment_digest=SHA,
                input_digest=SHA,
                output_namespace="attempt",
            )
        )
        lease = runtime.claim_next(capabilities={"local_process"})
        pending = {"kind": "repair_harness", "blocker_code": "harness_code_failure"}
        request = {
            "hard_pending": [pending, pending],
            "campaign_id": "campaign",
            "max_heal_attempts": 1,
            "playbooks_enabled": False,
            "lease": lease.model_dump(mode="json"),
            "source_digest": marker["source_digest"],
            "environment_digest": SHA,
            "parent_event": "parent",
            "repair_config": str(config_path),
            "repair_config_digest": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
        }
        result = repair_operation(
            request,
            cwd=execution,
            root=context.root,
            loop_id="loop",
            handle_hard_pending=lambda *args, **kwargs: {"any_healed": False},
        )
    assert len(calls) == 1 and result["any_healed"] is False
    assert result["agent_repairs"][0]["release_handoff"] == {"fixture": True}
