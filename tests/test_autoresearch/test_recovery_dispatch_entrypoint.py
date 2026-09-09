"""Supervisor-facing recovery parser and durable capability wait tests."""

import json
import time

from slm_training.autoresearch.heal.repair_contracts import RepairGrant
from slm_training.autoresearch.storage import CampaignStore

SHA = "a" * 64


def test_actual_supervisor_seam_missing_config_is_durable_scoped_wait(tmp_path):
    from slm_training.autoresearch.heal.recovery_dispatch import (
        RecoveryContext,
        dispatch_hard_pending,
    )

    context = RecoveryContext(
        tmp_path / "store", "loop", "campaign", tmp_path, SHA, SHA, "fence-1", "parent"
    )
    pending = {"kind": "repair_harness", "blocker_code": "harness_code_failure"}
    result = dispatch_hard_pending(
        pending, context, config=None, fence_valid=lambda _: True
    )
    assert result["reason"] == "agent_grant_missing"
    assert result["request_digest"] is None
    repeated = dispatch_hard_pending(
        pending, context, config=None, fence_valid=lambda _: True
    )
    assert repeated == result
    events = CampaignStore("campaign", tmp_path / "store").verify_event_chain()
    assert len(events) == 1
    assert events[0]["event_type"] == "repair_wait"


def test_config_parser_builds_full_request_without_manual_request_file(tmp_path):
    grant = RepairGrant(grant_id="fixture", provider="fixture", executable="/usr/bin/false",
        executable_sha256=SHA, expires_at=time.time() + 600, max_attempts=1,
        total_seconds=30.0, interrupt_seconds=10)
    allowed_paths = ("src/slm_training/harnesses/model_build/eval_runner.py",)
    from slm_training.autoresearch.heal.recovery_dispatch import (
        RecoveryContext,
        build_repair_request,
        load_recovery_config,
    )

    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    (source / "AGENTS.md").write_text("I6 preserve grammar. " * 100)
    (source / "RTK.md").write_text("bounded tools")
    (source / "docs/design/decode-invariants.md").write_text("zero singleton forwards")
    (source / "owner.md").write_text("immutable evidence and original reproducer")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "verifier_release": SHA,
                "grant": grant.model_dump(mode="json"),
                "recipes": {
                    "harness_code_failure": {
                        "allowed_paths": list(allowed_paths),
                        "input_digest": SHA,
                        "original": {
                            "check_id": "original",
                            "argv": ["python", "repro.py"],
                            "expected_stdout": "ready\n",
                        },
                        "checks": [
                            {
                                "check_id": "regression",
                                "argv": ["python", "regression.py"],
                                "expected_stdout": "passed\n",
                            }
                        ],
                        "owner_contract_path": "owner.md",
                        "failure_returncode": 1,
                        "failure_stdout_sha256": SHA,
                        "failure_stderr_sha256": SHA,
                    }
                },
            }
        )
    )
    context = RecoveryContext(
        tmp_path / "store", "loop", "campaign", source, SHA, SHA, "fence-1", "parent"
    )
    pending = {
        "kind": "repair_harness",
        "blocker_code": "harness_code_failure",
        "unmet_predicate": "rows_complete",
        "required_capability": "source_repair",
    }
    built = build_repair_request(pending, context, load_recovery_config(config_path))
    assert len(built.project_instructions) > 512
    assert built.blocker.reproducer == ("python", "repro.py")
    assert built.blocker.blocker_class == "code"
    assert built.grant == grant

