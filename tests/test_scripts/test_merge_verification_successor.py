"""Release-verifier successor carries only the predecessor's unused grant."""

import pytest

from scripts.merge_verification_controller import register_release, release_grant
from scripts.merge_verification_evidence import digest
from scripts.merge_verification_successor import release_successor_plan
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityOutcome


def test_release_successor_reuses_only_unspent_budget_and_attempts(tmp_path):
    old = {
        "schema": "release_verification_plan/v1",
        "activity_id": "release-r12",
        "source": str(tmp_path / "old-source"),
        "source_digest": "a" * 64,
        "environment_digest": "b" * 64,
        "runtime_digest": "c" * 64,
        "state_dir": str(tmp_path / "cache"),
        "base_ref": "main",
        "step_seconds": 30,
        "total_seconds": 200,
        "max_invocations": 3,
        "local_feedback": True,
        "require_js_runtime": False,
        "runtime_roots": [],
    }
    store = CampaignStore("shared-job", tmp_path / "events")
    with ActivityRuntime(store) as runtime:
        register_release(runtime, old)
        lease = runtime.claim_next(
            capabilities={"local_process"}, activity_id=old["activity_id"]
        )
        assert lease is not None
        runtime.finish(lease, outcome=ActivityOutcome.RETRY, spent_seconds=12)

        current = {
            **old,
            "activity_id": "release-r13",
            "source": str(tmp_path / "new-source"),
            "source_digest": "d" * 64,
            "environment_digest": "e" * 64,
            "runtime_digest": "f" * 64,
        }
        successor = release_successor_plan(
            runtime, current, old["activity_id"], release_grant
        )
        assert successor["total_seconds"] == 188
        assert successor["max_invocations"] == 2
        assert successor["successor_of"] == old["activity_id"]
        assert runtime.snapshot()[old["activity_id"]].status == "cancelled"

        assert (
            release_successor_plan(
                runtime, current, old["activity_id"], release_grant
            )
            == successor
        )
        with pytest.raises(ValueError, match="already_forked"):
            release_successor_plan(
                runtime,
                {**current, "activity_id": "release-r14"},
                old["activity_id"],
                release_grant,
            )
