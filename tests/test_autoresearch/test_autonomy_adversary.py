"""Independent activity and proof-boundary falsifiers; fixtures are not model wins."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from slm_training.autoresearch.runtime.activity_runtime import (
    ActivityRuntime,
    StaleLease,
)
from slm_training.harness_core.activity_contract import (
    ActivityEvent,
    ActivityLease,
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
    reduce_activity,
)
from slm_training.autoresearch.storage import CampaignStore


def test_runtime_package_is_canonical_without_legacy_shadows():
    from scripts.verify_autonomy import _plan

    source = Path(__file__).resolve().parents[2]
    owners = _plan(source)["source_files"]
    assert "src/slm_training/autoresearch/runtime/__init__.py" in owners
    for prefix in ("activity", "operations"):
        for path in (source / "src/slm_training/autoresearch/runtime").glob(
            prefix + "*.py"
        ):
            name = path.stem
            assert importlib.util.find_spec("slm_training.autoresearch.runtime." + name)
            assert importlib.util.find_spec("slm_training.autoresearch." + name) is None


def _spec() -> ActivitySpec:
    return ActivitySpec(
        activity_id="adversary",
        family="fixture",
        kind="control",
        source_digest="a" * 40,
        environment_digest="b" * 64,
        input_digest="c" * 64,
        output_namespace="runs/adversary",
        grant=ResourceGrant(
            interrupt_seconds=1,
            kill_grace_seconds=0.1,
            total_seconds=4,
            finalization_reserve_seconds=0.1,
        ),
    )


class ActivityFaultMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.at = 1.0
        self.state = reduce_activity(
            None,
            ActivityEvent(
                operation="register",
                activity_id="adversary",
                sequence=0,
                at=self.at,
                spec=_spec(),
            ),
        )
        self.reference_attempts = 0
        self.reference_charge = 0.0
        self.prior_leases = []

    def event(self, operation, **kwargs):
        self.at += 65
        return ActivityEvent(
            operation=operation,
            activity_id="adversary",
            sequence=self.state.sequence + 1,
            at=self.at,
            **kwargs,
        )

    @precondition(lambda self: self.state.status in {"runnable", "waiting_retry"})
    @rule()
    def claim(self):
        event = self.event("claim")
        lease = ActivityLease(
            activity_id="adversary",
            attempt_id=str(self.at),
            epoch=str(self.reference_attempts),
            generation=self.reference_attempts + 1,
            token=f"fence-{self.at}",
            owner_identity="fixture",
            expires_at=self.at + 1_000_000,
        )
        self.state = reduce_activity(
            self.state, event.model_copy(update={"lease": lease})
        )
        self.reference_attempts += 1
        self.reference_charge += 1.1
        self.prior_leases.append(lease)

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def retry(self):
        self.state = reduce_activity(
            self.state,
            self.event(
                "finish",
                lease=self.state.lease,
                outcome=ActivityOutcome.RETRY,
                spent_seconds=0.2,
            ),
        )
        self.reference_charge += 0.2 - 1.1

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def crash(self):
        self.state = reduce_activity(
            self.state, self.event("recover", lease=self.state.lease)
        )

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def complete(self):
        self.state = reduce_activity(
            self.state,
            self.event(
                "finish",
                lease=self.state.lease,
                outcome=ActivityOutcome.SUCCEEDED,
                spent_seconds=0.3,
                outputs={"result": "d" * 64},
            ),
        )
        self.reference_charge += 0.3 - 1.1

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def cancel_owned_attempt(self):
        self.state = reduce_activity(
            self.state,
            self.event(
                "cancel",
                lease=self.state.lease,
                cancel_reason="explicit user stop",
            ),
        )

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def heartbeat_is_not_scientific_progress(self):
        before = self.state.charged_seconds
        self.state = reduce_activity(
            self.state,
            self.event("heartbeat", lease=self.state.lease),
        )
        assert self.state.status == "running" and not self.state.outputs
        assert self.state.charged_seconds == before

    @precondition(lambda self: self.state.status == "running")
    @rule()
    def code_failure_requires_repair(self):
        self.state = reduce_activity(
            self.state,
            self.event(
                "finish",
                lease=self.state.lease,
                outcome=ActivityOutcome.CODE_FAILURE,
                spent_seconds=0.2,
                wake=WakeCondition(
                    predicate="original_reproducer_restored",
                    source="independent_verifier",
                    identity_digest="c" * 64,
                ),
            ),
        )
        self.reference_charge += 0.2 - 1.1
        assert self.state.action == "repair_harness"

    @precondition(
        lambda self: (
            self.state.status == "waiting_repair"
            and self.state.action == "repair_harness"
            and self.state.attempts < 3
            and self.state.charged_seconds + 1.2 <= 4
        )
    )
    @rule()
    def verified_wake_keeps_consumed_resource_history(self):
        before = self.state.charged_seconds
        with pytest.raises(ValueError):
            reduce_activity(
                self.state,
                self.event(
                    "wake",
                    wake=self.state.wake.model_copy(
                        update={"identity_digest": "e" * 64},
                    ),
                ),
            )
        self.state = reduce_activity(
            self.state,
            self.event("wake", wake=self.state.wake),
        )
        assert self.state.charged_seconds == before

    @rule()
    def invalid_fence_never_changes_state(self):
        if self.state.lease is not None:
            forged = self.state.lease.model_copy(update={"token": "forged"})
            with pytest.raises(ValueError):
                reduce_activity(
                    self.state,
                    self.event(
                        "finish",
                        lease=forged,
                        outcome=ActivityOutcome.SUCCEEDED,
                        spent_seconds=0,
                        outputs={"result": "d" * 64},
                    ),
                )

    @invariant()
    def charges_and_attempts_match_independent_accounting(self):
        assert self.state.attempts == self.reference_attempts
        assert self.state.charged_seconds == pytest.approx(self.reference_charge)
        assert self.state.charged_seconds >= 0
        assert (
            len({lease.token for lease in self.prior_leases}) == self.reference_attempts
        )


TestActivityFaultMachine = ActivityFaultMachine.TestCase
TestActivityFaultMachine.settings = settings(
    max_examples=50, stateful_step_count=30, deadline=None
)


def test_no_artifact_no_success_and_stale_fence_no_publication(tmp_path):
    with ActivityRuntime(CampaignStore("adversary", tmp_path)) as runtime:
        runtime.register(_spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        assert lease is not None
        with pytest.raises(ValueError):
            runtime.finish(
                lease,
                outcome=ActivityOutcome.SUCCEEDED,
                spent_seconds=0,
                outputs={"absent.json": "d" * 64},
            )
        assert runtime.snapshot()["adversary"].status == "running"
        with pytest.raises(StaleLease):
            runtime.finish(
                lease.model_copy(update={"token": "stale"}),
                outcome=ActivityOutcome.SUCCEEDED,
                spent_seconds=0,
                outputs={"absent.json": "d" * 64},
            )


def test_corrupt_output_digest_guard_is_load_bearing(tmp_path, monkeypatch):
    """Targeted mutation: bypass the actual shared output validator, rerun oracle."""

    def oracle():
        with ActivityRuntime(CampaignStore("mutation", tmp_path)) as runtime:
            runtime.register(_spec())
            lease = runtime.claim_next(capabilities={"local_process"})
            path = runtime.attempt_dir(lease) / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"actual":"wrong-content"}')
            with pytest.raises(ValueError):
                runtime.finish(
                    lease,
                    outcome=ActivityOutcome.SUCCEEDED,
                    spent_seconds=0.1,
                    outputs={"result.json": hashlib.sha256(b"different").hexdigest()},
                )

    oracle()
    monkeypatch.setattr(
        "slm_training.autoresearch.runtime.activity_process.verify_outputs",
        lambda *args: None,
    )
    # Use a distinct empty store so a prior leased trial cannot change the oracle.
    tmp_path = tmp_path / "mutant"
    with pytest.raises(pytest.fail.Exception):
        oracle()


def test_finite_runner_locks_faults_and_refuses_plan_drift(tmp_path):
    from pathlib import Path

    from scripts.verify_autonomy import _lock, _plan

    store = CampaignStore("locked", tmp_path)
    plan = _plan(Path(__file__).resolve().parents[2])
    _lock(store, plan)
    assert len(plan["faults"]) == 100
    assert plan["faults"].count("code_crash") == 5
    assert plan["schema_version"] == "autonomy_adversary_plan/v5"
    assert plan["faults"].count("repair_fixture") == plan["faults"].count("repair_replay") == 5
    assert plan["expected_succeeded"] == 65
    legacy = _plan(Path(__file__).resolve().parents[2], version=2)
    assert legacy["schema_version"] == "autonomy_adversary_plan/v2"
    assert legacy["expected_succeeded"] == 70 and len(legacy["faults"]) == 100
    assert not any(fault.startswith("repair_") for fault in legacy["faults"])
    with pytest.raises(ValueError, match="changed"):
        _lock(store, {**plan, "expected_succeeded": 99})


def test_concurrent_finite_batch_is_rejected_before_any_work(tmp_path):
    import fcntl

    from scripts.verify_autonomy import main

    store = CampaignStore("autonomy-adversary-finite-100", tmp_path)
    store.root.mkdir(parents=True)
    with (store.root / ".autonomy-batch.lock").open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="finite_verification_already_owned"):
            main(["--root", str(tmp_path), "--batch-size", "1"])
    assert not [
        e
        for e in store.verify_event_chain()
        if e["event_type"] == "activity_transition"
    ]


def test_actual_v3_repair_replays_original_activity_and_storage_interruptions(tmp_path):
    """Eleven selected real controller invocations; not the final 100-run claim."""
    import json
    import os
    import sys

    from scripts.autonomy_fixture_worker import fixture_spec
    from scripts.verify_autonomy import _plan, _summary
    from slm_training.autoresearch.heal.isolation import probe_isolation
    from slm_training.harness_core.bounded_process import run_bounded_process

    capability = probe_isolation()
    if not capability.available:
        if os.environ.get("SLM_REQUIRE_ISOLATION"):
            pytest.fail(capability.reason)
        pytest.skip(capability.reason)
    source = Path(__file__).resolve().parents[2]
    plan = _plan(source)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    store = CampaignStore("adversary", tmp_path / "events")
    with ActivityRuntime(store) as runtime:
        runtime.register(fixture_spec(100, plan).model_copy(update={
            "activity_id": "external-authority", "capabilities": ("unavailable_provider",),
            "output_namespace": "external-authority"}))
    entry = ("import json,sys; from pathlib import Path; "
             "from scripts.autonomy_fixture_worker import run_fixture_attempt; "
             "from slm_training.autoresearch.storage import CampaignStore; "
             "raise SystemExit(run_fixture_attempt(CampaignStore('adversary',Path(sys.argv[1])),"
             "json.loads(Path(sys.argv[2]).read_text()),int(sys.argv[3])))")
    for index in range(11):
        result = run_bounded_process(
            [sys.executable, "-c", entry, str(tmp_path / "events"), str(plan_file), str(index)],
            cwd=source, interrupt_after_seconds=plan["controller_interrupt_seconds"][index],
            kill_grace_seconds=10,
        )
        assert not result.timed_out, (index, result.stdout, result.stderr)
        assert result.returncode == {4: 0, 5: 73, 6: 74, 9: 76, 10: 75}.get(index, 0), result.stderr
    summary = _summary(store, plan)
    assert summary["actual_workload_attempts"] == 11 and not summary["complete"]
    assert summary["states"]["attempt-000"] == "succeeded"
    assert summary["states"]["attempt-009"] == "succeeded"
    assert summary["states"]["attempt-010"] == "waiting_repair"
    assert summary["fake_agent_predicate_repairs"] == summary["verified_activity_replays"] == 1
    assert summary["storage_interruptions"] == 2
    assert summary["source_releases_authorized"] == summary["live_agent_repairs"] == 0
    first, replay = summary["observations"][0], summary["observations"][8]
    assert first["returncode"] == 1 and replay["returncode"] == 0
    assert first["attempt_id"] != replay["attempt_id"]


def test_current_runtime_fixture_identity_is_content_bound():
    from scripts.verify_autonomy import _plan
    from slm_training.harness_core.activity_contract import contract_digest
    from tests.test_autoresearch.test_activity_runtime_demonstration import _spec

    identity = contract_digest({"runtime_owners": _plan(Path(__file__).resolve().parents[2])["source_files"],
        "fixture_test": hashlib.sha256(Path(__file__).with_name("test_activity_runtime_demonstration.py").read_bytes()).hexdigest()})
    assert _spec(0).source_digest == identity and len(identity) == 64
