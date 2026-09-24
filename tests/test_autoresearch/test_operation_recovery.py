"""Real operation/reproducer processes; explicit fake agent and mocked isolation."""

import hashlib
import sys
import time
from dataclasses import replace

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    WakeCondition,
    contract_digest,
)
from slm_training.autoresearch.heal import operation_recovery as bridge
from slm_training.autoresearch.heal import recovery_dispatch as dispatch
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairGrant,
)
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import KILL_GRACE_SECONDS
from tests.test_autoresearch import test_repair_dispatch as dispatch_fixtures

executor = dispatch_fixtures.executor
journal = dispatch_fixtures.journal
repair_request = dispatch_fixtures.repair_request

SHA = "a" * 64


@pytest.mark.parametrize("kind", case_values(__file__, "test_agent_cannot_ignore_other_repair_reservations"))
@pytest.mark.parametrize("legacy", [False, True])
def test_agent_cannot_ignore_other_repair_reservations(
    repair_request, journal, executor, kind, legacy,
):
    from slm_training.autoresearch.heal.dispatch import dispatch_repair

    detail = {"reserved_seconds": 30, "fingerprint": "different-blocker",
              "attempt_id": "previous"}
    if not legacy:
        detail["grant_digest"] = repair_request.grant.digest()
    journal.append_event(kind, detail=detail)
    result = dispatch_repair(repair_request, executor=executor, journal=journal,
                             fence_valid=lambda _: True)
    assert result.reason == "repair_grant_exhausted"
    assert executor.runner.calls == 0


def test_verification_cannot_ignore_other_blocker_reservations(repair_request, journal):
    from types import SimpleNamespace
    from slm_training.autoresearch.heal.repair_acceptance import _reserve_verification

    journal.append_event("repair_started", detail={"reserved_seconds": 40,
        "fingerprint": "different-blocker", "grant_digest": repair_request.grant.digest()})
    spec = SimpleNamespace(checks=(), equivalence_checks=(), timeout_seconds=1)
    assert _reserve_verification(repair_request, spec, journal) == "verification_grant_exhausted"


def test_diagnosis_cannot_ignore_agent_reservations(diagnosed):
    context, config, probes = diagnosed
    journal = CampaignStore(context.campaign_id, context.root)
    journal.append_event("repair_started", detail={"reserved_seconds": 100,
        "grant_digest": config.grant.digest(), "fingerprint": "different-blocker"})
    pending = {"original_operation": "inspect", "observed_outcome": "code_failure",
               "original_request_digest": SHA}
    _, reason, ran = bridge.diagnose_operation(pending, context, config, journal)
    assert reason == "diagnosis_grant_exhausted" and not ran and not probes


def test_other_explicit_grant_is_not_charged(repair_request, journal, executor):
    from slm_training.autoresearch.heal.dispatch import dispatch_repair

    other = repair_request.grant.model_copy(update={"grant_id": "other-grant"})
    journal.append_event("repair_started", detail={"reserved_seconds": 100,
        "grant_digest": "b" * 64, "grant_id": other.grant_id,
        "grant_accounting_digest": other.accounting_digest(),
        "fingerprint": "different-blocker", "attempt_id": "previous"})
    result = dispatch_repair(repair_request, executor=executor, journal=journal,
                             fence_valid=lambda _: True)
    assert result.status == "waiting_verification" and executor.runner.calls == 1


@pytest.fixture
def diagnosed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    for path in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / path).write_text("I6: immutable judge; I2: singleton bypass")
    (source / "fixture.py").write_text('print("broken"); raise SystemExit(1)\n')
    recipe = dispatch.RepairRecipe(
        allowed_paths=("fixture.py", "tests/test_fix.py"),
        input_digest=SHA,
        original=VerificationCheck(
            "original", (sys.executable, "fixture.py"), "ready\n"
        ),
        checks=(VerificationCheck("existing", (sys.executable, "check.py"), "pass\n"),),
        owner_contract_path="AGENTS.md",
        failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"broken\n").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    config = dispatch.RecoveryConfig(
        grant=RepairGrant(
            grant_id="fixture",
            provider="fixture",
            executable=sys.executable,
            executable_sha256=SHA,
            expires_at=time.time() + 100,
            max_attempts=1,
            total_seconds=100,
            interrupt_seconds=10,
        ),
        verifier_release=SHA,
        recipes={"harness_code_failure": recipe},
        operation_recipes={"inspect": "harness_code_failure"},
    )
    context = dispatch.RecoveryContext(
        tmp_path / "outputs",
        "loop",
        "campaign",
        source,
        manifest_digest(tree_manifest(source)),
        SHA,
        "fence",
        "parent",
    )
    calls = []

    def actual_process_with_mocked_isolation(spec, argv):
        calls.append(tuple(argv))
        return run_bounded_process(
            argv,
            cwd=spec.workspace,
            interrupt_after_seconds=spec.timeout_seconds,
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )

    monkeypatch.setattr(bridge, "run_isolated", actual_process_with_mocked_isolation)
    return context, config, calls


def test_failed_operation_queue_reaches_actual_diagnosis_and_executor(
    diagnosed, monkeypatch
):
    context, config, probes = diagnosed
    original = {
        "operation": "inspect",
        "loop_id": "loop",
        "cwd": str(context.source),
        "root": str(context.root),
        "source_digest": context.source_digest,
        "environment_digest": SHA,
    }
    with ActivityRuntime(CampaignStore("runtime", context.root)) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="original",
                family="loop",
                kind="control",
                source_digest=context.source_digest,
                environment_digest=SHA,
                input_digest=contract_digest(original),
                output_namespace="attempt",
            )
        )
        lease = runtime.claim_next(capabilities={"local_process"})
        result = runtime.run(
            lease,
            (sys.executable, "fixture.py"),
            cwd=context.source,
        )
        bridge.record_operation_failure(
            runtime, lease, original, result, outcome=ActivityOutcome.UNKNOWN_FAILURE
        )
        runtime.finish(
            lease,
            outcome=ActivityOutcome.UNKNOWN_FAILURE,
            spent_seconds=result.duration_seconds,
            wake=WakeCondition(
                predicate="original operation produces valid output",
                source="independent_repair_verification",
                identity_digest=contract_digest(original),
            ),
        )
        jobs = bridge.pending_operation_repairs(runtime)
        assert len(jobs) == 1 and jobs[0]["operation"] == "repair"
        pending = jobs[0]["hard_pending"][0]
        assert pending["affected_activity_id"] == "original"
        assert pending["observed_outcome"] == "unknown_failure"
        launched = []

        class ExplicitFakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def capability(self, request):
                return None

            def execute(self, request, **kwargs):
                launched.append(request)
                return RepairDispatchResult(
                    status="waiting_diagnosis",
                    request_digest=request.digest(),
                    reason="fake_agent_invoked",
                )

        monkeypatch.setattr(dispatch, "CodexExecutor", ExplicitFakeAgent)
        first = dispatch.dispatch_hard_pending(
            pending, context, config=config, fence_valid=lambda _: True
        )
        assert first["reason"] == "original_fault_reproduced" and not launched
        second = dispatch.dispatch_hard_pending(
            pending, context, config=config, fence_valid=lambda _: True
        )
        assert second["reason"] == "fake_agent_invoked"
        assert len(probes) == len(launched) == 1
        assert launched[0].blocked_activity_id == "original"
        assert launched[0].blocker.blocker_class == "code"
        events = CampaignStore("campaign", context.root).verify_event_chain()
        assert any(
            e["event_type"] == "operation_diagnosis_finished"
            and e["detail"]["spent_seconds"] > 0
            for e in events
        )
        assert any(e["event_type"] == "repair_started" for e in events)


@pytest.mark.parametrize("fault", ["argparse", "missing_recipe", "typed_formal"])
def test_unreproduced_or_wrong_authority_never_launches_agent(diagnosed, fault):
    context, config, probes = diagnosed
    pending = {
        "original_operation": "inspect",
        "original_request_digest": SHA,
        "observed_outcome": "unknown_failure",
        "affected_activity_id": "original",
    }
    if fault == "argparse":
        (context.source / "fixture.py").write_text("raise SystemExit(2)\n")
        context = replace(
            context, source_digest=manifest_digest(tree_manifest(context.source))
        )
    elif fault == "missing_recipe":
        config = config.model_copy(update={"operation_recipes": {}})
    else:
        pending["observed_outcome"] = "formal_contradiction"
    resolved, reason, _ = bridge.diagnose_operation(
        pending, context, config, CampaignStore("campaign", context.root)
    )
    assert resolved is None
    assert (
        reason
        == {
            "argparse": "original_fault_not_reproduced",
            "missing_recipe": "operation_reproducer_not_configured:inspect",
            "typed_formal": "typed_outcome_requires_original_owner:formal_contradiction",
        }[fault]
    )
    assert len(probes) == int(fault == "argparse")


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_unrelated_probe_cannot_repair_original_observation(diagnosed, returncode):
    context, config, probes = diagnosed
    pending = {
        "original_operation": "inspect",
        "original_request_digest": SHA,
        "observed_outcome": "unknown_failure",
        "affected_activity_id": "original",
        "failure_observation": {
            "returncode": returncode,
            "stdout_sha256": SHA,
            "stderr_sha256": SHA,
            "truncated": False,
        },
    }
    resolved, reason, _ = bridge.diagnose_operation(
        pending, context, config, CampaignStore("campaign", context.root)
    )
    assert resolved is None and reason == "operation_observation_not_covered_by_recipe"
    assert len(probes) == 1


def test_interrupted_diagnosis_retries_next_bounded_attempt_without_budget_reset(
    diagnosed, monkeypatch
):
    context, config, probes = diagnosed
    config = config.model_copy(
        update={"grant": config.grant.model_copy(update={"max_attempts": 2})}
    )
    recipe = config.recipes["harness_code_failure"]
    pending = {
        "original_operation": "inspect",
        "original_request_digest": SHA,
        "observed_outcome": "unknown_failure",
        "affected_activity_id": "original",
        "failure_observation": {
            "returncode": 1,
            "stdout_sha256": recipe.failure_stdout_sha256,
            "stderr_sha256": recipe.failure_stderr_sha256,
            "truncated": False,
        },
    }
    journal = CampaignStore("campaign", context.root)

    def crash(*args):
        raise SystemExit("injected worker crash")

    with monkeypatch.context() as patch:
        patch.setattr(bridge, "run_isolated", crash)
        with pytest.raises(SystemExit):
            bridge.diagnose_operation(pending, context, config, journal)
    assert (
        bridge.diagnose_operation(pending, context, config, journal)[1]
        == "interrupted_diagnosis_requires_reconciliation"
    )
    resolved, _, ran = bridge.diagnose_operation(
        pending, replace(context, attempt_id="next-attempt"), config, journal
    )
    assert resolved and ran and len(probes) == 1
    starts = [
        e
        for e in journal.verify_event_chain()
        if e["event_type"] == "operation_diagnosis_started"
    ]
    assert (
        len(starts) == 2 and sum(e["detail"]["reserved_seconds"] for e in starts) == 40
    )
    assert all(e["detail"]["grant_id"] == config.grant.grant_id for e in starts)
    assert all(
        e["detail"]["grant_accounting_digest"] == config.grant.accounting_digest()
        for e in starts
    )
