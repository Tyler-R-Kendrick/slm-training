"""Grant refreshes cannot reset repair attempt or time accounting."""

import hashlib
import time

import pytest

from tests.test_autoresearch.test_repair_dispatch import CHECK_MANIFEST, SHA, FakeSandbox
from slm_training.autoresearch.heal.agent_executor import CodexExecutor
from slm_training.autoresearch.heal.dispatch import (
    dispatch_repair,
    reserved_repair_seconds,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairBlocker,
    RepairGrant,
    RepairRequest,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.lineage.records import canonical_json


@pytest.fixture
def repair_request(tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("approved executable fixture")
    grant = RepairGrant(
        grant_id="local-test",
        provider="fixture",
        executable=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        expires_at=time.time() + 600,
        max_attempts=2,
        total_seconds=40.0,
        interrupt_seconds=10,
    )
    blocker = RepairBlocker(
        code="eval_missing_rows",
        blocker_class="code",
        owner="model_build",
        source_digest=SHA,
        environment_digest=SHA,
        input_digest=SHA,
        reproducer=("python", "-m", "pytest", "tests/test_original.py"),
        predicate="selected_rows_complete",
        needed_capability="source_repair",
        evidence=("Traceback: untrusted failure",),
    )
    return RepairRequest(
        activity_id="activity-1",
        attempt_id="attempt-1",
        campaign_id="fixture",
        fence="epoch:1",
        parent_event="event-1",
        blocker=blocker,
        grant=grant,
        allowed_paths=("src/slm_training/harnesses/model_build/eval_runner.py",),
        verification_manifest_digest=hashlib.sha256(
            canonical_json(CHECK_MANIFEST).encode()
        ).hexdigest(),
        project_instructions="AGENTS.md",
        owner_contract="docs/design/decode-invariants.md",
        existing_tests=("tests/test_original.py",),
        failure_returncode=1,
        failure_stdout_sha256=SHA,
        failure_stderr_sha256=SHA,
    )


@pytest.fixture
def journal(tmp_path):
    return CampaignStore("fixture", tmp_path / "store")


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(
        "slm_training.autoresearch.heal.agent_executor.probe_codex",
        lambda _: {"available": True},
    )
    return CodexExecutor(
        FakeSandbox(),
        instructions="AGENTS.md",
        contract="docs/design/decode-invariants.md",
        verification_manifest=CHECK_MANIFEST,
    )


def test_expiry_refresh_keeps_consumed_grant_reservation(
    repair_request, journal, executor
):
    grant = repair_request.grant.model_copy(update={"total_seconds": 30.0})
    request = repair_request.model_copy(update={"grant": grant})
    first = dispatch_repair(
        request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    refreshed = grant.model_copy(update={"expires_at": grant.expires_at + 3600})
    assert refreshed.digest() != grant.digest()
    assert refreshed.accounting_digest() == grant.accounting_digest()
    assert (
        reserved_repair_seconds(journal.verify_event_chain(), refreshed, journal) == 20
    )
    second = request.model_copy(
        update={
            "attempt_id": "after-refresh",
            "grant": refreshed,
        }
    )
    result = dispatch_repair(
        second, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    assert first.status == "waiting_verification"
    started = next(
        e for e in journal.verify_event_chain() if e["event_type"] == "repair_started"
    )
    assert started["detail"]["grant_id"] == grant.grant_id
    assert started["detail"]["grant_accounting_digest"] == grant.accounting_digest()
    assert result.reason == "repair_grant_exhausted"
    assert executor.runner.calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("total_seconds", 41.0), ("repair_classes", ("code", "data"))],
)
def test_grant_id_cannot_be_reused_with_changed_budget_or_scope(
    repair_request,
    journal,
    executor,
    field,
    value,
):
    dispatch_repair(
        repair_request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    changed = repair_request.grant.model_copy(update={field: value})
    with pytest.raises(ValueError, match="grant ID reused"):
        reserved_repair_seconds(journal.verify_event_chain(), changed, journal)


def test_legacy_reservation_resolves_grant_from_request_artifact(
    repair_request,
    journal,
):
    artifact = journal.write_artifact("repair_requests", repair_request)
    journal.append_event(
        "repair_started",
        artifact_sha256=artifact.stem,
        detail={
            "request_digest": repair_request.digest(),
            "grant_digest": repair_request.grant.digest(),
            "reserved_seconds": 17,
        },
    )
    refreshed = repair_request.grant.model_copy(
        update={"expires_at": repair_request.grant.expires_at + 3600}
    )
    assert (
        reserved_repair_seconds(journal.verify_event_chain(), refreshed, journal) == 17
    )


def test_unidentified_legacy_grant_reservation_is_charged(repair_request, journal):
    journal.append_event(
        "repair_started",
        detail={
            "grant_digest": "b" * 64,
            "reserved_seconds": 19,
        },
    )
    assert (
        reserved_repair_seconds(
            journal.verify_event_chain(), repair_request.grant, journal
        )
        == 19
    )


def test_successor_carries_predecessor_reservations_and_adds_only_its_budget(
    repair_request, journal, executor
):
    first = dispatch_repair(
        repair_request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    predecessor = repair_request.grant
    successor = predecessor.model_copy(
        update={
            "grant_id": "local-test-successor",
            "successor_of": predecessor.grant_id,
            "max_attempts": 1,
            "total_seconds": 20.0,
        }
    )
    events = journal.verify_event_chain()
    assert reserved_repair_seconds(events, successor, journal) == 20
    next_request = repair_request.model_copy(
        update={"attempt_id": "attempt-2", "grant": successor}
    )
    second = dispatch_repair(
        next_request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    assert first.status == second.status == "waiting_verification"
    assert executor.runner.calls == 2
    assert reserved_repair_seconds(journal.verify_event_chain(), successor, journal) == 40


def test_new_grant_cannot_reset_blocker_budget_without_successor_link(
    repair_request, journal, executor
):
    dispatch_repair(
        repair_request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    unrelated = repair_request.grant.model_copy(update={"grant_id": "unlinked"})
    request = repair_request.model_copy(
        update={"attempt_id": "attempt-2", "grant": unrelated}
    )
    with pytest.raises(ValueError, match="must extend the latest blocker grant"):
        dispatch_repair(
            request, executor=executor, journal=journal, fence_valid=lambda _: True
        )
    assert executor.runner.calls == 1


def test_successor_budget_and_attempts_are_cumulative(
    repair_request, journal, executor
):
    dispatch_repair(
        repair_request, executor=executor, journal=journal, fence_valid=lambda _: True
    )
    predecessor = repair_request.grant
    successor = predecessor.model_copy(
        update={
            "grant_id": "local-test-successor",
            "successor_of": predecessor.grant_id,
            "max_attempts": 1,
            "total_seconds": 20.0,
        }
    )
    successor_request = repair_request.model_copy(
        update={"attempt_id": "attempt-2", "grant": successor}
    )
    dispatch_repair(
        successor_request,
        executor=executor,
        journal=journal,
        fence_valid=lambda _: True,
    )
    fork = predecessor.model_copy(
        update={"grant_id": "fork", "successor_of": predecessor.grant_id}
    )
    fork_request = repair_request.model_copy(
        update={"attempt_id": "attempt-3", "grant": fork}
    )
    with pytest.raises(ValueError, match="latest blocker grant"):
        dispatch_repair(
            fork_request, executor=executor, journal=journal, fence_valid=lambda _: True
        )
    assert executor.runner.calls == 2


def test_successor_budget_covers_independent_verification(
    repair_request, journal
):
    from types import SimpleNamespace

    from slm_training.autoresearch.heal.repair_acceptance import _reserve_verification

    predecessor = repair_request.grant.model_copy(update={"total_seconds": 100.0})
    original = repair_request.model_copy(update={"grant": predecessor})
    artifact = journal.write_artifact("repair_requests", original)
    journal.append_event(
        "repair_started",
        artifact_sha256=artifact.stem,
        detail={
            "fingerprint": original.blocker.fingerprint(),
            "attempt_id": original.attempt_id,
            "request_digest": original.digest(),
            "grant_id": predecessor.grant_id,
            "grant_accounting_digest": predecessor.accounting_digest(),
            "reserved_seconds": 80,
        },
    )
    successor = predecessor.model_copy(
        update={
            "grant_id": "verification-successor",
            "successor_of": predecessor.grant_id,
            "max_attempts": 1,
            "total_seconds": 50.0,
        }
    )
    request = original.model_copy(update={"grant": successor})
    spec = SimpleNamespace(
        checks=(), equivalence_checks=(), timeout_seconds=1, fencing_token="fence"
    )
    assert _reserve_verification(request, spec, journal) is None
