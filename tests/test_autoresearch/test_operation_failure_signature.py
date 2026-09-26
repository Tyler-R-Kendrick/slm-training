"""Original operation faults stay bound when volatile driver logs differ."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.heal import operation_recovery as bridge
from slm_training.autoresearch.heal import operation_diagnosis as diagnosis
from slm_training.autoresearch.heal.operation_failure_signature import (
    capture_failure_signature,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityOutcome
from slm_training.harness_core.bounded_process import ProcessOutcome
from tests.test_autoresearch.test_operation_recovery import diagnosed as _diagnosed


@pytest.fixture(name="case")
def _case(tmp_path, monkeypatch):
    return _diagnosed.__wrapped__(tmp_path, monkeypatch)


def _result(
    source,
    *,
    pid=8,
    message="source identity changed",
    frame="scripts/run_autotrain_continuous.py",
    exception_type="RuntimeError",
    truncated=False,
):
    return SimpleNamespace(
        returncode=1,
        stdout=f"DRIVER_LOCK_ACQUIRED loop_id=r21 pid={pid} path={source}/lock\n",
        stderr=(
            "Traceback (most recent call last):\n"
            f'  File "{source}/{frame}", line 141, in run_cycle\n'
            f"{exception_type}: {message}\n"
        ),
        stdout_truncated=truncated,
        stderr_truncated=False,
        launch_error=None,
        outcome=ProcessOutcome.COMPLETED,
    )


def test_signature_ignores_driver_pid_and_source_root_but_binds_fault(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    expected = capture_failure_signature(_result(first), first)
    assert expected == capture_failure_signature(_result(second, pid=92), second)
    assert expected != capture_failure_signature(
        _result(second, message="different fault"), second
    )
    assert capture_failure_signature(_result(first, truncated=True), first) is None
    assert capture_failure_signature(_result(first), second) is None


def test_original_event_keeps_raw_hashes_and_signature(tmp_path):
    source = tmp_path / "source"
    result = _result(source)
    event = {}

    class Store:
        def write_artifact(self, *_args):
            return SimpleNamespace(stem="artifact")

        def append_event(self, *_args, **kwargs):
            event.update(kwargs)

    pending = bridge.record_operation_failure(
        SimpleNamespace(store=Store()),
        SimpleNamespace(activity_id="activity", token="fence"),
        {"cwd": str(source), "operation": "driver"},
        result,
        outcome=ActivityOutcome.CODE_FAILURE,
    )
    observed = pending["failure_observation"]
    assert (
        observed["stdout_sha256"] == hashlib.sha256(result.stdout.encode()).hexdigest()
    )
    assert (
        observed["stderr_sha256"] == hashlib.sha256(result.stderr.encode()).hexdigest()
    )
    assert observed["failure_signature"] == capture_failure_signature(
        result, source
    ).model_dump(mode="json")
    assert event["detail"]["pending"] == pending


def test_diagnosis_requires_exact_typed_fault_and_independent_probe(case, monkeypatch):
    context, config, probes = case
    original = _result(context.source)
    probe = _result("/workspace", pid=92)
    probe.duration_seconds = 0.01

    def isolated_probe(_spec, argv):
        probes.append(tuple(argv))
        return probe

    monkeypatch.setattr(diagnosis, "run_isolated", isolated_probe)
    signature = capture_failure_signature(original, context.source)
    recipe = config.recipes["harness_code_failure"].model_copy(
        update={
            "original_failure_signature": signature,
            "failure_stdout_sha256": hashlib.sha256(probe.stdout.encode()).hexdigest(),
            "failure_stderr_sha256": hashlib.sha256(probe.stderr.encode()).hexdigest(),
        }
    )
    assert "original_failure_signature" not in config.recipes[
        "harness_code_failure"
    ].model_dump(mode="json")
    config = config.model_copy(update={"recipes": {"harness_code_failure": recipe}})
    pending = {
        "original_operation": "inspect",
        "original_request_digest": "a" * 64,
        "observed_outcome": "code_failure",
        "affected_activity_id": "original",
        "failure_observation": {
            "returncode": 1,
            "stdout_sha256": hashlib.sha256(original.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(original.stderr.encode()).hexdigest(),
            "truncated": False,
            "launch_error": False,
            "process_outcome": "completed",
            "failure_signature": signature.model_dump(mode="json"),
        },
    }
    assert (
        pending["failure_observation"]["stdout_sha256"] != recipe.failure_stdout_sha256
    )
    journal = CampaignStore("campaign", context.root)
    resolved, reason, ran = bridge.diagnose_operation(pending, context, config, journal)
    assert resolved and reason == "original_fault_reproduced" and ran
    assert len(probes) == 1

    altered = {
        **pending,
        "failure_observation": {
            **pending["failure_observation"],
            "failure_signature": capture_failure_signature(
                _result(context.source, message="other"), context.source
            ).model_dump(mode="json"),
        },
    }
    result, reason, ran = bridge.diagnose_operation(
        altered, replace(context, attempt_id="other"), config, journal
    )
    assert result is None and reason != "original_fault_reproduced"
    assert not ran  # Prior success cannot be reused; grant already spent.


def test_signature_cannot_certify_unrelated_exact_hash_reproducer(case):
    context, config, probes = case
    signature = capture_failure_signature(_result(context.source), context.source)
    recipe = config.recipes["harness_code_failure"].model_copy(
        update={"original_failure_signature": signature}
    )
    config = config.model_copy(
        update={"recipes": {"harness_code_failure": recipe}}
    )
    # Real fixture process matches approved raw hashes and returncode, but
    # contains no traceback from original failing source frame.
    pending = {
        "original_operation": "inspect",
        "original_request_digest": "a" * 64,
        "observed_outcome": "code_failure",
        "affected_activity_id": "original",
        "failure_observation": {
            "returncode": 1,
            "stdout_sha256": "b" * 64,
            "stderr_sha256": "c" * 64,
            "truncated": False,
            "failure_signature": signature.model_dump(mode="json"),
        },
    }
    result, reason, ran = bridge.diagnose_operation(
        pending, context, config, CampaignStore("campaign", context.root)
    )
    assert result is None and reason == "original_fault_not_reproduced" and ran
    assert len(probes) == 1


@pytest.mark.parametrize(
    ("frame", "message", "exception_type"),
    case_values(__file__, "test_exact_probe_bytes_cannot_cover_wrong_typed_fault"),
)
def test_exact_probe_bytes_cannot_cover_wrong_typed_fault(
    case, monkeypatch, frame, message, exception_type
):
    context, config, probes = case
    original = _result(context.source)
    probe = _result(
        "/workspace", frame=frame, message=message, exception_type=exception_type
    )
    probe.duration_seconds = 0.01

    def isolated_probe(_spec, argv):
        probes.append(tuple(argv))
        return probe

    monkeypatch.setattr(diagnosis, "run_isolated", isolated_probe)
    recipe = config.recipes["harness_code_failure"].model_copy(
        update={
            "original_failure_signature": capture_failure_signature(
                original, context.source
            ),
            "failure_stdout_sha256": hashlib.sha256(probe.stdout.encode()).hexdigest(),
            "failure_stderr_sha256": hashlib.sha256(probe.stderr.encode()).hexdigest(),
        }
    )
    config = config.model_copy(
        update={"recipes": {"harness_code_failure": recipe}}
    )
    pending = {
        "original_operation": "inspect",
        "original_request_digest": "a" * 64,
        "observed_outcome": "code_failure",
        "affected_activity_id": "original",
        "failure_observation": {
            "returncode": 1,
            "stdout_sha256": hashlib.sha256(original.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(original.stderr.encode()).hexdigest(),
            "truncated": False,
            "failure_signature": recipe.original_failure_signature.model_dump(mode="json"),
        },
    }
    result, reason, ran = bridge.diagnose_operation(
        pending, context, config, CampaignStore("campaign", context.root)
    )
    assert result is None and reason == "original_fault_not_reproduced" and ran
    assert len(probes) == 1
