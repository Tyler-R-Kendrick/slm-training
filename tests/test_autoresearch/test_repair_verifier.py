"""Independent repair acceptance, using real workload processes when available."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.autoresearch.heal.isolation import probe_isolation
from slm_training.autoresearch.heal.isolation_workspace import (
    IsolationViolation,
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_scope import repair_classification
from slm_training.autoresearch.heal.repair_verifier import (
    VerificationCheck,
    VerificationRequest,
    verify_candidate,
)


@pytest.fixture
def repair(tmp_path: Path) -> tuple[Path, Path, VerificationRequest]:
    base = tmp_path / "base"
    base.mkdir()
    (base / "broken.py").write_text("def increment(x):\n    return x - 1\n")
    candidate = private_snapshot(base, tmp_path / "candidate")
    (candidate / "broken.py").write_text("def increment(x):\n    return x + 1\n")
    (candidate / "tests").mkdir()
    (candidate / "tests/test_fix.py").write_text(
        "from broken import increment\ndef test_increment():\n"
        "    assert increment(1) == 2\n"
    )
    original = VerificationCheck(
        "original",
        (
            "/usr/bin/python3",
            "-c",
            "from broken import increment; import sys; ok = increment(3) == 4; "
            "print('predicate-restored' if ok else 'predicate-miss'); "
            "sys.exit(0 if ok else 42)",
        ),
        "predicate-restored\n",
    )
    independent = VerificationCheck(
        "existing-contract",
        (
            "/usr/bin/python3",
            "-c",
            "from broken import increment; assert increment(-2) == -1; "
            "print('independent-pass')",
        ),
        "independent-pass\n",
    )
    request = VerificationRequest(
        hashlib.sha256(b"request").hexdigest(),
        hashlib.sha256(b"blocker").hexdigest(),
        manifest_digest(tree_manifest(base)),
        manifest_digest(tree_manifest(candidate)),
        hashlib.sha256(b"environment").hexdigest(),
        hashlib.sha256(b"verifier-release").hexdigest(),
        hashlib.sha256(b"authority").hexdigest(),
        "epoch:lease:1",
        ("broken.py", "tests/test_fix.py"),
        original,
        (independent,),
        42,
        hashlib.sha256(b"predicate-miss\n").hexdigest(),
        hashlib.sha256(b"").hexdigest(),
        timeout_seconds=5,
    )
    return base, candidate, request


def require_real() -> None:
    capability = probe_isolation()
    if not capability.available:
        if os.environ.get("SLM_REQUIRE_ISOLATION") == "1":
            pytest.fail(capability.reason)
        pytest.skip(capability.reason)


def test_original_failure_replayed_and_independent_predicate_restored(repair) -> None:
    require_real()
    base, candidate, request = repair
    evidence = verify_candidate(request, base, candidate)
    assert evidence.accepted and evidence.reason == "restored_predicate"
    assert not evidence.observations[0].passed
    assert all(row.passed for row in evidence.observations[1:])
    assert evidence.request.fencing_token == "epoch:lease:1"
    assert evidence.evidence_class == "independent_isolated_process"


def test_agent_says_fixed_but_original_fails(repair) -> None:
    require_real()
    base, candidate, request = repair
    (candidate / "broken.py").write_text("def increment(x):\n    return 2\n")
    request = replace(
        request, candidate_digest=manifest_digest(tree_manifest(candidate))
    )
    evidence = verify_candidate(request, base, candidate)
    assert not evidence.accepted and evidence.reason == "verification_failed"


def test_exit_zero_without_observation_is_not_verification(repair) -> None:
    require_real()
    base, candidate, request = repair
    (candidate / "broken.py").write_text("import os\nos._exit(0)\n")
    request = replace(
        request, candidate_digest=manifest_digest(tree_manifest(candidate))
    )
    evidence = verify_candidate(request, base, candidate)
    assert not evidence.accepted
    assert evidence.observations[1].returncode == 0


def test_different_original_failure_cannot_authorize_edit(repair) -> None:
    require_real()
    base, candidate, request = repair
    request = replace(request, failure_returncode=2)
    evidence = verify_candidate(request, base, candidate)
    assert (
        not evidence.accepted and evidence.reason == "original_failure_not_reproduced"
    )
    assert len(evidence.observations) == 1


def test_wrong_candidate_digest_never_runs(repair) -> None:
    base, candidate, request = repair
    with pytest.raises(IsolationViolation, match="identity mismatch"):
        verify_candidate(replace(request, candidate_digest="0" * 64), base, candidate)


def test_test_weakening_rejected_before_execution(repair) -> None:
    base, candidate, request = repair
    (candidate / "tests/test_fix.py").write_text(
        "import pytest\npytest.skip('fixed')\nassert True\n"
    )
    request = replace(
        request, candidate_digest=manifest_digest(tree_manifest(candidate))
    )
    with pytest.raises(IsolationViolation, match="skip/xfail"):
        verify_candidate(request, base, candidate)


def test_forged_worker_receipt_is_out_of_scope(repair) -> None:
    base, candidate, request = repair
    (candidate / "receipt.json").write_text('{"accepted":true}')
    request = replace(
        request, candidate_digest=manifest_digest(tree_manifest(candidate))
    )
    with pytest.raises(IsolationViolation, match="exceed exact repair grant"):
        verify_candidate(request, base, candidate)


def test_existing_tests_are_immutable(repair) -> None:
    base, candidate, request = repair
    (base / "tests").mkdir()
    (base / "tests/test_fix.py").write_text("assert False\n")
    request = replace(request, source_digest=manifest_digest(tree_manifest(base)))
    with pytest.raises(IsolationViolation, match="existing regression"):
        verify_candidate(request, base, candidate)


def test_policy_and_measurement_changes_require_separate_authority() -> None:
    assert repair_classification(("src/slm_training/__init__.py",)) == "trust_policy_change"
    assert (
        repair_classification(("src/slm_training/resources/evals/policy.json",))
        == "trust_policy_change"
    )
    assert (
        repair_classification(("src/slm_training/autoresearch/heal/isolation.py",))
        == "trust_policy_change"
    )
    assert (
        repair_classification(
            ("src/slm_training/harnesses/model_build/eval_runner.py",)
        )
        == "measurement_semantic_change"
    )


def test_empty_or_duplicate_verification_contract_is_refused(repair) -> None:
    _, _, request = repair
    with pytest.raises(ValueError, match="nonempty golden"):
        VerificationCheck("empty", ("true",), "")
    with pytest.raises(ValueError, match="unique"):
        replace(request, checks=(request.original,))


def test_eval_wiring_repair_requires_and_executes_unchanged_golden(repair) -> None:
    require_real()
    base, candidate, request = repair
    for root in (base, candidate):
        (root / "scripts").mkdir()
        (root / "scripts/evaluate_model.py").write_text(
            "def score():\n    return 7\n" + ("wired = True\n" if root == candidate else ""))
    golden = VerificationCheck("score-equivalence", ("/usr/bin/python3", "-c",
        "from scripts.evaluate_model import score; print(score())"), "7\n")
    request = replace(request,
        source_digest=manifest_digest(tree_manifest(base)),
        candidate_digest=manifest_digest(tree_manifest(candidate)),
        allowed_paths=(*request.allowed_paths, "scripts/evaluate_model.py"))
    with pytest.raises(IsolationViolation, match="separate science"):
        verify_candidate(request, base, candidate)
    request = replace(request, semantics_preserving_paths=("scripts/evaluate_model.py",),
                      equivalence_checks=(golden,))
    evidence = verify_candidate(request, base, candidate)
    assert evidence.accepted
    assert [row.phase for row in evidence.observations].count("baseline_equivalence") == 1
    (candidate / "scripts/evaluate_model.py").write_text("def score():\n    return 8\n")
    request = replace(request, candidate_digest=manifest_digest(tree_manifest(candidate)))
    assert not verify_candidate(request, base, candidate).accepted


def test_original_directory_mode_changes_are_not_hidden(repair) -> None:
    base, candidate, request = repair
    for root in (base, candidate):
        (root / "source").mkdir()
    (candidate / "source").chmod(0o777)
    request = replace(request, source_digest=manifest_digest(tree_manifest(base)),
                      candidate_digest=manifest_digest(tree_manifest(candidate)))
    with pytest.raises(IsolationViolation, match="exact repair grant"):
        verify_candidate(request, base, candidate)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1, 0, True, 171])
def test_verification_timeout_is_validated_before_any_work(repair, timeout):
    with pytest.raises(ValueError, match="timeout"):
        replace(repair[2], timeout_seconds=timeout)


@pytest.mark.parametrize("returncode", [True, 1.0, -1, 256])
def test_original_failure_exit_is_a_strict_ordinary_code(repair, returncode):
    with pytest.raises(ValueError, match="ordinary nonzero"):
        replace(repair[2], failure_returncode=returncode)


def test_verification_original_also_reserves_cleanup_and_finalization(repair, monkeypatch):
    from slm_training.autoresearch.heal import repair_verifier as verifier
    from slm_training.levers import (
        HARNESS_FINALIZATION_RESERVE_SECONDS, INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS,
    )

    base, candidate, request = repair
    request = replace(request, timeout_seconds=INTERRUPT_AFTER_SECONDS)
    clock, budgets = [0.0], []
    monkeypatch.setattr(verifier.time, "monotonic", lambda: clock[0])

    def observation(check, source, phase, runtimes, timeout):
        budgets.append(timeout)
        clock[0] += timeout
        return verifier.CheckObservation(
            check.check_id, phase, False, request.failure_returncode, "completed",
            request.failure_stdout_sha256, request.failure_stderr_sha256, timeout,
        )

    monkeypatch.setattr(verifier, "_observe", observation)
    result = verify_candidate(request, base, candidate)
    assert budgets == [INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS - KILL_GRACE_SECONDS]
    assert not result.accepted and result.reason == "verification_budget_exhausted"


def test_expired_controller_deadline_launches_no_work(repair, monkeypatch):
    from slm_training.autoresearch.heal import repair_verifier as verifier

    monkeypatch.setattr(verifier, "_observe", lambda *args: pytest.fail("expired workload launched"))
    base, candidate, request = repair
    result = verify_candidate(request, base, candidate, deadline=-1.0)
    assert not result.accepted and not result.observations
    assert result.reason == "verification_budget_exhausted"
