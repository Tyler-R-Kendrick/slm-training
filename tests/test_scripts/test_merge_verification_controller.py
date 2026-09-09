"""Finite controller wiring: real leases/processes plus explicit injected yields."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from scripts import merge_verification_controller as owner
from scripts.merge_verification_evidence import (
    digest,
    environment_identity,
    runtime_identity,
    source_identity,
)
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore


def fixture_plan(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "test_case.py").write_text("def test_case(): pass\n")
    # Source enumeration is exercised elsewhere against actual Git. This fixture
    # has only one enumerated input so no repository Git mutation is necessary.
    return {
        "schema": "release_verification_plan/v1",
        "source": str(root),
        "source_digest": "a" * 64,
        "environment_digest": "b" * 64,
        "runtime_digest": "c" * 64,
        "state_dir": str(tmp_path / "cache"),
        "base_ref": "HEAD",
        "step_seconds": 2,
        "total_seconds": 200,
        "max_invocations": 3,
        "local_feedback": True,
        "runtime_roots": [],
    }


def test_canonical_parser_exposes_finite_release_command(tmp_path):
    from scripts.autoresearch import build_parser

    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "verify-release",
            "--source",
            str(tmp_path),
            "--state-dir",
            str(tmp_path / "cache"),
            "--total-seconds",
            "600",
            "--max-invocations",
            "8",
            "--max-step-seconds",
            "15",
            "--identity",
            "d" * 64,
            "--activity-id",
            "source-verification-" + "d" * 64,
        ]
    )
    assert args.func is owner.cmd_verify_release
    assert args.total_seconds == 600 and args.max_invocations == 8
    assert args.max_step_seconds == 15
    assert not args.local_feedback
    assert args.identity == "d" * 64
    assert args.activity_id == "source-verification-" + args.identity


@pytest.mark.parametrize(
    "argv,handler,expected",
    case_values(__file__, "test_extracted_operations_keep_existing_parser_contract"),
)
def test_extracted_operations_keep_existing_parser_contract(argv, handler, expected):
    from scripts import autoresearch

    args = autoresearch.build_parser().parse_args(argv)
    assert args.func is getattr(autoresearch, handler)
    assert {key: getattr(args, key) for key in expected} == expected


@pytest.mark.parametrize(
    "argv",
    case_values(
        __file__, "test_extracted_operations_keep_required_and_exclusive_options"
    ),
)
def test_extracted_operations_keep_required_and_exclusive_options(argv):
    from scripts.autoresearch import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_locked_grant_cannot_reset_retries(tmp_path):
    plan = fixture_plan(tmp_path)
    store = CampaignStore("finite", tmp_path / "events")
    with ActivityRuntime(store) as runtime:
        original = owner.register_release(runtime, plan)
        assert owner.register_release(runtime, plan) == original
        with pytest.raises(ValueError, match="locked verification"):
            owner.register_release(runtime, {**plan, "max_invocations": 4})


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 91])
def test_unusable_controller_allowance_refused_before_execution(tmp_path, seconds):
    plan = fixture_plan(tmp_path)
    with pytest.raises(ValueError):
        owner.release_grant({**plan, "step_seconds": seconds})


def test_fake_success_without_authenticated_journal_is_rejected(tmp_path):
    plan = fixture_plan(tmp_path)
    result = SimpleNamespace(
        timed_out=False,
        cancelled=False,
        progress_stalled=False,
        returncode=0,
        stdout=json.dumps(
            {"identity": "d" * 64, "verification_complete": True, "status": "complete"}
        ),
    )
    with pytest.raises(ValueError, match="journal mismatch"):
        owner.validate_observation(result, plan)
    with pytest.raises(ValueError, match="result identity"):
        owner.validate_observation(result, {**plan, "identity": "e" * 64})


def test_child_invalidation_preserves_reason_without_accepting_claims(tmp_path):
    plan = fixture_plan(tmp_path)
    payload = {
        "status": "invalid_evidence",
        "reason": "source_changed_during_verification",
        "verification_complete": True,
        "release_authorized": True,
    }
    result = SimpleNamespace(
        timed_out=False,
        cancelled=False,
        progress_stalled=False,
        returncode=1,
        stdout=json.dumps(payload),
    )
    summary = owner.validate_observation(result, plan)
    assert summary["reason"] == "source_changed_during_verification"
    assert not summary["verification_complete"] and not summary["release_authorized"]
    result.stdout = json.dumps({**payload, "status": "complete"})
    with pytest.raises(ValueError, match="nonfailure outcome"):
        owner.validate_observation(result, plan)


@pytest.mark.parametrize(
    "status", ["waiting_capability", "waiting_dependency", "waiting_environment"]
)
def test_operational_wait_never_requires_or_claims_success_journal(tmp_path, status):
    result = SimpleNamespace(
        timed_out=False,
        cancelled=False,
        progress_stalled=False,
        returncode=20,
        stdout=json.dumps(
            {
                "status": status,
                "verification_complete": True,
                "release_authorized": True,
            }
        ),
    )
    summary = owner.validate_observation(result, fixture_plan(tmp_path))
    assert summary["status"] == status
    assert not summary["verification_complete"] and not summary["release_authorized"]


def test_finite_controller_resumes_actual_gate_after_yield(
    tmp_path, monkeypatch, capsys
):
    from scripts import merge_verification_evidence as evidence

    plan = fixture_plan(tmp_path)
    source = Path(__file__).resolve().parents[2]
    root = Path(plan["source"])
    monkeypatch.setattr(evidence, "source_paths", lambda _: ["test_case.py"])
    plan.update(
        source_digest=source_identity(root),
        environment_digest=digest(environment_identity()),
        runtime_roots=[sys.prefix],
        runtime_digest=runtime_identity((Path(sys.prefix),)),
    )
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(source)!r})
from scripts import merge_verification as owner, merge_verification_evidence as evidence
from scripts.verify_merge_ready import Step, run_step
evidence.source_paths = lambda _: ['test_case.py']
owner.changed_paths = lambda *_: ('fixture-base', ['test_case.py'])
owner.check_changed.select_tests = lambda *args, **kwargs: ['test_case.py']
if sys.argv[1] == '1': owner._run_shards = lambda *args: None
summary = owner.run_release_gate((Step('static', (sys.executable, '-c', 'pass')),),
 root=Path({str(root)!r}), base_ref='HEAD', state_dir=Path({plan["state_dir"]!r}),
 step_seconds=2, run_step=run_step, local_feedback=True)
print(json.dumps(summary))
sys.exit(0 if summary['verification_complete'] else 10)
"""
    calls = []

    def argv(_):
        calls.append(1)
        return [sys.executable, "-c", code, str(len(calls))]

    monkeypatch.setattr(owner, "verification_argv", argv)
    store = CampaignStore("finite", tmp_path / "events")
    from scripts.autoresearch import build_parser

    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path / "events"),
            "verify-release",
            "--source",
            str(root),
            "--state-dir",
            plan["state_dir"],
            "--job-id",
            "finite",
            "--max-step-seconds",
            "2",
            "--total-seconds",
            "200",
            "--max-invocations",
            "3",
            "--local-feedback",
        ]
    )
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verification_complete"], result
    assert result["attempts"] == 2
    assert not result["release_authorized"]
    assert result["charged_activity_seconds"] > 0
    locked = next(
        e
        for e in store.verify_event_chain()
        if e["event_type"] == "release_verification_locked"
    )
    plan = json.loads(Path(locked["detail"]["artifact"]).read_text())
    with ActivityRuntime(store) as runtime:
        replay = owner.drain_release(runtime, plan, deadline=time.monotonic() + 150)
        assert replay["verification_complete"]
        assert len(calls) == 2
        assert replay["charged_activity_seconds"] == result["charged_activity_seconds"]


def test_missing_isolation_parks_only_verification_activity(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import isolation

    plan = {**fixture_plan(tmp_path), "local_feedback": False}
    monkeypatch.setattr(
        isolation, "probe_isolation", lambda: SimpleNamespace(available=False)
    )
    store = CampaignStore("finite", tmp_path / "events")
    with ActivityRuntime(store) as runtime:
        result = owner.drain_release(runtime, plan, deadline=time.monotonic() + 150)
        assert result["status"] == "waiting_capability"
        assert result["attempts"] == 0
        assert result["wake"]["source"] == "capability_probe"


def test_verifier_bootstrap_does_not_import_candidate_sitecustomize(
    tmp_path, monkeypatch
):
    import subprocess

    plan = fixture_plan(tmp_path)
    root = Path(plan["source"])
    marker = tmp_path / "host-corruption"
    (root / "sitecustomize.py").write_text(
        f"open({str(marker)!r}, 'w').write('unsafe')"
    )
    monkeypatch.setenv("PYTHONPATH", str(root))
    argv = owner.verification_argv(plan)
    assert argv[1] == "-I"
    result = subprocess.run(
        argv[:4] + ["--help"], cwd=root, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert "--identity" in result.stdout
    assert not marker.exists()
