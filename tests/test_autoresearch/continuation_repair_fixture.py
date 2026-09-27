"""Real bounded repair verification/publication for a tiny protocol workload.

Source-gate discovery/isolation use existing explicit fixture adapters; acceptance
and publication are real owners. No provider, full gate, or scientific claim.
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest
from slm_training.autoresearch.heal.repair_acceptance import (
    VerificationWorkspace, check_manifest_digest, proposal_patch_digest, verify_repair,
)
from slm_training.autoresearch.heal.repair_contracts import RepairBlocker, RepairGrant, RepairProposal, RepairRequest
from slm_training.autoresearch.heal.repair_release import publish_verified_repair
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck, VerificationRequest
from slm_training.harness_core.activity_contract import ActivitySpec, ActivityOutcome, ResourceGrant
from slm_training.harness_core.bounded_process import run_bounded_process
from tests.test_autoresearch.repair_acceptance_helpers import source_gate_fixture

FAULT = "scripts/autotrain_fixture.py"
PROVENANCE = {"integration_commit": "c" * 40, "upstream_commit": "c" * 40, "code_dirty": False}


def fault_program(version):
    return (f"VERSION = {version}\n"
            "if __name__ == '__main__':\n"
            "    import sys\n"
            "    ready = VERSION >= int(sys.argv[1])\n"
            "    print('ready' if ready else 'broken')\n"
            "    raise SystemExit(0 if ready else 1)\n")


def workload_source(source, science):
    (source / "scripts").mkdir()
    (source / "scripts/__init__.py").write_text("")
    (source / "src/slm_training").mkdir(parents=True)
    (source / "src/slm_training/__init__.py").write_text("")
    (source / FAULT).write_text(fault_program(0))
    (source / "scripts/train_model.py").write_text(
        "import json\nfrom pathlib import Path\n"
        f"root = Path({str(science)!r}); root.mkdir(parents=True, exist_ok=True)\n"
        "with (root / 'trained').open('a') as f: f.write('x')\n"
        "(root / 'checkpoint').write_bytes(b'committed-six-updates')\n"
        "print(json.dumps({'stopped_on': 'steps', 'steps': 6, 'checkpoint': str(root / 'checkpoint')}))\n"
    )
    (source / "scripts/evaluate_model.py").write_text(
        "import json, os, signal, sys\nfrom pathlib import Path\n"
        "from scripts.autotrain_fixture import VERSION\n"
        "import slm_training\n"
        f"root = Path({str(science)!r})\n"
        "assert '--resume-run' in sys.argv\n"
        "assert (root / 'checkpoint').read_bytes() == b'committed-six-updates'\n"
        "rows = root / 'rows'; before = rows.read_text(); assert before.startswith('preserved\\n')\n"
        "row = f'repair-{VERSION}\\n'\n"
        "if row not in before:\n"
        "    with rows.open('a') as f: f.write(row); f.flush(); os.fsync(f.fileno())\n"
        "(root / f'origin-{VERSION}.json').write_text(json.dumps({'module': __file__, "
        "'package': slm_training.__file__, 'cwd': str(Path.cwd()), 'argv': sys.argv}))\n"
        "if VERSION == 1: os.kill(os.getppid(), signal.SIGKILL)\n"
        "print(json.dumps({'resume': {'pending_record_n': {'smoke': 6 - len(rows.read_text().splitlines())}}}))\n"
        "raise SystemExit(10)\n"
    )
    (source / "docs/design").mkdir(parents=True)
    for name in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / name).write_text("I6 fail closed; preserve scientific inputs and retained cursor.\n")


def _request(runtime, lease, state, base, index, campaign_id, input_digest):
    original = VerificationCheck("original", (sys.executable, FAULT, str(index)), "ready\n")
    checks = (VerificationCheck("existing", original.argv, "ready\n"),)
    grant = RepairGrant(grant_id=f"fixture-{index}", provider="fixture", executable=sys.executable,
        executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        expires_at=time.time() + 300, max_attempts=1, total_seconds=150, interrupt_seconds=15)
    request = RepairRequest(activity_id=lease.activity_id, blocked_activity_id=state.spec.activity_id,
        attempt_id=lease.attempt_id, campaign_id=campaign_id, fence=lease.token,
        parent_event=runtime.store.verify_event_chain()[-1]["event_id"],
        blocker=RepairBlocker(code="harness_code_failure", blocker_class="code", owner="fixture",
            source_digest=manifest_digest(tree_manifest(base)), environment_digest=state.spec.environment_digest,
            input_digest=input_digest, reproducer=original.argv, predicate="ready_stdout",
            needed_capability="source_repair", evidence=("bounded fixture failure",)),
        grant=grant, allowed_paths=(FAULT, f"tests/test_repair_{index}.py"),
        verification_manifest_digest=check_manifest_digest(original, checks),
        project_instructions="Fixture protocol only; preserve scientific files.", owner_contract="AGENTS.md",
        existing_tests=("existing",), failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"broken\n").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest())
    return request, original, checks


def _real_check(spec, argv, **_kwargs):
    argv = [str(Path(sys.prefix) / Path(arg).relative_to("/runtime/0"))
            if arg == "/runtime/0" or arg.startswith("/runtime/0/") else arg for arg in argv]
    return run_bounded_process(argv, cwd=spec.workspace, interrupt_after_seconds=spec.timeout_seconds,
                               kill_grace_seconds=5, env={"PYTHONDONTWRITEBYTECODE": "1"})


def publish_repair(fixture, runtime, state, index, monkeypatch):
    from slm_training.harness_core import execution_release

    base = fixture.base
    candidate = fixture.directory / f"candidate-{index}"
    shutil.copytree(base, candidate)
    (candidate / FAULT).write_text(fault_program(index))
    (candidate / "tests").mkdir(exist_ok=True)
    regression = f"tests/test_repair_{index}.py"
    (candidate / regression).write_text(
        "from scripts.autotrain_fixture import VERSION\n"
        f"def test_restored():\n    assert VERSION >= {index}\n"
    )
    runtime.register(ActivitySpec(activity_id=f"verify-{index}", family="fixture", kind="control",
        source_digest=state.spec.source_digest, environment_digest=state.spec.environment_digest,
        input_digest=state.spec.input_digest, output_namespace=f"attempts/verify-{index}",
        grant=ResourceGrant()))
    started = time.monotonic()
    lease = runtime.claim_next(activity_id=f"verify-{index}", capabilities={"local_process"})
    request, original, checks = _request(runtime, lease, state, base, index,
                                        fixture.store.campaign_id, fixture.input_digest)
    proposal = RepairProposal(request_digest=request.digest(), tree_digest=manifest_digest(tree_manifest(candidate)),
        patch_digest=proposal_patch_digest(base, candidate), root_cause="controlled orchestration defect",
        regression_test=regression, reproduction_artifacts=(fixture.input_digest,), classification="implementation")
    spec = VerificationRequest(request.digest(), request.blocker.fingerprint(), request.blocker.source_digest,
        proposal.tree_digest, request.blocker.environment_digest, "a" * 64, request.grant.digest(),
        lease.token, request.allowed_paths, original, checks, 1, request.failure_stdout_sha256,
        request.failure_stderr_sha256, timeout_seconds=15, regression_test_path=regression)
    workspace = VerificationWorkspace(base, candidate, (Path(sys.prefix),))
    with monkeypatch.context() as patch:
        patch.setattr("slm_training.autoresearch.heal.repair_verifier.run_isolated", _real_check)
        patch.setattr("slm_training.autoresearch.heal.repair_regression.run_isolated", _real_check)
        gate = source_gate_fixture(fixture.directory / f"gate-{index}", patch, workspace)
        verified = verify_repair(request, proposal, spec, workspace=workspace, journal=fixture.store,
            fence_valid=lambda token: token == lease.token, source_verification=gate)
    assert verified.status == "verified", verified.reason
    previous = runtime.store.root / "source_release_pointer.json"
    prior = json.loads(previous.read_text())["publication_id"] if previous.exists() else None
    with monkeypatch.context() as patch:
        # Existing local-materialization fixture readback; no remote/main assertion.
        patch.setattr(execution_release, "_checkout_provenance", lambda _: {**PROVENANCE, "code_dirty": True})
        handoff = publish_verified_repair(request, verified, runtime=runtime, lease=lease,
            candidate=candidate, destinations=(fixture.directory / "accepted", fixture.directory / "outputs"),
            expected_previous=prior, authenticated=lambda supplied: supplied is verified)
    result = runtime.attempt_dir(lease) / "verification.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(verified.model_dump_json())
    runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
        outputs={result.name: hashlib.sha256(result.read_bytes()).hexdigest()}, spent_seconds=time.monotonic() - started)
    fixture.base = Path(json.loads(previous.read_text())["verified_source"])
    return handoff
