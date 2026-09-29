"""Readonly-release staging and actual native apply_patch; never inference."""

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.heal import isolation_workspace
from slm_training.autoresearch.heal.isolated_agent import BubblewrapAgentRunner, _create_regression_mounts
from slm_training.autoresearch.heal.isolation import IsolationSpec, run_isolated
from slm_training.autoresearch.heal.isolation_workspace import (
    IsolationViolation, manifest_digest, private_snapshot, tree_manifest,
)
from slm_training.autoresearch.heal.repair_acceptance import proposal_patch_digest
from slm_training.autoresearch.heal.repair_contracts import RepairProposal
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck, VerificationRequest, verify_candidate
from tests.test_autoresearch.test_repair_dispatch import repair_request as base_request


@pytest.fixture
def readonly_source(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    for name in (".agents", ".codex", "tests"):
        (root / name).mkdir()
    (root / "broken.py").write_text('VALUE = 0\n')
    (root / "protected.py").write_text('PROTECTED = True\n')
    (root / "tests/test_original.py").write_text('assert True\n')
    for path in root.rglob("*.py"):
        path.chmod(0o444)
    return root


def test_private_staging_changes_only_granted_owner_write_bits(readonly_source, tmp_path):
    original = tree_manifest(readonly_source)
    candidate = private_snapshot(readonly_source, tmp_path / "candidate")
    (candidate / "executable.py").write_text("pass\n")
    (candidate / "executable.py").chmod(0o555)
    _create_regression_mounts(candidate, ("broken.py", "executable.py", "tests/test_added.py"))
    assert stat.S_IMODE((candidate / "broken.py").stat().st_mode) == 0o644
    assert stat.S_IMODE((candidate / "executable.py").stat().st_mode) == 0o755
    assert stat.S_IMODE((candidate / "protected.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((candidate / "tests/test_original.py").stat().st_mode) == 0o444
    assert tree_manifest(readonly_source) == original


@pytest.mark.parametrize("fault", case_values(__file__, "test_entire_grant_validated_before_any_mode_change"))
def test_entire_grant_validated_before_any_mode_change(readonly_source, tmp_path, fault):
    candidate = private_snapshot(readonly_source, tmp_path / "candidate")
    bad = "bad.py"
    if fault == "symlink":
        (candidate / bad).symlink_to("protected.py")
    elif fault == "parent_symlink":
        (candidate / "tests/alias").symlink_to(tmp_path, target_is_directory=True)
        bad = "tests/alias/test_outside.py"
    elif fault == "hardlink":
        os.link(candidate / "protected.py", candidate / bad)
    elif fault == "directory":
        (candidate / bad).mkdir()
    elif fault == "traversal":
        bad = "../escape.py"
    elif fault == "file_ancestor":
        bad = "tests/test_original.py/test_new.py"
    else:
        bad = "tests/test_original.py"
    with pytest.raises((ValueError, IsolationViolation)):
        _create_regression_mounts(candidate, ("broken.py", bad))
    assert stat.S_IMODE((candidate / "broken.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((candidate / "protected.py").stat().st_mode) == 0o444


def test_frozen_digest_is_checked_before_trusted_mode_preparation(readonly_source, tmp_path, monkeypatch):
    request = base_request.__wrapped__(tmp_path)
    request = request.model_copy(update={"allowed_paths": ("broken.py",)})
    runner = BubblewrapAgentRunner(source=readonly_source, attempt_root=tmp_path / "attempts")
    monkeypatch.setattr(runner, "capability", lambda _: None)
    monkeypatch.setattr("slm_training.autoresearch.heal.isolated_agent._create_regression_mounts",
                        lambda *a: pytest.fail("prepared modes before authenticating source"))
    with pytest.raises(ValueError, match="differs from pinned"):
        runner.run(request, ("fixture",), inputs={}, progress=lambda: None, cancelled=lambda: False)
    assert stat.S_IMODE((readonly_source / "broken.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((runner.attempt_root / request.digest() / "candidate/broken.py").stat().st_mode) == 0o444


def native_binary():
    native = os.environ.get("SLM_NATIVE_CODEX")
    if not native:
        pytest.skip("installed native Codex path not configured")
    return Path(native).resolve()


PATCH = "*** Begin Patch\n*** Update File: /workspace/broken.py\n@@\n-VALUE = 0\n+VALUE = 1\n*** End Patch"
REGRESSION = "*** Begin Patch\n*** Add File: /workspace/tests/test_added.py\n+from broken import VALUE\n+def test_value():\n+    assert VALUE == 1\n*** End Patch"


def test_native_patch_on_unprepared_0444_fails_without_changing_source(readonly_source, tmp_path):
    native = native_binary()
    candidate = private_snapshot(readonly_source, tmp_path / "unprepared")
    before = tree_manifest(candidate)
    result = run_isolated(IsolationSpec(candidate, writable_paths=("broken.py",),
                          runtime_roots=(native.parent.parent,), timeout_seconds=60, codex_mount_target=True),
                         ("/runtime/0/bin/codex", "sandbox", "-c", 'sandbox_mode="workspace-write"',
                          "--", "/runtime/0/bin/codex", "--codex-run-as-apply-patch", PATCH))
    assert result.returncode != 0 and "Failed to write file" in result.stderr
    assert (candidate / "broken.py").read_text() == "VALUE = 0\n"
    assert tree_manifest(readonly_source) == before
    (tmp_path / "unprepared-native-evidence.json").write_text(json.dumps({
        "returncode": result.returncode, "outcome": result.outcome.value,
        "stderr": result.stderr, "source_mode": oct(stat.S_IMODE(before["broken.py"][0])),
        "frozen_source_unchanged": True, "inference": False,
    }, sort_keys=True))


def test_native_patch_prepared_modes_match_worker_digest_and_independent_judge(readonly_source, tmp_path):
    native = native_binary()
    helpers = tmp_path / "readonly-helpers"
    helpers.mkdir()
    # Reuse the canonical manifest implementation inside the diagnostic worker.
    shutil.copyfile(isolation_workspace.__file__, helpers / "isolation_workspace.py")
    original = tree_manifest(readonly_source)
    request = base_request.__wrapped__(tmp_path)
    request = request.model_copy(update={
        "allowed_paths": ("broken.py", "tests/test_added.py"),
        "grant": request.grant.model_copy(update={"executable": str(native), "interrupt_seconds": 60,
            "executable_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}),
        "blocker": request.blocker.model_copy(update={"source_digest": manifest_digest(original)})})
    runner = BubblewrapAgentRunner(source=readonly_source, attempt_root=tmp_path / "attempts",
                                  runtime_roots=(native.parent.parent, helpers))
    script = '''import json, subprocess, sys
from pathlib import Path
for patch in sys.argv[1:]:
    subprocess.run(['/runtime/0/bin/codex', '--codex-run-as-apply-patch', patch], check=True)
for path in ('protected.py', 'tests/test_original.py', 'outside.py',
             'repair-input/prepared-manifest.json', 'repair-input/proposal-digest.py'):
    try:
        Path(path).write_text('not allowed')
    except OSError:
        pass
    else:
        raise AssertionError('off-scope write succeeded: ' + path)
digests = json.loads(subprocess.check_output(
    ['/usr/bin/python3', '-I', '-S', 'repair-input/proposal-digest.py'], text=True))
Path('repair-output/proposal.json').write_text(json.dumps({
    'worker_tree_digest': digests['tree_digest'],
    'worker_patch_digest': digests['patch_digest']}))
print('NATIVE_PATCH_AND_SCOPE_OK')
'''
    result = runner.run(request, (str(native), "sandbox", "-c", 'sandbox_mode="workspace-write"',
                        "--", "/usr/bin/python3", "-c", script, PATCH, REGRESSION),
                        inputs={"repair-instructions.json": {
                            "digest_contract": "Hash the candidate.",
                            "request": request.model_dump(mode="json"),
                        }},
                        progress=lambda: None, cancelled=lambda: False)
    attempt = runner.attempt_root / request.digest()
    assert result.outcome == "completed" and result.returncode == 0, (attempt / "execution/stderr.txt").read_text()
    candidate = attempt / "candidate"
    after = tree_manifest(candidate)
    digest_instructions = json.loads((attempt / "input/repair-instructions.json").read_text())["digest_contract"]
    assert "/usr/bin/python3 -I -S repair-input/proposal-digest.py" in digest_instructions
    assert "use its tree_digest and patch_digest verbatim" in digest_instructions
    assert "Never edit or reimplement it" in digest_instructions
    assert hashlib.sha256((attempt / "input/proposal-digest.py").read_bytes()).hexdigest() == hashlib.sha256(
        Path(isolation_workspace.__file__).read_bytes()
    ).hexdigest()
    assert stat.S_IMODE((attempt / "input/proposal-digest.py").stat().st_mode) == 0o444
    assert json.loads((attempt / "input/baseline-manifest.json").read_text())["broken.py"][0] == original["broken.py"][0]
    assert tree_manifest(readonly_source) == original
    assert after["broken.py"][0] == (original["broken.py"][0] | stat.S_IWUSR)
    assert after["protected.py"] == original["protected.py"]
    worker_digest = json.loads(result.final_json)["worker_tree_digest"]
    assert worker_digest == manifest_digest(after)
    proposal = RepairProposal(request_digest=request.digest(), tree_digest=worker_digest,
        patch_digest=json.loads(result.final_json)["worker_patch_digest"], root_cause="diagnostic fixture increment",
        regression_test="tests/test_added.py", reproduction_artifacts=(request.digest(),), classification="implementation")
    check = VerificationCheck("original", ("/usr/bin/python3", "-c",
        "import sys; from broken import VALUE; print('ready' if VALUE == 1 else 'broken'); sys.exit(0 if VALUE == 1 else 1)"), "ready\n")
    regression = VerificationCheck("regression", ("/usr/bin/python3", "-c",
        "exec(open('tests/test_added.py').read()); test_value(); print('checked')"), "checked\n")
    spec = VerificationRequest(request.digest(), request.blocker.fingerprint(), request.blocker.source_digest,
        proposal.tree_digest, request.blocker.environment_digest, "a" * 64, request.grant.digest(), "fixture-fence",
        request.allowed_paths, check, (regression,), 1, hashlib.sha256(b"broken\n").hexdigest(),
        hashlib.sha256(b"").hexdigest(), timeout_seconds=20,
        regression_test_path=proposal.regression_test)
    evidence = verify_candidate(spec, readonly_source, candidate)
    assert evidence.accepted, evidence
    assert evidence.changed_paths == ("broken.py", "tests/test_added.py")
    assert proposal.tree_digest == manifest_digest(tree_manifest(candidate))
    assert proposal.patch_digest == proposal_patch_digest(readonly_source, candidate)
    (tmp_path / "prepared-native-evidence.json").write_text(json.dumps({
        "tree_digest": proposal.tree_digest, "patch_digest": proposal.patch_digest,
        "source_digest": request.blocker.source_digest,
        "original_mode": oct(stat.S_IMODE(original["broken.py"][0])),
        "candidate_mode": oct(stat.S_IMODE(after["broken.py"][0])),
        "independent_accepted": evidence.accepted, "changed_paths": evidence.changed_paths,
        "frozen_source_unchanged": True, "inference": False,
    }, sort_keys=True))
    for protected in (".agents", ".codex", "protected.py"):
        target = candidate / protected
        old_mode = target.stat().st_mode
        target.chmod(stat.S_IMODE(old_mode) ^ stat.S_IWUSR)
        with pytest.raises(IsolationViolation, match="source/candidate identity mismatch"):
            verify_candidate(spec, readonly_source, candidate)
        modified = replace(spec, candidate_digest=manifest_digest(tree_manifest(candidate)))
        with pytest.raises(IsolationViolation, match="exceed exact repair grant"):
            verify_candidate(modified, readonly_source, candidate)
        target.chmod(stat.S_IMODE(old_mode))
