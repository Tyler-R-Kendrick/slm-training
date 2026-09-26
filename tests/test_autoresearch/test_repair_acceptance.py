"""Real subprocess verification plumbing; OS isolation is simulated explicitly."""

import hashlib
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.harness_core.activity_contract import ResourceGrant
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.heal.recovery_dispatch import (
    RecoveryConfig,
    RecoveryContext,
    RepairRecipe,
    build_repair_request,
)
from slm_training.autoresearch.heal.repair_acceptance import (
    VerificationWorkspace,
    proposal_patch_digest,
    verify_repair,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairGrant,
    RepairProposal,
)
from slm_training.autoresearch.heal.repair_verifier import (
    VerificationCheck,
    VerificationRequest,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import KILL_GRACE_SECONDS


@pytest.mark.parametrize("restored", [True, False])
@pytest.mark.parametrize("through_seam", [False, True])
@pytest.mark.parametrize("source_available", [False, None, True])
def test_real_original_reproducer_controls_acceptance(
    tmp_path, monkeypatch, restored, through_seam, source_available
):
    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    for name in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / name).write_text("I6 fail closed; preserve original reproduction")
    (source / "fixture.py").write_text('print("broken")\nraise SystemExit(1)\n')
    candidate = tmp_path / "candidate"
    shutil.copytree(source, candidate)
    if restored:
        (candidate / "fixture.py").write_text('print("ready")\n')
    (candidate / "tests").mkdir()
    (candidate / "tests/test_added.py").write_text(
        "import subprocess, sys\n"
        "def test_added():\n"
        "    result = subprocess.run([sys.executable, 'fixture.py'], capture_output=True, text=True)\n"
        "    assert result.returncode == 0 and result.stdout == 'ready\\n'\n"
        "print('checked')\n"
    )
    sha = "a" * 64
    grant = RepairGrant(
        grant_id="fixture",
        provider="fixture",
        executable=sys.executable,
        executable_sha256=sha,
        expires_at=time.time() + 60,
        max_attempts=2,
        total_seconds=200.0,
        interrupt_seconds=10,
    )
    recipe = RepairRecipe(
        allowed_paths=("fixture.py", "tests/test_added.py"),
        input_digest=sha,
        original=VerificationCheck(
            "original", (sys.executable, "fixture.py"), "ready\n"
        ),
        checks=(
            VerificationCheck(
                "regression", (sys.executable, "tests/test_added.py"), "checked\n"
            ),
        ),
        owner_contract_path="AGENTS.md",
        failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"broken\n").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    config = RecoveryConfig(
        grant=grant, recipes={"harness_code_failure": recipe}, verifier_release=sha,
        source_verification_grant=ResourceGrant(),
        runtime_roots=(sys.prefix,),
    )
    context = RecoveryContext(
        tmp_path / "store",
        "loop",
        "campaign",
        source,
        manifest_digest(tree_manifest(source)),
        sha,
        "fence",
        "parent",
    )
    request = build_repair_request(
        {
            "kind": "repair_harness",
            "affected_activity_id": "original-evaluation",
            "blocker_code": "harness_code_failure",
            "unmet_predicate": "ready_stdout",
            "required_capability": "source_repair",
        },
        context,
        config,
    )
    proposal = RepairProposal(
        request_digest=request.digest(),
        tree_digest=manifest_digest(tree_manifest(candidate)),
        patch_digest=proposal_patch_digest(source, candidate),
        root_cause="fixture ordinary code defect",
        regression_test="tests/test_added.py",
        reproduction_artifacts=(sha,),
        classification="implementation",
    )
    spec = VerificationRequest(
        request.digest(),
        request.blocker.fingerprint(),
        context.source_digest,
        proposal.tree_digest,
        sha,
        sha,
        grant.digest(),
        context.fence,
        request.allowed_paths,
        recipe.original,
        recipe.checks,
        1,
        recipe.failure_stdout_sha256,
        recipe.failure_stderr_sha256,
        timeout_seconds=10,
        regression_test_path=proposal.regression_test,
    )
    if restored and not through_seam and source_available is True:
        request = request.model_copy(update={
            "grant": request.grant.model_copy(update={"expires_at": 1.0})
        })
        proposal = proposal.model_copy(update={"request_digest": request.digest()})
        spec = replace(spec, request_digest=request.digest(),
                       authority_digest=request.grant.digest())

    def real_process_without_isolation(spec, argv, **kwargs):
        argv = tuple(
            str(Path(sys.prefix) / Path(arg).relative_to("/runtime/0"))
            if arg == "/runtime/0" or arg.startswith("/runtime/0/")
            else arg
            for arg in argv
        )
        return run_bounded_process(
            argv,
            cwd=spec.workspace,
            interrupt_after_seconds=spec.timeout_seconds,
            kill_grace_seconds=KILL_GRACE_SECONDS,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.repair_verifier.run_isolated", real_process_without_isolation
    )
    monkeypatch.setattr(
        "slm_training.autoresearch.heal.repair_regression.run_isolated", real_process_without_isolation
    )
    if through_seam:
        result = _two_lease_dispatch(
            context, config, candidate, proposal, monkeypatch, source_available, restored
        )
    else:
        workspace = VerificationWorkspace(
            source, candidate, runtime_roots=(Path(sys.prefix),)
        )
        result = verify_repair(
            request,
            proposal,
            spec,
            workspace=workspace,
            journal=CampaignStore("campaign", tmp_path / "store"),
            fence_valid=lambda _: True,
            source_verification=source_gate_fixture(
                tmp_path, monkeypatch, workspace,
                complete=source_available is True and restored,
            ) if source_available is not False else None,
        )
    if not source_available:
        assert result.status == "waiting_verification"
        assert result.reason == ("source_verification_not_configured" if source_available is False else "source_verification_pending")
        assert result.verification is None
        events = CampaignStore("campaign", tmp_path / "store").verify_event_chain()
        assert any(row["event_type"] == "repair_source_verification_wait" for row in events)
        assert not any(row["event_type"] == "repair_verification_started" for row in events)
        return
    if not restored and source_available is True:
        assert result.status == "waiting_verification"
        assert result.reason == "source_verification_pending"
        assert result.verification is None
        return
    assert result.status == ("verified" if restored else "rejected"), result.reason
    assert result.verification.original_predicate_restored is restored
    journal = CampaignStore("campaign", tmp_path / "store")
    import json

    combined = json.loads((journal.root / "artifacts/repair_verification" / (result.verification.evidence_digest + ".json")).read_text())
    assert combined["source_release_authorized"] is restored
    assert combined["source_verification"]["verification_complete"]
    assert combined["candidate_snapshot_digest"] == proposal.tree_digest


from tests.test_autoresearch.repair_acceptance_helpers import (
    _two_lease_dispatch,
    source_gate_fixture,
)
