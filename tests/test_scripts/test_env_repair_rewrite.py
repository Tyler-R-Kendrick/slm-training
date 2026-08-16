"""SELF_HEAL_ENV_REPAIR: verified environment heal retires repair_harness.

The rewrite is the ``SELF_HEAL_THRASH_TIMEOUT_REPAIR`` precedent, never a
receipt ack: an environment-incomplete blocker (missing JS bridge / AgentV
install) with a **verified** heal receipt is reclassified to
``next_experiment``; code-class crashes and unverified heals stay hard.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from slm_training.autoresearch.heal import write_heal_receipt
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealAttemptReceiptV1,
    HealStepResultV1,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

_ENV_REASON = (
    "AgentV SDK is unavailable; run npm ci at the repo root "
    "(ERR_MODULE_NOT_FOUND Cannot find package 'typebox')"
)
_CODE_REASON = "No module named slm_training.harnesses.model_build.missing"


def _init_git_repo(path: Path) -> None:
    subprocess.check_call(["git", "init"], cwd=path, stdout=subprocess.DEVNULL)
    subprocess.check_call(
        ["git", "config", "user.email", "test@example.com"], cwd=path
    )
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=path)
    (path / "README.md").write_text("# test\n", encoding="utf-8")
    (path / "docs" / "design").mkdir(parents=True)
    (path / "docs" / "design" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md", "docs"], cwd=path)
    subprocess.check_call(
        ["git", "commit", "-m", "init"], cwd=path, stdout=subprocess.DEVNULL
    )


def _write_handoff(
    root: Path, *, loop_id: str, campaign_id: str, reason: str
) -> Path:
    camp = root / campaign_id
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "campaign.json").write_text(
        json.dumps(
            {"campaign_id": campaign_id, "loop_id": loop_id, "cycle_index": 90}
        ),
        encoding="utf-8",
    )
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 90,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "harness_failure",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["harness_failure"],
        "actions": [
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": reason,
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
                "frozen_manifest_sha256": "c" * 64,
            },
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
        "checkpoint_paths": [],
        "checkpoint_documentation_required": False,
    }
    path = camp / "cycle_handoff.json"
    path.write_text(json.dumps(handoff) + "\n", encoding="utf-8")
    return path


def _healed_receipt(
    loop_id: str, campaign_id: str, reason: str
) -> HealAttemptReceiptV1:
    digest = hashlib.sha256(b"probe-ok").hexdigest()
    probe = HealStepResultV1(
        step_id="verify",
        returncode=0,
        outcome="completed",
        stdout_sha256=digest,
        stderr_sha256=digest,
    )
    return HealAttemptReceiptV1(
        loop_id=loop_id,
        campaign_id=campaign_id,
        playbook_id="npm_bridges/v1",
        plan_sha256="d" * 64,
        blocker_fingerprint=blocker_fingerprint("repair_harness", reason),
        attempts_prior=0,
        step_results=(probe,),
        verify_result=probe,
        outcome="healed",
    )


@pytest.fixture()
def loop_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    return repo, root


def test_verified_env_heal_rewrites_to_next_experiment(
    loop_repo: tuple[Path, Path],
) -> None:
    repo, root = loop_repo
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c90"
    _write_handoff(root, loop_id=loop_id, campaign_id=campaign_id, reason=_ENV_REASON)
    write_heal_receipt(root, _healed_receipt(loop_id, campaign_id, _ENV_REASON))

    kind = _mod._self_heal_env_repair_rewrite(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind in {"env_repair_rewrite", "document_closeout"}
    rebuilt = json.loads((root / campaign_id / "cycle_handoff.json").read_text())
    kinds = [a["kind"] for a in rebuilt["actions"]]
    assert "repair_harness" not in kinds
    assert "next_experiment" in kinds
    assert any(
        r.startswith("self_heal:env_repair_verified:") for r in rebuilt["reasons"]
    )
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_no_heal_receipt_stays_hard(loop_repo: tuple[Path, Path]) -> None:
    repo, root = loop_repo
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c91"
    _write_handoff(root, loop_id=loop_id, campaign_id=campaign_id, reason=_ENV_REASON)

    kind = _mod._self_heal_env_repair_rewrite(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind is None
    rebuilt = json.loads((root / campaign_id / "cycle_handoff.json").read_text())
    assert "repair_harness" in [a["kind"] for a in rebuilt["actions"]]


def test_unverified_receipt_stays_hard(loop_repo: tuple[Path, Path]) -> None:
    repo, root = loop_repo
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c92"
    _write_handoff(root, loop_id=loop_id, campaign_id=campaign_id, reason=_ENV_REASON)
    receipt = _healed_receipt(loop_id, campaign_id, _ENV_REASON).model_copy(
        update={"outcome": "verify_failed"}
    )
    write_heal_receipt(root, receipt)

    kind = _mod._self_heal_env_repair_rewrite(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind is None


def test_code_class_crash_never_rewritten(loop_repo: tuple[Path, Path]) -> None:
    # `npm ci` cannot fix a repo-internal import error; even a (spurious)
    # healed receipt must not soften a code-class repair_harness.
    repo, root = loop_repo
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c93"
    _write_handoff(
        root, loop_id=loop_id, campaign_id=campaign_id, reason=_CODE_REASON
    )
    write_heal_receipt(root, _healed_receipt(loop_id, campaign_id, _CODE_REASON))

    kind = _mod._self_heal_env_repair_rewrite(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind is None
    rebuilt = json.loads((root / campaign_id / "cycle_handoff.json").read_text())
    assert "repair_harness" in [a["kind"] for a in rebuilt["actions"]]


def test_other_hard_actions_block_rewrite(loop_repo: tuple[Path, Path]) -> None:
    repo, root = loop_repo
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c94"
    path = _write_handoff(
        root, loop_id=loop_id, campaign_id=campaign_id, reason=_ENV_REASON
    )
    handoff = json.loads(path.read_text())
    handoff["actions"].append(
        {
            "kind": "repair_formal",
            "owner": "improve-lean-optimums",
            "reason": "theorem_backed_band_miss",
            "evidence_ids": [f"campaign:{campaign_id}"],
        }
    )
    path.write_text(json.dumps(handoff) + "\n", encoding="utf-8")
    write_heal_receipt(root, _healed_receipt(loop_id, campaign_id, _ENV_REASON))

    kind = _mod._self_heal_env_repair_rewrite(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind is None
