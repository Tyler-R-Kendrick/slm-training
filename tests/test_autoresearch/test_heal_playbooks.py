"""Heal runner + playbook laws: discovery, scope, verify-decides, budgets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pydantic
import pytest

from tests.casefiles import case_values

from slm_training.autoresearch import heal
from slm_training.autoresearch.heal import run_playbooks
from slm_training.autoresearch.heal.escalation import (
    EscalationLedger,
    blocker_fingerprint,
)
from slm_training.autoresearch.heal.playbooks.npm_bridges import (
    PLAYBOOK as NPM_PLAYBOOK,
)
from slm_training.autoresearch.heal.playbooks.quarantine_dirt import (
    PLAYBOOK as QUARANTINE_PLAYBOOK,
    worktree_sentinel_path,
)
from slm_training.autoresearch.heal.schemas import (
    HealPlanV1,
    HealScopeError,
    HealStepV1,
    HealVerifyV1,
    plan_sha256,
    reject_forbidden_heal,
)


# Isolate from ambient Git config (gpgsign, hooksPath) so setup commits and
# stash behavior never depend on the developer machine or CI image.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["git", "init"], cwd=path, env=_GIT_ENV, stdout=subprocess.DEVNULL
    )
    subprocess.check_call(
        ["git", "config", "user.email", "t@example.com"], cwd=path, env=_GIT_ENV
    )
    subprocess.check_call(
        ["git", "config", "user.name", "t"], cwd=path, env=_GIT_ENV
    )
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "-A"], cwd=path, env=_GIT_ENV)
    subprocess.check_call(
        ["git", "commit", "-m", "init"],
        cwd=path,
        env=_GIT_ENV,
        stdout=subprocess.DEVNULL,
    )
    return path


class _FakePlaybook:
    """Test playbook with injectable plan."""

    playbook_id = "fake/v1"
    handles = frozenset({"environment"})

    def __init__(self, plan_factory) -> None:
        self._factory = plan_factory

    def matches(self, blocker: dict) -> bool:
        return True

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        return self._factory(blocker, cwd)


def _env_blocker(reason: str = "AgentV SDK is unavailable; npm ci") -> dict:
    return {
        "kind": "repair_harness",
        "campaign_id": "c1",
        "index": 0,
        "reason": reason,
        "blocker_class": "environment",
    }


def _plan(
    *,
    steps: tuple[HealStepV1, ...],
    verify_argv: tuple[str, ...] = ("true",),
    reason: str = "AgentV SDK is unavailable; npm ci",
) -> HealPlanV1:
    return HealPlanV1(
        playbook_id="fake/v1",
        blocker_fingerprint=blocker_fingerprint("repair_harness", reason),
        blocker_class="environment",
        steps=steps,
        verify=HealVerifyV1(argv=verify_argv, timeout_seconds=30),
    )


class TestDiscovery:
    def test_builtin_playbooks_discovered(self) -> None:
        ids = {p.playbook_id for p in heal.discovered_playbooks()}
        assert "npm_bridges/v1" in ids
        assert "quarantine_dirt/v1" in ids


class TestForbiddenScope:
    @pytest.mark.parametrize(
        "prefix",
        case_values(__file__, "test_frozen_surfaces_rejected"),
    )
    def test_frozen_surfaces_rejected(self, prefix: str) -> None:
        with pytest.raises(HealScopeError):
            reject_forbidden_heal((prefix,))

    def test_environment_prefixes_allowed(self) -> None:
        reject_forbidden_heal(("node_modules/", ".lake/", "package-lock.json"))

    @pytest.mark.parametrize(
        "prefix",
        case_values(__file__, "test_traversal_and_absolute_prefixes_rejected"),
    )
    def test_traversal_and_absolute_prefixes_rejected(self, prefix: str) -> None:
        # A traversal or absolute spelling must not alias a frozen surface
        # past the prefix comparison.
        with pytest.raises(HealScopeError):
            reject_forbidden_heal((prefix,))

    def test_plan_without_verify_fails_validation(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            HealPlanV1.model_validate(
                {
                    "playbook_id": "x/v1",
                    "blocker_fingerprint": "a" * 64,
                    "blocker_class": "environment",
                    "steps": [
                        {
                            "step_id": "s",
                            "argv": ["true"],
                            "timeout_seconds": 5,
                        }
                    ],
                }
            )


class TestRunnerLaws:
    def test_verify_decides_healed(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"

        ok = _FakePlaybook(
            lambda b, cwd: _plan(
                steps=(
                    HealStepV1(
                        step_id="noop",
                        argv=("true",),
                        timeout_seconds=30,
                    ),
                ),
                verify_argv=("true",),
            )
        )
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[ok],
        )
        assert [r.outcome for r in receipts] == ["healed"]
        assert receipts[0].verify_result is not None

    def test_step_success_alone_is_never_healed(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        bad_verify = _FakePlaybook(
            lambda b, cwd: _plan(
                steps=(
                    HealStepV1(
                        step_id="noop", argv=("true",), timeout_seconds=30
                    ),
                ),
                verify_argv=("false",),
            )
        )
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[bad_verify],
        )
        assert [r.outcome for r in receipts] == ["verify_failed"]

    def test_scope_violation_refused(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        naughty = _FakePlaybook(
            lambda b, cwd: _plan(
                steps=(
                    HealStepV1(
                        step_id="write_outside_allowlist",
                        argv=("touch", "unauthorized.txt"),
                        timeout_seconds=30,
                        writes_allowed=("node_modules/",),
                    ),
                ),
            )
        )
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[naughty],
        )
        assert [r.outcome for r in receipts] == ["refused_scope"]

    def test_forbidden_allowlist_yields_refused_scope(
        self, tmp_path: Path
    ) -> None:
        # A playbook naming a frozen surface is recorded and escalated —
        # never a runner escape that would discard earlier receipts.
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        frozen = _FakePlaybook(
            lambda b, cwd: _plan(
                steps=(
                    HealStepV1(
                        step_id="claims_source_scope",
                        argv=("true",),
                        timeout_seconds=30,
                        writes_allowed=("src/slm_training/",),
                    ),
                ),
            )
        )
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[frozen],
        )
        assert [r.outcome for r in receipts] == ["refused_scope"]
        ledger = EscalationLedger.load(root, "loop-1")
        record = next(iter(ledger.records.values()))
        assert record.status == "escalated"

    def test_crash_on_one_blocker_preserves_prior_receipts(
        self, tmp_path: Path
    ) -> None:
        # Incremental persistence: blocker B crashing must not discard
        # blocker A's receipt or attempt counts.
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"

        class _Exploding(_FakePlaybook):
            def plan(self, blocker: dict, *, cwd: Path):
                if "second" in str(blocker.get("reason")):
                    raise RuntimeError("boom")
                return _plan(
                    steps=(
                        HealStepV1(
                            step_id="noop", argv=("true",), timeout_seconds=30
                        ),
                    ),
                    verify_argv=("true",),
                )

        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[
                _env_blocker(),
                _env_blocker("second blocker: AgentV SDK is unavailable"),
            ],
            cwd=repo,
            playbooks=[_Exploding(None)],
        )
        assert [r.outcome for r in receipts] == ["healed", "step_failed"]
        persisted = heal.load_heal_receipts(root, "loop-1")
        assert [r.outcome for r in persisted] == ["healed", "step_failed"]

    def test_healed_plan_may_rerun_after_regression(
        self, tmp_path: Path
    ) -> None:
        # A deterministic repair that healed once must stay available when
        # the environment regresses — healed receipts are not cycles.
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        factory = lambda b, cwd: _plan(  # noqa: E731
            steps=(
                HealStepV1(step_id="noop", argv=("true",), timeout_seconds=30),
            ),
            verify_argv=("true",),
        )
        first = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[_FakePlaybook(factory)],
        )
        assert [r.outcome for r in first] == ["healed"]
        second = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c2",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[_FakePlaybook(factory)],
        )
        assert [r.outcome for r in second] == ["healed"]

    def test_cycle_detection_on_identical_plan(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        factory = lambda b, cwd: _plan(  # noqa: E731
            steps=(
                HealStepV1(step_id="noop", argv=("true",), timeout_seconds=30),
            ),
            verify_argv=("false",),
        )
        first = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[_FakePlaybook(factory)],
        )
        assert [r.outcome for r in first] == ["verify_failed"]
        second = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[_FakePlaybook(factory)],
        )
        assert [r.outcome for r in second] == ["cycle_detected"]

    def test_budget_exhaustion_escalates(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        counter = {"n": 0}

        def varying(b, cwd):
            counter["n"] += 1
            return _plan(
                steps=(
                    HealStepV1(
                        step_id=f"attempt_{counter['n']}",
                        argv=("true",),
                        timeout_seconds=30,
                    ),
                ),
                verify_argv=("false",),
            )

        for _ in range(3):
            receipts = run_playbooks(
                root=root,
                loop_id="loop-1",
                campaign_id="c1",
                blockers=[_env_blocker()],
                cwd=repo,
                playbooks=[_FakePlaybook(varying)],
                max_attempts_per_fingerprint=2,
            )
        assert [r.outcome for r in receipts] == ["budget_exhausted"]
        ledger = EscalationLedger.load(root, "loop-1")
        fingerprint = blocker_fingerprint(
            "repair_harness", _env_blocker()["reason"]
        )
        assert ledger.records[fingerprint].status == "escalated"

    def test_step_timeout_is_not_evidence_and_escalates(
        self, tmp_path: Path
    ) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        slow = _FakePlaybook(
            lambda b, cwd: _plan(
                steps=(
                    HealStepV1(
                        step_id="hangs",
                        argv=("sleep", "30"),
                        timeout_seconds=1,
                    ),
                ),
            )
        )
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[_env_blocker()],
            cwd=repo,
            playbooks=[slow],
        )
        # Kill is not evidence: neither healed nor a plain failure — an
        # immediate escalation with zero further retries.
        assert [r.outcome for r in receipts] == ["step_timeout"]
        ledger = EscalationLedger.load(root, "loop-1")
        fingerprint = blocker_fingerprint(
            "repair_harness", _env_blocker()["reason"]
        )
        assert ledger.records[fingerprint].status == "escalated"

    def test_no_matching_playbook_escalates(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        blocker = {
            "kind": "repair_formal",
            "campaign_id": "c1",
            "reason": "theorem_backed_band_miss",
            "blocker_class": "formal_contradiction",
        }
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[blocker],
            cwd=repo,
            playbooks=[NPM_PLAYBOOK],
        )
        assert receipts == ()
        ledger = EscalationLedger.load(root, "loop-1")
        record = next(iter(ledger.records.values()))
        assert record.status == "escalated"
        assert record.owner_skill == "improve-lean-optimums"


class TestNoAckSurface:
    def test_heal_package_cannot_write_action_receipts(self) -> None:
        # The heal layer may never acknowledge an action: the only legal
        # consumer of a healed receipt is the driver's emission-time rewrite.
        import slm_training.autoresearch.heal as heal_pkg
        import slm_training.autoresearch.heal.escalation as esc_mod
        import slm_training.autoresearch.heal.schemas as schemas_mod

        for module in (heal_pkg, esc_mod, schemas_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "append_autotrain_action_receipt" not in source, module
            assert "AutotrainActionReceiptV1(" not in source, module


class TestNpmPlaybook:
    def test_plan_targets_only_existing_package_roots(
        self, tmp_path: Path
    ) -> None:
        repo = _git_repo(tmp_path / "repo")
        (repo / "package.json").write_text("{}", encoding="utf-8")
        plan = NPM_PLAYBOOK.plan(_env_blocker(), cwd=repo)
        assert plan is not None
        assert [s.argv for s in plan.steps] == [("npm", "ci")]
        assert plan.steps[0].env_clears == ("NODE_OPTIONS",)
        assert plan.verify.argv[0] == "node"

    def test_plan_is_deterministic(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        (repo / "package.json").write_text("{}", encoding="utf-8")
        a = NPM_PLAYBOOK.plan(_env_blocker(), cwd=repo)
        b = NPM_PLAYBOOK.plan(_env_blocker(), cwd=repo)
        assert a is not None and b is not None
        assert plan_sha256(a) == plan_sha256(b)

    def test_verify_probes_named_missing_module(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        (repo / "package.json").write_text("{}", encoding="utf-8")
        plan = NPM_PLAYBOOK.plan(
            _env_blocker(
                "ERR_MODULE_NOT_FOUND Cannot find package 'typebox' "
                "imported from @agentv/core"
            ),
            cwd=repo,
        )
        assert plan is not None
        assert "typebox" in plan.verify.argv[-1]

    def test_no_package_roots_declines(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        assert NPM_PLAYBOOK.plan(_env_blocker(), cwd=repo) is None


class TestQuarantinePlaybook:
    def test_refuses_without_sentinel(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        blocker = {
            "kind": "foreign_dirty_tree",
            "reason": "non-closeout dirty paths: ['wip.py']",
            "_root": root,
            "_loop_id": "loop-1",
        }
        assert QUARANTINE_PLAYBOOK.plan(blocker, cwd=repo) is None

    def test_refuses_with_mismatched_sentinel(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        sentinel = worktree_sentinel_path(root, "loop-1")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("/somewhere/else/.git\n", encoding="utf-8")
        blocker = {
            "kind": "foreign_dirty_tree",
            "reason": "non-closeout dirty paths: ['wip.py']",
            "_root": root,
            "_loop_id": "loop-1",
        }
        assert QUARANTINE_PLAYBOOK.plan(blocker, cwd=repo) is None

    def test_quarantines_with_sentinel_and_records_clean_tree(
        self, tmp_path: Path
    ) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=repo,
            env=_GIT_ENV,
            text=True,
        ).strip()
        sentinel = worktree_sentinel_path(root, "loop-1")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(git_dir + "\n", encoding="utf-8")
        (repo / "wip.py").write_text("print('foreign')\n", encoding="utf-8")
        blocker = {
            "kind": "foreign_dirty_tree",
            "reason": "non-closeout dirty paths: ['wip.py']",
            "blocker_class": "dirty_tree",
            "_root": root,
            "_loop_id": "loop-1",
        }
        receipts = run_playbooks(
            root=root,
            loop_id="loop-1",
            campaign_id="c1",
            blockers=[blocker],
            cwd=repo,
            playbooks=[QUARANTINE_PLAYBOOK],
        )
        assert [r.outcome for r in receipts] == ["healed"]
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, env=_GIT_ENV, text=True
        )
        assert status.strip() == ""
        # The stash still holds the quarantined work (reversible by law).
        stashes = subprocess.check_output(
            ["git", "stash", "list"], cwd=repo, env=_GIT_ENV, text=True
        )
        assert "heal-quarantine:loop-1" in stashes

    def test_refuses_mid_merge(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path / "repo")
        root = tmp_path / "ar"
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=repo,
            env=_GIT_ENV,
            text=True,
        ).strip()
        sentinel = worktree_sentinel_path(root, "loop-1")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(git_dir + "\n", encoding="utf-8")
        (Path(git_dir) / "MERGE_HEAD").write_text("a" * 40, encoding="utf-8")
        blocker = {
            "kind": "foreign_dirty_tree",
            "reason": "non-closeout dirty paths: ['wip.py']",
            "_root": root,
            "_loop_id": "loop-1",
        }
        assert QUARANTINE_PLAYBOOK.plan(blocker, cwd=repo) is None
