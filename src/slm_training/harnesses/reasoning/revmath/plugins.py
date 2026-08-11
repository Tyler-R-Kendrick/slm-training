"""Revmath task plugins (HARN-03).

Plugins plan a bounded check command and interpret capture; the shared
runner owns orchestration, judgment classification, replay, and reports.
Task-kind logic must not fork a second orchestrator.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from slm_training.harness_core.bounded_process import BoundedProcessResult
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskKind,
    RevmathTaskV1,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LEAN_ROOT = REPO_ROOT / "src" / "leverproof_lean"
HERMETIC_CHECKER = (
    REPO_ROOT
    / "src"
    / "slm_training"
    / "resources"
    / "revmath"
    / "fixtures"
    / "hermetic_checker.py"
)


@dataclass(frozen=True)
class RevmathCheckPlan:
    """Pinned command + cwd for one bounded check."""

    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] | None = None
    requires_lean_tool: bool = False
    lean_project_root: Path | None = None
    checker_id: str = "revmath.plugin"


@dataclass(frozen=True)
class PluginCheckEvidence:
    """Plugin-local check evidence; runner maps this to SolverJudgmentV1."""

    checker_id: str
    checked: bool
    proof_present: bool = False
    proof_sha256: str | None = None
    checker_report_sha256: str | None = None
    malformed_proof: bool = False
    incomplete: bool = False
    refuted: bool = False
    refutation_digest: str | None = None
    detail: str = ""


class RevmathTaskPlugin(Protocol):
    """Task-kind seam — plan + interpret only."""

    task_kind: RevmathTaskKind

    def supports(self, task: RevmathTaskV1) -> bool: ...

    def plan_check(
        self,
        task: RevmathTaskV1,
        *,
        lean_root: Path,
        hermetic: bool,
    ) -> RevmathCheckPlan: ...

    def interpret_capture(
        self,
        task: RevmathTaskV1,
        proc: BoundedProcessResult,
        *,
        plan: RevmathCheckPlan,
    ) -> PluginCheckEvidence: ...


def lean_tool_available(lean_root: Path | None = None) -> bool:
    """True when a Lean/Lake toolchain is resolvable for the pinned project."""

    root = (lean_root or DEFAULT_LEAN_ROOT).resolve()
    if not root.is_dir():
        return False
    if shutil.which("lake") is None and shutil.which("lean") is None:
        return False
    return (root / "lakefile.toml").is_file() or (root / "lakefile.lean").is_file()


def _parse_checker_json(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    # Last non-empty line wins so Lake noise can precede a trailer JSON object.
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


@dataclass(frozen=True)
class HermeticForwardPlugin:
    """Hermetic forward_theorem checks via the committed fixture checker."""

    task_kind: RevmathTaskKind = "forward_theorem"

    def supports(self, task: RevmathTaskV1) -> bool:
        return task.task_kind == self.task_kind

    def plan_check(
        self,
        task: RevmathTaskV1,
        *,
        lean_root: Path,
        hermetic: bool,
    ) -> RevmathCheckPlan:
        if hermetic:
            if not HERMETIC_CHECKER.is_file():
                raise RevmathSchemaError(
                    f"hermetic checker missing at {HERMETIC_CHECKER}"
                )
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(HERMETIC_CHECKER),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--mode",
                    "ok",
                ),
                cwd=HERMETIC_CHECKER.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_forward",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_forward",
        )

    def interpret_capture(
        self,
        task: RevmathTaskV1,
        proc: BoundedProcessResult,
        *,
        plan: RevmathCheckPlan,
    ) -> PluginCheckEvidence:
        del task
        report_sha = content_sha(
            {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "outcome": proc.outcome.value,
            }
        )
        if plan.checker_id == "revmath.hermetic_forward":
            payload = _parse_checker_json(proc.stdout)
            if payload is None:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=False,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail="hermetic checker emitted no JSON trailer",
                )
            status = str(payload.get("status", ""))
            if status == "malformed":
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail=str(payload.get("detail", "malformed proof")),
                )
            if status == "incomplete":
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=False,
                    incomplete=True,
                    checker_report_sha256=report_sha,
                    detail=str(payload.get("detail", "incomplete check")),
                )
            if status == "refuted":
                digest = str(payload.get("refutation_digest") or report_sha)
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    refuted=True,
                    refutation_digest=digest,
                    proof_present=True,
                    proof_sha256=str(payload.get("proof_sha256") or digest),
                    checker_report_sha256=report_sha,
                    detail=str(payload.get("detail", "refuted")),
                )
            if status == "ok":
                proof_sha = payload.get("proof_sha256")
                if not isinstance(proof_sha, str) or len(proof_sha) != 64:
                    return PluginCheckEvidence(
                        checker_id=plan.checker_id,
                        checked=True,
                        malformed_proof=True,
                        checker_report_sha256=report_sha,
                        detail="ok status missing proof_sha256",
                    )
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    proof_present=True,
                    proof_sha256=proof_sha,
                    checker_report_sha256=report_sha,
                    detail=str(payload.get("detail", "ok")),
                )
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail=f"unrecognized hermetic status {status!r}",
            )

        # Non-hermetic lean path: make test success is project-ok, not a
        # task-specific witness — leave incomplete so unknown stays unknown
        # until HARN-04+ task validators attach real theorem evidence.
        if proc.returncode == 0:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean project check ok; task-level witness deferred",
            )
        return PluginCheckEvidence(
            checker_id=plan.checker_id,
            checked=False,
            incomplete=True,
            checker_report_sha256=report_sha,
            detail="lean project check failed or incomplete",
        )


@dataclass(frozen=True)
class UnsupportedKindPlugin:
    """Explicit unsupported marker for kinds without a validator yet."""

    task_kind: RevmathTaskKind

    def supports(self, task: RevmathTaskV1) -> bool:
        return False

    def plan_check(
        self,
        task: RevmathTaskV1,
        *,
        lean_root: Path,
        hermetic: bool,
    ) -> RevmathCheckPlan:
        del task, lean_root, hermetic
        raise RevmathSchemaError(f"task kind {self.task_kind!r} has no check plan")

    def interpret_capture(
        self,
        task: RevmathTaskV1,
        proc: BoundedProcessResult,
        *,
        plan: RevmathCheckPlan,
    ) -> PluginCheckEvidence:
        del task, proc, plan
        raise RevmathSchemaError(f"task kind {self.task_kind!r} cannot interpret capture")


_DEFAULT_PLUGINS: tuple[RevmathTaskPlugin, ...] = (
    HermeticForwardPlugin(),
    UnsupportedKindPlugin("assumption_ablation"),
    UnsupportedKindPlugin("reversal"),
    UnsupportedKindPlugin("constructivization"),
    UnsupportedKindPlugin("computable_finite_counterexample"),
    UnsupportedKindPlugin("quantitative_bound_extraction"),
    UnsupportedKindPlugin("computability_classification"),
)


def default_plugin_registry() -> dict[RevmathTaskKind, RevmathTaskPlugin]:
    return {plugin.task_kind: plugin for plugin in _DEFAULT_PLUGINS}


def resolve_plugin(
    task: RevmathTaskV1,
    registry: Mapping[RevmathTaskKind, RevmathTaskPlugin] | None = None,
) -> RevmathTaskPlugin | None:
    table: Mapping[RevmathTaskKind, RevmathTaskPlugin] = (
        registry if registry is not None else default_plugin_registry()
    )
    plugin = table.get(task.task_kind)
    if plugin is None or not plugin.supports(task):
        return None
    return plugin


def register_plugin(
    registry: MutableMapping[RevmathTaskKind, RevmathTaskPlugin],
    plugin: RevmathTaskPlugin,
) -> None:
    registry[plugin.task_kind] = plugin


def override_hermetic_mode(plan_command: Sequence[str], mode: str) -> tuple[str, ...]:
    """Rewrite ``--mode`` for focused hermetic fixture tests."""

    argv = list(plan_command)
    if "--mode" in argv:
        idx = argv.index("--mode")
        if idx + 1 < len(argv):
            argv[idx + 1] = mode
            return tuple(argv)
    return tuple([*argv, "--mode", mode])


__all__ = [
    "DEFAULT_LEAN_ROOT",
    "HERMETIC_CHECKER",
    "HermeticForwardPlugin",
    "PluginCheckEvidence",
    "RevmathCheckPlan",
    "RevmathTaskPlugin",
    "UnsupportedKindPlugin",
    "default_plugin_registry",
    "lean_tool_available",
    "override_hermetic_mode",
    "register_plugin",
    "resolve_plugin",
]
