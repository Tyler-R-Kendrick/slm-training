"""Revmath task plugins (HARN-03/HARN-04/HARN-05).

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
from typing import Any, Literal, Protocol

from slm_training.harness_core.bounded_process import BoundedProcessResult
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.assumption_ablation import (
    AblationCandidateV1,
    AblationMetaV1,
    audit_hidden_reintroduction,
    parse_ablation_meta,
)
from slm_training.harnesses.reasoning.revmath.reversal import (
    ReversalMetaV1,
    ReversalObligationV1,
    audit_hidden_stronger,
    parse_reversal_meta,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskKind,
    RevmathTaskV1,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LEAN_ROOT = REPO_ROOT / "src" / "leverproof_lean"
FIXTURES_DIR = (
    REPO_ROOT / "src" / "slm_training" / "resources" / "revmath" / "fixtures"
)
HERMETIC_CHECKER = FIXTURES_DIR / "hermetic_checker.py"
HERMETIC_ABLATION_CHECKER = FIXTURES_DIR / "hermetic_ablation_checker.py"
HERMETIC_REVERSAL_CHECKER = FIXTURES_DIR / "hermetic_reversal_checker.py"

# Frozen fixture task_id prefix → sidecar meta filename stem.
_ABLATION_META_BY_TASK_PREFIX: Mapping[str, str] = {
    "task.ablation.positive": "ablation_positive",
    "task.ablation.redundant": "ablation_redundant",
    "task.ablation.necessary": "ablation_necessary",
    "task.ablation.timeout": "ablation_timeout",
    "task.ablation.hidden_import": "ablation_hidden_import",
}
_REVERSAL_META_BY_TASK_PREFIX: Mapping[str, str] = {
    "task.reversal.equivalence": "reversal_equivalence",
    "task.reversal.one_way_forward": "reversal_one_way_forward",
    "task.reversal.timeout": "reversal_timeout",
    "task.reversal.hidden_stronger": "reversal_hidden_stronger",
    "task.reversal.strength_mismatch": "reversal_strength_mismatch",
}


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


def load_ablation_meta_for_task(
    task: RevmathTaskV1,
    *,
    meta: AblationMetaV1 | None = None,
    fixtures_dir: Path | None = None,
) -> AblationMetaV1 | None:
    """Load frozen ablation sidecar meta for hermetic fixture tasks."""

    if meta is not None:
        return meta
    root = fixtures_dir or FIXTURES_DIR
    stem: str | None = None
    for prefix, name in _ABLATION_META_BY_TASK_PREFIX.items():
        if task.task_id == prefix or task.task_id.startswith(prefix + "."):
            stem = name
            break
    if stem is None:
        return None
    path = root / f"{stem}.meta.json"
    if not path.is_file():
        raise RevmathSchemaError(f"ablation meta missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevmathSchemaError(f"ablation meta at {path} must be an object")
    return parse_ablation_meta(payload)


def _ablation_removed_ids(
    task: RevmathTaskV1,
    meta: AblationMetaV1 | None,
) -> tuple[str, ...]:
    if meta is None:
        return ()
    remaining = set(task.base_theory.allowed_assumption_ids)
    return tuple(
        sorted(aid for aid in meta.baseline_assumption_ids if aid not in remaining)
    )


@dataclass(frozen=True)
class AssumptionAblationPlugin:
    """Hermetic assumption-ablation checks (HARN-04); plan/interpret only."""

    task_kind: RevmathTaskKind = "assumption_ablation"
    meta_override: AblationMetaV1 | None = None
    removed_override: tuple[str, ...] | None = None
    fixtures_dir: Path | None = None

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
            checker = HERMETIC_ABLATION_CHECKER
            if self.fixtures_dir is not None:
                checker = self.fixtures_dir / "hermetic_ablation_checker.py"
            if not checker.is_file():
                raise RevmathSchemaError(f"hermetic ablation checker missing at {checker}")
            meta = load_ablation_meta_for_task(
                task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
            )
            removed = (
                self.removed_override
                if self.removed_override is not None
                else _ablation_removed_ids(task, meta)
            )
            scenario = meta.hermetic_scenario if meta is not None else "positive"
            remaining = ",".join(sorted(task.base_theory.allowed_assumption_ids))
            removed_s = ",".join(removed)
            proof_deps = ""
            if meta is not None:
                key = ",".join(removed)
                deps = meta.proof_deps_by_removed.get(key)
                if deps is not None:
                    proof_deps = ",".join(deps)
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(checker),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--remaining-assumptions",
                    remaining,
                    "--removed-assumptions",
                    removed_s,
                    "--scenario",
                    scenario,
                    "--proof-deps",
                    proof_deps,
                    "--mode",
                    "auto",
                ),
                cwd=checker.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_ablation",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_ablation",
        )

    def interpret_capture(
        self,
        task: RevmathTaskV1,
        proc: BoundedProcessResult,
        *,
        plan: RevmathCheckPlan,
    ) -> PluginCheckEvidence:
        report_sha = content_sha(
            {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "outcome": proc.outcome.value,
            }
        )
        if plan.checker_id != "revmath.hermetic_ablation":
            if proc.returncode == 0:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=False,
                    incomplete=True,
                    checker_report_sha256=report_sha,
                    detail="lean ablation check ok; theorem witness deferred",
                )
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean ablation check failed or incomplete",
            )

        payload = _parse_checker_json(proc.stdout)
        if payload is None:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail="hermetic ablation checker emitted no JSON trailer",
            )

        deps_raw = payload.get("proof_dependency_ids") or []
        if isinstance(deps_raw, str):
            deps = tuple(
                sorted({p.strip() for p in deps_raw.split(",") if p.strip()})
            )
        else:
            deps = tuple(sorted(str(x) for x in deps_raw))

        meta = load_ablation_meta_for_task(
            task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
        )
        removed = (
            self.removed_override
            if self.removed_override is not None
            else _ablation_removed_ids(task, meta)
        )
        hidden: tuple[str, ...] = ()
        if meta is not None:
            hidden = audit_hidden_reintroduction(
                removed_assumption_ids=removed,
                proof_dependency_ids=deps,
                import_edges=meta.import_edges,
                strength_bearing_lemmas=meta.strength_bearing_lemmas,
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
            if hidden:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail=(
                        "hidden_reintroduction:"
                        + ",".join(hidden)
                        + f"; deps={','.join(deps)}"
                    ),
                )
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
            detail=f"unrecognized hermetic ablation status {status!r}",
        )



def load_reversal_meta_for_task(
    task: RevmathTaskV1,
    *,
    meta: ReversalMetaV1 | None = None,
    fixtures_dir: Path | None = None,
) -> ReversalMetaV1 | None:
    """Load frozen reversal sidecar meta for hermetic fixture tasks."""

    if meta is not None:
        return meta
    root = fixtures_dir or FIXTURES_DIR
    stem: str | None = None
    task_id = task.task_id
    # Obligation clones keep the parent stem prefix before ".obl.".
    lookup_id = task_id.split(".obl.", 1)[0]
    for prefix, name in _REVERSAL_META_BY_TASK_PREFIX.items():
        if lookup_id == prefix or lookup_id.startswith(prefix + "."):
            stem = name
            break
    if stem is None:
        return None
    path = root / f"{stem}.meta.json"
    if not path.is_file():
        raise RevmathSchemaError(f"reversal meta missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevmathSchemaError(f"reversal meta at {path} must be an object")
    return parse_reversal_meta(payload)


def _infer_reversal_direction(
    task: RevmathTaskV1,
    meta: ReversalMetaV1 | None,
    override: Literal["forward", "reverse"] | None,
) -> Literal["forward", "reverse"]:
    if override is not None:
        return override
    if task.task_id.endswith(".obl.forward") or ".obl.forward." in task.task_id:
        return "forward"
    if task.task_id.endswith(".obl.reverse") or ".obl.reverse." in task.task_id:
        return "reverse"
    if meta is not None:
        if task.proposition.statement_sha256 == meta.theorem.statement_sha256:
            return "forward"
        if task.proposition.statement_sha256 == meta.principle.statement_sha256:
            return "reverse"
    raise RevmathSchemaError(
        f"cannot infer reversal direction for task_id={task.task_id!r}"
    )


@dataclass(frozen=True)
class ReversalPlugin:
    """Hermetic bidirectional reversal checks (HARN-05); plan/interpret only."""

    task_kind: RevmathTaskKind = "reversal"
    meta_override: ReversalMetaV1 | None = None
    direction_override: Literal["forward", "reverse"] | None = None
    fixtures_dir: Path | None = None

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
            checker = HERMETIC_REVERSAL_CHECKER
            if self.fixtures_dir is not None:
                checker = self.fixtures_dir / "hermetic_reversal_checker.py"
            if not checker.is_file():
                raise RevmathSchemaError(
                    f"hermetic reversal checker missing at {checker}"
                )
            meta = load_reversal_meta_for_task(
                task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
            )
            direction = _infer_reversal_direction(task, meta, self.direction_override)
            scenario = meta.hermetic_scenario if meta is not None else "equivalence"
            coding_id = meta.coding_id if meta is not None else ""
            proof_deps = ""
            if meta is not None:
                deps = meta.proof_deps_by_direction.get(direction)
                if deps is not None:
                    proof_deps = ",".join(deps)
            antecedents = ",".join(sorted(task.base_theory.allowed_assumption_ids))
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(checker),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--direction",
                    direction,
                    "--base-theory-id",
                    task.base_theory.base_theory_id,
                    "--interpretation-status",
                    task.base_theory.interpretation_status,
                    "--coding-id",
                    coding_id,
                    "--antecedent-assumptions",
                    antecedents,
                    "--scenario",
                    scenario,
                    "--proof-deps",
                    proof_deps,
                    "--mode",
                    "auto",
                ),
                cwd=checker.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_reversal",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_reversal",
        )

    def interpret_capture(
        self,
        task: RevmathTaskV1,
        proc: BoundedProcessResult,
        *,
        plan: RevmathCheckPlan,
    ) -> PluginCheckEvidence:
        report_sha = content_sha(
            {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "outcome": proc.outcome.value,
            }
        )
        if plan.checker_id != "revmath.hermetic_reversal":
            if proc.returncode == 0:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=False,
                    incomplete=True,
                    checker_report_sha256=report_sha,
                    detail="lean reversal check ok; theorem witness deferred",
                )
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean reversal check failed or incomplete",
            )

        payload = _parse_checker_json(proc.stdout)
        if payload is None:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail="hermetic reversal checker emitted no JSON trailer",
            )

        deps_raw = payload.get("proof_dependency_ids") or []
        if isinstance(deps_raw, str):
            deps = tuple(
                sorted({p.strip() for p in deps_raw.split(",") if p.strip()})
            )
        else:
            deps = tuple(sorted(str(x) for x in deps_raw))

        meta = load_reversal_meta_for_task(
            task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
        )
        direction = _infer_reversal_direction(task, meta, self.direction_override)
        hidden: tuple[str, ...] = ()
        if meta is not None:
            forbidden = (
                (meta.theorem.statement_id,)
                if direction == "forward"
                else (meta.principle.statement_id,)
            )
            hidden = audit_hidden_stronger(
                direction=direction,
                proof_dependency_ids=deps,
                import_edges=meta.import_edges,
                strength_bearing_lemmas=meta.strength_bearing_lemmas,
                forbidden_assumption_ids=forbidden,
            )

        status = str(payload.get("status", ""))
        if payload.get("unsupported"):
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "unsupported direction")),
            )
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
            if hidden:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail=(
                        "hidden_stronger:"
                        + ",".join(hidden)
                        + f"; deps={','.join(deps)}"
                    ),
                )
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
            detail=f"unrecognized hermetic reversal status {status!r}",
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


def ablation_plugin_for_candidate(
    meta: AblationMetaV1,
    candidate: AblationCandidateV1,
) -> AssumptionAblationPlugin:
    """Plugin bound to one lattice candidate's removed set (tests / lattice)."""

    return AssumptionAblationPlugin(
        meta_override=meta,
        removed_override=candidate.removed_assumption_ids,
    )


def reversal_plugin_for_obligation(
    meta: ReversalMetaV1,
    obligation: ReversalObligationV1,
) -> ReversalPlugin:
    """Plugin bound to one reversal direction (tests / evaluate_reversal)."""

    return ReversalPlugin(
        meta_override=meta,
        direction_override=obligation.direction,
    )


_DEFAULT_PLUGINS: tuple[RevmathTaskPlugin, ...] = (
    HermeticForwardPlugin(),
    AssumptionAblationPlugin(),
    ReversalPlugin(),
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
    "FIXTURES_DIR",
    "HERMETIC_ABLATION_CHECKER",
    "HERMETIC_CHECKER",
    "HERMETIC_REVERSAL_CHECKER",
    "AssumptionAblationPlugin",
    "HermeticForwardPlugin",
    "PluginCheckEvidence",
    "ReversalPlugin",
    "RevmathCheckPlan",
    "RevmathTaskPlugin",
    "UnsupportedKindPlugin",
    "ablation_plugin_for_candidate",
    "default_plugin_registry",
    "lean_tool_available",
    "load_ablation_meta_for_task",
    "load_reversal_meta_for_task",
    "override_hermetic_mode",
    "register_plugin",
    "resolve_plugin",
    "reversal_plugin_for_obligation",
]
