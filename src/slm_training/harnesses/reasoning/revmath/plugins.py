"""Revmath task plugins (HARN-03/HARN-04/HARN-05/HARN-06/HARN-07).

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
from slm_training.harnesses.reasoning.revmath.quantitative_bound import (
    QuantitativeBoundMetaV1,
    parse_quantitative_bound_meta,
)
from slm_training.harnesses.reasoning.revmath.constructivization import (
    ConstructivizationMetaV1,
    parse_constructivization_meta,
)
from slm_training.harnesses.reasoning.revmath.counterexample import (
    CounterexampleMetaV1,
    parse_counterexample_meta,
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
HERMETIC_QUANT_CHECKER = FIXTURES_DIR / "hermetic_quantitative_bound_checker.py"
HERMETIC_CONSTRUCTIVIZATION_CHECKER = FIXTURES_DIR / "hermetic_constructivization_checker.py"
HERMETIC_COUNTEREXAMPLE_CHECKER = FIXTURES_DIR / "hermetic_counterexample_checker.py"

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
_QUANT_META_BY_TASK_PREFIX: Mapping[str, str] = {
    "task.quant.finite_search": "quant_finite_search",
    "task.quant.closure_live_upper": "quant_closure_live_upper",
    "task.quant.nonextractable": "quant_nonextractable",
}

_CONSTRUCTIVIZATION_META_BY_TASK_PREFIX: Mapping[str, str] = {
    "task.constructivization.bounded": "constructivization_bounded",
    "task.constructivization.witness": "constructivization_witness",
    "task.constructivization.oracle": "constructivization_oracle",
    "task.constructivization.remainder": "constructivization_remainder",
    "task.constructivization.timeout": "constructivization_timeout",
}
_COUNTEREXAMPLE_META_BY_TASK_PREFIX: Mapping[str, str] = {
    "task.counterexample.checked": "counterexample_checked",
    "task.counterexample.search_failed": "counterexample_search_failed",
    "task.counterexample.no_counterexample": "counterexample_no_counterexample",
    "task.counterexample.mismatch_prop": "counterexample_mismatch_prop",
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




def load_quantitative_bound_meta_for_task(
    task: RevmathTaskV1,
    *,
    meta: QuantitativeBoundMetaV1 | None = None,
    fixtures_dir: Path | None = None,
) -> QuantitativeBoundMetaV1 | None:
    """Load frozen quantitative-bound sidecar meta for hermetic fixtures."""

    if meta is not None:
        return meta
    root = fixtures_dir or FIXTURES_DIR
    stem: str | None = None
    for prefix, name in _QUANT_META_BY_TASK_PREFIX.items():
        if task.task_id == prefix or task.task_id.startswith(prefix + "."):
            stem = name
            break
    if stem is None:
        return None
    path = root / f"{stem}.meta.json"
    if not path.is_file():
        raise RevmathSchemaError(f"quantitative-bound meta missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevmathSchemaError(f"quantitative-bound meta at {path} must be an object")
    return parse_quantitative_bound_meta(payload)


@dataclass(frozen=True)
class QuantitativeBoundPlugin:
    """Hermetic quantitative-bound extraction checks (HARN-07); plan/interpret only."""

    task_kind: RevmathTaskKind = "quantitative_bound_extraction"
    meta_override: QuantitativeBoundMetaV1 | None = None
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
            checker = HERMETIC_QUANT_CHECKER
            if self.fixtures_dir is not None:
                checker = self.fixtures_dir / "hermetic_quantitative_bound_checker.py"
            if not checker.is_file():
                raise RevmathSchemaError(
                    f"hermetic quantitative-bound checker missing at {checker}"
                )
            meta = load_quantitative_bound_meta_for_task(
                task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
            )
            if meta is None:
                raise RevmathSchemaError(
                    f"no quantitative-bound meta for task_id={task.task_id!r}"
                )
            stem: str | None = None
            for prefix, name in _QUANT_META_BY_TASK_PREFIX.items():
                if task.task_id == prefix or task.task_id.startswith(prefix + "."):
                    stem = name
                    break
            if stem is None and self.meta_override is not None:
                for name in _QUANT_META_BY_TASK_PREFIX.values():
                    candidate = (self.fixtures_dir or FIXTURES_DIR) / f"{name}.meta.json"
                    if not candidate.is_file():
                        continue
                    loaded = parse_quantitative_bound_meta(
                        json.loads(candidate.read_text(encoding="utf-8"))
                    )
                    if loaded.meta_id == self.meta_override.meta_id:
                        stem = name
                        break
            if stem is None:
                raise RevmathSchemaError(
                    f"cannot resolve meta path for task_id={task.task_id!r}"
                )
            meta_path = (self.fixtures_dir or FIXTURES_DIR) / f"{stem}.meta.json"
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(checker),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--meta-path",
                    str(meta_path),
                    "--scenario",
                    meta.hermetic_scenario,
                    "--mode",
                    "auto",
                ),
                cwd=checker.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_quantitative_bound",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_quantitative_bound",
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
        if plan.checker_id != "revmath.hermetic_quantitative_bound":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean quantitative-bound check deferred to hermetic/EVID-04 path",
            )

        payload = _parse_checker_json(proc.stdout)
        if payload is None:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail="hermetic quantitative-bound checker emitted no JSON trailer",
            )

        status = str(payload.get("status", ""))
        if status == "malformed":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "malformed")),
            )
        if status == "incomplete":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "incomplete")),
            )
        if status == "nonextractable":
            digest = str(payload.get("proof_sha256") or report_sha)
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                proof_present=True,
                proof_sha256=digest,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "nonextractable")),
            )
        if status == "ok":
            if payload.get("empirical_timing_claimed"):
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail="empirical_timing_claimed must be false for theorem-derived bounds",
                )
            if not payload.get("theorem_derived_bound"):
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail="ok status requires theorem_derived_bound=true",
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
            detail=f"unrecognized hermetic quantitative-bound status {status!r}",
        )


def load_constructivization_meta_for_task(
    task: RevmathTaskV1,
    *,
    meta: ConstructivizationMetaV1 | None = None,
    fixtures_dir: Path | None = None,
) -> ConstructivizationMetaV1 | None:
    if meta is not None:
        return meta
    root = fixtures_dir or FIXTURES_DIR
    stem: str | None = None
    for prefix, name in _CONSTRUCTIVIZATION_META_BY_TASK_PREFIX.items():
        if task.task_id == prefix or task.task_id.startswith(prefix + "."):
            stem = name
            break
    if stem is None:
        return None
    path = root / f"{stem}.meta.json"
    if not path.is_file():
        raise RevmathSchemaError(f"constructivization meta missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevmathSchemaError(f"constructivization meta at {path} must be an object")
    return parse_constructivization_meta(payload)


def load_counterexample_meta_for_task(
    task: RevmathTaskV1,
    *,
    meta: CounterexampleMetaV1 | None = None,
    fixtures_dir: Path | None = None,
) -> CounterexampleMetaV1 | None:
    if meta is not None:
        return meta
    root = fixtures_dir or FIXTURES_DIR
    stem: str | None = None
    for prefix, name in _COUNTEREXAMPLE_META_BY_TASK_PREFIX.items():
        if task.task_id == prefix or task.task_id.startswith(prefix + "."):
            stem = name
            break
    if stem is None:
        return None
    path = root / f"{stem}.meta.json"
    if not path.is_file():
        raise RevmathSchemaError(f"counterexample meta missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RevmathSchemaError(f"counterexample meta at {path} must be an object")
    return parse_counterexample_meta(payload)


@dataclass(frozen=True)
class ConstructivizationPlugin:
    """Hermetic constructivization checks (HARN-06); plan/interpret only."""

    task_kind: RevmathTaskKind = "constructivization"
    meta_override: ConstructivizationMetaV1 | None = None
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
            checker = HERMETIC_CONSTRUCTIVIZATION_CHECKER
            if self.fixtures_dir is not None:
                checker = self.fixtures_dir / "hermetic_constructivization_checker.py"
            if not checker.is_file():
                raise RevmathSchemaError(
                    f"hermetic constructivization checker missing at {checker}"
                )
            meta = load_constructivization_meta_for_task(
                task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
            )
            scenario = meta.hermetic_scenario if meta is not None else "bounded_ok"
            original_sha = (
                meta.original_statement_sha256 if meta is not None else ""
            )
            constructivized_sha = (
                meta.constructivized_statement_sha256
                if meta is not None
                else task.proposition.statement_sha256
            )
            weakening = meta.weakening_description if meta is not None else ""
            bound_ast = meta.bound_ast_id if meta is not None and meta.bound_ast_id else ""
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(checker),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--original-statement-sha256",
                    original_sha,
                    "--constructivized-statement-sha256",
                    constructivized_sha,
                    "--scenario",
                    scenario,
                    "--weakening-description",
                    weakening,
                    "--bound-ast-id",
                    bound_ast,
                    "--mode",
                    "auto",
                ),
                cwd=checker.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_constructivization",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_constructivization",
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
        if plan.checker_id != "revmath.hermetic_constructivization":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean constructivization check incomplete; witness deferred",
            )
        payload = _parse_checker_json(proc.stdout)
        if payload is None:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail="hermetic constructivization checker emitted no JSON trailer",
            )
        status = str(payload.get("status", ""))
        if status in ("malformed", "masquerade"):
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", status)),
            )
        if status == "incomplete":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "incomplete check")),
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
            detail=f"unrecognized hermetic constructivization status {status!r}",
        )


@dataclass(frozen=True)
class CounterexamplePlugin:
    """Hermetic computable/finite counterexample checks (HARN-06)."""

    task_kind: RevmathTaskKind = "computable_finite_counterexample"
    meta_override: CounterexampleMetaV1 | None = None
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
            checker = HERMETIC_COUNTEREXAMPLE_CHECKER
            if self.fixtures_dir is not None:
                checker = self.fixtures_dir / "hermetic_counterexample_checker.py"
            if not checker.is_file():
                raise RevmathSchemaError(
                    f"hermetic counterexample checker missing at {checker}"
                )
            meta = load_counterexample_meta_for_task(
                task, meta=self.meta_override, fixtures_dir=self.fixtures_dir
            )
            scenario = (
                meta.hermetic_scenario if meta is not None else "search_failed"
            )
            search_status = meta.search_status if meta is not None else "unknown"
            model_digest = ""
            model_kind = ""
            model_payload = ""
            model_target_sha = ""
            model_target_assumps = ""
            if meta is not None and meta.model is not None:
                model_digest = meta.model.model_digest
                model_kind = meta.model.model_kind
                model_payload = meta.model.payload
                model_target_sha = meta.model.target_statement_sha256
                model_target_assumps = ",".join(meta.model.target_assumption_ids)
            assumptions = ",".join(sorted(task.base_theory.allowed_assumption_ids))
            return RevmathCheckPlan(
                command=(
                    "python",
                    "-u",
                    str(checker),
                    "--task-id",
                    task.task_id,
                    "--statement-sha256",
                    task.proposition.statement_sha256,
                    "--assumptions",
                    assumptions,
                    "--scenario",
                    scenario,
                    "--model-digest",
                    model_digest,
                    "--model-kind",
                    model_kind,
                    "--model-payload",
                    model_payload,
                    "--model-target-statement-sha256",
                    model_target_sha,
                    "--model-target-assumptions",
                    model_target_assumps,
                    "--search-status",
                    search_status,
                    "--mode",
                    "auto",
                ),
                cwd=checker.parent,
                requires_lean_tool=False,
                lean_project_root=None,
                checker_id="revmath.hermetic_counterexample",
            )
        root = lean_root.resolve()
        return RevmathCheckPlan(
            command=("make", "test"),
            cwd=root,
            requires_lean_tool=True,
            lean_project_root=root,
            checker_id="revmath.lean_counterexample",
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
        if plan.checker_id != "revmath.hermetic_counterexample":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail="lean counterexample check incomplete; witness deferred",
            )
        payload = _parse_checker_json(proc.stdout)
        if payload is None:
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail="hermetic counterexample checker emitted no JSON trailer",
            )
        status = str(payload.get("status", ""))
        if status == "malformed":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                malformed_proof=True,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "malformed")),
            )
        if status == "incomplete":
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=False,
                incomplete=True,
                checker_report_sha256=report_sha,
                detail=str(
                    payload.get(
                        "detail",
                        "search failed; unknown (not refutation)",
                    )
                ),
            )
        if status == "refuted":
            digest = str(payload.get("refutation_digest") or payload.get("proof_sha256") or "")
            if len(digest) != 64:
                return PluginCheckEvidence(
                    checker_id=plan.checker_id,
                    checked=True,
                    malformed_proof=True,
                    checker_report_sha256=report_sha,
                    detail="refuted status missing refutation digest",
                )
            return PluginCheckEvidence(
                checker_id=plan.checker_id,
                checked=True,
                refuted=True,
                refutation_digest=digest,
                proof_present=True,
                proof_sha256=digest,
                checker_report_sha256=report_sha,
                detail=str(payload.get("detail", "refuted")),
            )
        return PluginCheckEvidence(
            checker_id=plan.checker_id,
            checked=False,
            incomplete=True,
            checker_report_sha256=report_sha,
            detail=f"unrecognized hermetic counterexample status {status!r}",
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
    QuantitativeBoundPlugin(),
    ConstructivizationPlugin(),
    CounterexamplePlugin(),
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
    "HERMETIC_QUANT_CHECKER",
    "AssumptionAblationPlugin",
    "CounterexamplePlugin",
    "ConstructivizationPlugin",
    "HermeticForwardPlugin",
    "PluginCheckEvidence",
    "ReversalPlugin",
    "QuantitativeBoundPlugin",
    "RevmathCheckPlan",
    "RevmathTaskPlugin",
    "UnsupportedKindPlugin",
    "ablation_plugin_for_candidate",
    "default_plugin_registry",
    "lean_tool_available",
    "load_ablation_meta_for_task",
    "load_counterexample_meta_for_task",
    "load_constructivization_meta_for_task",
    "HERMETIC_COUNTEREXAMPLE_CHECKER",
    "HERMETIC_CONSTRUCTIVIZATION_CHECKER",
    "load_reversal_meta_for_task",
    "load_quantitative_bound_meta_for_task",
    "override_hermetic_mode",
    "register_plugin",
    "resolve_plugin",
    "reversal_plugin_for_obligation",
]
