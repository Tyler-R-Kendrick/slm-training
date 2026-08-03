"""Lean-backed preflight contracts for autoresearch hypotheses.

Formal preflights prove structural implications under explicit assumptions.
They never predict empirical loss, accuracy, latency, or ship-gate outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.schemas import (
    ExperimentSpec,
    FormalClaimV1,
    FormalEvidenceScope,
    FormalObligationV1,
    FormalPreflightV1,
    FormalProofStatus,
    FormalTraceStepV1,
)
from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import (
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_RUN_SECONDS,
)
from slm_training.lineage.records import canonical_json

if TYPE_CHECKING:
    from slm_training.dsl.solver.closure import ClosureResult

REPO_ROOT = Path(__file__).resolve().parents[3]
LEAN_ROOT = Path(__file__).resolve().parents[1] / "formal" / "lean"
LEVERPROOF_ROOT = REPO_ROOT / "src" / "leverproof_lean"
_FORBIDDEN_PROOF_TOKENS = re.compile(r"\b(?:sorry|admit|axiom|unsafe|native_decide)\b")


@dataclass(frozen=True)
class _ProjectChecks:
    """Successful checks that can be reused by later claims in this process."""

    build: subprocess.CompletedProcess[str]
    audit: subprocess.CompletedProcess[str]
    version: subprocess.CompletedProcess[str]
    output: str
    proof_total: bool


# ``cmd formalize`` can carry several claims for the same pinned Lean project.
# Re-running ``make test`` for every claim is redundant, but a failed or stale
# check must never become sticky.  The key includes the complete proof-project
# digest and the runner identity (the latter keeps monkeypatched/test runners
# isolated).  This cache is intentionally process-local: no proof authority is
# persisted without the normal content-addressed preflight artifact.
_PROJECT_CHECK_CACHE: dict[tuple[str, str, object], _ProjectChecks] = {}


def clear_project_check_cache() -> None:
    """Clear the process-local Lean project-check cache (tests/tools)."""

    _PROJECT_CHECK_CACHE.clear()


@dataclass(frozen=True)
class FormalTemplate:
    template_id: str
    version: str
    theorem: str
    proof_target: str
    evidence_scope: FormalEvidenceScope
    status: FormalProofStatus
    assumptions: tuple[str, ...]
    open_assumptions: tuple[str, ...]
    source_paths: tuple[str, ...]
    lean_project: Literal["openui_proofs", "leverproof"] = "openui_proofs"
    proof_digest_scope: Literal["project_bundle", "template_sources"] = "project_bundle"
    checker_contract: str | None = None
    counterexample: dict[str, Any] | None = None


def _sff_template(template_id: str, theorem: str, *source_paths: str) -> FormalTemplate:
    return FormalTemplate(
        template_id=template_id,
        version="v1",
        theorem=theorem,
        proof_target="LeverProofLean.AdvisoryResidual",
        evidence_scope="universal",
        status="proved",
        assumptions=(
            "LeverProofLean Mathlib-free closed development",
            "AdvisoryResidual theorems certified via make proofs",
        ),
        open_assumptions=(),
        source_paths=(
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            *source_paths,
        ),
        lean_project="leverproof",
        proof_digest_scope="template_sources",
        checker_contract="make -C src/leverproof_lean proofs",
    )


FORMAL_TEMPLATES: dict[str, FormalTemplate] = {
    "sff.advisory-keys-legal": _sff_template(
        "sff.advisory-keys-legal",
        "LeverProofLean.AdvisoryResidual.filterLegal_subset",
        "src/slm_training/models/semantic_residual_scorer.py",
    ),
    "sff.advisory-singleton-zero-work": _sff_template(
        "sff.advisory-singleton-zero-work",
        "LeverProofLean.AdvisoryResidual.singleton_is_zero_work",
        "src/slm_training/models/semantic_residual_scorer.py",
    ),
    "sff.factor-membership-roundtrip": _sff_template(
        "sff.factor-membership-roundtrip",
        "LeverProofLean.AdvisoryResidual.reconstruct_encode_example",
        "src/slm_training/data/progspec/semantic_evidence.py",
    ),
    "sff.golden-incidence-degrees": _sff_template(
        "sff.golden-incidence-degrees",
        "LeverProofLean.AdvisoryResidual.golden_S_column_masses_from_B",
        "src/slm_training/resources/experiments/semantic_factor_frontier/golden_vectors.v1.json",
        "src/slm_training/models/semantic_factor_propagation.py",
    ),
    "sff.role-shuffle-membership": _sff_template(
        "sff.role-shuffle-membership",
        "LeverProofLean.AdvisoryResidual.role_shuffle_preserves_membership",
        "src/slm_training/data/progspec/semantic_evidence.py",
    ),
    "sff.soft-token-collision": _sff_template(
        "sff.soft-token-collision",
        "LeverProofLean.AdvisoryResidual.soft_token_collision",
        "src/slm_training/resources/experiments/semantic_factor_frontier/math_probes.v1.json",
    ),
    "metrics.structural_similarity_monotone": FormalTemplate(
        template_id="metrics.structural_similarity_monotone",
        version="v2",
        theorem="LeverProofLean.StructuralMetrics.structural_similarity_mono",
        proof_target="LeverProofLean.StructuralMetrics",
        evidence_scope="universal",
        status="proved",
        assumptions=(
            "the computed jaccard component does not decrease",
            "the computed depth-similarity component does not decrease",
        ),
        open_assumptions=(),
        source_paths=(
            "src/leverproof_lean/LeverProofLean/StructuralMetrics.lean",
            "src/slm_training/harnesses/model_build/eval_runner.py",
        ),
        lean_project="leverproof",
        checker_contract="make -C src/leverproof_lean test",
    ),
    "forest.history_preservation": FormalTemplate(
        template_id="forest.history_preservation",
        version="v1",
        theorem="OpenUIProofs.Trace.valid_trace_all_steps",
        proof_target="OpenUIProofs.Trace",
        evidence_scope="universal",
        status="proved",
        assumptions=(
            "every declared certified removal already passed certificate replay",
            "each history is a prefix of the next history",
            "adjacent trace states agree",
        ),
        open_assumptions=(),
        source_paths=(
            "src/slm_training/formal/lean/OpenUIProofs/Forest.lean",
            "src/slm_training/formal/lean/OpenUIProofs/Trace.lean",
            "src/slm_training/dsl/solver/closure.py",
        ),
        checker_contract="OpenUIProofs.Trace.validTrace/v1",
    ),
    "forest.lossy_history_counterexample": FormalTemplate(
        template_id="forest.lossy_history_counterexample",
        version="v1",
        theorem="OpenUIProofs.Forest.lossy_history_counterexample",
        proof_target="OpenUIProofs.Forest",
        evidence_scope="bounded_instance",
        status="refuted",
        assumptions=("history is reconstructed from only the current live set",),
        open_assumptions=(),
        source_paths=(
            "src/slm_training/formal/lean/OpenUIProofs/Forest.lean",
            "src/slm_training/dsl/solver/closure.py",
        ),
        counterexample={
            "initial": [0, 1],
            "removed": [],
            "original_history": ["decision 1"],
            "lossy_history": ["rollback 0"],
            "original_is_prefix": False,
        },
    ),
    "recurrence.layerscale_stability": FormalTemplate(
        template_id="recurrence.layerscale_stability",
        version="v1",
        theorem="OpenUIProofs.Recurrence.layerscale_bound",
        proof_target="OpenUIProofs.Recurrence",
        evidence_scope="conditional",
        status="conditional",
        assumptions=(
            "the update has the modeled delta form",
            "the declared layerscale bound holds",
        ),
        open_assumptions=(
            "establish a global contraction or task-local margin bound for the "
            "trained transition",
        ),
        source_paths=(
            "src/slm_training/formal/lean/OpenUIProofs/Recurrence.lean",
            "src/slm_training/models/recursive_denoiser.py",
        ),
    ),
}


def formal_obligation_id(
    campaign_id: str, experiment_id: str, claim: FormalClaimV1
) -> str:
    payload = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "template_id": claim.template_id,
        "claim": claim.claim,
        "policy": claim.policy,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"formal-{digest[:16]}"


def check_formal_trace(steps: tuple[FormalTraceStepV1, ...]) -> bool:
    """Executable JSON checker matching ``OpenUIProofs.Trace.validTrace``."""

    for index, step in enumerate(steps):
        if set(step.after_removed) != set(step.before_removed) | set(step.certified):
            return False
        if step.after_history[: len(step.before_history)] != step.before_history:
            return False
        if index:
            previous = steps[index - 1]
            if set(previous.after_removed) != set(step.before_removed):
                return False
            if previous.after_history != step.before_history:
                return False
    return True


def formal_trace_from_closure(
    result: ClosureResult,
) -> tuple[FormalTraceStepV1, ...]:
    """Project replay-checked closure deductions to stable trace ordinals."""

    ordinals: dict[str, int] = {}
    removed: set[int] = set()
    history: tuple[str, ...] = ()
    steps: list[FormalTraceStepV1] = []
    for deduction in result.deductions:
        certified: list[int] = []
        for value in deduction.removed:
            identity = canonical_json(
                {
                    "hole_id": deduction.hole_id.to_dict(),
                    "value": value.to_dict(),
                }
            )
            ordinal = ordinals.setdefault(identity, len(ordinals))
            certified.append(ordinal)
        before_removed = tuple(sorted(removed))
        before_history = history
        removed.update(certified)
        history = (*history, deduction.after_fingerprint)
        steps.append(
            FormalTraceStepV1(
                before_removed=before_removed,
                after_removed=tuple(sorted(removed)),
                certified=tuple(sorted(set(certified))),
                before_history=before_history,
                after_history=history,
            )
        )
    trace = tuple(steps)
    if not check_formal_trace(trace):
        raise ValueError("closure result does not satisfy the formal trace contract")
    return trace


def _digest_path(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"formal source does not exist: {relative_path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digests(template: FormalTemplate) -> dict[str, str]:
    return {path: _digest_path(path) for path in template.source_paths}


def _lean_root(template: FormalTemplate) -> Path:
    return LEVERPROOF_ROOT if template.lean_project == "leverproof" else LEAN_ROOT


def _proof_paths(template: FormalTemplate) -> tuple[tuple[str, Path], ...]:
    root = _lean_root(template)
    if template.lean_project == "leverproof":
        relative_paths = (
            "lakefile.toml",
            "lake-manifest.json",
            "lean-toolchain",
            "Makefile",
            "Main.lean",
            "LeverProofLean.lean",
            "Test/Proofs.lean",
            "Test/run.sh",
            *(
                str(path.relative_to(root))
                for path in sorted((root / "LeverProofLean").glob("*.lean"))
            ),
        )
        return (
            ("autoresearch/formal.py", Path(__file__).resolve()),
            *((relative, root / relative) for relative in relative_paths),
        )
    return (
        ("autoresearch/formal.py", Path(__file__).resolve()),
        ("lakefile.toml", root / "lakefile.toml"),
        ("lake-manifest.json", root / "lake-manifest.json"),
        ("lean-toolchain", root / "lean-toolchain"),
        ("OpenUIProofs.lean", root / "OpenUIProofs.lean"),
        *(
            (str(path.relative_to(root)), path)
            for path in sorted((root / "OpenUIProofs").glob("*.lean"))
        ),
    )


def _proof_digest(template: FormalTemplate) -> str:
    if template.proof_digest_scope == "template_sources":
        payload = {
            "theorem": template.theorem,
            "template_id": template.template_id,
            "source_digests": _source_digests(template),
            "lean_project": "src/leverproof_lean",
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for label, path in _proof_paths(template):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _project_digest(template: FormalTemplate) -> str:
    """Digest every file that can change the pinned Lean project check."""

    digest = hashlib.sha256()
    for label, path in _proof_paths(template):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mathlib_version(template: FormalTemplate) -> str:
    if template.lean_project == "leverproof":
        return "none"
    manifest = json.loads(
        (LEAN_ROOT / "lake-manifest.json").read_text(encoding="utf-8")
    )
    for package in manifest.get("packages", ()):
        if package.get("name") == "mathlib":
            return str(package.get("rev", "unknown"))
    return "unknown"


def _proof_sources_are_total(template: FormalTemplate) -> bool:
    root = _lean_root(template)
    paths = (
        (root / "LeverProofLean.lean",)
        + tuple(sorted((root / "LeverProofLean").glob("*.lean")))
        if template.lean_project == "leverproof"
        else (root / "OpenUIProofs.lean",)
        + tuple(sorted((root / "OpenUIProofs").glob("*.lean")))
    )
    return not any(_FORBIDDEN_PROOF_TOKENS.search(path.read_text()) for path in paths)


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    total = min(float(MAX_RUN_SECONDS), max(0.001, timeout_seconds))
    grace = min(float(KILL_GRACE_SECONDS), total * 0.1)
    interrupt_after = min(float(INTERRUPT_AFTER_SECONDS), max(0.001, total - grace))
    result = run_bounded_process(
        command,
        cwd=cwd,
        interrupt_after_seconds=interrupt_after,
        kill_grace_seconds=grace,
        on_start=on_start,
        on_heartbeat=on_heartbeat,
    )
    if result.timed_out:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=result.stdout,
            stderr=(
                f"{result.stderr}\nformal command timed out after {total:.3f}s"
            ).strip(),
        )
    if result.outcome is ProcessOutcome.LAUNCH_FAILED:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=result.launch_error or "formal command launch failed",
        )
    return subprocess.CompletedProcess(
        command,
        int(result.returncode or 0),
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _is_timeout_result(proc: subprocess.CompletedProcess[str]) -> bool:
    """True when a formal subprocess hit its wall (rc 124 or timeout stderr)."""
    if int(getattr(proc, "returncode", 0) or 0) == 124:
        return True
    err = str(getattr(proc, "stderr", "") or "")
    out = str(getattr(proc, "stdout", "") or "")
    return "timed out after" in err or "timed out after" in out


def _project_checks(
    template: FormalTemplate,
    *,
    remaining: Callable[[], float],
    on_start: Callable[[int], None] | None,
    on_heartbeat: Callable[[int], None] | None,
) -> _ProjectChecks:
    """Run or reuse the expensive project-wide Lean checks.

    Only a fully successful, total proof project is cached.  A timeout,
    compiler failure, failed axiom audit, or forbidden source token is always
    retried for the next claim rather than converted into durable negative
    authority.
    """

    key = (str(_lean_root(template).resolve()), _project_digest(template), _run)
    cached = _PROJECT_CHECK_CACHE.get(key)
    if cached is not None:
        return cached

    lean_root = _lean_root(template)
    build_command = (
        ["make", "test"]
        if template.lean_project == "leverproof"
        else ["lake", "build", "OpenUIProofs"]
    )
    build = _run(
        build_command,
        cwd=lean_root,
        timeout_seconds=remaining(),
        on_start=on_start,
        on_heartbeat=on_heartbeat,
    )
    build_timed_out = _is_timeout_result(build)
    audit = (
        subprocess.CompletedProcess([], 0, "", "")
        if template.lean_project == "leverproof" and build.returncode == 0
        else _run(
            ["lake", "env", "lean", "OpenUIProofs/Axioms.lean"],
            cwd=lean_root,
            timeout_seconds=remaining(),
            on_start=on_start,
            on_heartbeat=on_heartbeat,
        )
        if build.returncode == 0
        else subprocess.CompletedProcess(
            [],
            124 if build_timed_out else build.returncode,
            "",
            build.stderr or ("build timed out" if build_timed_out else "build failed"),
        )
    )
    audit_timed_out = build_timed_out or _is_timeout_result(audit)
    version = (
        _run(
            ["lake", "env", "lean", "--version"],
            cwd=lean_root,
            timeout_seconds=min(remaining(), 30.0),
            on_start=on_start,
            on_heartbeat=on_heartbeat,
        )
        if build.returncode == 0 and audit.returncode == 0
        else subprocess.CompletedProcess([], 1, "", "proof audit failed")
    )
    output = f"{build.stdout}\n{build.stderr}\n{audit.stdout}\n{audit.stderr}"
    proof_total = _proof_sources_are_total(template)
    result = _ProjectChecks(
        build=build,
        audit=audit,
        version=version,
        output=output,
        proof_total=proof_total,
    )
    if (
        build.returncode == 0
        and audit.returncode == 0
        and version.returncode == 0
        and not audit_timed_out
        and "sorryAx" not in output
        and proof_total
    ):
        _PROJECT_CHECK_CACHE[key] = result
    return result


def run_formal_preflight(
    campaign_id: str,
    experiment: ExperimentSpec,
    claim: FormalClaimV1,
    *,
    timeout_seconds: float = float(MAX_RUN_SECONDS),
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> tuple[FormalPreflightV1, FormalObligationV1]:
    """Build the pinned Lean project and emit a content-addressable result.

    ``timeout_seconds`` is caller-owned but repository-capped. A wall
    hit records status ``timed_out`` — incomplete measurement, not a proof
    rejection.
    """

    template = FORMAL_TEMPLATES.get(claim.template_id)
    if template is None:
        raise ValueError(f"unknown formal template: {claim.template_id}")
    if experiment.campaign_id != campaign_id:
        raise ValueError("formal claim belongs to a different campaign")
    started = time.monotonic()
    budget_seconds = min(float(MAX_RUN_SECONDS), max(0.001, float(timeout_seconds)))
    command_deadline = started + max(0.001, budget_seconds - 1.0)

    def remaining() -> float:
        return max(0.001, command_deadline - time.monotonic())

    checks = _project_checks(
        template,
        remaining=remaining,
        on_start=on_start,
        on_heartbeat=on_heartbeat,
    )
    build = checks.build
    audit = checks.audit
    version = checks.version
    output = checks.output
    proof_total = checks.proof_total
    build_timed_out = _is_timeout_result(build)
    audit_timed_out = build_timed_out or _is_timeout_result(audit)
    version_timed_out = _is_timeout_result(version)
    if audit_timed_out or build_timed_out or version_timed_out:
        status: FormalProofStatus = "timed_out"
    elif (
        build.returncode == 0
        and audit.returncode == 0
        and version.returncode == 0
        and "sorryAx" not in output
        and proof_total
    ):
        status = template.status
    else:
        status = "unknown"
    obligation_id = formal_obligation_id(campaign_id, experiment.experiment_id, claim)
    preflight = FormalPreflightV1(
        campaign_id=campaign_id,
        experiment_id=experiment.experiment_id,
        obligation_id=obligation_id,
        template_id=template.template_id,
        template_version=template.version,
        claim=claim.claim,
        policy=claim.policy,
        status=status,
        evidence_scope=template.evidence_scope,
        theorem=template.theorem,
        proof_target=template.proof_target,
        checker_contract=template.checker_contract,
        assumptions=template.assumptions,
        open_assumptions=template.open_assumptions,
        source_digests=_source_digests(template),
        proof_sha256=_proof_digest(template),
        lean_version=(version.stdout.strip() or version.stderr.strip() or "unknown"),
        mathlib_version=_mathlib_version(template),
        build_output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        counterexample=template.counterexample,
        duration_seconds=time.monotonic() - started,
    )
    obligation = FormalObligationV1(
        obligation_id=obligation_id,
        template_id=claim.template_id,
        policy=claim.policy,
        preflight_sha256="0" * 64,
    )
    return preflight, obligation


def bind_preflight(
    obligation: FormalObligationV1, preflight_sha256: str
) -> FormalObligationV1:
    return obligation.model_copy(update={"preflight_sha256": preflight_sha256})


def validate_formal_preflight_artifact(
    path: Path,
    *,
    campaign_id: str,
    experiment_id: str,
    claim: FormalClaimV1,
    expected_sha256: str,
) -> FormalPreflightV1:
    """Validate one cached preflight against its current claim and proof bundle."""

    obligation_id = formal_obligation_id(campaign_id, experiment_id, claim)
    preflight = FormalPreflightV1.model_validate_json(path.read_text(encoding="utf-8"))
    content_sha = hashlib.sha256(
        canonical_json(preflight.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    if content_sha != expected_sha256:
        raise ValueError(f"formal preflight digest mismatch: {obligation_id}")
    template = FORMAL_TEMPLATES.get(claim.template_id)
    if template is None:
        raise ValueError(f"unknown formal template: {claim.template_id}")
    if (
        preflight.obligation_id != obligation_id
        or preflight.campaign_id != campaign_id
        or preflight.experiment_id != experiment_id
        or preflight.template_id != claim.template_id
        or preflight.template_version != template.version
        or preflight.claim != claim.claim
        or preflight.policy != claim.policy
        or preflight.theorem != template.theorem
        or preflight.proof_target != template.proof_target
        or preflight.checker_contract != template.checker_contract
        or preflight.evidence_scope != template.evidence_scope
        or preflight.assumptions != template.assumptions
        or preflight.open_assumptions != template.open_assumptions
        or preflight.counterexample != template.counterexample
    ):
        raise ValueError(f"formal preflight binding mismatch: {obligation_id}")
    if preflight.source_digests != _source_digests(template):
        raise ValueError(f"formal preflight sources are stale: {obligation_id}")
    if preflight.proof_sha256 != _proof_digest(template):
        raise ValueError(f"formal proof bundle is stale: {obligation_id}")
    if claim.policy == "required" and preflight.status != "proved":
        raise ValueError(
            f"required formal claim is not proved: {claim.template_id} "
            f"({preflight.status})"
        )
    return preflight


def validate_formal_preflights(
    campaign_root: Path,
    experiment: ExperimentSpec,
    manifest: ExperimentCampaignV1,
) -> tuple[FormalPreflightV1, ...]:
    """Verify every lock-bound artifact and enforce the tiered gate."""

    claims_by_id = {
        formal_obligation_id(
            manifest.campaign_id, experiment.experiment_id, claim
        ): claim
        for claim in experiment.formal_claims
    }
    obligations_by_id = {
        obligation.obligation_id: obligation
        for obligation in manifest.formal_obligations
    }
    if set(claims_by_id) != set(obligations_by_id):
        raise ValueError(
            "campaign formal obligations do not exactly match experiment formal claims"
        )
    validated: list[FormalPreflightV1] = []
    for obligation_id, claim in claims_by_id.items():
        obligation = obligations_by_id[obligation_id]
        path = (
            campaign_root
            / "artifacts"
            / "formal_preflights"
            / f"{obligation.preflight_sha256}.json"
        )
        preflight = validate_formal_preflight_artifact(
            path,
            campaign_id=manifest.campaign_id,
            experiment_id=experiment.experiment_id,
            claim=claim,
            expected_sha256=obligation.preflight_sha256,
        )
        validated.append(preflight)
    return tuple(validated)
