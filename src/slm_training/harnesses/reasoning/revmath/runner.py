"""Bounded deterministic revmath runner (HARN-03 / SLM-536).

Executes frozen ``RevmathTaskV1`` schemas through pinned Lean/project context
(or hermetic fixtures) with explicit budgets. Maps process + plugin evidence
through ``SolverJudgmentV1`` so timeout / missing tool / incomplete / unsupported
never become proof or refutation. Does not mutate proposition or campaign state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.formal.checkers import formal_process_budget
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    InvalidReason,
    JudgmentOutcome,
    RefutationPayloadV1,
    SolverJudgmentV1,
    UnknownReason,
    WitnessPayloadV1,
    classify,
)
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.plugins import (
    DEFAULT_LEAN_ROOT,
    PluginCheckEvidence,
    RevmathCheckPlan,
    RevmathTaskPlugin,
    default_plugin_registry,
    lean_tool_available,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.replay import (
    assert_revmath_replay_integrity,
    build_revmath_replay_bundle,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
    RevmathProofArtifactRefV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskKind,
    RevmathTaskV1,
)
from slm_training.models.decode_stats import (
    DecodeIdentityV1,
    DecodeStats,
    build_decode_stats_record,
)
from slm_training.runtime.telemetry.replay_bundle import ReplayBundleV1


def _identity_snapshot(task: RevmathTaskV1) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_identity_digest": task.identity_digest(),
        "proposition_id": task.proposition.proposition_id,
        "statement_sha256": task.proposition.statement_sha256,
        "base_theory_id": task.base_theory.base_theory_id,
        "allowed_assumption_ids": list(task.base_theory.allowed_assumption_ids),
        "campaign_id": task.campaign.campaign_id,
        "experiment_id": task.campaign.experiment_id,
        "campaign_manifest_sha256": task.campaign.campaign_manifest_sha256,
        "locked_arm_ids": list(task.campaign.locked_arm_ids),
        "task_canonical_hash": task.canonical_hash(),
    }


def _default_four_axis(*, incomplete: bool = False) -> RevmathFourAxisAnalysisV1:
    status = "unknown_axis" if incomplete else "not_claimed"

    def axis(name: str) -> AxisCertificateRefV1:
        return AxisCertificateRefV1(axis=name, status=status)  # type: ignore[arg-type]

    return RevmathFourAxisAnalysisV1(
        assumption_strength=axis("assumption_strength"),
        computability=axis("computability"),
        resource_bounds=axis("resource_bounds"),
        implementation_refinement=axis("implementation_refinement"),
    )


def _capture_dict(proc: BoundedProcessResult | None, *, note: str = "") -> dict[str, Any]:
    if proc is None:
        return {
            "outcome": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "stdout_sha256": content_sha(""),
            "stderr_sha256": content_sha(""),
            "duration_seconds": 0.0,
            "timed_out": False,
            "launch_error": None,
            "note": note,
        }
    return {
        "outcome": proc.outcome.value,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_sha256": content_sha(proc.stdout),
        "stderr_sha256": content_sha(proc.stderr),
        "duration_seconds": proc.duration_seconds,
        "timed_out": proc.timed_out,
        "interrupted": proc.interrupted,
        "killed": proc.killed,
        "launch_error": proc.launch_error,
        "stdout_truncated": proc.stdout_truncated,
        "stderr_truncated": proc.stderr_truncated,
        "command": list(proc.command),
        "note": note,
    }


def _judgment_from_process(
    proc: BoundedProcessResult | None,
    *,
    missing_tool: bool = False,
    unsupported: bool = False,
    evidence: PluginCheckEvidence | None = None,
) -> SolverJudgmentV1:
    """Map process + plugin evidence to KERN-01 judgment (fail closed)."""

    if unsupported:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.UNSUPPORTED_CAPABILITY,
            checked=False,
        )
    if missing_tool:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.MISSING_TOOL,
            checked=False,
        )
    if proc is None:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.INCONCLUSIVE,
            checked=False,
        )
    if proc.outcome is ProcessOutcome.LAUNCH_FAILED:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.MISSING_TOOL,
            checked=False,
        )
    if (
        proc.timed_out
        or proc.outcome
        in (
            ProcessOutcome.TIMED_OUT,
            ProcessOutcome.INTERRUPTED,
            ProcessOutcome.KILLED,
        )
    ):
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.TIMEOUT,
            checked=False,
        )

    if evidence is not None and evidence.malformed_proof:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.INVALID,
            payload=InvalidReason.MALFORMED_INPUT,
            checked=bool(evidence.checked),
        )
    if evidence is not None and evidence.incomplete:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.INCOMPLETE_COVERAGE,
            checked=bool(evidence.checked),
        )
    if evidence is not None and evidence.refuted:
        digest = evidence.refutation_digest or evidence.proof_sha256 or content_sha(
            {"stderr": proc.stderr, "stdout": proc.stdout}
        )
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.REFUTED,
            payload=RefutationPayloadV1(
                refutation_digest=digest,
                source="counterexample",
            ),
            checked=True,
        )
    if evidence is not None and evidence.proof_present and evidence.proof_sha256:
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.WITNESSED,
            payload=WitnessPayloadV1(
                witness_digest=evidence.proof_sha256,
                source="formal_preflight",
            ),
            checked=True,
        )

    # Completed without decisive evidence → unknown, never refuted.
    inputs = ClassificationInputsV1(
        incomplete_coverage=True,
        replay_checked=False,
        replay_ok=False,
    )
    return classify(inputs, checked=False)


def _result_id(task: RevmathTaskV1, judgment: SolverJudgmentV1) -> str:
    digest = content_sha(
        {
            "task_id": task.task_id,
            "task_identity_digest": task.identity_digest(),
            "judgment": judgment.to_dict(),
        }
    )
    return f"result.{task.task_id}.{digest[:12]}"


def _build_result(
    task: RevmathTaskV1,
    judgment: SolverJudgmentV1,
    evidence: PluginCheckEvidence | None,
) -> RevmathResultV1:
    proof = None
    if judgment.outcome in (JudgmentOutcome.WITNESSED, JudgmentOutcome.REFUTED):
        if evidence is None or not evidence.proof_sha256:
            raise RevmathSchemaError(
                "witnessed/refuted requires proof artifact (fail closed)"
            )
        proof = RevmathProofArtifactRefV1(
            proof_sha256=evidence.proof_sha256,
            checker_id=evidence.checker_id,
            checker_report_sha256=evidence.checker_report_sha256,
        )
    incomplete = judgment.outcome is JudgmentOutcome.UNKNOWN
    return RevmathResultV1(
        result_id=_result_id(task, judgment),
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=judgment.to_dict(),
        proof=proof,
        four_axis=_default_four_axis(incomplete=incomplete),
    )


@dataclass(frozen=True)
class RevmathRunRecord:
    """One deterministic run: result + replay + evidence envelopes."""

    task: RevmathTaskV1
    result: RevmathResultV1
    replay_bundle: ReplayBundleV1
    capture: dict[str, Any]
    identity_snapshot: dict[str, Any]
    decode_stats_record: dict[str, Any]
    repair_history_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "revmath_run_record/v1",
            "task": self.task.model_dump(mode="json"),
            "result": self.result.model_dump(mode="json"),
            "replay_bundle": self.replay_bundle.to_dict(),
            "capture": self.capture,
            "identity_snapshot": self.identity_snapshot,
            "decode_stats_record": self.decode_stats_record,
            "repair_history_ref": self.repair_history_ref,
        }


def _emit_decode_stats(
    task: RevmathTaskV1,
    result: RevmathResultV1,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    del capture  # capture digests live in the replay bundle final_evidence
    identity = DecodeIdentityV1(
        checkpoint_sha256=task.identity_digest(),
        pack_id=task.corpus.corpus_id,
        evaluator_version="revmath_runner/v1",
        artifact_sha256=task.canonical_hash(),
    )
    stats = DecodeStats()
    decisive = result.solver_judgment().outcome in (
        JudgmentOutcome.WITNESSED,
        JudgmentOutcome.REFUTED,
    )
    record = build_decode_stats_record(
        stats,
        identity=identity,
        measurement_stage="steady_state",
        completeness="complete" if decisive else "partial_timeout",
        legal_domain_size=1,
        legal_domain_status="complete",
        witness_ids=(
            (result.proof.proof_sha256,) if result.proof is not None else ()
        ),
    )
    return record.to_dict()


def run_revmath_task(
    task: RevmathTaskV1,
    *,
    lean_root: Path | None = None,
    hermetic: bool = True,
    plugin_registry: Mapping[RevmathTaskKind, RevmathTaskPlugin] | None = None,
    repair_history_ref: str | None = None,
    plan_override: RevmathCheckPlan | None = None,
    env: Mapping[str, str] | None = None,
) -> RevmathRunRecord:
    """Execute one frozen task; return typed result + deterministic replay.

    ``hermetic=True`` (default for CI) uses the committed fixture checker.
    Production Lean paths set ``hermetic=False`` and require lake/lean.
    """

    before = _identity_snapshot(task)
    root = (lean_root or DEFAULT_LEAN_ROOT).resolve()
    plugin = resolve_plugin(task, plugin_registry)
    if plugin is None:
        judgment = _judgment_from_process(None, unsupported=True)
        result = _build_result(task, judgment, None)
        capture = _capture_dict(None, note="unsupported_task_kind")
        bundle = build_revmath_replay_bundle(
            task=task,
            result=result,
            capture=capture,
            identity_snapshot=before,
            repair_history_ref=repair_history_ref,
        )
        assert_revmath_replay_integrity(bundle)
        after = _identity_snapshot(task)
        if after != before:
            raise RevmathSchemaError(
                "runner mutated task identity fields (proposition/campaign freeze)"
            )
        return RevmathRunRecord(
            task=task,
            result=result,
            replay_bundle=bundle,
            capture=capture,
            identity_snapshot=before,
            decode_stats_record=_emit_decode_stats(task, result, capture),
            repair_history_ref=repair_history_ref,
        )

    plan = plan_override or plugin.plan_check(task, lean_root=root, hermetic=hermetic)
    if plan.requires_lean_tool and not lean_tool_available(plan.lean_project_root or root):
        judgment = _judgment_from_process(None, missing_tool=True)
        result = _build_result(task, judgment, None)
        capture = _capture_dict(None, note="missing_lean_tool")
        bundle = build_revmath_replay_bundle(
            task=task,
            result=result,
            capture=capture,
            identity_snapshot=before,
            repair_history_ref=repair_history_ref,
        )
        assert_revmath_replay_integrity(bundle)
        return RevmathRunRecord(
            task=task,
            result=result,
            replay_bundle=bundle,
            capture=capture,
            identity_snapshot=before,
            decode_stats_record=_emit_decode_stats(task, result, capture),
            repair_history_ref=repair_history_ref,
        )

    _total, interrupt_after, grace = formal_process_budget(task.budget.max_wall_seconds)
    merged_env = dict(env) if env is not None else None
    proc = run_bounded_process(
        plan.command,
        interrupt_after_seconds=interrupt_after,
        kill_grace_seconds=grace,
        cwd=plan.cwd,
        env=merged_env,
    )
    evidence = plugin.interpret_capture(task, proc, plan=plan)
    judgment = _judgment_from_process(proc, evidence=evidence)
    result = _build_result(task, judgment, evidence)
    capture = _capture_dict(proc, note=evidence.detail)
    bundle = build_revmath_replay_bundle(
        task=task,
        result=result,
        capture=capture,
        identity_snapshot=before,
        repair_history_ref=repair_history_ref,
    )
    assert_revmath_replay_integrity(bundle)
    after = _identity_snapshot(task)
    if after != before:
        raise RevmathSchemaError(
            "runner mutated task identity fields (proposition/campaign freeze)"
        )
    return RevmathRunRecord(
        task=task,
        result=result,
        replay_bundle=bundle,
        capture=capture,
        identity_snapshot=before,
        decode_stats_record=_emit_decode_stats(task, result, capture),
        repair_history_ref=repair_history_ref,
    )


__all__ = [
    "RevmathRunRecord",
    "default_plugin_registry",
    "run_revmath_task",
]
