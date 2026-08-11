"""INTEG-05 / SLM-571: route revmath failures through VerifierWitnessV1.

Adapter only — does not invent a parallel failure store, never overrides
``SolverJudgmentV1`` / checker authority, and never turns failed/unknown
judgments into proof success. Localizes revmath-shaped failures into the
existing ``evals.semantic_failure`` witness taxonomy and feeds immutable
HARN-09 before/after repair refs into the existing disposition surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal

from slm_training.evals.semantic_failure import (
    TAXONOMY_VERSION,
    WITNESS_SCHEMA_VERSION,
    SemanticFailureFamily,
    VerifierLocalizationV1,
    VerifierWitnessV1,
    seal_verifier_witness,
    witness_integrity_ok,
)
from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.mechanism_disposition_report import (
    build_mechanism_disposition_record,
)
from slm_training.harnesses.experiments.slm160_spv_disposition import Disposition
from slm_training.harnesses.reasoning.revmath.runner import RevmathRunRecord
from slm_training.harnesses.reasoning.revmath.schemas import (
    REVMATH_MUTABLE_REPAIR_KNOBS,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
)
from slm_training.harnesses.reasoning.revmath.self_healing import (
    RevmathRepairSessionV1,
    semantic_repair_lineage_from_session,
)

REVMATH_FAILURE_TAXONOMY_VERSION = "revmath_failure_taxonomy/v1"
EVIDENCE_SCHEMA = "revmath_failure_evidence/v1"
FEEDBACK_SCHEMA = "revmath_repair_disposition_feedback/v1"
EVALUATOR_VERSION = "revmath_failure_witness/v1"

CompletenessClass = Literal["EXACT", "HEURISTIC", "UNKNOWN"]
_ALLOWED_COMPLETENESS = frozenset({"EXACT", "HEURISTIC", "UNKNOWN"})


class RevmathFailureClass(str, Enum):
    """Closed localization classes for revmath-shaped failures (INTEG-05)."""

    MISSING_NECESSARY_ASSUMPTION_CANDIDATE = "missing_necessary_assumption_candidate"
    FAILED_REVERSAL_DIRECTION = "failed_reversal_direction"
    NONCONSTRUCTIVE_REMAINDER = "nonconstructive_remainder"
    QUANTITATIVE_BOUND_UNAVAILABLE = "quantitative_bound_unavailable"
    CHECKED_COUNTEREXAMPLE = "checked_counterexample"
    CHECKER_TOOL_ENVIRONMENT_ERROR = "checker_tool_environment_error"
    BLOCKED_SEMANTIC_MUTATION = "blocked_semantic_mutation"
    UNLOCALIZED = "unlocalized"


# Map into the existing OpenUI semantic-failure family vocabulary so HARN-09
# repair preferences stay on one taxonomy — never a second failure enum owner.
_FAMILY_BY_CLASS: Mapping[RevmathFailureClass, SemanticFailureFamily] = {
    RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE: (
        SemanticFailureFamily.PROMPT_CONTRACT_INVENTORY
    ),
    RevmathFailureClass.FAILED_REVERSAL_DIRECTION: (
        SemanticFailureFamily.SEARCH_OR_PRUNING_REGRET
    ),
    RevmathFailureClass.NONCONSTRUCTIVE_REMAINDER: (
        SemanticFailureFamily.SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE
    ),
    RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE: (
        SemanticFailureFamily.SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE
    ),
    RevmathFailureClass.CHECKED_COUNTEREXAMPLE: (
        SemanticFailureFamily.DUPLICATE_FILLER_OR_METRIC_GAMING
    ),
    RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR: (
        SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE
    ),
    RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION: SemanticFailureFamily.UNKNOWN,
    RevmathFailureClass.UNLOCALIZED: SemanticFailureFamily.UNKNOWN,
}

# Bounded mutable-knob suggestions only — empty where repair would change
# frozen identity or invent missing mathematical content.
_ACTIONS_BY_CLASS: Mapping[RevmathFailureClass, tuple[str, ...]] = {
    RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE: (),
    RevmathFailureClass.FAILED_REVERSAL_DIRECTION: (
        "proof_search_strategy",
        "tactic_sequence",
        "retrieval_choices",
    ),
    RevmathFailureClass.NONCONSTRUCTIVE_REMAINDER: ("temporary_decomposition",),
    RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE: (),
    RevmathFailureClass.CHECKED_COUNTEREXAMPLE: (),
    RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR: (
        "serialization_bridge_generation",
        "resource_allocation_within_locked_budget",
    ),
    RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION: (),
    RevmathFailureClass.UNLOCALIZED: (
        "tactic_sequence",
        "proof_search_strategy",
        "retrieval_choices",
    ),
}


@dataclass(frozen=True)
class BoundedRepairActionV1:
    """Advisory mutable-knob suggestion — never changes authority."""

    knob: str
    rationale: str
    failure_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundedRepairActionV1:
        knob = str(payload.get("knob", ""))
        if knob not in REVMATH_MUTABLE_REPAIR_KNOBS:
            raise RevmathSchemaError(
                f"bounded repair action knob {knob!r} is not mutable; "
                f"allowed={sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
            )
        return cls(
            knob=knob,
            rationale=str(payload.get("rationale", "")),
            failure_class=str(payload.get("failure_class", "")),
        )


@dataclass(frozen=True)
class RevmathFailureEvidenceV1:
    """Queryable/replayable revmath failure routed through VerifierWitnessV1."""

    schema_version: str
    failure_taxonomy_version: str
    failure_class: str
    semantic_failure_family: str
    completeness_class: str
    proposition_id: str
    task_id: str
    task_identity_digest: str
    checker_id: str | None
    judgment_outcome: str
    judgment_checked: bool
    lineage: dict[str, Any]
    unresolved: tuple[str, ...]
    witness: VerifierWitnessV1
    repair_action_suggestions: tuple[BoundedRepairActionV1, ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "failure_taxonomy_version": self.failure_taxonomy_version,
            "failure_class": self.failure_class,
            "semantic_failure_family": self.semantic_failure_family,
            "completeness_class": self.completeness_class,
            "proposition_id": self.proposition_id,
            "task_id": self.task_id,
            "task_identity_digest": self.task_identity_digest,
            "checker_id": self.checker_id,
            "judgment_outcome": self.judgment_outcome,
            "judgment_checked": self.judgment_checked,
            "lineage": dict(self.lineage),
            "unresolved": list(self.unresolved),
            "witness": self.witness.to_dict(),
            "repair_action_suggestions": [
                a.to_dict() for a in self.repair_action_suggestions
            ],
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RevmathFailureEvidenceV1:
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            raise RevmathSchemaError(
                f"unsupported revmath failure evidence schema: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("failure_taxonomy_version") != REVMATH_FAILURE_TAXONOMY_VERSION:
            raise RevmathSchemaError(
                f"unsupported revmath failure taxonomy: "
                f"{payload.get('failure_taxonomy_version')!r}"
            )
        completeness = payload.get("completeness_class")
        if completeness not in _ALLOWED_COMPLETENESS:
            raise RevmathSchemaError(
                f"unsupported completeness_class: {completeness!r}"
            )
        failure_class = str(payload.get("failure_class", ""))
        if failure_class not in {c.value for c in RevmathFailureClass}:
            raise RevmathSchemaError(f"unknown revmath failure class: {failure_class!r}")
        witness = VerifierWitnessV1.from_dict(dict(payload["witness"]))
        if not witness_integrity_ok(witness):
            raise RevmathSchemaError("embedded verifier witness digest mismatch")
        evidence = cls(
            schema_version=EVIDENCE_SCHEMA,
            failure_taxonomy_version=REVMATH_FAILURE_TAXONOMY_VERSION,
            failure_class=failure_class,
            semantic_failure_family=str(payload["semantic_failure_family"]),
            completeness_class=str(completeness),
            proposition_id=str(payload["proposition_id"]),
            task_id=str(payload["task_id"]),
            task_identity_digest=str(payload["task_identity_digest"]),
            checker_id=(
                str(payload["checker_id"])
                if payload.get("checker_id") is not None
                else None
            ),
            judgment_outcome=str(payload["judgment_outcome"]),
            judgment_checked=bool(payload["judgment_checked"]),
            lineage=dict(payload.get("lineage") or {}),
            unresolved=tuple(payload.get("unresolved") or ()),
            witness=witness,
            repair_action_suggestions=tuple(
                BoundedRepairActionV1.from_dict(item)
                for item in payload.get("repair_action_suggestions") or ()
            ),
            content_sha256=str(payload.get("content_sha256") or ""),
        )
        if _evidence_digest(evidence) != evidence.content_sha256:
            raise RevmathSchemaError(
                "revmath failure evidence digest mismatch (tampered or corrupted)"
            )
        # Diagnostic envelope must never claim proof success for non-decisive runs.
        if evidence.judgment_outcome not in (
            JudgmentOutcome.WITNESSED.value,
            JudgmentOutcome.REFUTED.value,
        ):
            if evidence.completeness_class == "EXACT" and evidence.failure_class in {
                RevmathFailureClass.UNLOCALIZED.value,
            }:
                raise RevmathSchemaError(
                    "UNLOCALIZED failures cannot claim EXACT completeness"
                )
        return evidence


def _evidence_digest(evidence: RevmathFailureEvidenceV1) -> str:
    payload = evidence.to_dict()
    payload.pop("content_sha256", None)
    return content_sha(payload)


def _scrub(text: str) -> str:
    if not text:
        return text
    if len(text) > 240:
        dig = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"{text[:240]}...[truncated:{dig}]"
    return text


def semantic_family_for_revmath_class(
    failure_class: RevmathFailureClass | str,
) -> SemanticFailureFamily:
    if isinstance(failure_class, str):
        failure_class = RevmathFailureClass(failure_class)
    return _FAMILY_BY_CLASS[failure_class]


def bounded_repair_actions_for(
    failure_class: RevmathFailureClass | str,
) -> tuple[BoundedRepairActionV1, ...]:
    """Return mutable-knob suggestions only where semantically safe."""
    if isinstance(failure_class, str):
        failure_class = RevmathFailureClass(failure_class)
    knobs = _ACTIONS_BY_CLASS[failure_class]
    return tuple(
        BoundedRepairActionV1(
            knob=knob,
            rationale=(
                f"bounded advisory repair for {failure_class.value}; "
                "does not change proposition/theory/campaign authority"
            ),
            failure_class=failure_class.value,
        )
        for knob in knobs
    )


def _classify_from_signals(
    *,
    task: RevmathTaskV1,
    result: RevmathResultV1,
    capture: Mapping[str, Any],
    blocked_semantic_mutation: bool,
    localization_hints: Mapping[str, Any] | None,
) -> tuple[RevmathFailureClass, CompletenessClass, str, tuple[str, ...]]:
    """Return (class, completeness, detail, unresolved)."""
    hints = dict(localization_hints or {})
    note = str(capture.get("note") or "")
    unresolved: list[str] = []
    judgment = result.solver_judgment()

    if blocked_semantic_mutation or hints.get("blocked_semantic_mutation"):
        return (
            RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION,
            "EXACT",
            str(hints.get("detail") or "blocked semantic mutation"),
            (),
        )

    if (
        capture.get("launch_error")
        or capture.get("timed_out")
        or note in {"missing_lean_tool", "unsupported_task_kind"}
        or "timeout" in note.lower()
        or "missing" in note.lower() and "tool" in note.lower()
    ):
        return (
            RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR,
            "EXACT" if capture.get("launch_error") or note == "missing_lean_tool" else "HEURISTIC",
            _scrub(note or str(capture.get("launch_error") or "tool/environment error")),
            (),
        )

    kind = task.task_kind
    if kind == "assumption_ablation" or hints.get("necessary_assumption_ids"):
        ids = hints.get("necessary_assumption_ids") or hints.get("missing_assumption_ids")
        if ids or "necessary" in note.lower() or hints.get("failure_class") == (
            RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE.value
        ):
            detail = _scrub(
                str(hints.get("detail") or note or f"necessary assumptions={ids}")
            )
            completeness: CompletenessClass = "EXACT" if ids else "HEURISTIC"
            if not ids:
                unresolved.append("revmath:necessary_assumption_ids")
            return (
                RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE,
                completeness,
                detail,
                tuple(unresolved),
            )

    if kind == "reversal" or hints.get("failed_direction"):
        direction = hints.get("failed_direction")
        if direction or "reversal" in note.lower() or hints.get("failure_class") == (
            RevmathFailureClass.FAILED_REVERSAL_DIRECTION.value
        ):
            if not direction:
                unresolved.append("revmath:failed_direction")
            return (
                RevmathFailureClass.FAILED_REVERSAL_DIRECTION,
                "EXACT" if direction else "HEURISTIC",
                _scrub(str(hints.get("detail") or note or f"failed_direction={direction}")),
                tuple(unresolved),
            )

    if kind == "constructivization" or hints.get("remainder_note"):
        if hints.get("remainder_note") or "nonconstructive" in note.lower() or (
            hints.get("failure_class")
            == RevmathFailureClass.NONCONSTRUCTIVE_REMAINDER.value
        ):
            remainder = hints.get("remainder_note")
            if not remainder:
                unresolved.append("revmath:remainder_note")
            return (
                RevmathFailureClass.NONCONSTRUCTIVE_REMAINDER,
                "EXACT" if remainder else "HEURISTIC",
                _scrub(str(remainder or note or "documented nonconstructive remainder")),
                tuple(unresolved),
            )

    if kind == "quantitative_bound_extraction" or hints.get("missing_assumptions"):
        missing = hints.get("missing_assumptions")
        if missing is not None or "nonextractable" in note.lower() or (
            hints.get("failure_class")
            == RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE.value
        ):
            if not missing:
                unresolved.append("revmath:missing_assumptions")
            return (
                RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE,
                "EXACT" if missing else "HEURISTIC",
                _scrub(
                    str(
                        hints.get("detail")
                        or note
                        or f"nonextractable missing={missing}"
                    )
                ),
                tuple(unresolved),
            )

    if kind == "computable_finite_counterexample" or hints.get(
        "independently_checked"
    ):
        checked = bool(hints.get("independently_checked"))
        if checked or judgment.outcome is JudgmentOutcome.REFUTED:
            if not checked and judgment.outcome is JudgmentOutcome.REFUTED:
                unresolved.append("revmath:independently_checked")
            return (
                RevmathFailureClass.CHECKED_COUNTEREXAMPLE,
                "EXACT" if checked else "HEURISTIC",
                _scrub(str(hints.get("detail") or note or "checked counterexample")),
                tuple(unresolved),
            )

    # Explicit hint always wins over blind UNLOCALIZED.
    hinted = hints.get("failure_class")
    if hinted:
        try:
            cls = RevmathFailureClass(str(hinted))
        except ValueError as exc:
            raise RevmathSchemaError(f"unknown failure_class hint: {hinted!r}") from exc
        if cls is RevmathFailureClass.UNLOCALIZED:
            unresolved.append("revmath:unlocalized")
            return cls, "UNKNOWN", _scrub(note or "unlocalized"), tuple(unresolved)
        return (
            cls,
            "HEURISTIC",
            _scrub(str(hints.get("detail") or note or cls.value)),
            tuple(unresolved),
        )

    unresolved.append("revmath:unlocalized")
    return (
        RevmathFailureClass.UNLOCALIZED,
        "UNKNOWN",
        _scrub(note or "unlocalized revmath failure"),
        tuple(unresolved),
    )


def _lineage_payload(
    *,
    task: RevmathTaskV1,
    result: RevmathResultV1,
    capture: Mapping[str, Any],
    repair_session: RevmathRepairSessionV1 | None,
    repair_history_ref: str | None,
) -> dict[str, Any]:
    proof = result.proof
    lineage: dict[str, Any] = {
        "proposition_id": task.proposition.proposition_id,
        "statement_sha256": task.proposition.statement_sha256,
        "task_id": task.task_id,
        "task_identity_digest": task.identity_digest(),
        "task_kind": task.task_kind,
        "base_theory_id": task.base_theory.base_theory_id,
        "campaign_id": task.campaign.campaign_id,
        "experiment_id": task.campaign.experiment_id,
        "campaign_manifest_sha256": task.campaign.campaign_manifest_sha256,
        "result_id": result.result_id,
        "checker_id": proof.checker_id if proof is not None else None,
        "proof_sha256": proof.proof_sha256 if proof is not None else None,
        "checker_report_sha256": (
            proof.checker_report_sha256 if proof is not None else None
        ),
        "capture_stdout_sha256": capture.get("stdout_sha256"),
        "capture_stderr_sha256": capture.get("stderr_sha256"),
        "repair_history_ref": repair_history_ref,
    }
    if repair_session is not None:
        lineage["repair_session"] = {
            "session_id": repair_session.session_id,
            "status": repair_session.status,
            "attempts": [
                {
                    "attempt_id": a.attempt_id,
                    "before_result_id": a.before_result_id,
                    "after_result_id": a.after_result_id,
                    "before_proof_sha256": (
                        a.repair.before_proof_sha256 if a.repair is not None else None
                    ),
                    "after_proof_sha256": (
                        a.repair.after_proof_sha256 if a.repair is not None else None
                    ),
                    "outcome": a.outcome,
                }
                for a in repair_session.attempts
            ],
            "semantic_lineage": repair_session.semantic_lineage
            or semantic_repair_lineage_from_session(repair_session),
        }
    return lineage


def build_revmath_verifier_witness(
    *,
    failure_class: RevmathFailureClass,
    completeness_class: CompletenessClass,
    detail: str,
    source_trace_fingerprint: str,
    judgment_outcome: str,
    certificate_id: str | None,
    unresolved: Sequence[str],
) -> VerifierWitnessV1:
    """Seal a VerifierWitnessV1 localization for one revmath failure class.

    Fail-closed: EXACT requires a concrete certificate/localization handle;
    fabricated completeness outside {EXACT,HEURISTIC,UNKNOWN} is rejected by
    ``seal_verifier_witness``.
    """
    if completeness_class not in _ALLOWED_COMPLETENESS:
        raise RevmathSchemaError(
            f"fabricated completeness_class rejected: {completeness_class!r}"
        )
    if completeness_class == "EXACT" and not certificate_id and failure_class not in {
        RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION,
        RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR,
    }:
        # Exact localization must name a certificate or stay unresolved.
        raise RevmathSchemaError(
            "EXACT revmath localization requires certificate_id "
            f"(failure_class={failure_class.value})"
        )
    if failure_class is RevmathFailureClass.UNLOCALIZED and completeness_class != "UNKNOWN":
        raise RevmathSchemaError(
            "UNLOCALIZED failures must use completeness_class=UNKNOWN"
        )

    loc = VerifierLocalizationV1(
        source="revmath",
        gate=None,
        reason_code=f"revmath.{failure_class.value}",
        ast_path=None,
        source_span=None,
        expected=None,
        observed=judgment_outcome,
        completeness_class=completeness_class,
        certificate_id=certificate_id,
        detail=_scrub(detail),
    )
    unresolved_list = list(unresolved)
    if completeness_class == "UNKNOWN":
        unresolved_list.append(f"revmath:{failure_class.value}")
    return seal_verifier_witness(
        source_trace_fingerprint=source_trace_fingerprint,
        taxonomy_version=TAXONOMY_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        evaluator_hash=content_sha(
            {
                "evaluator": EVALUATOR_VERSION,
                "failure_taxonomy": REVMATH_FAILURE_TAXONOMY_VERSION,
            }
        ),
        first_failed_gate=None,
        localizations=(loc,),
        unresolved_localizations=tuple(dict.fromkeys(unresolved_list)),
        schema_version=WITNESS_SCHEMA_VERSION,
    )


def route_revmath_failure(
    run: RevmathRunRecord,
    *,
    localization_hints: Mapping[str, Any] | None = None,
    repair_session: RevmathRepairSessionV1 | None = None,
    blocked_semantic_mutation: bool = False,
) -> RevmathFailureEvidenceV1:
    """Map one revmath run into existing witness/action taxonomies.

    Pure adapter: returns a diagnostic envelope. The run's ``SolverJudgmentV1``
    is copied into the envelope and never rewritten.
    """
    task = run.task
    result = run.result
    judgment = result.solver_judgment()
    failure_class, completeness, detail, unresolved = _classify_from_signals(
        task=task,
        result=result,
        capture=run.capture,
        blocked_semantic_mutation=blocked_semantic_mutation,
        localization_hints=localization_hints,
    )
    family = semantic_family_for_revmath_class(failure_class)
    hints = dict(localization_hints or {})
    certificate_id = None
    if result.proof is not None:
        certificate_id = result.proof.checker_report_sha256 or result.proof.proof_sha256
    elif hints.get("certificate_id"):
        certificate_id = str(hints["certificate_id"])
    elif completeness == "EXACT":
        # Content-address a localization handle from concrete hint fields so
        # EXACT stays exact without inventing checker authority.
        handle_fields = {
            "necessary_assumption_ids": hints.get("necessary_assumption_ids"),
            "missing_assumption_ids": hints.get("missing_assumption_ids"),
            "failed_direction": hints.get("failed_direction"),
            "remainder_note": hints.get("remainder_note"),
            "missing_assumptions": hints.get("missing_assumptions"),
            "independently_checked": hints.get("independently_checked"),
            "blocked_semantic_mutation": blocked_semantic_mutation
            or hints.get("blocked_semantic_mutation"),
            "note": run.capture.get("note"),
            "launch_error": run.capture.get("launch_error"),
            "class": failure_class.value,
            "task": task.identity_digest(),
        }
        if any(v not in (None, False, (), [], "") for k, v in handle_fields.items() if k != "class" and k != "task"):
            certificate_id = content_sha(handle_fields)

    # Downgrade fabricated EXACT when we still lack a certificate handle.
    if completeness == "EXACT" and not certificate_id:
        completeness = "HEURISTIC"
        unresolved = tuple(dict.fromkeys((*unresolved, "revmath:certificate_id")))

    trace_fp = content_sha(
        {
            "task_identity_digest": task.identity_digest(),
            "result_id": result.result_id,
            "judgment": judgment.to_dict(),
            "capture_stdout_sha256": run.capture.get("stdout_sha256"),
            "capture_stderr_sha256": run.capture.get("stderr_sha256"),
        }
    )
    witness = build_revmath_verifier_witness(
        failure_class=failure_class,
        completeness_class=completeness,
        detail=detail,
        source_trace_fingerprint=trace_fp,
        judgment_outcome=judgment.outcome.value,
        certificate_id=certificate_id,
        unresolved=unresolved,
    )
    lineage = _lineage_payload(
        task=task,
        result=result,
        capture=run.capture,
        repair_session=repair_session,
        repair_history_ref=run.repair_history_ref,
    )
    actions = bounded_repair_actions_for(failure_class)
    draft = RevmathFailureEvidenceV1(
        schema_version=EVIDENCE_SCHEMA,
        failure_taxonomy_version=REVMATH_FAILURE_TAXONOMY_VERSION,
        failure_class=failure_class.value,
        semantic_failure_family=family.value,
        completeness_class=completeness,
        proposition_id=task.proposition.proposition_id,
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        checker_id=result.proof.checker_id if result.proof is not None else None,
        judgment_outcome=judgment.outcome.value,
        judgment_checked=bool(judgment.checked),
        lineage=lineage,
        unresolved=tuple(witness.unresolved_localizations),
        witness=witness,
        repair_action_suggestions=actions,
        content_sha256="",
    )
    sealed = RevmathFailureEvidenceV1(
        schema_version=draft.schema_version,
        failure_taxonomy_version=draft.failure_taxonomy_version,
        failure_class=draft.failure_class,
        semantic_failure_family=draft.semantic_failure_family,
        completeness_class=draft.completeness_class,
        proposition_id=draft.proposition_id,
        task_id=draft.task_id,
        task_identity_digest=draft.task_identity_digest,
        checker_id=draft.checker_id,
        judgment_outcome=draft.judgment_outcome,
        judgment_checked=draft.judgment_checked,
        lineage=draft.lineage,
        unresolved=draft.unresolved,
        witness=draft.witness,
        repair_action_suggestions=draft.repair_action_suggestions,
        content_sha256=_evidence_digest(draft),
    )
    assert_diagnostic_cannot_prove_success(sealed, result)
    return sealed


def assert_diagnostic_cannot_prove_success(
    evidence: RevmathFailureEvidenceV1,
    result: RevmathResultV1,
) -> None:
    """Acceptance: a diagnostic witness never upgrades failed/unknown → success."""
    outcome = result.solver_judgment().outcome
    if evidence.judgment_outcome != outcome.value:
        raise RevmathSchemaError(
            "failure evidence judgment_outcome drifted from SolverJudgmentV1"
        )
    if outcome not in (JudgmentOutcome.WITNESSED, JudgmentOutcome.REFUTED):
        # Envelope must not advertise a decisive proof success.
        if evidence.judgment_outcome in (
            JudgmentOutcome.WITNESSED.value,
        ) and outcome is not JudgmentOutcome.WITNESSED:
            raise RevmathSchemaError(
                "diagnostic witness cannot turn non-witnessed into witnessed"
            )
        # Refuted is only legal when the checker actually refuted.
        if (
            evidence.failure_class == RevmathFailureClass.CHECKED_COUNTEREXAMPLE.value
            and outcome not in (JudgmentOutcome.REFUTED, JudgmentOutcome.UNKNOWN)
        ):
            raise RevmathSchemaError(
                "checked_counterexample class requires refuted/unknown judgment"
            )


def feed_repair_session_to_disposition(
    session: RevmathRepairSessionV1,
    *,
    failure: RevmathFailureEvidenceV1 | None = None,
    evidence_cutoff: str = "integ-05/revmath-self-healing",
) -> dict[str, Any]:
    """Project immutable HARN-09 before/after refs into disposition feedback.

    Uses ``MechanismDispositionRecordV1`` (existing owner) with
    ``retain_diagnostic`` — never adopt/promote from repair diagnostics alone.
    """
    lineage = session.semantic_lineage or semantic_repair_lineage_from_session(session)
    before_after = [
        {
            "attempt_id": a.attempt_id,
            "before_proof_sha256": (
                a.repair.before_proof_sha256 if a.repair is not None else None
            ),
            "after_proof_sha256": (
                a.repair.after_proof_sha256 if a.repair is not None else None
            ),
            "before_result_id": a.before_result_id,
            "after_result_id": a.after_result_id,
            "outcome": a.outcome,
            "failure_family": a.failure_family,
            "witness_digest": a.witness_digest,
        }
        for a in session.attempts
    ]
    activation = json.dumps(
        {
            "schema": FEEDBACK_SCHEMA,
            "session_id": session.session_id,
            "task_identity_digest": session.task_identity_digest,
            "status": session.status,
            "before_after_refs": before_after,
            "semantic_lineage": lineage,
            "failure_witness_digest": (
                failure.witness.witness_digest if failure is not None else None
            ),
            "failure_class": failure.failure_class if failure is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    disposition = Disposition.RETAIN_DIAGNOSTIC
    if session.status == "invalid":
        disposition = Disposition.BLOCKED
    elif session.status in {"cycle_detected", "budget_exhausted", "unknown"}:
        disposition = Disposition.INCONCLUSIVE
    record = build_mechanism_disposition_record(
        mechanism_id="revmath.self_healing",
        owner_module="src/slm_training/harnesses/reasoning/revmath/self_healing.py",
        owner_version="harn-09/v1",
        default_state="off",
        authority_class="oracle-diagnostic",
        evidence_cutoff=evidence_cutoff,
        evidence_class="fixture",
        disposition=disposition,
        rationale=(
            "INTEG-05: HARN-09 immutable repair before/after refs retained as "
            "diagnostic disposition; never promotes proof success"
        ),
        activation_evidence=activation,
        applicable_packs=("reasoning/revmath",),
        matched_controls=False,
        semantic_effect="none_identity_preserving_knobs_only",
        cost_cold_effect="bounded_recheck",
        safety_invariants=(
            "proposition_frozen",
            "campaign_frozen",
            "exact_checker_rerun",
            "diagnostic_witness_non_authoritative",
        ),
        issues=("SLM-571", "SLM-560"),
    )
    return {
        "schema": FEEDBACK_SCHEMA,
        "disposition_record": record.to_dict(),
        "before_after_refs": before_after,
        "semantic_lineage": lineage,
        "failure_evidence": failure.to_dict() if failure is not None else None,
    }


def failure_family_from_revmath_witness(
    witness: VerifierWitnessV1 | Mapping[str, Any] | None,
) -> str | None:
    """Extract mapped SemanticFailureFamily value from a revmath witness."""
    if witness is None:
        return None
    localizations: Sequence[Any]
    if isinstance(witness, VerifierWitnessV1):
        localizations = witness.localizations
    else:
        localizations = witness.get("localizations") or ()
    for item in localizations:
        reason = (
            item.reason_code
            if isinstance(item, VerifierLocalizationV1)
            else (item.get("reason_code") if isinstance(item, Mapping) else None)
        )
        if not reason or not str(reason).startswith("revmath."):
            continue
        cls_name = str(reason).removeprefix("revmath.")
        try:
            return semantic_family_for_revmath_class(cls_name).value
        except ValueError:
            return SemanticFailureFamily.UNKNOWN.value
    return None


__all__ = [
    "EVIDENCE_SCHEMA",
    "FEEDBACK_SCHEMA",
    "REVMATH_FAILURE_TAXONOMY_VERSION",
    "BoundedRepairActionV1",
    "RevmathFailureClass",
    "RevmathFailureEvidenceV1",
    "assert_diagnostic_cannot_prove_success",
    "bounded_repair_actions_for",
    "build_revmath_verifier_witness",
    "failure_family_from_revmath_witness",
    "feed_repair_session_to_disposition",
    "route_revmath_failure",
    "semantic_family_for_revmath_class",
]
