"""OpenUI wiring for the enumerative support oracle (VSS0-04).

Adapts the deterministic compiler forest and the lang-core validity check to the
problem-independent :class:`~slm_training.dsl.solver.support.EnumerativeSupportOracle`.

* :class:`OpenUIForestExpander` advances a token prefix through
  :func:`build_completion_forest`, projecting each next decision with the VSS0-03
  :func:`completion_forest_state` adapter. A chosen ``eos`` path terminates the
  program; a ``coverage == "none"`` forest is bottom; a ``partial``/``none``
  child is reported ``INCOMPLETE`` so the oracle keeps it ``UNKNOWN`` (it can
  never be exhaustively covered).
* :class:`OpenUIWellFormedVerifier` runs the deterministic lang-core parse/schema
  check. A genuine ``ParseError`` is a hard ``REJECT``; a missing bridge, timeout,
  or other runtime fault is ``UNAVAILABLE`` (→ ``UNKNOWN``), **never**
  ``UNSUPPORTED`` — the timeout-vs-UNSAT distinction the contract requires.
* :class:`OpenUIGoalVerifier` composes well-formedness, ``ProgramSpec`` /
  ``SemanticPlanV1`` extraction, goal-constraint evaluation, G0–G12 gates, and
  pinned evaluators into explicit ``ACCEPT`` / ``REJECT`` / ``UNAVAILABLE``
  terminal semantics (PGS-B02 / SLM-498). It emits redacted
  :class:`~slm_training.dsl.solver.goal_support.GoalTerminalEvidenceV1` in a
  fresh query-local trace per call.

This module is Torch-free and is not invoked by decode by default.
"""

from __future__ import annotations

import hashlib
from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.semantic_plan.requirements_compile import (
    evaluate_goal_constraints,
    compile_goal_constraints,
    request_production_digest,
    source_digest,
)
from slm_training.data.verify.stack import Gate, GateStatus, VerificationContext, evaluate_gate
from slm_training.dsl.grammar.fastpath.completion_artifact import (
    completion_artifact_checkpoint_identity,
)
from slm_training.dsl.grammar.fastpath.completion_kernel import CompletionSession
from slm_training.dsl.language_contract import current_contract
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.dsl.solver.goal_support import (
    GOAL_SUPPORT_IMPLEMENTATION_VERSION,
    EvaluatorIdentityV1,
    GoalEvaluatorResultV1,
    GoalFailureAtomV1,
    GoalGateResultV1,
    GoalTerminalEvidenceV1,
    GoalTerminalStatus,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    MetricIdentityV1,
    compute_pack_identity_digest,
    redact_bounded_string,
    validate_profile_against_constraint_set,
)
from slm_training.dsl.grammar.fastpath.token_map import decode_prefix
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    ExpandStatus,
    ExpandStep,
    VerifyOutcome,
    VerifyStatus,
)

WELL_FORMED_PROFILE = "openui/lang-core-validate/well-formed@0.2.x"
GOAL_SUPPORT_PROFILE_PREFIX = "openui/goal-support/v1"


def current_openui_pack_identity() -> PackIdentityV1:
    """Return the complete live identity required by production goal support."""
    contract = current_contract()
    artifact = completion_artifact_checkpoint_identity()
    return PackIdentityV1(
        pack_id="openui",
        contract_version=contract.output_contract_version,
        grammar_sha256=contract.grammar_sha256,
        tokenizer_id=(
            f"dsl-native/v{contract.dsl_tokenizer_version}:"
            f"{artifact['tokenizer_authority_sha256']}"
        ),
        canonicalizer_id=f"openui/canonical/{contract.contract_id}",
        artifact_schema=artifact["schema"],
        artifact_sha256=artifact["sha256"],
    )

__all__ = [
    "WELL_FORMED_PROFILE",
    "GOAL_SUPPORT_PROFILE_PREFIX",
    "OpenUIWellFormedVerifier",
    "OpenUIForestExpander",
    "GoalTerminalEvidenceTrace",
    "OpenUIGoalVerifier",
    "current_openui_pack_identity",
    "goal_verifier_status_table",
    "goal_verifier_invocation_map",
]

_GATE_BY_ID: dict[str, Gate] = {gate.value: gate for gate in Gate}

_EVALUATOR_ALIASES: dict[str, str] = {
    "binding_aware_meaningful_v2": "binding_aware_meaningful_v2",
    "meaningful_program/v2": "binding_aware_meaningful_v2",
}


def goal_verifier_status_table() -> dict[str, str]:
    """Map structural/constraint/gate/evaluator observations to terminal status."""
    return {
        "structural_reject": "REJECT",
        "structural_unavailable": "UNAVAILABLE",
        "mandatory_exact_fail": "REJECT",
        "mandatory_skip_or_unknown": "UNAVAILABLE",
        "mandatory_pass": "ACCEPT",
        "optional_unknown": "no_change",
    }


def goal_verifier_invocation_map() -> dict[str, str]:
    """Canonical owners invoked by :class:`OpenUIGoalVerifier` (identity only)."""
    return {
        "structural": "OpenUIWellFormedVerifier.verify",
        "program_spec": "ProgramSpec.from_openui",
        "semantic_plan": "OpenUISemanticPlanExtractor.extract",
        "constraints": "evaluate_goal_constraints",
        "gates": "evaluate_gate",
        "evaluator_binding_aware_meaningful_v2": "binding_aware_meaningful_v2",
    }


class GoalTerminalEvidenceTrace:
    """Fresh, query-local terminal-evidence records (never shared across queries)."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[GoalTerminalEvidenceV1] = []

    def append(self, evidence: GoalTerminalEvidenceV1) -> None:
        self._records.append(evidence)

    def latest(self) -> GoalTerminalEvidenceV1 | None:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> tuple[GoalTerminalEvidenceV1, ...]:
        return tuple(self._records)


def _terminal_from_structural(outcome: VerifyOutcome) -> GoalTerminalStatus:
    if outcome.status is VerifyStatus.REJECT:
        return "REJECT"
    if outcome.status is VerifyStatus.UNAVAILABLE:
        return "UNAVAILABLE"
    return "ACCEPT"


def _terminal_from_gate(status: GateStatus) -> GoalTerminalStatus:
    if status is GateStatus.FAIL:
        return "REJECT"
    if status is GateStatus.SKIP:
        return "UNAVAILABLE"
    return "ACCEPT"


def _terminal_from_constraint(
    evaluation: GoalConstraintEvaluationV1,
    *,
    profile: GoalVerifierProfileV1,
) -> GoalTerminalStatus:
    if evaluation.status == "FAIL":
        if profile.mode == "production_exact" and evaluation.completeness_achieved != "EXACT":
            return "UNAVAILABLE"
        return "REJECT"
    if evaluation.status in {"UNKNOWN", "SKIPPED"}:
        return "UNAVAILABLE"
    if evaluation.status == "NOT_APPLICABLE":
        return "UNAVAILABLE"
    return "ACCEPT"


def _gate_id_from_constraint(constraint: GoalConstraintV1) -> str | None:
    if constraint.kind != "required_gate_passes":
        return None
    gate = constraint.parameters.get("gate")
    return str(gate) if gate is not None else None


def _patch_gate_constraints(
    evaluations: tuple[GoalConstraintEvaluationV1, ...],
    *,
    constraints_by_id: dict[str, GoalConstraintV1],
    gate_results: dict[str, GoalGateResultV1],
) -> tuple[GoalConstraintEvaluationV1, ...]:
    patched: list[GoalConstraintEvaluationV1] = []
    for item in evaluations:
        constraint = constraints_by_id.get(item.constraint_id)
        gate_id = _gate_id_from_constraint(constraint) if constraint else None
        if gate_id is None or gate_id not in gate_results:
            patched.append(item)
            continue
        gate_status = gate_results[gate_id].status
        if gate_status == "REJECT":
            status = "FAIL"
            reason = "required_gate_failed"
        elif gate_status == "UNAVAILABLE":
            status = "UNKNOWN"
            reason = "required_gate_unavailable"
        else:
            status = "PASS"
            reason = "required_gate_passed"
        patched.append(
            GoalConstraintEvaluationV1(
                constraint_id=item.constraint_id,
                status=status,
                reason_code=reason,
                evidence=item.evidence,
                evaluator_id=item.evaluator_id,
                evaluator_digest=item.evaluator_digest,
                completeness_achieved="EXACT",
                authority_tier=item.authority_tier,
                may_prune=item.may_prune,
            )
        )
    return tuple(patched)


def _evaluator_affects_overall(profile: GoalVerifierProfileV1) -> bool:
    return profile.mode in {"evaluation_oracle", "advisory_diagnostic"}


def _aggregate_terminal_status(
    *,
    structural: GoalTerminalStatus,
    constraint_results: tuple[GoalConstraintEvaluationV1, ...],
    gate_results: tuple[GoalGateResultV1, ...],
    evaluator_results: tuple[GoalEvaluatorResultV1, ...],
    profile: GoalVerifierProfileV1,
) -> GoalTerminalStatus:
    if structural == "REJECT":
        return "REJECT"
    if structural == "UNAVAILABLE":
        return "UNAVAILABLE"

    has_reject = False
    has_unavailable = False

    for evaluation in constraint_results:
        terminal = _terminal_from_constraint(evaluation, profile=profile)
        if terminal == "REJECT":
            has_reject = True
        elif terminal == "UNAVAILABLE":
            has_unavailable = True

    for gate in gate_results:
        if gate.status == "REJECT":
            has_reject = True
        elif gate.status == "UNAVAILABLE":
            has_unavailable = True

    if _evaluator_affects_overall(profile):
        for result in evaluator_results:
            if result.status == "REJECT":
                has_reject = True
            elif result.status == "UNAVAILABLE":
                has_unavailable = True

    if has_reject:
        return "REJECT"
    if has_unavailable:
        return "UNAVAILABLE"
    return "ACCEPT"


def _build_failure_atoms(
    *,
    constraint_results: tuple[GoalConstraintEvaluationV1, ...],
    gate_results: tuple[GoalGateResultV1, ...],
    profile: GoalVerifierProfileV1,
) -> tuple[GoalFailureAtomV1, ...]:
    atoms: list[GoalFailureAtomV1] = []
    for evaluation in constraint_results:
        if evaluation.status != "FAIL":
            continue
        if (
            profile.mode != "production_exact"
            or evaluation.authority_tier not in {"compiler-hard", "verifier-hard"}
            or not evaluation.may_prune
            or evaluation.completeness_achieved != "EXACT"
        ):
            continue
        atoms.append(
            GoalFailureAtomV1(
                atom_id=f"constraint:{evaluation.constraint_id}",
                reason_code=evaluation.reason_code or "constraint_fail",
                source_kind="constraint",
                completeness_class="EXACT",
                source_identity=evaluation.constraint_id,
                implementation_version=GOAL_SUPPORT_IMPLEMENTATION_VERSION,
            )
        )
    for gate in gate_results:
        if gate.status != "REJECT":
            continue
        atoms.append(
            GoalFailureAtomV1(
                atom_id=f"gate:{gate.gate_id}",
                reason_code=gate.reason_code or "gate_fail",
                source_kind="gate",
                completeness_class="EXACT",
                source_identity=gate.gate_id,
                implementation_version=GOAL_SUPPORT_IMPLEMENTATION_VERSION,
            )
        )
    return tuple(atoms)


def _build_unknown_atoms(
    *,
    constraint_results: tuple[GoalConstraintEvaluationV1, ...],
    gate_results: tuple[GoalGateResultV1, ...],
    evaluator_results: tuple[GoalEvaluatorResultV1, ...],
    profile: GoalVerifierProfileV1,
) -> tuple[GoalUnknownAtomV1, ...]:
    atoms: list[GoalUnknownAtomV1] = []
    for evaluation in constraint_results:
        if evaluation.status not in {"UNKNOWN", "SKIPPED", "NOT_APPLICABLE"}:
            continue
        atoms.append(
            GoalUnknownAtomV1(
                atom_id=f"constraint:{evaluation.constraint_id}",
                reason_code=evaluation.reason_code or "constraint_unknown",
                source_kind="constraint",
                source_identity=evaluation.constraint_id,
            )
        )
    for gate in gate_results:
        if gate.status != "UNAVAILABLE":
            continue
        atoms.append(
            GoalUnknownAtomV1(
                atom_id=f"gate:{gate.gate_id}",
                reason_code=gate.reason_code or "gate_unavailable",
                source_kind="gate",
                source_identity=gate.gate_id,
            )
        )
    if _evaluator_affects_overall(profile):
        for result in evaluator_results:
            if result.status != "UNAVAILABLE":
                continue
            atoms.append(
                GoalUnknownAtomV1(
                    atom_id=f"evaluator:{result.evaluator_id}",
                    reason_code=result.reason_code or "evaluator_unavailable",
                    source_kind="evaluator",
                    source_identity=result.evaluator_id,
                )
            )
    return tuple(atoms)


def _run_pinned_evaluator(
    identity: EvaluatorIdentityV1,
    *,
    program: str,
    record: Any,
    request: GenerationRequest,
    metric_identities: tuple[MetricIdentityV1, ...] = (),
) -> GoalEvaluatorResultV1:
    evaluator_id = identity.evaluator_id
    canonical = _EVALUATOR_ALIASES.get(evaluator_id)
    if canonical != "binding_aware_meaningful_v2":
        return GoalEvaluatorResultV1(
            evaluator_id=evaluator_id,
            status="UNAVAILABLE",
            reason_code="evaluator_unsupported",
        )
    from slm_training.evals.meaningful_program import CheckStatus, binding_aware_meaningful_v2

    report = binding_aware_meaningful_v2(program, record=record, request=request)
    identity_matches = (
        (identity.version is None or identity.version == report.metric_version)
        and (
            identity.implementation_hash is None
            or identity.implementation_hash == report.metric_implementation_hash
        )
        and all(
            metric.version == report.metric_version
            and (
                metric.implementation_hash is None
                or metric.implementation_hash == report.metric_implementation_hash
            )
            for metric in metric_identities
        )
    )
    if not identity_matches:
        return GoalEvaluatorResultV1(
            evaluator_id=evaluator_id,
            status="UNAVAILABLE",
            reason_code="evaluator_identity_mismatch",
        )
    if any(check.status is CheckStatus.UNKNOWN for check in report.checks):
        return GoalEvaluatorResultV1(
            evaluator_id=evaluator_id,
            status="UNAVAILABLE",
            reason_code="evaluator_unknown",
        )
    if report.verdict:
        return GoalEvaluatorResultV1(
            evaluator_id=evaluator_id,
            status="ACCEPT",
            reason_code="evaluator_pass",
        )
    return GoalEvaluatorResultV1(
        evaluator_id=evaluator_id,
        status="REJECT",
        reason_code="evaluator_fail",
    )


def _bounded_outcome_detail(
    *,
    overall: GoalTerminalStatus,
    reason_code: str,
    evidence_digest: str,
) -> str:
    detail = f"overall={overall};reason={reason_code};evidence={evidence_digest}"
    return redact_bounded_string(detail)


class OpenUIGoalVerifier:
    """Profile-bound OpenUI terminal goal verifier (PGS-B02 / SLM-498)."""

    def __init__(
        self,
        *,
        profile: GoalVerifierProfileV1,
        constraint_set: CompiledGoalConstraintSetV1,
        problem: VerifiedSynthesisProblemV1,
        request: GenerationRequest,
        pack_identity: PackIdentityV1,
        pack: DslPack | None = None,
        verification_context: VerificationContext | None = None,
        structural_verifier: OpenUIWellFormedVerifier | None = None,
    ) -> None:
        validate_profile_against_constraint_set(profile, constraint_set)
        if problem.digest != profile.problem_digest:
            raise ValueError("live synthesis problem does not match goal-verifier profile")
        if request_production_digest(request) != constraint_set.request_digest:
            raise ValueError("live request does not match compiled constraint set")
        if compute_pack_identity_digest(pack_identity) != profile.pack_identity_digest:
            raise ValueError("live pack identity does not match goal-verifier profile")
        if pack_identity != current_openui_pack_identity():
            raise ValueError("goal-verifier profile does not match the live OpenUI identity")
        self._profile = profile
        self._constraint_set = constraint_set
        self._problem = problem
        self._request = request
        self._pack = pack or get_pack("openui")
        self._constraints_by_id = {
            item.constraint_id: item for item in constraint_set.constraints
        }
        if self._pack.pack_id != pack_identity.pack_id:
            raise ValueError("live pack id does not match goal-verifier profile")
        if profile.mode == "production_exact":
            live_constraints = {
                item.constraint_id: item
                for item in compile_goal_constraints(problem, request, self._pack).constraints
            }
            for constraint_id in profile.required_constraint_ids:
                if live_constraints.get(constraint_id) != self._constraints_by_id.get(
                    constraint_id
                ):
                    raise ValueError(
                        "required hard constraint does not replay from live sources: "
                        f"{constraint_id}"
                    )
        self._verification_context = verification_context
        self._structural = structural_verifier or OpenUIWellFormedVerifier()
        self._plan_extractor = OpenUISemanticPlanExtractor()
        self._trace = GoalTerminalEvidenceTrace()

    @property
    def profile(self) -> str:
        digest = self._profile.digest or self._profile.compute_digest()
        return f"{GOAL_SUPPORT_PROFILE_PREFIX}/{digest}"

    @property
    def last_trace(self) -> GoalTerminalEvidenceTrace | None:
        return self._trace

    @property
    def terminal_evidence(self) -> tuple[GoalTerminalEvidenceV1, ...]:
        return self._trace.records

    def verify(self, program: str, *, program_id: str = "goal_terminal") -> VerifyOutcome:
        validate_profile_against_constraint_set(self._profile, self._constraint_set)
        trace = self._trace

        try:
            structural_outcome = self._structural.verify(program)
        except TimeoutError:
            structural_outcome = VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="timeout")
        structural_status = _terminal_from_structural(structural_outcome)
        if structural_status != "ACCEPT":
            structural_reason = (
                "structural_reject"
                if structural_status == "REJECT"
                else "structural_unavailable"
            )
            evidence = self._emit_evidence(
                program=program,
                program_id=program_id,
                structural_status=structural_status,
                overall_status=structural_status,
                constraint_results=(),
                gate_results=(),
                evaluator_results=(),
                spec=None,
                plan=None,
                reason_code=structural_reason,
            )
            trace.append(evidence)
            return VerifyOutcome(
                _verify_status_from_terminal(structural_status),
                detail=_bounded_outcome_detail(
                    overall=structural_status,
                    reason_code=structural_reason,
                    evidence_digest=evidence.evidence_digest or evidence.compute_digest(),
                ),
            )

        try:
            spec = ProgramSpec.from_openui(
                id=program_id,
                openui=program,
                facts={
                    "output_kind": self._request.output_kind,
                    "output_category": self._request.output_category,
                },
                program_family_id=program_id,
                lineage_id=program_id,
                split_group_id=program_id,
            )
            plan = self._plan_extractor.extract(spec, self._pack)
            record = emit_record(
                spec,
                prompt=self._request.prompt,
                task="generation",
                openui=program,
            )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            evidence = self._emit_evidence(
                program=program,
                program_id=program_id,
                structural_status="ACCEPT",
                overall_status="UNAVAILABLE",
                constraint_results=(),
                gate_results=(),
                evaluator_results=(),
                spec=None,
                plan=None,
                reason_code=type(exc).__name__,
            )
            trace.append(evidence)
            digest = evidence.evidence_digest or evidence.compute_digest()
            return VerifyOutcome(
                VerifyStatus.UNAVAILABLE,
                detail=_bounded_outcome_detail(
                    overall="UNAVAILABLE",
                    reason_code="projection_unavailable",
                    evidence_digest=digest,
                ),
            )

        all_evaluations = evaluate_goal_constraints(
            source=program,
            program_spec=spec,
            plan=plan,
            constraints=self._constraint_set,
        )
        required_ids = set(self._profile.required_constraint_ids)
        constraint_results = tuple(
            item for item in all_evaluations if item.constraint_id in required_ids
        )

        gate_results_list: list[GoalGateResultV1] = []
        gate_result_by_id: dict[str, GoalGateResultV1] = {}
        for gate_id in self._profile.required_gates:
            gate = _GATE_BY_ID.get(gate_id)
            if gate is None:
                result = GoalGateResultV1(
                    gate_id=gate_id,
                    status="UNAVAILABLE",
                    reason_code="gate_identity_unknown",
                )
            else:
                try:
                    raw = evaluate_gate(gate, record, self._verification_context)
                except (RuntimeError, ValueError, TimeoutError):
                    result = GoalGateResultV1(
                        gate_id=gate_id,
                        status="UNAVAILABLE",
                        reason_code="gate_unavailable",
                    )
                else:
                    result = GoalGateResultV1(
                        gate_id=gate_id,
                        status=_terminal_from_gate(raw.status),
                        reason_code=f"gate_{raw.status.value.lower()}",
                    )
            gate_results_list.append(result)
            gate_result_by_id[gate_id] = result
        gate_results = tuple(gate_results_list)

        constraint_results = _patch_gate_constraints(
            constraint_results,
            constraints_by_id=self._constraints_by_id,
            gate_results=gate_result_by_id,
        )

        evaluator_results_list: list[GoalEvaluatorResultV1] = []
        for item in self._profile.required_evaluators:
            try:
                result = _run_pinned_evaluator(
                    item,
                    program=program,
                    record=record,
                    request=self._request,
                    metric_identities=tuple(
                        metric
                        for metric in self._profile.metric_identities
                        if _EVALUATOR_ALIASES.get(metric.metric_id)
                        == _EVALUATOR_ALIASES.get(item.evaluator_id)
                    ),
                )
            except (RuntimeError, ValueError, TimeoutError):
                result = GoalEvaluatorResultV1(
                    evaluator_id=item.evaluator_id,
                    status="UNAVAILABLE",
                    reason_code="evaluator_unavailable",
                )
            evaluator_results_list.append(result)
        matched_metrics = {
            metric.metric_id
            for metric in self._profile.metric_identities
            if any(
                _EVALUATOR_ALIASES.get(metric.metric_id)
                == _EVALUATOR_ALIASES.get(item.evaluator_id)
                for item in self._profile.required_evaluators
            )
        }
        evaluator_results_list.extend(
            GoalEvaluatorResultV1(
                evaluator_id=f"metric:{metric.metric_id}",
                status="UNAVAILABLE",
                reason_code="metric_evaluator_unavailable",
            )
            for metric in self._profile.metric_identities
            if metric.metric_id not in matched_metrics
        )
        evaluator_results = tuple(evaluator_results_list)

        overall = _aggregate_terminal_status(
            structural=structural_status,
            constraint_results=constraint_results,
            gate_results=gate_results,
            evaluator_results=evaluator_results,
            profile=self._profile,
        )
        reason_code = "goal_terminal"
        if overall == "REJECT":
            reason_code = "mandatory_fail"
        elif overall == "UNAVAILABLE":
            reason_code = "mandatory_unavailable"

        evidence = self._emit_evidence(
            program=program,
            program_id=program_id,
            structural_status=structural_status,
            overall_status=overall,
            constraint_results=constraint_results,
            gate_results=gate_results,
            evaluator_results=evaluator_results,
            spec=spec,
            plan=plan,
            reason_code=reason_code,
        )
        trace.append(evidence)
        digest = evidence.evidence_digest or evidence.compute_digest()
        return VerifyOutcome(
            _verify_status_from_terminal(overall),
            detail=_bounded_outcome_detail(
                overall=overall,
                reason_code=reason_code,
                evidence_digest=digest,
            ),
        )

    def _emit_evidence(
        self,
        *,
        program: str,
        program_id: str,
        structural_status: GoalTerminalStatus,
        overall_status: GoalTerminalStatus,
        constraint_results: tuple[GoalConstraintEvaluationV1, ...],
        gate_results: tuple[GoalGateResultV1, ...],
        evaluator_results: tuple[GoalEvaluatorResultV1, ...],
        spec: ProgramSpec | None,
        plan: SemanticPlanV1 | None,
        reason_code: str,
    ) -> GoalTerminalEvidenceV1:
        profile_digest = self._profile.digest or self._profile.compute_digest()
        canonical = spec.canonical_openui if spec is not None else program.strip()
        program_digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
        canonical_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        spec_digest = source_digest(spec.to_dict()) if spec is not None else program_digest
        plan_digest = source_digest(plan.to_dict()) if plan is not None else spec_digest
        failure_atoms = _build_failure_atoms(
            constraint_results=constraint_results,
            gate_results=gate_results,
            profile=self._profile,
        )
        unknown_atoms = _build_unknown_atoms(
            constraint_results=constraint_results,
            gate_results=gate_results,
            evaluator_results=evaluator_results,
            profile=self._profile,
        )
        if overall_status == "UNAVAILABLE" and not unknown_atoms:
            unknown_atoms = (
                GoalUnknownAtomV1(
                    atom_id="structural:terminal_projection",
                    reason_code=reason_code or "terminal_projection_unavailable",
                    source_kind="structural",
                    source_identity=GOAL_SUPPORT_IMPLEMENTATION_VERSION,
                ),
            )
        payload = GoalTerminalEvidenceV1(
            profile_digest=profile_digest,
            program_digest=program_digest,
            canonical_program_digest=canonical_digest,
            program_spec_digest=spec_digest,
            semantic_plan_digest=plan_digest,
            constraint_evaluations=constraint_results,
            required_gate_results=gate_results,
            required_evaluator_results=evaluator_results,
            mandatory_failure_atoms=failure_atoms,
            mandatory_unknown_atoms=unknown_atoms,
            structural_status=structural_status,
            overall_status=overall_status,
        )
        return GoalTerminalEvidenceV1.from_dict(payload.to_dict(include_digest=True))


def _verify_status_from_terminal(status: GoalTerminalStatus) -> VerifyStatus:
    if status == "REJECT":
        return VerifyStatus.REJECT
    if status == "UNAVAILABLE":
        return VerifyStatus.UNAVAILABLE
    return VerifyStatus.ACCEPT

class OpenUIWellFormedVerifier:
    """Deterministic lang-core well-formedness verifier (G0-G2 surface).

    ``profile`` is the recorded verifier identity. This checks structural
    validity only (``reward_label`` is ``well_formed_not_behavioral``); it is not
    a behavioral or ship verdict.
    """

    def __init__(self, *, profile: str = WELL_FORMED_PROFILE) -> None:
        self._profile = profile

    @property
    def profile(self) -> str:
        return self._profile

    def verify(self, program: str) -> VerifyOutcome:
        from slm_training.dsl import lang_core

        if not lang_core.bridge_available():
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="bridge_unavailable")
        try:
            lang_core.validate(program)
        except lang_core.ParseError:
            return VerifyOutcome(VerifyStatus.REJECT, detail="parse_error")
        except (RuntimeError, TimeoutError) as exc:  # capability fault, not a rejection
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail=type(exc).__name__)
        return VerifyOutcome(VerifyStatus.ACCEPT)


class OpenUIForestExpander:
    """Bounded deterministic expander over the OpenUI choice/compiler forest.

    Holds one request-local packed
    :class:`~slm_training.dsl.grammar.fastpath.completion_kernel.CompletionSession`:
    ``_project``/``successor`` reuse interned states (advance along the chosen
    path's token ids, then project the target state's outgoing edges) instead
    of building a fresh forest per node with no shared state.  Certificates
    remain payload-based (kind, token ids, versions, digests) — interned
    state ids never leave the expander.
    """

    def __init__(
        self,
        tokenizer: Any,
        prefix_ids: tuple[int, ...] | list[int],
        *,
        pack_id: str,
        constraint_version: str,
        bounds: SolverBounds,
        max_path_tokens: int = 8,
    ) -> None:
        self._tok = tokenizer
        self._pack_id = pack_id
        self._cv = constraint_version
        self._bounds = bounds
        self._mpt = max(1, int(max_path_tokens))
        self._eos = int(tokenizer.eos_id)
        self._session = CompletionSession(tokenizer, max_path_tokens=self._mpt)
        self._prefix_by_fp: dict[str, tuple[int, ...]] = {}
        self._sid_by_fp: dict[str, int] = {}
        self._root = self._project(tuple(int(t) for t in prefix_ids), None)

    def _project(
        self, prefix: tuple[int, ...], state_id: int | None
    ) -> FiniteDomainState:
        sid = self._session.seed(prefix) if state_id is None else state_id
        forest = self._session.outgoing(sid)
        state = completion_forest_state(
            prefix_ids=prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[state.fingerprint] = prefix
        self._sid_by_fp[state.fingerprint] = sid
        return state

    # --- ProblemExpander protocol ---------------------------------------- #
    @property
    def problem_id(self) -> str:
        return self._root.problem_id

    @property
    def pack_id(self) -> str:
        return self._pack_id

    @property
    def constraint_version(self) -> str:
        return self._cv

    @property
    def bounds(self) -> SolverBounds:
        return self._bounds

    def root_state(self) -> FiniteDomainState:
        return self._root

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        prefix = self._prefix_by_fp.get(state.fingerprint)
        sid = self._sid_by_fp.get(state.fingerprint)
        if prefix is None or sid is None:
            # The oracle only expands states this expander created; a miss means
            # an out-of-band state we cannot faithfully advance -> UNKNOWN.
            return ExpandStep(
                ExpandStatus.INCOMPLETE, coverage="none", detail="unknown_state"
            )
        payload = value.payload
        token_ids = tuple(int(tok) for tok in payload.get("token_ids", ()))
        kind = str(payload.get("kind", ""))
        advertised = any(
            domain.hole_id == hole_id and value in domain.values
            for domain in state.holes
        )
        if kind == "eos" or (len(token_ids) == 1 and token_ids[0] == self._eos):
            program = decode_prefix(self._tok, list(prefix))
            return ExpandStep(
                ExpandStatus.TERMINAL,
                program=program,
                coverage="complete",
                detail=f"prefix_len={len(prefix)}",
            )
        new_prefix = prefix + token_ids
        child_sid = self._session.advance_path(sid, token_ids)
        if child_sid is None:
            # This edge was advertised by the exact finite domain.  A failed
            # replay is an authority inconsistency, not proof that the branch
            # is impossible; verified-solver law requires UNKNOWN/incomplete.
            if advertised:
                return ExpandStep(
                    ExpandStatus.INCOMPLETE,
                    coverage="none",
                    detail="advertised_edge_transition_failed",
                )
            return ExpandStep(
                ExpandStatus.DEAD, coverage="none", detail="illegal_prefix"
            )
        forest = self._session.outgoing(child_sid)
        if forest.coverage == "none":
            return ExpandStep(ExpandStatus.DEAD, coverage="none", detail="illegal_prefix")
        if not forest.paths:
            return ExpandStep(
                ExpandStatus.INCOMPLETE,
                coverage=forest.coverage,
                detail="no_enumerated_actions",
            )
        child = completion_forest_state(
            prefix_ids=new_prefix,
            forest=forest,
            pack_id=self._pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
        )
        self._prefix_by_fp[child.fingerprint] = new_prefix
        self._sid_by_fp[child.fingerprint] = child_sid
        return ExpandStep(
            ExpandStatus.CONTINUE, next_state=child, coverage=forest.coverage
        )
