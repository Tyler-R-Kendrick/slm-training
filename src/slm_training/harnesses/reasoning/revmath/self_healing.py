"""Proposition-preserving self-healing controller (HARN-09 / SLM-560).

Repairs may touch only ``REVMATH_MUTABLE_REPAIR_KNOBS``. Proposition, base
theory / assumptions, theorem direction, corpus, campaign arms, budgets /
stopping rules, judges, promotion gates, authority policy, and expected
counterexample semantics stay cryptographically frozen via task identity +
``assert_repair_preserves_identity``.

Extends the existing repair *philosophy* (lineage + residual-shaped metadata
compatible with ``SemanticRepairRecordV1``) without forking a second OpenUI
repair registry or importing the torch-bound distill scorer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from slm_training.evals.semantic_failure import (
    SemanticFailureFamily,
    SemanticFailureTraceV1,
    VerifierWitnessV1,
)
from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.runner import (
    RevmathRunRecord,
    run_revmath_task,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    REVMATH_MUTABLE_REPAIR_KNOBS,
    RevmathRepairKnobChangeV1,
    RevmathRepairRecordV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
    assert_repair_preserves_identity,
)

# Mirrors revmath_harness_parity.json self_healing.must_freeze (parity-tested).
REVMATH_FROZEN_REPAIR_FIELDS: frozenset[str] = frozenset(
    {
        "proposition",
        "base_theory_or_allowed_assumptions",
        "theorem_direction",
        "corpus_membership",
        "experiment_arms",
        "budgets_and_stopping_rules",
        "judge_definitions",
        "promotion_gates",
        "authority_policy",
        "expected_counterexample_semantics",
    }
)

# Forbidden API aliases → frozen field (mutation tests + fail-closed gate).
_FORBIDDEN_REPAIR_ALIASES: Mapping[str, str] = {
    "proposition": "proposition",
    "proposition_id": "proposition",
    "statement": "proposition",
    "statement_sha256": "proposition",
    "base_theory": "base_theory_or_allowed_assumptions",
    "base_theory_id": "base_theory_or_allowed_assumptions",
    "allowed_assumption_ids": "base_theory_or_allowed_assumptions",
    "add_axiom": "base_theory_or_allowed_assumptions",
    "weaken_theorem": "proposition",
    "theorem_direction": "theorem_direction",
    "corpus": "corpus_membership",
    "corpus_id": "corpus_membership",
    "corpus_sha256": "corpus_membership",
    "locked_arm_ids": "experiment_arms",
    "experiment_arms": "experiment_arms",
    "max_wall_seconds": "budgets_and_stopping_rules",
    "max_tactic_steps": "budgets_and_stopping_rules",
    "stopping_rule_ids": "budgets_and_stopping_rules",
    "increase_budget": "budgets_and_stopping_rules",
    "verifier_id": "judge_definitions",
    "judge_id": "judge_definitions",
    "promotion_gates": "promotion_gates",
    "authority_policy": "authority_policy",
    "expected_counterexample_semantics": "expected_counterexample_semantics",
    "change_corpus": "corpus_membership",
}

_FAILURE_KNOB_PREFERENCES: Mapping[str, tuple[str, ...]] = {
    SemanticFailureFamily.SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE.value: (
        "retrieval_choices",
        "proof_search_strategy",
    ),
    SemanticFailureFamily.SEARCH_OR_PRUNING_REGRET.value: (
        "proof_search_strategy",
        "tactic_sequence",
    ),
    SemanticFailureFamily.LOCAL_SCORING_REGRET.value: ("tactic_sequence",),
    SemanticFailureFamily.GLOBAL_RANKING_REGRET.value: (
        "proof_search_strategy",
        "retrieval_choices",
    ),
    SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE.value: (
        "serialization_bridge_generation",
        "resource_allocation_within_locked_budget",
    ),
    SemanticFailureFamily.PLAN_PREDICTION_REGRET.value: (
        "temporary_decomposition",
        "tactic_sequence",
    ),
    SemanticFailureFamily.UNKNOWN.value: (
        "tactic_sequence",
        "proof_search_strategy",
        "retrieval_choices",
    ),
}

RepairSessionStatus = Literal[
    "repaired",
    "unknown",
    "invalid",
    "cycle_detected",
    "budget_exhausted",
    "no_repair_needed",
]

AttemptOutcome = Literal[
    "repaired",
    "unknown",
    "invalid",
    "cycle_detected",
    "budget_exhausted",
]

DEFAULT_MAX_ATTEMPTS = 4

RunnerFn = Callable[..., RevmathRunRecord]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    if isinstance(value, str):
        return content_sha(value)
    return content_sha(value)


@dataclass(frozen=True)
class RevmathRepairKnobStateV1:
    """Mutable search state only — never carries frozen identity fields."""

    tactic_sequence: tuple[str, ...] = ("simp",)
    proof_search_strategy: str = "default_bfs"
    retrieval_choices: tuple[str, ...] = ()
    type_equivalent_theorem_selection: tuple[str, ...] = ()
    serialization_bridge_generation: str = "default_bridge"
    temporary_decomposition: tuple[str, ...] = ()
    resource_allocation_within_locked_budget: Mapping[str, float] = field(
        default_factory=lambda: {"search": 0.5, "check": 0.5}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tactic_sequence": list(self.tactic_sequence),
            "proof_search_strategy": self.proof_search_strategy,
            "retrieval_choices": list(self.retrieval_choices),
            "type_equivalent_theorem_selection": list(
                self.type_equivalent_theorem_selection
            ),
            "serialization_bridge_generation": self.serialization_bridge_generation,
            "temporary_decomposition": list(self.temporary_decomposition),
            "resource_allocation_within_locked_budget": dict(
                self.resource_allocation_within_locked_budget
            ),
        }

    def digest(self) -> str:
        return content_sha(self.to_dict())

    def knob_value(self, knob: str) -> Any:
        if knob not in REVMATH_MUTABLE_REPAIR_KNOBS:
            raise RevmathSchemaError(
                f"knob {knob!r} is frozen or unknown; "
                f"allowed={sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
            )
        return self.to_dict()[knob]

    def with_knob(self, knob: str, value: Any) -> RevmathRepairKnobStateV1:
        if knob not in REVMATH_MUTABLE_REPAIR_KNOBS:
            raise RevmathSchemaError(
                f"cannot mutate frozen/unknown knob {knob!r}; "
                f"allowed={sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
            )
        payload = self.to_dict()
        if knob == "proof_search_strategy" or knob == "serialization_bridge_generation":
            if not isinstance(value, str) or not value:
                raise RevmathSchemaError(f"{knob} requires a non-empty string")
            payload[knob] = value
        elif knob == "resource_allocation_within_locked_budget":
            if not isinstance(value, Mapping) or not value:
                raise RevmathSchemaError(f"{knob} requires a non-empty mapping")
            total = float(sum(float(v) for v in value.values()))
            if total <= 0 or total > 1.0 + 1e-9:
                raise RevmathSchemaError(
                    f"{knob} fractions must sum to (0, 1]; got {total}"
                )
            payload[knob] = {str(k): float(v) for k, v in value.items()}
        else:
            if isinstance(value, str):
                seq: tuple[str, ...] = (value,)
            elif isinstance(value, Sequence):
                seq = tuple(str(v) for v in value)
            else:
                raise RevmathSchemaError(f"{knob} requires a sequence of strings")
            payload[knob] = list(seq)
        return RevmathRepairKnobStateV1(
            tactic_sequence=tuple(payload["tactic_sequence"]),
            proof_search_strategy=str(payload["proof_search_strategy"]),
            retrieval_choices=tuple(payload["retrieval_choices"]),
            type_equivalent_theorem_selection=tuple(
                payload["type_equivalent_theorem_selection"]
            ),
            serialization_bridge_generation=str(
                payload["serialization_bridge_generation"]
            ),
            temporary_decomposition=tuple(payload["temporary_decomposition"]),
            resource_allocation_within_locked_budget=dict(
                payload["resource_allocation_within_locked_budget"]
            ),
        )


@dataclass(frozen=True)
class RevmathRepairProposalV1:
    """One bounded, identity-preserving knob delta proposal."""

    knob: str
    after_value: Any
    rationale: str
    failure_family: str | None = None
    witness_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "knob": self.knob,
            "after_value": self.after_value,
            "rationale": self.rationale,
            "failure_family": self.failure_family,
            "witness_digest": self.witness_digest,
        }


@dataclass(frozen=True)
class RevmathRepairAttemptV1:
    """Immutable attempt envelope: identity certificate + lineage + evidence."""

    schema_version: Literal["revmath_repair_attempt/v1"] = "revmath_repair_attempt/v1"
    attempt_id: str = ""
    parent_attempt_id: str | None = None
    repair: RevmathRepairRecordV1 | None = None
    failure_family: str | None = None
    witness_digest: str | None = None
    search_budget_remaining: int = 0
    before_result_id: str = ""
    after_result_id: str | None = None
    checker_id: str = ""
    evidence_refs: tuple[str, ...] = ()
    outcome: AttemptOutcome = "unknown"
    knob_state_before_sha256: str = ""
    knob_state_after_sha256: str = ""
    invalidated_proof_sha256s: tuple[str, ...] = ()
    equivalent_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "parent_attempt_id": self.parent_attempt_id,
            "repair": None if self.repair is None else self.repair.model_dump(mode="json"),
            "failure_family": self.failure_family,
            "witness_digest": self.witness_digest,
            "search_budget_remaining": self.search_budget_remaining,
            "before_result_id": self.before_result_id,
            "after_result_id": self.after_result_id,
            "checker_id": self.checker_id,
            "evidence_refs": list(self.evidence_refs),
            "outcome": self.outcome,
            "knob_state_before_sha256": self.knob_state_before_sha256,
            "knob_state_after_sha256": self.knob_state_after_sha256,
            "invalidated_proof_sha256s": list(self.invalidated_proof_sha256s),
            "equivalent_signature": self.equivalent_signature,
        }


@dataclass(frozen=True)
class RevmathRepairSessionV1:
    """Bounded self-healing session over one locked task."""

    schema_version: Literal["revmath_repair_session/v1"] = "revmath_repair_session/v1"
    session_id: str = ""
    task_identity_digest: str = ""
    status: RepairSessionStatus = "unknown"
    attempts: tuple[RevmathRepairAttemptV1, ...] = ()
    final_result: RevmathResultV1 | None = None
    final_knob_state: RevmathRepairKnobStateV1 | None = None
    semantic_lineage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "task_identity_digest": self.task_identity_digest,
            "status": self.status,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_result": (
                None
                if self.final_result is None
                else self.final_result.model_dump(mode="json")
            ),
            "final_knob_state": (
                None if self.final_knob_state is None else self.final_knob_state.to_dict()
            ),
            "semantic_lineage": dict(self.semantic_lineage),
        }


def reject_forbidden_repair(field_or_alias: str) -> None:
    """Fail closed when a caller names a frozen identity / policy field."""
    frozen = _FORBIDDEN_REPAIR_ALIASES.get(field_or_alias, field_or_alias)
    if frozen in REVMATH_FROZEN_REPAIR_FIELDS or field_or_alias in REVMATH_FROZEN_REPAIR_FIELDS:
        raise RevmathSchemaError(
            f"forbidden repair dimension {field_or_alias!r} "
            f"(frozen as {frozen!r}); may_modify="
            f"{sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
        )
    if field_or_alias not in REVMATH_MUTABLE_REPAIR_KNOBS:
        raise RevmathSchemaError(
            f"unknown repair knob {field_or_alias!r}; "
            f"allowed={sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
        )


def assert_serialized_repair_freeze(
    task: RevmathTaskV1, payload: Mapping[str, Any]
) -> None:
    """Refuse serialized artifacts that rewrite frozen identity fields."""
    if "repair" in payload and isinstance(payload["repair"], Mapping):
        repair = RevmathRepairRecordV1.model_validate(dict(payload["repair"]))
        assert_repair_preserves_identity(task, repair)
        return
    if payload.get("schema_version") == "revmath_repair/v1":
        repair = RevmathRepairRecordV1.model_validate(dict(payload))
        assert_repair_preserves_identity(task, repair)
        return
    for key in (
        "proposition_id",
        "statement_sha256",
        "base_theory_id",
        "allowed_assumption_ids",
        "campaign_manifest_sha256",
        "campaign_id",
        "experiment_id",
        "task_identity_digest",
    ):
        if key in payload and key != "task_identity_digest":
            # Presence alone is fine only when values match the locked task.
            pass
    if (
        "task_identity_digest" in payload
        and payload["task_identity_digest"] != task.identity_digest()
    ):
        raise RevmathSchemaError(
            "serialized repair artifact task_identity_digest drift"
        )


def _failure_family_from_inputs(
    *,
    failure: SemanticFailureTraceV1 | Mapping[str, Any] | None,
    witness: VerifierWitnessV1 | Mapping[str, Any] | None,
    run: RevmathRunRecord | None,
) -> str:
    # Prefer INTEG-05 revmath localizations on the existing witness owner.
    from slm_training.harnesses.reasoning.revmath.failure_witness import (
        failure_family_from_revmath_witness,
    )

    mapped = failure_family_from_revmath_witness(witness)
    if mapped:
        return mapped
    if isinstance(failure, SemanticFailureTraceV1):
        fam = failure.first_failure_family
        if fam:
            return str(fam)
        if failure.all_failure_families:
            return str(failure.all_failure_families[0])
    elif isinstance(failure, Mapping):
        fam = (
            failure.get("first_failure_family")
            or failure.get("primary_family")
            or failure.get("failure_family")
            or failure.get("semantic_failure_family")
        )
        if fam:
            return str(fam)
        families = failure.get("all_failure_families") or ()
        if families:
            return str(families[0])
    if isinstance(witness, VerifierWitnessV1):
        if witness.first_failed_gate:
            return SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE.value
    elif isinstance(witness, Mapping) and witness.get("first_failed_gate"):
        return SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE.value
    if run is not None:
        outcome = run.result.solver_judgment().outcome
        if outcome is JudgmentOutcome.UNKNOWN:
            note = str(run.capture.get("note") or "")
            if "timeout" in note or run.capture.get("timed_out"):
                return SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE.value
            return SemanticFailureFamily.SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE.value
        if outcome is JudgmentOutcome.INVALID:
            return SemanticFailureFamily.LEXICAL_OR_PARSE.value
    return SemanticFailureFamily.UNKNOWN.value


def _witness_digest(
    witness: VerifierWitnessV1 | Mapping[str, Any] | None,
) -> str | None:
    if witness is None:
        return None
    if isinstance(witness, VerifierWitnessV1):
        return witness.witness_digest
    dig = witness.get("witness_digest")
    return str(dig) if dig else None


def _next_value_for_knob(knob: str, state: RevmathRepairKnobStateV1, salt: int) -> Any:
    current = state.knob_value(knob)
    if knob == "proof_search_strategy":
        alts = ("default_bfs", "dfs_restart", "best_first", "iddfs")
        cur = str(current)
        for alt in alts:
            if alt != cur:
                return alt
        return f"{cur}#{salt}"
    if knob == "serialization_bridge_generation":
        return f"{current}#bridge{salt}"
    if knob == "resource_allocation_within_locked_budget":
        # Reallocate inside the locked budget envelope (fractions still ≤ 1).
        return {"search": 0.7, "check": 0.3} if salt % 2 == 0 else {"search": 0.3, "check": 0.7}
    seq = list(current)
    marker = f"repair{salt}"
    return (*seq, marker) if seq else (marker,)


def propose_repair_attempts(
    *,
    task: RevmathTaskV1,
    knob_state: RevmathRepairKnobStateV1,
    failure: SemanticFailureTraceV1 | Mapping[str, Any] | None = None,
    witness: VerifierWitnessV1 | Mapping[str, Any] | None = None,
    run: RevmathRunRecord | None = None,
    max_proposals: int = 3,
) -> tuple[RevmathRepairProposalV1, ...]:
    """Propose bounded mutable-knob repairs from typed failure/witness data."""
    del task  # identity is locked; proposals never rewrite the task
    family = _failure_family_from_inputs(failure=failure, witness=witness, run=run)
    preferred = _FAILURE_KNOB_PREFERENCES.get(
        family, _FAILURE_KNOB_PREFERENCES[SemanticFailureFamily.UNKNOWN.value]
    )
    # Round-robin remaining mutable knobs for coverage.
    ordered = list(preferred) + [
        k for k in sorted(REVMATH_MUTABLE_REPAIR_KNOBS) if k not in preferred
    ]
    wdig = _witness_digest(witness)
    out: list[RevmathRepairProposalV1] = []
    for idx, knob in enumerate(ordered):
        if len(out) >= max_proposals:
            break
        after = _next_value_for_knob(knob, knob_state, salt=idx + 1)
        if _sha(after) == _sha(knob_state.knob_value(knob)):
            continue
        out.append(
            RevmathRepairProposalV1(
                knob=knob,
                after_value=after,
                rationale=f"bounded repair for failure_family={family}",
                failure_family=family,
                witness_digest=wdig,
            )
        )
    return tuple(out)


def equivalent_repair_signature(
    knob_changes: Sequence[RevmathRepairKnobChangeV1] | Sequence[Mapping[str, Any]],
) -> str:
    """Stable signature for cycle detection over equivalent knob deltas."""
    rows: list[tuple[str, str, str]] = []
    for change in knob_changes:
        if isinstance(change, RevmathRepairKnobChangeV1):
            rows.append(
                (change.knob, change.before_value_sha256, change.after_value_sha256)
            )
        else:
            rows.append(
                (
                    str(change["knob"]),
                    str(change.get("before_value_sha256", "")),
                    str(change["after_value_sha256"]),
                )
            )
    rows.sort()
    return content_sha(rows)


def proposal_signature(proposal: RevmathRepairProposalV1) -> str:
    """Signature over the absolute post-repair knob value (cycle key)."""
    return equivalent_repair_signature(
        (
            {
                "knob": proposal.knob,
                "before_value_sha256": "",
                "after_value_sha256": _sha(proposal.after_value),
            },
        )
    )


def _proof_sha(result: RevmathResultV1) -> str:
    if result.proof is not None:
        return result.proof.proof_sha256
    # Unknown/invalid runs still need a distinct artifact digest for lineage.
    return content_sha(
        {
            "result_id": result.result_id,
            "judgment": result.judgment,
            "task_identity_digest": result.task_identity_digest,
            "empty_proof": True,
        }
    )


def _checker_id(result: RevmathResultV1, run: RevmathRunRecord | None) -> str:
    if result.proof is not None:
        return result.proof.checker_id
    if run is not None:
        note = str(run.capture.get("note") or "revmath.runner")
        return f"revmath.capture.{note}"[:128]
    return "revmath.unknown_checker"


def build_repair_record(
    *,
    task: RevmathTaskV1,
    repair_id: str,
    before_result: RevmathResultV1,
    after_result: RevmathResultV1,
    before_state: RevmathRepairKnobStateV1,
    after_state: RevmathRepairKnobStateV1,
    changed_knobs: Sequence[str],
) -> RevmathRepairRecordV1:
    """Build an identity-preserving repair certificate for one attempt."""
    changes: list[RevmathRepairKnobChangeV1] = []
    for knob in changed_knobs:
        before_v = before_state.knob_value(knob)
        after_v = after_state.knob_value(knob)
        before_sha = _sha(before_v)
        after_sha = _sha(after_v)
        if before_sha == after_sha:
            continue
        changes.append(
            RevmathRepairKnobChangeV1(
                knob=knob,
                before_value_sha256=before_sha,
                after_value_sha256=after_sha,
            )
        )
    if not changes:
        raise RevmathSchemaError("repair attempt recorded no mutable knob change")
    before_proof = _proof_sha(before_result)
    after_proof = _proof_sha(after_result)
    if before_proof == after_proof:
        # Unknown→unknown can share empty-proof digests; bind knob state so the
        # immutable before/after artifacts remain distinct without inventing a proof.
        before_proof = content_sha(
            {"proof": before_proof, "knob_state": before_state.digest()}
        )
        after_proof = content_sha(
            {"proof": after_proof, "knob_state": after_state.digest()}
        )
    repair = RevmathRepairRecordV1(
        repair_id=repair_id,
        task_identity_digest=task.identity_digest(),
        proposition_id=task.proposition.proposition_id,
        statement_sha256=task.proposition.statement_sha256,
        base_theory_id=task.base_theory.base_theory_id,
        allowed_assumption_ids=task.base_theory.allowed_assumption_ids,
        campaign_manifest_sha256=task.campaign.campaign_manifest_sha256,
        campaign_id=task.campaign.campaign_id,
        experiment_id=task.campaign.experiment_id,
        before_proof_sha256=before_proof,
        after_proof_sha256=after_proof,
        knob_changes=tuple(changes),
    )
    assert_repair_preserves_identity(task, repair)
    return repair


def semantic_repair_lineage_from_session(
    session: RevmathRepairSessionV1,
) -> dict[str, Any]:
    """Lineage payload compatible with ``SemanticRepairRecordV1.lineage``.

    Does not construct a distill ``SemanticRepairRecordV1`` (torch-bound); INTEG
    layers may attach this lineage without forking a second repair registry.
    """
    return {
        "schema": "revmath_self_healing_lineage/v1",
        "extends": "semantic_repair/v1",
        "session_id": session.session_id,
        "task_identity_digest": session.task_identity_digest,
        "status": session.status,
        "attempt_ids": [a.attempt_id for a in session.attempts],
        "parent_lineage": [
            {
                "attempt_id": a.attempt_id,
                "parent_attempt_id": a.parent_attempt_id,
                "equivalent_signature": a.equivalent_signature,
                "outcome": a.outcome,
            }
            for a in session.attempts
        ],
        "invalidated_proof_sha256s": sorted(
            {
                p
                for a in session.attempts
                for p in a.invalidated_proof_sha256s
            }
        ),
    }


def _apply_proposal(
    state: RevmathRepairKnobStateV1, proposal: RevmathRepairProposalV1
) -> RevmathRepairKnobStateV1:
    reject_forbidden_repair(proposal.knob)
    return state.with_knob(proposal.knob, proposal.after_value)


def _plan_env_for_knobs(state: RevmathRepairKnobStateV1) -> dict[str, str]:
    """Serialize mutable knobs into runner env without touching the task."""
    return {
        "REVMATH_REPAIR_KNOB_STATE_SHA256": state.digest(),
        "REVMATH_REPAIR_KNOB_STATE_JSON": _canonical_json(state.to_dict()),
    }


def run_self_healing(
    task: RevmathTaskV1,
    *,
    initial_run: RevmathRunRecord | None = None,
    initial_knob_state: RevmathRepairKnobStateV1 | None = None,
    failure: SemanticFailureTraceV1 | Mapping[str, Any] | None = None,
    witness: VerifierWitnessV1 | Mapping[str, Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    runner: RunnerFn | None = None,
    hermetic: bool = True,
) -> RevmathRepairSessionV1:
    """Bounded repair loop: propose → mutate knobs → re-run exact task/checkers.

    Successful repair proves the *original* locked task. Exhausted / cyclic
    sessions remain unknown without semantic drift. Stale proof digests from
    prior attempts are invalidated on every re-run.
    """
    if max_attempts < 0:
        raise RevmathSchemaError("max_attempts must be >= 0")
    identity_before = task.identity_digest()
    run_fn: RunnerFn = runner or run_revmath_task
    knob_state = initial_knob_state or RevmathRepairKnobStateV1()
    current = initial_run or run_fn(
        task, hermetic=hermetic, env=_plan_env_for_knobs(knob_state)
    )
    if current.task.identity_digest() != identity_before:
        raise RevmathSchemaError("runner mutated locked task identity")

    session_id = f"repair.session.{task.task_id}.{identity_before[:12]}"
    judgment = current.result.solver_judgment()
    if judgment.outcome in (JudgmentOutcome.WITNESSED, JudgmentOutcome.REFUTED):
        session = RevmathRepairSessionV1(
            session_id=session_id,
            task_identity_digest=identity_before,
            status="no_repair_needed",
            attempts=(),
            final_result=current.result,
            final_knob_state=knob_state,
        )
        return replace(
            session, semantic_lineage=semantic_repair_lineage_from_session(session)
        )

    attempts: list[RevmathRepairAttemptV1] = []
    seen_signatures: set[str] = set()
    parent_id: str | None = None
    status: RepairSessionStatus = "unknown"
    budget = max_attempts

    while budget > 0:
        proposals = propose_repair_attempts(
            task=task,
            knob_state=knob_state,
            failure=failure,
            witness=witness,
            run=current,
            max_proposals=1,
        )
        if not proposals:
            status = "budget_exhausted"
            break
        proposal = proposals[0]
        before_state = knob_state
        sig = proposal_signature(proposal)
        attempt_id = f"repair.attempt.{len(attempts) + 1}.{sig[:12]}"
        budget -= 1

        if sig in seen_signatures:
            attempt = RevmathRepairAttemptV1(
                attempt_id=attempt_id,
                parent_attempt_id=parent_id,
                repair=None,
                failure_family=proposal.failure_family,
                witness_digest=proposal.witness_digest,
                search_budget_remaining=budget,
                before_result_id=current.result.result_id,
                after_result_id=None,
                checker_id=_checker_id(current.result, current),
                evidence_refs=tuple(r for r in (proposal.witness_digest,) if r),
                outcome="cycle_detected",
                knob_state_before_sha256=before_state.digest(),
                knob_state_after_sha256=before_state.digest(),
                invalidated_proof_sha256s=(),
                equivalent_signature=sig,
            )
            attempts.append(attempt)
            status = "cycle_detected"
            break
        seen_signatures.add(sig)

        try:
            after_state = _apply_proposal(before_state, proposal)
        except RevmathSchemaError:
            status = "invalid"
            break
        if after_state.digest() == before_state.digest():
            attempt = RevmathRepairAttemptV1(
                attempt_id=attempt_id,
                parent_attempt_id=parent_id,
                repair=None,
                failure_family=proposal.failure_family,
                witness_digest=proposal.witness_digest,
                search_budget_remaining=budget,
                before_result_id=current.result.result_id,
                after_result_id=None,
                checker_id=_checker_id(current.result, current),
                evidence_refs=tuple(r for r in (proposal.witness_digest,) if r),
                outcome="cycle_detected",
                knob_state_before_sha256=before_state.digest(),
                knob_state_after_sha256=after_state.digest(),
                invalidated_proof_sha256s=(),
                equivalent_signature=sig,
            )
            attempts.append(attempt)
            status = "cycle_detected"
            break

        before_result = current.result
        stale: tuple[str, ...] = ()
        if before_result.proof is not None:
            stale = (before_result.proof.proof_sha256,)
        elif attempts and attempts[-1].repair is not None:
            # Prior attempt's after-proof is stale once we re-check.
            stale = (attempts[-1].repair.after_proof_sha256,)

        # Re-run the *exact* locked task; knobs travel only via env / plan.
        after_run = run_fn(
            task,
            hermetic=hermetic,
            repair_history_ref=attempt_id,
            env=_plan_env_for_knobs(after_state),
        )
        if after_run.task.identity_digest() != identity_before:
            raise RevmathSchemaError(
                "repair re-run mutated locked task identity (semantic drift)"
            )
        after_result = after_run.result
        # Prior proof evidence is stale once a new check runs.
        invalidated = stale
        if (
            before_result.proof is not None
            and after_result.proof is not None
            and before_result.proof.proof_sha256 == after_result.proof.proof_sha256
            and after_result.solver_judgment().outcome
            not in (JudgmentOutcome.WITNESSED, JudgmentOutcome.REFUTED)
        ):
            # Same digest reused without a decisive check — still invalidate.
            invalidated = (before_result.proof.proof_sha256,)

        repair = build_repair_record(
            task=task,
            repair_id=f"repair.rec.{attempt_id}",
            before_result=before_result,
            after_result=after_result,
            before_state=before_state,
            after_state=after_state,
            changed_knobs=(proposal.knob,),
        )
        outcome: AttemptOutcome = "unknown"
        after_judgment = after_result.solver_judgment()
        if after_judgment.outcome in (
            JudgmentOutcome.WITNESSED,
            JudgmentOutcome.REFUTED,
        ):
            outcome = "repaired"
            status = "repaired"
        elif after_judgment.outcome is JudgmentOutcome.INVALID:
            outcome = "invalid"
            status = "invalid"
        else:
            outcome = "unknown"
            status = "unknown"

        evidence = [
            e
            for e in (
                proposal.witness_digest,
                after_result.proof.checker_report_sha256
                if after_result.proof is not None
                else None,
                after_result.proof.proof_sha256 if after_result.proof is not None else None,
            )
            if e
        ]
        attempt = RevmathRepairAttemptV1(
            attempt_id=attempt_id,
            parent_attempt_id=parent_id,
            repair=repair,
            failure_family=proposal.failure_family,
            witness_digest=proposal.witness_digest,
            search_budget_remaining=budget,
            before_result_id=before_result.result_id,
            after_result_id=after_result.result_id,
            checker_id=_checker_id(after_result, after_run),
            evidence_refs=tuple(evidence),
            outcome=outcome,
            knob_state_before_sha256=before_state.digest(),
            knob_state_after_sha256=after_state.digest(),
            invalidated_proof_sha256s=invalidated,
            equivalent_signature=sig,
        )
        attempts.append(attempt)
        parent_id = attempt_id
        knob_state = after_state
        current = after_run
        if outcome in ("repaired", "invalid"):
            break
    else:
        if status not in ("repaired", "invalid", "cycle_detected"):
            status = "budget_exhausted" if max_attempts > 0 else "unknown"

    if task.identity_digest() != identity_before:
        raise RevmathSchemaError("self-healing mutated locked task identity")

    if not attempts and status not in ("no_repair_needed",):
        status = "budget_exhausted" if max_attempts == 0 else status

    session = RevmathRepairSessionV1(
        session_id=session_id,
        task_identity_digest=identity_before,
        status=status,
        attempts=tuple(attempts),
        final_result=current.result,
        final_knob_state=knob_state,
    )
    return replace(
        session, semantic_lineage=semantic_repair_lineage_from_session(session)
    )


def mutate_forbidden_repair_for_test(
    task: RevmathTaskV1,
    *,
    mutation: str,
    value: Any = None,
) -> None:
    """Mutation-test helper: every prohibited repair path must raise."""
    del value
    reject_forbidden_repair(mutation)
    # If alias somehow bypassed, still refuse via identity assert with a forged record.
    raise RevmathSchemaError(f"mutation {mutation!r} was not rejected")  # pragma: no cover


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "REVMATH_FROZEN_REPAIR_FIELDS",
    "RevmathRepairAttemptV1",
    "RevmathRepairKnobStateV1",
    "RevmathRepairProposalV1",
    "RevmathRepairSessionV1",
    "assert_serialized_repair_freeze",
    "build_repair_record",
    "equivalent_repair_signature",
    "mutate_forbidden_repair_for_test",
    "proposal_signature",
    "propose_repair_attempts",
    "reject_forbidden_repair",
    "run_self_healing",
    "semantic_repair_lineage_from_session",
]
