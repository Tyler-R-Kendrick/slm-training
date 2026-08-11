"""Reversal task generation + bidirectional validation (HARN-05 / SLM-545).

Given base theory ``B``, candidate principle ``A``, and theorem ``T``, generate
and validate both proof obligations:

* forward: ``B + A ⊢ T``
* reverse: ``B + T ⊢ A``

A report may label the pair **equivalent** only when both checked directions
succeed over the same base theory and explicit interpretation status. One-way
implications, timeouts/unsupported, strength mismatches, hidden stronger
assumptions, and changed propositions are preserved explicitly and never
promoted to equivalence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    InterpretationStatus,
    RevmathPropositionIdentityV1,
    RevmathSchemaError,
    RevmathTaskV1,
)

REVERSAL_META_SCHEMA = "revmath_reversal_meta/v1"
REVERSAL_OBLIGATION_SCHEMA = "revmath_reversal_obligation/v1"
REVERSAL_REPORT_SCHEMA = "revmath_reversal_report/v1"

ReversalDirection = Literal["forward", "reverse"]
HermeticReversalScenario = Literal[
    "equivalence",
    "one_way_forward",
    "one_way_reverse",
    "timeout",
    "unsupported",
    "strength_mismatch",
    "hidden_stronger",
    "changed_proposition",
]
ReversalDirectionOutcome = Literal["witnessed", "refuted", "unknown", "invalid"]
# Report-level label — never "equivalent" unless both directions witnessed.
ReversalEquivalenceLabel = Literal[
    "equivalent",
    "one_way_forward",
    "one_way_reverse",
    "non_equivalent",
    "unknown",
    "invalid",
]

_EXPLICIT_INTERPRETATION: frozenset[str] = frozenset(
    {"explicit_reversal", "explicit_interpretation"}
)


def _sorted_ids(ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(ids)))


def _require_statement_digest(statement: str, digest: str, *, label: str) -> None:
    actual = content_sha(statement)
    if actual != digest:
        raise RevmathSchemaError(
            f"{label} statement_sha256 mismatch "
            f"(expected {digest!r}, got {actual!r})"
        )


@dataclass(frozen=True)
class ReversalStatementV1:
    """Frozen principle or theorem identity (exact statement + digest)."""

    statement_id: str
    statement: str
    statement_sha256: str
    strength_class: str = "unclassified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "statement": self.statement,
            "statement_sha256": self.statement_sha256,
            "strength_class": self.strength_class,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, label: str) -> ReversalStatementV1:
        statement = str(data["statement"])
        digest = str(data["statement_sha256"])
        _require_statement_digest(statement, digest, label=label)
        sid = str(data["statement_id"])
        if not sid:
            raise RevmathSchemaError(f"{label}.statement_id must be non-empty")
        return cls(
            statement_id=sid,
            statement=statement,
            statement_sha256=digest,
            strength_class=str(data.get("strength_class") or "unclassified"),
        )


@dataclass(frozen=True)
class ReversalMetaV1:
    """Frozen reversal sidecar (not part of RevmathTaskV1 identity)."""

    schema_version: str
    meta_id: str
    coding_id: str
    base_theory_id: str
    interpretation_status: InterpretationStatus
    principle: ReversalStatementV1
    theorem: ReversalStatementV1
    hermetic_scenario: HermeticReversalScenario
    oracle_by_direction: Mapping[str, str]
    proof_deps_by_direction: Mapping[str, tuple[str, ...]]
    import_edges: Mapping[str, tuple[str, ...]]
    strength_bearing_lemmas: tuple[str, ...]
    # When set, task.proposition.statement_sha256 must match (changed-prop guard).
    claim_statement_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "coding_id": self.coding_id,
            "base_theory_id": self.base_theory_id,
            "interpretation_status": self.interpretation_status,
            "principle": self.principle.to_dict(),
            "theorem": self.theorem.to_dict(),
            "hermetic_scenario": self.hermetic_scenario,
            "oracle_by_direction": dict(sorted(self.oracle_by_direction.items())),
            "proof_deps_by_direction": {
                key: list(deps)
                for key, deps in sorted(self.proof_deps_by_direction.items())
            },
            "import_edges": {
                lemma: list(assumps) for lemma, assumps in sorted(self.import_edges.items())
            },
            "strength_bearing_lemmas": list(self.strength_bearing_lemmas),
            "claim_statement_sha256": self.claim_statement_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReversalMetaV1:
        if data.get("schema_version") != REVERSAL_META_SCHEMA:
            raise RevmathSchemaError(
                f"expected schema {REVERSAL_META_SCHEMA!r}, "
                f"got {data.get('schema_version')!r}"
            )
        status = str(data.get("interpretation_status", ""))
        if status not in _EXPLICIT_INTERPRETATION:
            raise RevmathSchemaError(
                "reversal meta interpretation_status must be "
                "explicit_reversal or explicit_interpretation "
                f"(got {status!r}; never infer RM labels from axioms alone)"
            )
        scenario = str(data.get("hermetic_scenario", ""))
        allowed = (
            "equivalence",
            "one_way_forward",
            "one_way_reverse",
            "timeout",
            "unsupported",
            "strength_mismatch",
            "hidden_stronger",
            "changed_proposition",
        )
        if scenario not in allowed:
            raise RevmathSchemaError(f"unknown hermetic_scenario {scenario!r}")
        oracle_raw = data.get("oracle_by_direction") or {}
        deps_raw = data.get("proof_deps_by_direction") or {}
        edges_raw = data.get("import_edges") or {}
        if not isinstance(oracle_raw, Mapping) or not isinstance(deps_raw, Mapping):
            raise RevmathSchemaError(
                "oracle_by_direction and proof_deps_by_direction must be objects"
            )
        if not isinstance(edges_raw, Mapping):
            raise RevmathSchemaError("import_edges must be an object")
        coding_id = str(data.get("coding_id") or "")
        if not coding_id:
            raise RevmathSchemaError("coding_id must be non-empty")
        claim = data.get("claim_statement_sha256")
        claim_sha = str(claim) if claim else None
        if claim_sha is not None and len(claim_sha) != 64:
            raise RevmathSchemaError("claim_statement_sha256 must be 64 hex chars")
        return cls(
            schema_version=REVERSAL_META_SCHEMA,
            meta_id=str(data["meta_id"]),
            coding_id=coding_id,
            base_theory_id=str(data["base_theory_id"]),
            interpretation_status=status,  # type: ignore[arg-type]
            principle=ReversalStatementV1.from_dict(
                data["principle"], label="principle"
            ),
            theorem=ReversalStatementV1.from_dict(data["theorem"], label="theorem"),
            hermetic_scenario=scenario,  # type: ignore[arg-type]
            oracle_by_direction={str(k): str(v) for k, v in oracle_raw.items()},
            proof_deps_by_direction={
                str(k): _sorted_ids(v if isinstance(v, Sequence) else ())
                for k, v in deps_raw.items()
            },
            import_edges={
                str(lemma): _sorted_ids(assumps)
                for lemma, assumps in edges_raw.items()
            },
            strength_bearing_lemmas=_sorted_ids(
                data.get("strength_bearing_lemmas") or ()
            ),
            claim_statement_sha256=claim_sha,
        )


@dataclass(frozen=True)
class ReversalObligationV1:
    """One checked direction: forward ``B+A⊢T`` or reverse ``B+T⊢A``."""

    schema_version: str
    obligation_id: str
    direction: ReversalDirection
    base_theory_id: str
    coding_id: str
    interpretation_status: InterpretationStatus
    antecedent_assumption_ids: tuple[str, ...]
    consequent_statement_id: str
    consequent_statement_sha256: str
    obligation_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "obligation_id": self.obligation_id,
            "direction": self.direction,
            "base_theory_id": self.base_theory_id,
            "coding_id": self.coding_id,
            "interpretation_status": self.interpretation_status,
            "antecedent_assumption_ids": list(self.antecedent_assumption_ids),
            "consequent_statement_id": self.consequent_statement_id,
            "consequent_statement_sha256": self.consequent_statement_sha256,
            "obligation_digest": self.obligation_digest,
        }


@dataclass(frozen=True)
class ReversalDirectionRecordV1:
    """Separate evidence for one direction."""

    obligation: ReversalObligationV1
    outcome: ReversalDirectionOutcome
    judgment: Mapping[str, Any]
    proof_dependency_ids: tuple[str, ...]
    hidden_stronger_assumptions: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation.to_dict(),
            "outcome": self.outcome,
            "judgment": dict(self.judgment),
            "proof_dependency_ids": list(self.proof_dependency_ids),
            "hidden_stronger_assumptions": list(self.hidden_stronger_assumptions),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ReversalReportV1:
    """Bidirectional reversal report — equivalence only when both succeed."""

    schema_version: str
    task_id: str
    meta_id: str
    coding_id: str
    base_theory_id: str
    interpretation_status: InterpretationStatus
    forward: ReversalDirectionRecordV1
    reverse: ReversalDirectionRecordV1
    equivalence_label: ReversalEquivalenceLabel
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "meta_id": self.meta_id,
            "coding_id": self.coding_id,
            "base_theory_id": self.base_theory_id,
            "interpretation_status": self.interpretation_status,
            "forward": self.forward.to_dict(),
            "reverse": self.reverse.to_dict(),
            "equivalence_label": self.equivalence_label,
            "report_digest": self.report_digest,
        }

    @property
    def claims_equivalence(self) -> bool:
        return self.equivalence_label == "equivalent"


def parse_reversal_meta(data: Mapping[str, Any]) -> ReversalMetaV1:
    return ReversalMetaV1.from_dict(data)


def assert_meta_matches_task(task: RevmathTaskV1, meta: ReversalMetaV1) -> None:
    if task.task_kind != "reversal":
        raise RevmathSchemaError(
            f"reversal meta requires task_kind=reversal, got {task.task_kind!r}"
        )
    if task.base_theory.base_theory_id != meta.base_theory_id:
        raise RevmathSchemaError(
            "reversal meta base_theory_id must equal task.base_theory.base_theory_id "
            f"(task={task.base_theory.base_theory_id!r} meta={meta.base_theory_id!r})"
        )
    if task.base_theory.interpretation_status != meta.interpretation_status:
        raise RevmathSchemaError(
            "reversal meta interpretation_status must equal "
            "task.base_theory.interpretation_status "
            f"(task={task.base_theory.interpretation_status!r} "
            f"meta={meta.interpretation_status!r})"
        )
    if task.base_theory.interpretation_status not in _EXPLICIT_INTERPRETATION:
        raise RevmathSchemaError(
            "reversal tasks require explicit_reversal or explicit_interpretation "
            f"(got {task.base_theory.interpretation_status!r})"
        )
    if meta.claim_statement_sha256 is not None:
        if task.proposition.statement_sha256 != meta.claim_statement_sha256:
            raise RevmathSchemaError(
                "changed proposition: task.proposition.statement_sha256 does not "
                "match meta.claim_statement_sha256"
            )
    if meta.principle.statement_id == meta.theorem.statement_id:
        raise RevmathSchemaError(
            "principle.statement_id and theorem.statement_id must differ"
        )
    for lemma, assumps in meta.import_edges.items():
        spill = set(assumps) - {
            meta.principle.statement_id,
            meta.theorem.statement_id,
            *task.base_theory.allowed_assumption_ids,
        }
        if spill:
            raise RevmathSchemaError(
                f"import_edges[{lemma!r}] references unknown assumptions "
                f"{sorted(spill)}"
            )
    missing_strength = set(meta.strength_bearing_lemmas) - set(meta.import_edges)
    if missing_strength:
        raise RevmathSchemaError(
            "strength_bearing_lemmas missing import_edges: "
            f"{sorted(missing_strength)}"
        )


def generate_reversal_obligations(
    task: RevmathTaskV1,
    meta: ReversalMetaV1,
) -> tuple[ReversalObligationV1, ReversalObligationV1]:
    """Deterministically emit forward and reverse obligations (exact A/T)."""

    assert_meta_matches_task(task, meta)
    base_ids = _sorted_ids(task.base_theory.allowed_assumption_ids)

    def _one(
        direction: ReversalDirection,
        *,
        antecedent_extra: str,
        consequent: ReversalStatementV1,
    ) -> ReversalObligationV1:
        antecedents = _sorted_ids((*base_ids, antecedent_extra))
        body = {
            "direction": direction,
            "base_theory_id": meta.base_theory_id,
            "coding_id": meta.coding_id,
            "interpretation_status": meta.interpretation_status,
            "antecedent_assumption_ids": list(antecedents),
            "consequent_statement_id": consequent.statement_id,
            "consequent_statement_sha256": consequent.statement_sha256,
            "meta_id": meta.meta_id,
            "task_kind": "reversal",
        }
        digest = content_sha(body)
        return ReversalObligationV1(
            schema_version=REVERSAL_OBLIGATION_SCHEMA,
            obligation_id=f"{task.task_id}.obl.{direction}",
            direction=direction,
            base_theory_id=meta.base_theory_id,
            coding_id=meta.coding_id,
            interpretation_status=meta.interpretation_status,
            antecedent_assumption_ids=antecedents,
            consequent_statement_id=consequent.statement_id,
            consequent_statement_sha256=consequent.statement_sha256,
            obligation_digest=digest,
        )

    forward = _one(
        "forward",
        antecedent_extra=meta.principle.statement_id,
        consequent=meta.theorem,
    )
    reverse = _one(
        "reverse",
        antecedent_extra=meta.theorem.statement_id,
        consequent=meta.principle,
    )
    return forward, reverse


def materialize_obligation_task(
    task: RevmathTaskV1,
    meta: ReversalMetaV1,
    obligation: ReversalObligationV1,
) -> RevmathTaskV1:
    """Clone task for one direction; consequent statement is exact."""

    if obligation.direction == "forward":
        consequent = meta.theorem
    else:
        consequent = meta.principle
    if consequent.statement_sha256 != obligation.consequent_statement_sha256:
        raise RevmathSchemaError(
            "obligation consequent digest drifted from meta "
            f"(direction={obligation.direction!r})"
        )
    if consequent.statement_id != obligation.consequent_statement_id:
        raise RevmathSchemaError(
            "obligation consequent id drifted from meta "
            f"(direction={obligation.direction!r})"
        )
    prop = RevmathPropositionIdentityV1(
        proposition_id=consequent.statement_id,
        statement=consequent.statement,
        statement_sha256=consequent.statement_sha256,
    )
    direction_label = "equivalence" if task.theorem_direction == "equivalence" else "reversal"
    return task.model_copy(
        update={
            "task_id": obligation.obligation_id,
            "task_kind": "reversal",
            "theorem_direction": direction_label,
            "proposition": prop,
            "base_theory": task.base_theory.model_copy(
                update={
                    "allowed_assumption_ids": obligation.antecedent_assumption_ids,
                    "interpretation_status": meta.interpretation_status,
                    "base_theory_id": meta.base_theory_id,
                }
            ),
        }
    )


def audit_hidden_stronger(
    *,
    direction: ReversalDirection,
    proof_dependency_ids: Sequence[str],
    import_edges: Mapping[str, Sequence[str]],
    strength_bearing_lemmas: Sequence[str],
    forbidden_assumption_ids: Sequence[str],
) -> tuple[str, ...]:
    """Detect lemmas that smuggle forbidden (stronger) assumptions.

    For reverse ``B+T⊢A``, the principle ``A`` must not re-enter via a
    strength-bearing lemma. For forward ``B+A⊢T``, the theorem must not
    re-enter as a hidden stronger hypothesis.
    """

    forbidden = set(forbidden_assumption_ids)
    if not forbidden:
        return ()
    strength = set(strength_bearing_lemmas or import_edges.keys())
    hits: list[str] = []
    for dep in proof_dependency_ids:
        if dep in forbidden:
            hits.append(dep)
            continue
        if dep not in strength and dep not in import_edges:
            continue
        carried = set(import_edges.get(dep, ()))
        if carried & forbidden:
            hits.append(dep)
    return _sorted_ids(hits)


def _outcome_name(judgment: Mapping[str, Any]) -> ReversalDirectionOutcome:
    raw = str(judgment.get("outcome", "unknown"))
    if raw in ("witnessed", "refuted", "unknown", "invalid"):
        return raw  # type: ignore[return-value]
    return "unknown"


def proof_deps_from_capture(capture: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract proof_dependency_ids from a runner capture dict."""

    deps_raw = capture.get("proof_dependency_ids")
    if deps_raw is not None:
        if isinstance(deps_raw, str):
            return _sorted_ids(deps_raw.split(",")) if deps_raw else ()
        if isinstance(deps_raw, Sequence):
            return _sorted_ids(deps_raw)
    stdout = str(capture.get("stdout") or "")
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "proof_dependency_ids" in payload:
            raw = payload.get("proof_dependency_ids") or ()
            if isinstance(raw, str):
                return _sorted_ids(raw.split(",")) if raw else ()
            return _sorted_ids(raw)
    return ()


def _label_equivalence(
    *,
    forward: ReversalDirectionRecordV1,
    reverse: ReversalDirectionRecordV1,
    meta: ReversalMetaV1,
) -> ReversalEquivalenceLabel:
    """Fail closed: equivalent only when both directions witnessed cleanly."""

    reasons = forward.rejection_reasons + reverse.rejection_reasons
    if any(
        r.startswith("changed_proposition")
        or r.startswith("strength_mismatch")
        or r.startswith("hidden_stronger")
        for r in reasons
    ):
        return "invalid"
    if forward.outcome == "invalid" or reverse.outcome == "invalid":
        return "invalid"
    if meta.principle.strength_class != meta.theorem.strength_class:
        # Distinct strength classes cannot be equivalence even if both "prove".
        if forward.outcome == "witnessed" and reverse.outcome == "witnessed":
            return "invalid"
    if forward.outcome == "unknown" or reverse.outcome == "unknown":
        return "unknown"
    if forward.outcome == "witnessed" and reverse.outcome == "witnessed":
        if forward.hidden_stronger_assumptions or reverse.hidden_stronger_assumptions:
            return "invalid"
        return "equivalent"
    if forward.outcome == "witnessed" and reverse.outcome != "witnessed":
        return "one_way_forward"
    if reverse.outcome == "witnessed" and forward.outcome != "witnessed":
        return "one_way_reverse"
    return "non_equivalent"


def evaluate_reversal(
    task: RevmathTaskV1,
    meta: ReversalMetaV1,
    *,
    run_obligation: Callable[[RevmathTaskV1, ReversalObligationV1], Any],
) -> ReversalReportV1:
    """Generate both obligations, run each, and label equivalence fail-closed."""

    assert_meta_matches_task(task, meta)
    if task.theorem_direction not in ("reversal", "equivalence"):
        raise RevmathSchemaError(
            "evaluate_reversal requires theorem_direction in "
            "{'reversal','equivalence'}, "
            f"got {task.theorem_direction!r}"
        )

    forward_obl, reverse_obl = generate_reversal_obligations(task, meta)

    def _run_one(obligation: ReversalObligationV1) -> ReversalDirectionRecordV1:
        record = run_obligation(task, obligation)
        judgment = record.result.solver_judgment().to_dict()
        capture = getattr(record, "capture", {}) or {}
        deps = proof_deps_from_capture(capture)
        # Forbidden: the *other* side's strength smuggled back in.
        if obligation.direction == "forward":
            forbidden = (meta.theorem.statement_id,)
        else:
            forbidden = (meta.principle.statement_id,)
        hidden = audit_hidden_stronger(
            direction=obligation.direction,
            proof_dependency_ids=deps,
            import_edges=meta.import_edges,
            strength_bearing_lemmas=meta.strength_bearing_lemmas,
            forbidden_assumption_ids=forbidden,
        )
        reasons: list[str] = []
        if meta.hermetic_scenario == "changed_proposition":
            reasons.append("changed_proposition")
        if meta.hermetic_scenario == "strength_mismatch" or (
            meta.principle.strength_class != meta.theorem.strength_class
            and meta.hermetic_scenario == "strength_mismatch"
        ):
            reasons.append("strength_mismatch")
        if hidden:
            reasons.append("hidden_stronger:" + ",".join(hidden))
            if judgment.get("outcome") == JudgmentOutcome.WITNESSED.value:
                judgment = {
                    "schema": judgment.get("schema", "solver_judgment/v1"),
                    "outcome": "invalid",
                    "payload": {
                        "kind": "invalid_reason",
                        "reason": "payload_mismatch",
                    },
                    "checked": True,
                }
        if meta.hermetic_scenario == "changed_proposition" and judgment.get(
            "outcome"
        ) == JudgmentOutcome.WITNESSED.value:
            judgment = {
                "schema": judgment.get("schema", "solver_judgment/v1"),
                "outcome": "invalid",
                "payload": {
                    "kind": "invalid_reason",
                    "reason": "payload_mismatch",
                },
                "checked": True,
            }
            reasons.append("changed_proposition:reject_witness")
        if (
            meta.hermetic_scenario == "strength_mismatch"
            and judgment.get("outcome") == JudgmentOutcome.WITNESSED.value
        ):
            judgment = {
                "schema": judgment.get("schema", "solver_judgment/v1"),
                "outcome": "invalid",
                "payload": {
                    "kind": "invalid_reason",
                    "reason": "payload_mismatch",
                },
                "checked": True,
            }
            if "strength_mismatch" not in reasons:
                reasons.append("strength_mismatch")

        outcome = _outcome_name(judgment)
        if hidden and outcome == "witnessed":
            outcome = "invalid"
        return ReversalDirectionRecordV1(
            obligation=obligation,
            outcome=outcome,
            judgment=judgment,
            proof_dependency_ids=deps,
            hidden_stronger_assumptions=hidden,
            rejection_reasons=_sorted_ids(reasons),
        )

    forward_rec = _run_one(forward_obl)
    reverse_rec = _run_one(reverse_obl)
    label = _label_equivalence(forward=forward_rec, reverse=reverse_rec, meta=meta)

    payload = {
        "schema_version": REVERSAL_REPORT_SCHEMA,
        "task_id": task.task_id,
        "meta_id": meta.meta_id,
        "coding_id": meta.coding_id,
        "base_theory_id": meta.base_theory_id,
        "interpretation_status": meta.interpretation_status,
        "forward": forward_rec.to_dict(),
        "reverse": reverse_rec.to_dict(),
        "equivalence_label": label,
    }
    return ReversalReportV1(
        schema_version=REVERSAL_REPORT_SCHEMA,
        task_id=task.task_id,
        meta_id=meta.meta_id,
        coding_id=meta.coding_id,
        base_theory_id=meta.base_theory_id,
        interpretation_status=meta.interpretation_status,
        forward=forward_rec,
        reverse=reverse_rec,
        equivalence_label=label,
        report_digest=content_sha(payload),
    )


__all__ = [
    "REVERSAL_META_SCHEMA",
    "REVERSAL_OBLIGATION_SCHEMA",
    "REVERSAL_REPORT_SCHEMA",
    "ReversalDirectionRecordV1",
    "ReversalEquivalenceLabel",
    "ReversalMetaV1",
    "ReversalObligationV1",
    "ReversalReportV1",
    "ReversalStatementV1",
    "assert_meta_matches_task",
    "audit_hidden_stronger",
    "evaluate_reversal",
    "generate_reversal_obligations",
    "materialize_obligation_task",
    "parse_reversal_meta",
    "proof_deps_from_capture",
]
