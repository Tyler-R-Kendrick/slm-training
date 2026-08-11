"""Conservative reverse-mathematics labeling + anti-overclaim (HARN-08 / SLM-537).

Big-Five subsystem labels (``RCA0`` / ``WKL0`` / …) are never inferred from
Lean compile success, ``#print axioms`` footprints, finite bounded analogues,
classical existence proofs, or empirical measurements. Genuine equivalence
requires an explicit base theory, interpretation/coding, forward implication,
and reversal evidence (HARN-05). Assumption ablation stays
``rm_inspired_assumption_minimization`` unless that full package is present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha
from slm_training.formal.computability_classification import (
    PRACTICAL_CLASSES,
    ComputabilityClassificationError,
    is_rm_research_label,
    normalize_class_id,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    KNOWN_RM_SUBSYSTEMS,
    InterpretationStatus,
    RevmathSchemaError,
    RevmathTaskKind,
)

LABEL_CLAIM_SCHEMA = "revmath_label_claim/v1"
LABEL_VERDICT_SCHEMA = "revmath_label_verdict/v1"

RmLabelingState = Literal[
    "practical_computability_only",
    "candidate_upper_bound",
    "interpreted",
    "reversed_equivalent",
    "counterexample_known",
    "unclassified",
]

AnalysisKind = Literal[
    "genuine_reverse_mathematics",
    "rm_inspired_assumption_minimization",
    "practical_computability",
    "unclassified_analysis",
]

EvidenceKind = Literal[
    "print_axioms_only",
    "lean_compile_success",
    "no_project_axioms",
    "finite_bounded_analogue",
    "classical_existence",
    "empirical_claim",
    "interpretation_package",
    "forward_implication",
    "reversal",
    "checked_counterexample",
    "assumption_ablation_lattice",
    "practical_computability_certificate",
]

# Evidence that can never alone justify a Big-Five subsystem label.
INSUFFICIENT_FOR_BIG_FIVE: frozenset[str] = frozenset(
    {
        "print_axioms_only",
        "lean_compile_success",
        "no_project_axioms",
        "finite_bounded_analogue",
        "classical_existence",
        "empirical_claim",
        "assumption_ablation_lattice",
    }
)

_EXPLICIT_INTERPRETATION: frozenset[InterpretationStatus] = frozenset(
    {"explicit_reversal", "explicit_interpretation"}
)

_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX64


@dataclass(frozen=True)
class RmLabelClaimV1:
    """Proposed RM / computability label with declared evidence."""

    schema_version: str
    claim_id: str
    labeling_state: RmLabelingState
    analysis_kind: AnalysisKind
    proposed_subsystem_id: str | None
    base_theory_id: str | None
    interpretation_status: InterpretationStatus
    coding_id: str | None
    forward_evidence_sha256: str | None
    reversal_evidence_sha256: str | None
    evidence_kinds: tuple[EvidenceKind, ...]
    practical_class: str | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "labeling_state": self.labeling_state,
            "analysis_kind": self.analysis_kind,
            "proposed_subsystem_id": self.proposed_subsystem_id,
            "base_theory_id": self.base_theory_id,
            "interpretation_status": self.interpretation_status,
            "coding_id": self.coding_id,
            "forward_evidence_sha256": self.forward_evidence_sha256,
            "reversal_evidence_sha256": self.reversal_evidence_sha256,
            "evidence_kinds": list(self.evidence_kinds),
            "practical_class": self.practical_class,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RmLabelClaimV1:
        if not isinstance(data, Mapping):
            raise RevmathSchemaError("label claim must be a mapping")
        schema = str(data.get("schema_version", ""))
        if schema != LABEL_CLAIM_SCHEMA:
            raise RevmathSchemaError(
                f"label claim schema_version must be {LABEL_CLAIM_SCHEMA!r} "
                f"(got {schema!r})"
            )
        claim_id = str(data.get("claim_id", "")).strip()
        if not claim_id:
            raise RevmathSchemaError("label claim requires claim_id")
        state = str(data.get("labeling_state", ""))
        if state not in (
            "practical_computability_only",
            "candidate_upper_bound",
            "interpreted",
            "reversed_equivalent",
            "counterexample_known",
            "unclassified",
        ):
            raise RevmathSchemaError(f"unknown labeling_state {state!r}")
        analysis = str(data.get("analysis_kind", ""))
        if analysis not in (
            "genuine_reverse_mathematics",
            "rm_inspired_assumption_minimization",
            "practical_computability",
            "unclassified_analysis",
        ):
            raise RevmathSchemaError(f"unknown analysis_kind {analysis!r}")
        status = str(data.get("interpretation_status", ""))
        if status not in (
            "explicit_reversal",
            "explicit_interpretation",
            "uninterpreted",
            "not_applicable",
        ):
            raise RevmathSchemaError(f"unknown interpretation_status {status!r}")
        raw_kinds = data.get("evidence_kinds", ())
        if not isinstance(raw_kinds, Sequence) or isinstance(raw_kinds, (str, bytes)):
            raise RevmathSchemaError("evidence_kinds must be a sequence")
        kinds: list[EvidenceKind] = []
        allowed = {
            "print_axioms_only",
            "lean_compile_success",
            "no_project_axioms",
            "finite_bounded_analogue",
            "classical_existence",
            "empirical_claim",
            "interpretation_package",
            "forward_implication",
            "reversal",
            "checked_counterexample",
            "assumption_ablation_lattice",
            "practical_computability_certificate",
        }
        for kind in raw_kinds:
            text = str(kind)
            if text not in allowed:
                raise RevmathSchemaError(f"unknown evidence_kind {text!r}")
            kinds.append(text)  # type: ignore[arg-type]
        subsystem = data.get("proposed_subsystem_id")
        base = data.get("base_theory_id")
        coding = data.get("coding_id")
        forward = data.get("forward_evidence_sha256")
        reverse = data.get("reversal_evidence_sha256")
        practical = data.get("practical_class")
        for digest_name, digest in (
            ("forward_evidence_sha256", forward),
            ("reversal_evidence_sha256", reverse),
        ):
            if digest is not None and not _is_hex64(str(digest)):
                raise RevmathSchemaError(f"{digest_name} must be 64 lowercase hex")
        practical_norm: str | None
        if practical is None:
            practical_norm = None
        else:
            text = str(practical)
            if is_rm_research_label(text):
                # HARN-08 stays production-facing; research labels need KERN-12
                # interpretation packages via computability_classification.
                raise RevmathSchemaError(
                    "practical_class cannot be rm_research:* on an RM label claim; "
                    "use proposed_subsystem_id + interpretation package, or the "
                    "KERN-12 computability_classification claim schema"
                )
            try:
                practical_norm = normalize_class_id(text)
            except ComputabilityClassificationError as exc:
                raise RevmathSchemaError(str(exc)) from exc
            if practical_norm not in PRACTICAL_CLASSES:
                raise RevmathSchemaError(
                    f"unknown practical_class {text!r} "
                    "(KERN-12 vocabulary; no ambiguous fall-through)"
                )
        return cls(
            schema_version=schema,
            claim_id=claim_id,
            labeling_state=state,  # type: ignore[arg-type]
            analysis_kind=analysis,  # type: ignore[arg-type]
            proposed_subsystem_id=None if subsystem is None else str(subsystem),
            base_theory_id=None if base is None else str(base),
            interpretation_status=status,  # type: ignore[arg-type]
            coding_id=None if coding is None else str(coding),
            forward_evidence_sha256=None if forward is None else str(forward),
            reversal_evidence_sha256=None if reverse is None else str(reverse),
            evidence_kinds=tuple(kinds),
            practical_class=practical_norm,
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class RmLabelVerdictV1:
    """Validated labeling outcome (fail-closed)."""

    schema_version: str
    claim_id: str
    accepted: bool
    labeling_state: RmLabelingState
    analysis_kind: AnalysisKind
    proposed_subsystem_id: str | None
    rejection_reasons: tuple[str, ...]
    verdict_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "accepted": self.accepted,
            "labeling_state": self.labeling_state,
            "analysis_kind": self.analysis_kind,
            "proposed_subsystem_id": self.proposed_subsystem_id,
            "rejection_reasons": list(self.rejection_reasons),
            "verdict_digest": self.verdict_digest,
        }


def parse_label_claim(data: Mapping[str, Any]) -> RmLabelClaimV1:
    return RmLabelClaimV1.from_dict(data)


def has_big_five_equivalence_package(claim: RmLabelClaimV1) -> bool:
    """True when base + coding + forward + reversal + explicit interpretation exist."""

    if claim.proposed_subsystem_id not in KNOWN_RM_SUBSYSTEMS:
        return False
    if not claim.base_theory_id:
        return False
    if claim.interpretation_status not in _EXPLICIT_INTERPRETATION:
        return False
    if not claim.coding_id:
        return False
    if not claim.forward_evidence_sha256 or not claim.reversal_evidence_sha256:
        return False
    kinds = set(claim.evidence_kinds)
    return (
        "interpretation_package" in kinds
        and "forward_implication" in kinds
        and "reversal" in kinds
    )


def anti_overclaim_reasons(claim: RmLabelClaimV1) -> tuple[str, ...]:
    """Return rejection reasons when the claim overreaches its evidence."""

    reasons: list[str] = []
    kinds = set(claim.evidence_kinds)
    subsystem = claim.proposed_subsystem_id
    wants_big_five = subsystem in KNOWN_RM_SUBSYSTEMS
    wants_equivalence = claim.labeling_state == "reversed_equivalent"
    wants_interpreted_rm = claim.labeling_state in (
        "interpreted",
        "reversed_equivalent",
        "candidate_upper_bound",
    ) and wants_big_five

    if wants_big_five and kinds and kinds <= INSUFFICIENT_FOR_BIG_FIVE:
        reasons.append(
            "Big-Five subsystem label cannot rest solely on insufficient evidence "
            f"{sorted(kinds)} (never #print axioms / compile / finite / classical / "
            "empirical alone)"
        )

    if "print_axioms_only" in kinds and wants_big_five:
        reasons.append(
            "#print axioms footprint alone cannot classify a theorem as "
            f"{subsystem!r}"
        )
    if "lean_compile_success" in kinds and wants_big_five and not has_big_five_equivalence_package(
        claim
    ):
        reasons.append(
            "Lean compile success cannot justify Big-Five label "
            f"{subsystem!r} without interpretation + reversal package"
        )
    if "no_project_axioms" in kinds and wants_big_five and not has_big_five_equivalence_package(
        claim
    ):
        reasons.append(
            "absence of project-specific axioms cannot justify Big-Five label "
            f"{subsystem!r}"
        )
    if "finite_bounded_analogue" in kinds and wants_big_five:
        if claim.interpretation_status not in _EXPLICIT_INTERPRETATION or not claim.coding_id:
            reasons.append(
                "finite bounded analogue is labeled by constructive/computability "
                "status unless a real interpretation/coding exists"
            )
    if "classical_existence" in kinds and wants_big_five:
        reasons.append(
            "classical existence proof cannot be labeled as Big-Five subsystem "
            f"{subsystem!r} without interpretation + reversal"
        )
    if "empirical_claim" in kinds and (
        wants_big_five or claim.analysis_kind == "genuine_reverse_mathematics"
    ):
        reasons.append(
            "empirical claims cannot support genuine reverse-mathematics or "
            "Big-Five labels"
        )

    if wants_equivalence and not has_big_five_equivalence_package(claim):
        reasons.append(
            "reversed_equivalent / Big-Five equivalence requires explicit "
            "base_theory_id, coding_id, interpretation_status in "
            "{explicit_interpretation, explicit_reversal}, forward_evidence_sha256, "
            "reversal_evidence_sha256, and evidence_kinds "
            "{interpretation_package, forward_implication, reversal}"
        )

    if wants_interpreted_rm and claim.labeling_state == "interpreted":
        if claim.interpretation_status not in _EXPLICIT_INTERPRETATION or not claim.coding_id:
            reasons.append(
                "interpreted state requires explicit interpretation_status and coding_id"
            )
        if not claim.forward_evidence_sha256 or "forward_implication" not in kinds:
            reasons.append(
                "interpreted state requires forward implication evidence "
                "(reversal not yet claimed)"
            )
        if claim.reversal_evidence_sha256 or "reversal" in kinds:
            reasons.append(
                "interpreted (non-equivalent) claims must not attach reversal "
                "evidence; use reversed_equivalent"
            )

    if claim.analysis_kind == "rm_inspired_assumption_minimization":
        if wants_big_five or claim.labeling_state in (
            "interpreted",
            "reversed_equivalent",
        ):
            reasons.append(
                "RM-inspired assumption minimization (HARN-04 ablation) is not "
                "genuine reverse mathematics; refuse Big-Five / interpreted / "
                "reversed_equivalent labels"
            )

    if claim.analysis_kind == "genuine_reverse_mathematics":
        if claim.labeling_state == "practical_computability_only":
            reasons.append(
                "genuine_reverse_mathematics analysis_kind conflicts with "
                "practical_computability_only state"
            )
        if "assumption_ablation_lattice" in kinds and not has_big_five_equivalence_package(
            claim
        ):
            reasons.append(
                "assumption-ablation lattice alone is RM-inspired minimization, "
                "not genuine reverse mathematics"
            )

    if claim.labeling_state == "practical_computability_only":
        if wants_big_five:
            reasons.append(
                "practical_computability_only cannot carry a Big-Five subsystem id"
            )
        if claim.analysis_kind not in (
            "practical_computability",
            "unclassified_analysis",
        ):
            reasons.append(
                "practical_computability_only requires analysis_kind "
                "practical_computability (or unclassified_analysis)"
            )

    if claim.labeling_state == "counterexample_known":
        if "checked_counterexample" not in kinds:
            reasons.append(
                "counterexample_known requires checked_counterexample evidence"
            )

    if claim.labeling_state == "candidate_upper_bound" and wants_big_five:
        if claim.interpretation_status not in _EXPLICIT_INTERPRETATION:
            reasons.append(
                "candidate_upper_bound for a Big-Five id still needs explicit "
                "interpretation (forward upper bound, not equivalence)"
            )

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)


def validate_label_claim(claim: RmLabelClaimV1) -> RmLabelVerdictV1:
    """Accept or reject a labeling claim; never silently upgrade evidence."""

    reasons = anti_overclaim_reasons(claim)
    accepted = not reasons
    body = {
        "claim_id": claim.claim_id,
        "accepted": accepted,
        "labeling_state": claim.labeling_state,
        "analysis_kind": claim.analysis_kind,
        "proposed_subsystem_id": claim.proposed_subsystem_id,
        "rejection_reasons": list(reasons),
        "claim_digest": content_sha(claim.to_dict()),
    }
    digest = content_sha(body)
    return RmLabelVerdictV1(
        schema_version=LABEL_VERDICT_SCHEMA,
        claim_id=claim.claim_id,
        accepted=accepted,
        labeling_state=claim.labeling_state,
        analysis_kind=claim.analysis_kind,
        proposed_subsystem_id=claim.proposed_subsystem_id,
        rejection_reasons=reasons,
        verdict_digest=digest,
    )


def assert_label_claim_accepted(claim: RmLabelClaimV1) -> RmLabelVerdictV1:
    verdict = validate_label_claim(claim)
    if not verdict.accepted:
        raise RevmathSchemaError(
            "RM labeling overclaim rejected: " + "; ".join(verdict.rejection_reasons)
        )
    return verdict


def default_analysis_kind_for_task(task_kind: RevmathTaskKind) -> AnalysisKind:
    """Reports: ablation ≠ genuine RM; classification ≠ Big-Five."""

    if task_kind == "assumption_ablation":
        return "rm_inspired_assumption_minimization"
    if task_kind in ("reversal", "forward_theorem"):
        return "genuine_reverse_mathematics"
    if task_kind in (
        "computability_classification",
        "constructivization",
        "computable_finite_counterexample",
        "quantitative_bound_extraction",
    ):
        return "practical_computability"
    return "unclassified_analysis"


def report_labeling_disclaimer(
    *,
    analysis_kind: AnalysisKind,
    labeling_state: RmLabelingState,
) -> str:
    """Human-facing distinction for reports / docs examples."""

    if analysis_kind == "rm_inspired_assumption_minimization":
        return (
            "RM-inspired assumption minimization over an explored finite lattice "
            f"(state={labeling_state}); not a Big-Five reverse-mathematics "
            "equivalence claim."
        )
    if analysis_kind == "genuine_reverse_mathematics":
        return (
            "Genuine reverse mathematics requires explicit base theory, "
            "interpretation/coding, forward implication, and (for equivalence) "
            f"reversal; current state={labeling_state}."
        )
    if analysis_kind == "practical_computability":
        return (
            "Practical computability / constructive status only "
            f"(state={labeling_state}); not an RCA0/WKL0/… classification."
        )
    return f"Unclassified analysis (state={labeling_state})."


__all__ = [
    "AnalysisKind",
    "EvidenceKind",
    "INSUFFICIENT_FOR_BIG_FIVE",
    "LABEL_CLAIM_SCHEMA",
    "LABEL_VERDICT_SCHEMA",
    "RmLabelClaimV1",
    "RmLabelVerdictV1",
    "RmLabelingState",
    "anti_overclaim_reasons",
    "assert_label_claim_accepted",
    "default_analysis_kind_for_task",
    "has_big_five_equivalence_package",
    "parse_label_claim",
    "report_labeling_disclaimer",
    "validate_label_claim",
]
