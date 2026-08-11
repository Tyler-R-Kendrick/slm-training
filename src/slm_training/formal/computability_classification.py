"""KERN-12 / SLM-532: production practical computability classification.

Extends HARN-08 conservative RM labeling and KERN-10 / EVID-03 parity labels
into a single vocabulary with testable evidence contracts. This is **not** a
parallel reverse-mathematics classifier: Big-Five / ``rm_research:*`` labels
require an explicit interpretation package (never ``#print axioms`` alone).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from slm_training.harness_core.lineage.records import content_sha

SCHEMA = "computability_classification/v1"
ISSUE = "SLM-532"
KERN = "KERN-12"

# Production practical classes (explicit unclassified is allowed; no silent default).
PracticalComputabilityClass = Literal[
    "finite_decidable",
    "primitive_recursive",
    "bounded_search",
    "total_recursive",
    "semidecidable",
    "co_semidecidable",
    "oracle_relative",
    "classical_propositional",
    "noncomputable_existence",
    "unclassified",
]

PRACTICAL_CLASSES: frozenset[str] = frozenset(
    {
        "finite_decidable",
        "primitive_recursive",
        "bounded_search",
        "total_recursive",
        "semidecidable",
        "co_semidecidable",
        "oracle_relative",
        "classical_propositional",
        "noncomputable_existence",
        "unclassified",
    }
)

# KERN-10 / EVID-03 alias kept for fixture continuity; normalizes to
# noncomputable_existence (never a distinct production class).
LEGACY_ALIASES: dict[str, str] = {
    "classical_noncomputable_existence": "noncomputable_existence",
}

# Labels admitted on KERN-10 parity fixtures (canonical + legacy alias).
COMPUTABILITY_LABELS: frozenset[str] = PRACTICAL_CLASSES | frozenset(LEGACY_ALIASES)

EvidenceContractKind = Literal[
    "finite_vss_certificate",
    "finite_search_bound",
    "bounded_closure_certificate",
    "primitive_recursive_bound",
    "total_recursive_certificate",
    "semidecision_procedure",
    "co_semidecision_procedure",
    "oracle_declaration",
    "classical_propositional_marker",
    "noncomputable_existence_claim",
    "explicit_unclassified",
    "interpretation_package",
    "print_axioms_only",
    "lean_compile_success",
]

# Evidence that never upgrades to Big-Five / rm_research:* by itself.
INSUFFICIENT_FOR_RM_RESEARCH: frozenset[str] = frozenset(
    {
        "print_axioms_only",
        "lean_compile_success",
        "finite_vss_certificate",
        "finite_search_bound",
        "bounded_closure_certificate",
        "classical_propositional_marker",
        "noncomputable_existence_claim",
        "explicit_unclassified",
    }
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "computability_classification_fixtures.v1.json"
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "computability_classification.schema.json"
)

_HEX64 = frozenset("0123456789abcdef")


class ComputabilityClassificationError(ValueError):
    """Fail-closed classification / evidence contract error."""


@dataclass(frozen=True)
class ClassContractV1:
    """Meaning + evidence + implication law for one practical class."""

    class_id: str
    meaning: str
    required_evidence: frozenset[str]
    permitted_implications: frozenset[str]
    non_implications: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "meaning": self.meaning,
            "required_evidence": sorted(self.required_evidence),
            "permitted_implications": sorted(self.permitted_implications),
            "non_implications": sorted(self.non_implications),
        }


# Every production class has an explicit contract — no ambiguous fall-through.
CLASS_CONTRACTS: dict[str, ClassContractV1] = {
    "finite_decidable": ClassContractV1(
        class_id="finite_decidable",
        meaning=(
            "Decision procedure over a proved-complete finite domain "
            "(finite VSS / finite assignment product)."
        ),
        required_evidence=frozenset(
            {"finite_vss_certificate", "finite_search_bound"}
        ),
        permitted_implications=frozenset({"bounded_search"}),
        non_implications=frozenset(
            {
                "total_recursive",
                "semidecidable",
                "RCA0",
                "WKL0",
                "rm_research",
            }
        ),
    ),
    "primitive_recursive": ClassContractV1(
        class_id="primitive_recursive",
        meaning=(
            "Primitive-recursive / bounded-primitive computation justified by an "
            "explicit PR-shaped resource bound (not an unbounded recursion claim)."
        ),
        required_evidence=frozenset({"primitive_recursive_bound"}),
        permitted_implications=frozenset({"total_recursive", "bounded_search"}),
        non_implications=frozenset(
            {"finite_decidable", "semidecidable", "RCA0", "rm_research"}
        ),
    ),
    "bounded_search": ClassContractV1(
        class_id="bounded_search",
        meaning=(
            "Search over an explicitly bounded finite space (prefix-tree / "
            "closure-height bound); may be weaker than full finite decidability "
            "when the domain certificate is incomplete."
        ),
        required_evidence=frozenset(
            {"finite_search_bound", "bounded_closure_certificate"}
        ),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "finite_decidable",
                "semidecidable",
                "oracle_relative",
                "RCA0",
                "rm_research",
            }
        ),
    ),
    "total_recursive": ClassContractV1(
        class_id="total_recursive",
        meaning="Total recursive / computable total function with a totality certificate.",
        required_evidence=frozenset({"total_recursive_certificate"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {"finite_decidable", "primitive_recursive", "RCA0", "rm_research"}
        ),
    ),
    "semidecidable": ClassContractV1(
        class_id="semidecidable",
        meaning=(
            "Recursively enumerable / Σ₁: a procedure that halts on yes-instances; "
            "non-halting on no is not a decision."
        ),
        required_evidence=frozenset({"semidecision_procedure"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "co_semidecidable",
                "finite_decidable",
                "total_recursive",
                "RCA0",
                "rm_research",
            }
        ),
    ),
    "co_semidecidable": ClassContractV1(
        class_id="co_semidecidable",
        meaning=(
            "Co-RE / Π₁: a procedure that halts on no-instances; not by itself "
            "semidecidable or decidable."
        ),
        required_evidence=frozenset({"co_semidecision_procedure"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "semidecidable",
                "finite_decidable",
                "total_recursive",
                "RCA0",
                "rm_research",
            }
        ),
    ),
    "oracle_relative": ClassContractV1(
        class_id="oracle_relative",
        meaning=(
            "Computability relative to a named external oracle / black box; "
            "absolute computability is not claimed."
        ),
        required_evidence=frozenset({"oracle_declaration"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "finite_decidable",
                "total_recursive",
                "semidecidable",
                "RCA0",
                "rm_research",
            }
        ),
    ),
    "classical_propositional": ClassContractV1(
        class_id="classical_propositional",
        meaning=(
            "Classical propositional reasoning (LEM / DNE) without a constructive "
            "witness or decision procedure."
        ),
        required_evidence=frozenset({"classical_propositional_marker"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "finite_decidable",
                "total_recursive",
                "noncomputable_existence",
                "RCA0",
                "rm_research",
            }
        ),
    ),
    "noncomputable_existence": ClassContractV1(
        class_id="noncomputable_existence",
        meaning=(
            "Existence claim without a computable witness (classical existence / "
            "non-extractable witness)."
        ),
        required_evidence=frozenset({"noncomputable_existence_claim"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(
            {
                "finite_decidable",
                "total_recursive",
                "classical_propositional",
                "RCA0",
                "ACA0",
                "rm_research",
            }
        ),
    ),
    "unclassified": ClassContractV1(
        class_id="unclassified",
        meaning=(
            "Explicit refusal to classify; incomplete domain / missing evidence / "
            "unsupported construct. Never a silent default for a positive class."
        ),
        required_evidence=frozenset({"explicit_unclassified"}),
        permitted_implications=frozenset(),
        non_implications=frozenset(PRACTICAL_CLASSES - {"unclassified"}),
    ),
}

assert set(CLASS_CONTRACTS) == PRACTICAL_CLASSES


def fixture_path() -> Path:
    return _FIXTURE_PATH


def schema_path() -> Path:
    return _SCHEMA_PATH


def normalize_class_id(label: str) -> str:
    """Map legacy aliases onto canonical production ids."""

    text = str(label).strip()
    if not text:
        raise ComputabilityClassificationError("empty computability class id")
    if text.startswith("rm_research:"):
        return text
    return LEGACY_ALIASES.get(text, text)


def is_rm_research_label(label: str) -> bool:
    return str(label).startswith("rm_research:")


def parse_rm_research_subsystem(label: str) -> str:
    if not is_rm_research_label(label):
        raise ComputabilityClassificationError(
            f"not an rm_research label: {label!r}"
        )
    subsystem = label.split(":", 1)[1].strip()
    if not subsystem:
        raise ComputabilityClassificationError("rm_research label missing subsystem")
    return subsystem


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX64


@dataclass(frozen=True)
class InterpretationPackageV1:
    """Minimal package required before any ``rm_research:*`` label."""

    base_theory_id: str
    coding_id: str
    interpretation_status: str
    forward_evidence_sha256: str
    reversal_evidence_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_theory_id": self.base_theory_id,
            "coding_id": self.coding_id,
            "interpretation_status": self.interpretation_status,
            "forward_evidence_sha256": self.forward_evidence_sha256,
            "reversal_evidence_sha256": self.reversal_evidence_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> InterpretationPackageV1 | None:
        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ComputabilityClassificationError(
                "interpretation_package must be a mapping"
            )
        base = str(data.get("base_theory_id", "")).strip()
        coding = str(data.get("coding_id", "")).strip()
        status = str(data.get("interpretation_status", "")).strip()
        forward = data.get("forward_evidence_sha256")
        reverse = data.get("reversal_evidence_sha256")
        if not base or not coding:
            raise ComputabilityClassificationError(
                "interpretation_package requires base_theory_id and coding_id"
            )
        if status not in {"explicit_interpretation", "explicit_reversal"}:
            raise ComputabilityClassificationError(
                "interpretation_package requires interpretation_status in "
                "{explicit_interpretation, explicit_reversal}"
            )
        if forward is None or not _is_hex64(str(forward)):
            raise ComputabilityClassificationError(
                "interpretation_package requires forward_evidence_sha256 "
                "(64 lowercase hex)"
            )
        if reverse is not None and not _is_hex64(str(reverse)):
            raise ComputabilityClassificationError(
                "reversal_evidence_sha256 must be 64 lowercase hex when present"
            )
        return cls(
            base_theory_id=base,
            coding_id=coding,
            interpretation_status=status,
            forward_evidence_sha256=str(forward),
            reversal_evidence_sha256=None if reverse is None else str(reverse),
        )


@dataclass(frozen=True)
class ClassificationClaimV1:
    """A proposed practical / research classification with evidence."""

    schema_version: str
    claim_id: str
    class_id: str
    evidence_kinds: tuple[str, ...]
    source_family: str
    notes: str = ""
    interpretation_package: InterpretationPackageV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "class_id": self.class_id,
            "evidence_kinds": list(self.evidence_kinds),
            "source_family": self.source_family,
            "notes": self.notes,
        }
        if self.interpretation_package is not None:
            out["interpretation_package"] = self.interpretation_package.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClassificationClaimV1:
        if not isinstance(data, Mapping):
            raise ComputabilityClassificationError("claim must be a mapping")
        schema = str(data.get("schema_version", ""))
        if schema != SCHEMA:
            raise ComputabilityClassificationError(
                f"schema_version must be {SCHEMA!r} (got {schema!r})"
            )
        claim_id = str(data.get("claim_id", "")).strip()
        if not claim_id:
            raise ComputabilityClassificationError("claim_id required")
        class_id = normalize_class_id(str(data.get("class_id", "")))
        raw_kinds = data.get("evidence_kinds", ())
        if not isinstance(raw_kinds, Sequence) or isinstance(raw_kinds, (str, bytes)):
            raise ComputabilityClassificationError("evidence_kinds must be a sequence")
        kinds = tuple(str(k) for k in raw_kinds)
        source = str(data.get("source_family", "")).strip()
        if not source:
            raise ComputabilityClassificationError("source_family required")
        package = InterpretationPackageV1.from_dict(data.get("interpretation_package"))
        return cls(
            schema_version=schema,
            claim_id=claim_id,
            class_id=class_id,
            evidence_kinds=kinds,
            source_family=source,
            notes=str(data.get("notes", "")),
            interpretation_package=package,
        )


@dataclass(frozen=True)
class ClassificationVerdictV1:
    accepted: bool
    claim_id: str
    class_id: str
    rejection_reasons: tuple[str, ...]
    verdict_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "claim_id": self.claim_id,
            "class_id": self.class_id,
            "rejection_reasons": list(self.rejection_reasons),
            "verdict_digest": self.verdict_digest,
        }


def contract_for(class_id: str) -> ClassContractV1:
    canonical = normalize_class_id(class_id)
    if is_rm_research_label(canonical):
        raise ComputabilityClassificationError(
            "rm_research labels use interpretation_package validation, not "
            "CLASS_CONTRACTS"
        )
    if canonical not in CLASS_CONTRACTS:
        raise ComputabilityClassificationError(
            f"unknown computability class {class_id!r} "
            f"(canonical={canonical!r}); no ambiguous fall-through"
        )
    return CLASS_CONTRACTS[canonical]


def validate_practical_class_id(label: str) -> str:
    """Accept canonical / legacy practical labels; reject unknown and bare RM ids."""

    canonical = normalize_class_id(label)
    if is_rm_research_label(canonical):
        raise ComputabilityClassificationError(
            f"{label!r} is an rm_research label; use validate_classification_claim "
            "with an interpretation_package"
        )
    if canonical not in PRACTICAL_CLASSES:
        raise ComputabilityClassificationError(
            f"unknown practical computability class {label!r}; "
            "no ambiguous fall-through"
        )
    return canonical


def evidence_satisfies_contract(
    class_id: str, evidence_kinds: Sequence[str]
) -> tuple[bool, str | None]:
    """Return whether evidence meets the OR-of-required contract for ``class_id``."""

    contract = contract_for(class_id)
    kinds = set(evidence_kinds)
    if not kinds.intersection(contract.required_evidence):
        return (
            False,
            f"{class_id} requires at least one of "
            f"{sorted(contract.required_evidence)}; got {sorted(kinds)}",
        )
    return True, None


def anti_rm_research_reasons(
    *,
    class_id: str,
    evidence_kinds: Sequence[str],
    package: InterpretationPackageV1 | None,
) -> tuple[str, ...]:
    """Reject Big-Five / rm_research upgrades from insufficient evidence."""

    reasons: list[str] = []
    kinds = set(evidence_kinds)
    if not is_rm_research_label(class_id):
        return ()

    if "print_axioms_only" in kinds:
        subsystem = parse_rm_research_subsystem(class_id)
        reasons.append(
            f"#print axioms alone cannot classify a theorem as "
            f"rm_research:{subsystem} / {subsystem}"
        )
    if kinds and kinds <= INSUFFICIENT_FOR_RM_RESEARCH:
        reasons.append(
            "rm_research:* cannot rest solely on insufficient evidence "
            f"{sorted(kinds)} (need interpretation_package)"
        )
    if package is None:
        reasons.append(
            "rm_research:* requires an explicit interpretation_package "
            "(base_theory_id, coding_id, explicit interpretation/reversal, "
            "forward_evidence_sha256)"
        )
    elif "interpretation_package" not in kinds:
        reasons.append(
            "rm_research:* evidence_kinds must include interpretation_package"
        )
    return tuple(reasons)


def validate_classification_claim(claim: ClassificationClaimV1) -> ClassificationVerdictV1:
    """Accept or reject; never silently upgrade evidence or invent a class."""

    reasons: list[str] = []
    class_id = claim.class_id

    if is_rm_research_label(class_id):
        reasons.extend(
            anti_rm_research_reasons(
                class_id=class_id,
                evidence_kinds=claim.evidence_kinds,
                package=claim.interpretation_package,
            )
        )
    else:
        try:
            validate_practical_class_id(class_id)
        except ComputabilityClassificationError as exc:
            reasons.append(str(exc))
        else:
            ok, detail = evidence_satisfies_contract(class_id, claim.evidence_kinds)
            if not ok and detail:
                reasons.append(detail)
            if claim.interpretation_package is not None:
                reasons.append(
                    "practical computability classes must not carry an "
                    "interpretation_package; use rm_research:* only when a "
                    "genuine interpretation exists"
                )
            if "print_axioms_only" in claim.evidence_kinds and class_id != "unclassified":
                # print axioms never justifies a positive practical class alone
                if set(claim.evidence_kinds) <= {"print_axioms_only", "lean_compile_success"}:
                    reasons.append(
                        "#print axioms / Lean compile alone cannot assign "
                        f"practical class {class_id!r}"
                    )

    accepted = not reasons
    body = {
        "claim_id": claim.claim_id,
        "accepted": accepted,
        "class_id": class_id,
        "rejection_reasons": list(reasons),
        "claim_digest": content_sha(claim.to_dict()),
    }
    return ClassificationVerdictV1(
        accepted=accepted,
        claim_id=claim.claim_id,
        class_id=class_id,
        rejection_reasons=tuple(reasons),
        verdict_digest=content_sha(body),
    )


def assert_classification_accepted(claim: ClassificationClaimV1) -> ClassificationVerdictV1:
    verdict = validate_classification_claim(claim)
    if not verdict.accepted:
        raise ComputabilityClassificationError(
            "computability classification rejected: "
            + "; ".join(verdict.rejection_reasons)
        )
    return verdict


def load_fixtures() -> dict[str, Any]:
    doc = json.loads(fixture_path().read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ComputabilityClassificationError(
            f"stale_schema_version: expected {SCHEMA!r} got {doc.get('schema')!r}"
        )
    if doc.get("issue") != ISSUE or doc.get("kern") != KERN:
        raise ComputabilityClassificationError("fixture issue/kern mismatch")
    return doc


def evaluate_fixture_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a fixture case; compare to frozen expect.class_id."""

    claim = ClassificationClaimV1.from_dict(case["claim"])
    verdict = validate_classification_claim(claim)
    expect = case.get("expect", {})
    expected_accepted = bool(expect.get("accepted", True))
    expected_class = expect.get("class_id")
    if expected_class is not None:
        expected_class = normalize_class_id(str(expected_class))
    if verdict.accepted != expected_accepted:
        raise ComputabilityClassificationError(
            f"parity mismatch case={case.get('id')} field=accepted "
            f"expected={expected_accepted!r} got={verdict.accepted!r}"
        )
    if expected_accepted and expected_class is not None and verdict.class_id != expected_class:
        raise ComputabilityClassificationError(
            f"parity mismatch case={case.get('id')} field=class_id "
            f"expected={expected_class!r} got={verdict.class_id!r}"
        )
    if not expected_accepted:
        pattern = str(expect.get("rejection_contains", ""))
        blob = " | ".join(verdict.rejection_reasons)
        if pattern and pattern.lower() not in blob.lower():
            raise ComputabilityClassificationError(
                f"parity mismatch case={case.get('id')} field=rejection_contains "
                f"expected substring {pattern!r} in {blob!r}"
            )
    return {
        "case_id": case["id"],
        "accepted": verdict.accepted,
        "class_id": verdict.class_id,
        "rejection_reasons": list(verdict.rejection_reasons),
        "lean_label": case.get("lean_label"),
        "digest": verdict.verdict_digest,
    }


def evaluate_all() -> dict[str, Any]:
    doc = load_fixtures()
    results = [evaluate_fixture_case(case) for case in doc["cases"]]
    return {
        "schema": SCHEMA,
        "issue": ISSUE,
        "kern": KERN,
        "results": results,
        "results_sha256": content_sha(results),
        "contracts_sha256": content_sha(
            {k: v.to_dict() for k, v in sorted(CLASS_CONTRACTS.items())}
        ),
    }


def lean_label_string(class_id: str) -> str:
    """Stable string Lean ``#guard`` mirrors (canonical id only)."""

    if is_rm_research_label(class_id):
        return class_id
    return validate_practical_class_id(class_id)


__all__ = [
    "CLASS_CONTRACTS",
    "COMPUTABILITY_LABELS",
    "ClassificationClaimV1",
    "ClassificationVerdictV1",
    "ClassContractV1",
    "ComputabilityClassificationError",
    "EvidenceContractKind",
    "INSUFFICIENT_FOR_RM_RESEARCH",
    "ISSUE",
    "InterpretationPackageV1",
    "KERN",
    "LEGACY_ALIASES",
    "PRACTICAL_CLASSES",
    "PracticalComputabilityClass",
    "SCHEMA",
    "anti_rm_research_reasons",
    "assert_classification_accepted",
    "contract_for",
    "evaluate_all",
    "evaluate_fixture_case",
    "evidence_satisfies_contract",
    "fixture_path",
    "is_rm_research_label",
    "lean_label_string",
    "load_fixtures",
    "normalize_class_id",
    "parse_rm_research_subsystem",
    "schema_path",
    "validate_classification_claim",
    "validate_practical_class_id",
]
