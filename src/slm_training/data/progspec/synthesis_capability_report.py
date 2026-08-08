"""SGS-008 (SLM-455): fail-closed SyGuS/SemGuS capability reports.

Reports which parts of a ``VerifiedSynthesisProblemV1`` (SGS-007/SLM-444) can
be faithfully represented in a SyGuS-IF-inspired structural fragment, and
marks everything else with a typed, reasoned capability finding instead of
silently dropping it. This is a pure reporting/export adapter: it never
imports or calls an external SMT/SyGuS solver, so nothing here can widen
exact compiler support (AC4) -- there is no solver to consult in the first
place.

Fidelity boundary (deliberately narrow): a runtime symbol's *existence and
name* is representable as an opaque ``String``-sorted SMT-LIB variable
declaration; a ``symbolic_regression`` pack's finite, arity-checked operator
vocabulary (``dsl/symbolic_expr_ir.py``, SRP-002) is representable as a
``synth-fun`` grammar. Everything else this envelope can carry --
verification-gate/evaluator identities, prompt-side requirement facts and
ambiguity groups, multi-objective terms, search/resource budgets, evidence
provenance -- has no SyGuS-IF equivalent and is reported
``non_representable``/``extension_required`` with an explicit reason. This
follows the same typed-failure idiom ``dsl/grammar_capabilities.py``'s
``UnsupportedCapabilityV1`` already uses (``status``/``reason``), generalized
to three outcomes instead of one.

The emitted ``sygus_program`` text is **inspired interoperability, not
SyGuS-IF standards conformance** (AC5): it declares variables and a
synth-fun grammar but never emits ``(check-synth)`` (this module has no
access to the concrete example table a real synthesis run would need for
``(constraint ...)`` rows), and grammar productions may name functions
(``sin``/``cos``/``exp``/...) outside any decidable SMT-LIB background
logic. ``conformance``/``conformance_note`` on the report make this
distinction machine-readable, not just prose.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from slm_training.dsl.symbolic_expr_ir import BINARY_OPERATORS, UNARY_OPERATORS

if TYPE_CHECKING:
    from slm_training.data.progspec.synthesis_problem import VerifiedSynthesisProblemV1

CAPABILITY_REPORT_SCHEMA = "sygus_capability_report/v1"

CapabilityStatus = Literal["representable", "non_representable", "extension_required"]

_SMT_SIMPLE_SYMBOL_RE = re.compile(r"^[A-Za-z~!@$%^&*_+=<>.?/-][A-Za-z0-9~!@$%^&*_+=<>.?/-]*$")


def _smt_quoted_symbol(name: str) -> str:
    """Render ``name`` as an SMT-LIB symbol, pipe-quoting it when required.

    SMT-LIB simple symbols exclude many characters OpenUI runtime-symbol
    surfaces use freely (``:``, ``.``); a quoted symbol never needs an
    escape for anything but ``|``/``\\``, which this repo's surfaces never
    contain, so this always produces a valid token.
    """
    if _SMT_SIMPLE_SYMBOL_RE.match(name):
        return name
    return f"|{name}|"


@dataclass(frozen=True)
class CapabilityFindingV1:
    """One element's representability verdict -- never silently omitted."""

    element: str
    status: CapabilityStatus
    reason: str
    sygus_fragment: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "element": self.element,
            "status": self.status,
            "reason": self.reason,
            "sygus_fragment": self.sygus_fragment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CapabilityFindingV1":
        return cls(
            element=str(data["element"]),
            status=data["status"],  # type: ignore[arg-type]
            reason=str(data["reason"]),
            sygus_fragment=data.get("sygus_fragment"),  # type: ignore[arg-type]
        )


def _digest(report: "SyGuSCapabilityReportV1") -> str:
    payload = report.to_dict()
    payload.pop("report_hash", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SyGuSCapabilityReportV1:
    """Immutable, tamper-evident capability report over one problem envelope."""

    schema_version: str
    problem_id: str
    problem_digest: str
    pack_id: str
    findings: tuple[CapabilityFindingV1, ...]
    sygus_program: str | None
    conformance: Literal["inspired_interoperability"]
    conformance_note: str
    report_hash: str

    def representable_elements(self) -> tuple[str, ...]:
        return tuple(f.element for f in self.findings if f.status == "representable")

    def unsupported_elements(self) -> tuple[str, ...]:
        return tuple(f.element for f in self.findings if f.status != "representable")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "problem_digest": self.problem_digest,
            "pack_id": self.pack_id,
            "findings": [f.to_dict() for f in self.findings],
            "sygus_program": self.sygus_program,
            "conformance": self.conformance,
            "conformance_note": self.conformance_note,
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SyGuSCapabilityReportV1":
        if data.get("schema_version") != CAPABILITY_REPORT_SCHEMA:
            raise ValueError(
                f"unsupported SyGuS capability report schema: {data.get('schema_version')!r}"
            )
        report = cls(
            schema_version=data.get("schema_version", CAPABILITY_REPORT_SCHEMA),  # type: ignore[arg-type]
            problem_id=str(data["problem_id"]),
            problem_digest=str(data["problem_digest"]),
            pack_id=str(data["pack_id"]),
            findings=tuple(
                CapabilityFindingV1.from_dict(f) for f in data.get("findings", [])  # type: ignore[arg-type]
            ),
            sygus_program=data.get("sygus_program"),  # type: ignore[arg-type]
            conformance=data.get("conformance", "inspired_interoperability"),  # type: ignore[arg-type]
            conformance_note=str(data.get("conformance_note", "")),
            report_hash=str(data.get("report_hash", "")),
        )
        if _digest(report) != report.report_hash:
            raise ValueError(
                "SyGuS capability report hash mismatch (tampered or corrupted payload)"
            )
        return report


def capability_report_integrity_ok(report: SyGuSCapabilityReportV1) -> bool:
    """Recompute the report hash and compare; used by replay/tamper checks."""
    return _digest(report) == report.report_hash


_CONFORMANCE_NOTE = (
    "Inspired interoperability only, not SyGuS-IF standards conformance: "
    "sygus_program never emits (check-synth) (no concrete example table is "
    "available from the envelope alone), and synth-fun grammar productions "
    "may name functions outside any decidable SMT-LIB background logic."
)


def _symbolic_regression_grammar_fragment() -> tuple[str, str]:
    """A faithful ``synth-fun`` grammar mirroring symbolic_expr_ir.py's
    finite, arity-checked operator vocabulary (SRP-002)."""
    binary_rules = " ".join(f"(Start {op} Start)" for op in sorted(BINARY_OPERATORS))
    unary_rules = " ".join(f"({op} Start)" for op in sorted(UNARY_OPERATORS))
    grammar = (
        "((Start Real (Var Coef " + binary_rules + " " + unary_rules + ")))"
    )
    return grammar, (
        "Mirrors dsl/symbolic_expr_ir.py's BINARY_OPERATORS/UNARY_OPERATORS "
        "exactly -- every production here has a corresponding OpApply arity "
        "check in the typed IR."
    )


def build_sygus_capability_report(
    problem: "VerifiedSynthesisProblemV1",
) -> SyGuSCapabilityReportV1:
    """Classify every requirement-bearing element of ``problem`` and, where
    fidelity is demonstrable, emit a SyGuS-IF-inspired structural fragment.

    Never silently drops an element: every runtime symbol, verification
    requirement, prompt-side fact, ambiguity group, objective term, and any
    declared search-budget field gets exactly one :class:`CapabilityFindingV1`.
    """
    findings: list[CapabilityFindingV1] = []
    program_lines: list[str] = []
    pack_id = problem.pack_identity.pack_id
    is_symbolic_regression = pack_id == "symbolic_regression"

    if is_symbolic_regression:
        program_lines.append("(set-logic NRA)")

    for symbol in problem.runtime_symbols:
        smt_name = _smt_quoted_symbol(symbol.surface)
        if is_symbolic_regression:
            fragment = f"(declare-var {smt_name} Real)"
            reason = (
                "symbolic_regression runtime symbols are numeric input "
                "variables; a Real-sorted SMT-LIB declaration is exact."
            )
        else:
            fragment = f"(declare-var {smt_name} String)"
            reason = (
                "only the symbol's existence and name are representable "
                "as an opaque String-sorted declaration; its role "
                f"({symbol.role!r}) and any scope/content semantics are not."
            )
        findings.append(
            CapabilityFindingV1(
                element=f"runtime_symbols[{symbol.surface}]",
                status="representable",
                reason=reason,
                sygus_fragment=fragment,
            )
        )
        program_lines.append(fragment)

    if is_symbolic_regression and problem.runtime_symbols:
        params = " ".join(
            f"({_smt_quoted_symbol(s.surface)} Real)" for s in problem.runtime_symbols
        )
        grammar, grammar_reason = _symbolic_regression_grammar_fragment()
        synth_fun = f"(synth-fun expr ({params}) Real {grammar})"
        findings.append(
            CapabilityFindingV1(
                element="operator_vocabulary",
                status="representable",
                reason=grammar_reason,
                sygus_fragment=synth_fun,
            )
        )
        program_lines.append(synth_fun)

    for req in problem.verification_requirements:
        findings.append(
            CapabilityFindingV1(
                element=f"verification_requirements[{req.requirement_id}]",
                status="extension_required",
                reason=(
                    "this repo's verifier gate/evaluator identities "
                    "(G0-G12 gate names, evaluator version/hash pinning) "
                    "have no SyGuS-IF equivalent; the envelope alone also "
                    "carries no concrete example table to emit "
                    "(constraint ...) rows even for an evaluator "
                    "requirement, so this is extension-required, not "
                    "merely absent from a fixed vocabulary."
                ),
            )
        )

    for fact in problem.requirements.facts:
        findings.append(
            CapabilityFindingV1(
                element=f"requirements.facts[{fact.fact_id}]",
                status="non_representable",
                reason=(
                    "prompt-derived semantic requirement "
                    f"(factor_family={fact.factor_family!r}, "
                    f"disposition={fact.disposition!r}); SyGuS-IF has no "
                    "vocabulary for prompt ambiguity, preference, or "
                    "advisory disposition semantics."
                ),
            )
        )
    for group in problem.requirements.ambiguity_groups:
        findings.append(
            CapabilityFindingV1(
                element=f"requirements.ambiguity_groups[{group.group_id}]",
                status="non_representable",
                reason=(
                    "mutually exclusive prompt interpretations have no "
                    "SyGuS-IF representation."
                ),
            )
        )

    for term in problem.objective:
        findings.append(
            CapabilityFindingV1(
                element=f"objective[{term.name}]",
                status="extension_required",
                reason=(
                    "SyGuS-IF's core (check-synth) is satisfaction-only; "
                    "weighted multi-objective optimization requires a "
                    "SyGuS optimization extension not implemented here."
                ),
            )
        )

    budget = problem.search_budget
    declared_budget_fields = {
        name: value
        for name, value in (
            ("max_forwards", budget.max_forwards),
            ("max_wall_seconds", budget.max_wall_seconds),
            ("max_states", budget.max_states),
            ("max_verifier_calls", budget.max_verifier_calls),
        )
        if value is not None
    }
    if declared_budget_fields:
        findings.append(
            CapabilityFindingV1(
                element="search_budget",
                status="extension_required",
                reason=(
                    "resource/search budgets bound this repo's own search "
                    "authority; they are not part of SyGuS-IF's synthesis "
                    f"semantics (declared fields: {sorted(declared_budget_fields)})."
                ),
            )
        )

    for provenance in problem.evidence_provenance:
        findings.append(
            CapabilityFindingV1(
                element=f"evidence_provenance[{provenance.source}]",
                status="non_representable",
                reason=(
                    "evidence provenance/authority is process/audit "
                    "metadata, not a formal synthesis constraint."
                ),
            )
        )

    sygus_program = "\n".join(program_lines) if program_lines else None

    report = SyGuSCapabilityReportV1(
        schema_version=CAPABILITY_REPORT_SCHEMA,
        problem_id=problem.problem_id,
        problem_digest=problem.digest,
        pack_id=pack_id,
        findings=tuple(findings),
        sygus_program=sygus_program,
        conformance="inspired_interoperability",
        conformance_note=_CONFORMANCE_NOTE,
        report_hash="",
    )
    return replace(report, report_hash=_digest(report))


__all__ = [
    "CAPABILITY_REPORT_SCHEMA",
    "CapabilityFindingV1",
    "SyGuSCapabilityReportV1",
    "build_sygus_capability_report",
    "capability_report_integrity_ok",
]
