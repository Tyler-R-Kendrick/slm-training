"""Deterministic G0-G12 verifier stack for OpenUI corpus rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

from slm_training.data.verify.runtime import RuntimeEvidence
from slm_training.dsl.grammar.backends.openui_lark import OpenUILarkBackend
from slm_training.dsl.lang_core import ParseError, validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord


class Gate(str, Enum):
    LEXICAL = "G0"
    GRAMMAR = "G1"
    SCHEMA = "G2"
    REFERENCES = "G3"
    DATAFLOW = "G4"
    RUNTIME = "G5"
    BEHAVIOR = "G6"
    GROUNDING = "G7"
    CANONICAL = "G8"
    PATCH = "G9"
    PROVENANCE = "G10"
    INDEPENDENT_JUDGE = "G11"
    HUMAN_AUDIT = "G12"


GATE_NAMES: dict[Gate, str] = {
    Gate.LEXICAL: "lexical",
    Gate.GRAMMAR: "grammar",
    Gate.SCHEMA: "schema",
    Gate.REFERENCES: "reference_graph",
    Gate.DATAFLOW: "dataflow",
    Gate.RUNTIME: "runtime",
    Gate.BEHAVIOR: "behavior",
    Gate.GROUNDING: "grounding",
    Gate.CANONICAL: "canonicalization",
    Gate.PATCH: "patch_correctness",
    Gate.PROVENANCE: "provenance",
    Gate.INDEPENDENT_JUDGE: "independent_judge",
    Gate.HUMAN_AUDIT: "human_audit",
}


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class Tier(str, Enum):
    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"
    QUARANTINE = "Quarantine"


PatchApplier = Callable[[str, Any], str]


@dataclass(frozen=True)
class VerificationContext:
    source_kind: str | None = None
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    runtime: RuntimeEvidence | None = None
    require_runtime: bool = False
    require_behavior: bool = False
    patch_before: str | None = None
    patch: Any = None
    patch_after: str | None = None
    patch_applier: PatchApplier | None = None
    provenance_complete: bool = True
    ambiguous: bool = False
    independent_judge_passed: bool | None = None
    human_audit_passed: bool | None = None

    @classmethod
    def from_record(cls, record: ExampleRecord) -> VerificationContext:
        """Build the default gate context from stable row metadata."""
        meta = record.meta
        runtime_data = meta.get("runtime_evidence")
        runtime = (
            RuntimeEvidence.from_dict(runtime_data)
            if isinstance(runtime_data, dict)
            else None
        )
        return cls(
            source_kind=str(meta.get("source_kind") or record.source),
            required_facts=tuple(str(x) for x in meta.get("required_facts") or ()),
            forbidden_facts=tuple(str(x) for x in meta.get("forbidden_facts") or ()),
            runtime=runtime,
            require_runtime=bool(meta.get("require_runtime", False)),
            require_behavior=bool(meta.get("require_behavior", False)),
            provenance_complete=bool(meta.get("provenance_complete", True)),
            ambiguous=bool(meta.get("ambiguous", False)),
            independent_judge_passed=(
                bool(meta["independent_judge_passed"])
                if "independent_judge_passed" in meta
                else None
            ),
            human_audit_passed=(
                bool(meta["human_audit_passed"])
                if "human_audit_passed" in meta
                else None
            ),
        )


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    status: GateStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not GateStatus.FAIL

    def to_dict(self) -> dict[str, str]:
        result = {
            "gate": self.gate.value,
            "name": GATE_NAMES[self.gate],
            "status": self.status.value,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass(frozen=True)
class VerificationReport:
    tier: Tier
    results: tuple[GateResult, ...]

    @property
    def failing_gate(self) -> Gate | None:
        return next(
            (result.gate for result in self.results if result.status is GateStatus.FAIL),
            None,
        )

    @property
    def ok(self) -> bool:
        return self.tier is not Tier.QUARANTINE

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "failing_gate": self.failing_gate.value if self.failing_gate else None,
            "gates": [result.to_dict() for result in self.results],
        }


_ASSIGNMENT_RE = re.compile(r"^\s*([a-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_NAME_RE = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\b")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_UNSUPPORTED_DATAFLOW_RE = re.compile(
    r"(?m)^\s*\$|\b(?:Query|Mutation|Action)\s*\(|@[A-Za-z_]"
)
_DETERMINISTIC_SOURCES = frozenset(
    {"fixture", "program", "program-first", "deterministic", "extracted", "rico"}
)
_WEAK_SOURCES = frozenset({"teacher", "web", "frontier", "distilled"})


def _pass(gate: Gate, detail: str = "") -> GateResult:
    return GateResult(gate, GateStatus.PASS, detail)


def _fail(gate: Gate, detail: str) -> GateResult:
    return GateResult(gate, GateStatus.FAIL, detail)


def _skip(gate: Gate, detail: str) -> GateResult:
    return GateResult(gate, GateStatus.SKIP, detail)


@lru_cache(maxsize=1)
def _grammar_backend() -> OpenUILarkBackend:
    return OpenUILarkBackend()


def _lexical(source: str) -> GateResult:
    gate = Gate.LEXICAL
    if not source.strip():
        return _fail(gate, "empty program")
    controls = [ch for ch in source if ord(ch) < 32 and ch not in "\n\r\t"]
    if controls:
        return _fail(gate, "forbidden control character")
    escaped = False
    in_string = False
    for char in source:
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
    if in_string or escaped:
        return _fail(gate, "unterminated string literal")
    return _pass(gate)


def _grammar(source: str) -> GateResult:
    try:
        _grammar_backend().validate(source)
    except (ParseError, ValueError) as exc:
        return _fail(Gate.GRAMMAR, str(exc).splitlines()[0][:200])
    return _pass(Gate.GRAMMAR)


def _schema(source: str) -> GateResult:
    try:
        validate(source)
    except (ParseError, RuntimeError, ValueError) as exc:
        return _fail(Gate.SCHEMA, str(exc).splitlines()[0][:200])
    return _pass(Gate.SCHEMA)


def _reference_graph(source: str) -> GateResult:
    expressions: dict[str, str] = {}
    for raw_line in source.splitlines():
        line = _strip_line_comment(raw_line).strip()
        if not line:
            continue
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        name, expression = match.groups()
        if name in expressions:
            return _fail(Gate.REFERENCES, f"duplicate binder: {name}")
        expressions[name] = expression
    if "root" not in expressions:
        return _fail(Gate.REFERENCES, "missing root binder")

    known = set(expressions)
    graph: dict[str, set[str]] = {}
    for name, expression in expressions.items():
        without_strings = _STRING_RE.sub("", expression)
        refs = {
            token
            for token in _NAME_RE.findall(without_strings)
            if token not in {"true", "false", "null"}
        }
        unresolved = refs - known
        if unresolved:
            return _fail(
                Gate.REFERENCES,
                f"unresolved reference: {sorted(unresolved)[0]}",
            )
        graph[name] = refs

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> str | None:
        if name in visiting:
            return name
        if name in visited:
            return None
        visiting.add(name)
        for child in sorted(graph[name]):
            cycle = visit(child)
            if cycle:
                return cycle
        visiting.remove(name)
        visited.add(name)
        return None

    cycle = visit("root")
    if cycle:
        return _fail(Gate.REFERENCES, f"reference cycle: {cycle}")
    unreachable = known - visited
    if unreachable:
        return _fail(
            Gate.REFERENCES,
            f"unreachable binder: {sorted(unreachable)[0]}",
        )
    return _pass(Gate.REFERENCES)


def _strip_line_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif char == "/" and not in_string and line[index : index + 2] == "//":
            return line[:index]
    return line


def _dataflow(source: str) -> GateResult:
    if _UNSUPPORTED_DATAFLOW_RE.search(_STRING_RE.sub("", source)):
        return _fail(
            Gate.DATAFLOW,
            "state/query/mutation/action/tool syntax is outside the pinned 0.2.x contract",
        )
    return _pass(Gate.DATAFLOW, "0.2.x layout-only contract")


def _runtime(context: VerificationContext) -> GateResult:
    evidence = context.runtime
    if evidence is None:
        if context.require_runtime:
            return _fail(Gate.RUNTIME, "runtime evidence required but missing")
        return _skip(Gate.RUNTIME, "not required")
    if not evidence.rendered:
        return _fail(Gate.RUNTIME, "preview did not render")
    if evidence.console_errors:
        return _fail(Gate.RUNTIME, evidence.console_errors[0][:200])
    return _pass(Gate.RUNTIME)


def _behavior(context: VerificationContext) -> GateResult:
    evidence = context.runtime
    if evidence is None:
        if context.require_behavior:
            return _fail(Gate.BEHAVIOR, "behavior evidence required but missing")
        return _skip(Gate.BEHAVIOR, "not required")
    if evidence.behavior_errors:
        return _fail(Gate.BEHAVIOR, evidence.behavior_errors[0][:200])
    if context.require_behavior and not evidence.interaction_trace:
        return _fail(Gate.BEHAVIOR, "required interaction trace is empty")
    return _pass(Gate.BEHAVIOR)


def _fact_present(source: str, fact: str) -> bool:
    kind, separator, value = fact.partition(":")
    if not separator:
        return fact in source
    if kind == "component":
        return value in _COMPONENT_RE.findall(source)
    if kind == "placeholder":
        token = value if value.startswith(":") else f":{value}"
        return token in extract_placeholders(source)
    if kind in {"contains", "text"}:
        return value in source
    return fact in source


def _grounding(source: str, context: VerificationContext) -> GateResult:
    if not context.required_facts and not context.forbidden_facts:
        return _skip(Gate.GROUNDING, "no fact contract")
    missing = [fact for fact in context.required_facts if not _fact_present(source, fact)]
    if missing:
        return _fail(Gate.GROUNDING, f"missing required fact: {missing[0]}")
    present = [fact for fact in context.forbidden_facts if _fact_present(source, fact)]
    if present:
        return _fail(Gate.GROUNDING, f"forbidden fact present: {present[0]}")
    return _pass(Gate.GROUNDING)


def _canonical(source: str) -> GateResult:
    try:
        first = (validate(source).serialized or source).strip()
        second = (validate(first).serialized or first).strip()
    except (ParseError, RuntimeError, ValueError) as exc:
        return _fail(Gate.CANONICAL, str(exc).splitlines()[0][:200])
    if first != second:
        return _fail(Gate.CANONICAL, "canonicalization is not idempotent")
    return _pass(Gate.CANONICAL)


def _canonical_or_strip(source: str) -> str:
    try:
        return (validate(source).serialized or source).strip()
    except (ParseError, RuntimeError, ValueError):
        return source.strip()


def _patch(context: VerificationContext) -> GateResult:
    values = (
        context.patch_before,
        context.patch_after,
        context.patch_applier,
    )
    if all(value is None for value in values) and context.patch is None:
        return _skip(Gate.PATCH, "no patch transition")
    if any(value is None for value in values):
        return _fail(Gate.PATCH, "incomplete patch transition evidence")
    assert context.patch_before is not None
    assert context.patch_after is not None
    assert context.patch_applier is not None
    try:
        actual = context.patch_applier(context.patch_before, context.patch)
    except Exception as exc:  # noqa: BLE001 - gate records the applier failure
        return _fail(Gate.PATCH, f"patch applier failed: {exc}")
    if _canonical_or_strip(actual) != _canonical_or_strip(context.patch_after):
        return _fail(Gate.PATCH, "apply(before, patch) != after")
    return _pass(Gate.PATCH)


def _provenance(context: VerificationContext) -> GateResult:
    if context.ambiguous:
        return _fail(Gate.PROVENANCE, "ambiguous record")
    if not context.provenance_complete:
        return _fail(Gate.PROVENANCE, "provenance incomplete")
    return _pass(Gate.PROVENANCE)


def _evidence_gate(gate: Gate, value: bool | None) -> GateResult:
    if value is None:
        return _skip(gate, "not supplied")
    if not value:
        return _fail(gate, f"{GATE_NAMES[gate]} rejected")
    return _pass(gate)


def evaluate_gate(
    gate: Gate,
    record: ExampleRecord,
    context: VerificationContext | None = None,
) -> GateResult:
    """Evaluate one gate independently for targeted tests and diagnostics."""
    context = context or VerificationContext.from_record(record)
    source = record.openui
    evaluators: dict[Gate, Callable[[], GateResult]] = {
        Gate.LEXICAL: lambda: _lexical(source),
        Gate.GRAMMAR: lambda: _grammar(source),
        Gate.SCHEMA: lambda: _schema(source),
        Gate.REFERENCES: lambda: _reference_graph(source),
        Gate.DATAFLOW: lambda: _dataflow(source),
        Gate.RUNTIME: lambda: _runtime(context),
        Gate.BEHAVIOR: lambda: _behavior(context),
        Gate.GROUNDING: lambda: _grounding(source, context),
        Gate.CANONICAL: lambda: _canonical(source),
        Gate.PATCH: lambda: _patch(context),
        Gate.PROVENANCE: lambda: _provenance(context),
        Gate.INDEPENDENT_JUDGE: lambda: _evidence_gate(
            gate, context.independent_judge_passed
        ),
        Gate.HUMAN_AUDIT: lambda: _evidence_gate(gate, context.human_audit_passed),
    }
    return evaluators[gate]()


def _tier(record: ExampleRecord, context: VerificationContext, results: tuple[GateResult, ...]) -> Tier:
    if any(result.status is GateStatus.FAIL for result in results):
        return Tier.QUARANTINE
    if context.human_audit_passed is True:
        return Tier.GOLD
    source_kind = (context.source_kind or record.source).lower()
    if source_kind in _DETERMINISTIC_SOURCES:
        return Tier.SILVER
    if source_kind in _WEAK_SOURCES:
        return Tier.BRONZE
    return Tier.BRONZE


def verify_record(
    record: ExampleRecord,
    context: VerificationContext | None = None,
) -> VerificationReport:
    context = context or VerificationContext.from_record(record)
    results = tuple(evaluate_gate(gate, record, context) for gate in Gate)
    return VerificationReport(tier=_tier(record, context, results), results=results)


def stamp_record(
    record: ExampleRecord,
    context: VerificationContext | None = None,
) -> ExampleRecord:
    """Return a copy carrying tier, first failure, and the complete gate report."""
    report = verify_record(record, context)
    verification = report.to_dict()
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=record.openui,
        placeholders=list(record.placeholders),
        split=record.split,
        source=record.source,
        meta={
            **dict(record.meta),
            "verification_tier": report.tier.value,
            "failing_gate": verification["failing_gate"],
            "verification": verification,
        },
        design_md=record.design_md,
    )
