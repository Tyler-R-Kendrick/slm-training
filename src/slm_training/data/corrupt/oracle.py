"""Formal corruption taxonomy and minimal-repair row projection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.data.verify import VerificationContext, verify_record
from slm_training.dsl.lang_core import ParseError, validate
from slm_training.dsl.schema import ExampleRecord


class OperatorFamily(str, Enum):
    LEXICAL = "lexical"
    GRAMMAR = "grammar"
    SCHEMA = "schema"
    REFERENCE_GRAPH = "reference_graph"
    DATAFLOW = "dataflow"
    PATCH = "patch"


class CorruptionOperator(str, Enum):
    MISSING_QUOTE = "missing_quote"
    EXTRA_QUOTE = "extra_quote"
    INVALID_ESCAPE = "invalid_escape"
    MISSING_DELIMITER = "missing_delimiter"
    BROKEN_NUMBER_BOOL = "broken_number_bool"
    TRUNCATED_LINE = "truncated_line"
    CONCATENATED_STATEMENTS = "concatenated_statements"

    MISSING_ASSIGNMENT = "missing_assignment"
    TWO_STATEMENTS_PER_LINE = "two_statements_per_line"
    NAMED_WHERE_POSITIONAL = "named_where_positional"
    MALFORMED_ARRAY = "malformed_array"
    MALFORMED_OBJECT = "malformed_object"
    INVALID_NESTING = "invalid_nesting"
    MISSING_ROOT = "missing_root"
    DUPLICATE_ROOT = "duplicate_root"

    UNKNOWN_COMPONENT = "unknown_component"
    WRONG_CAPITALIZATION = "wrong_capitalization"
    WRONG_ARG_COUNT = "wrong_arg_count"
    SWAPPED_POSITIONAL_ARGS = "swapped_positional_args"
    WRONG_PROP_TYPE = "wrong_prop_type"
    INVALID_ENUM = "invalid_enum"
    INVALID_CHILD_TYPE = "invalid_child_type"

    UNDEFINED_REF = "undefined_ref"
    DUPLICATE_NAME = "duplicate_name"
    RENAMED_WITHOUT_REFS = "renamed_without_updating_refs"
    REFERENCE_CYCLE = "reference_cycle"
    UNREACHABLE_NODE = "unreachable_node"
    ROOT_TO_MISSING = "root_to_missing"

    MUTATION_WITHOUT_TRIGGER = "mutation_without_trigger"
    INVALID_RUN_TARGET = "invalid_run_target"
    INVALID_SET_TARGET = "invalid_set_target"
    INVALID_RESET_TARGET = "invalid_reset_target"
    QUERY_BAD_DEFAULT = "query_bad_default"
    INVALID_MEMBER_ACCESS = "invalid_member_access"
    UNSUPPORTED_TOOL_ARG = "unsupported_tool_arg"

    PATCH_UNRELATED_STATEMENTS = "patch_touches_unrelated_statements"
    PATCH_NONEXISTENT_OLD = "patch_nonexistent_old_statement"
    PATCH_REMOVAL_DISCONNECTS = "patch_removal_does_not_disconnect_node"
    PATCH_PARENT_NOT_UPDATED = "patch_parent_list_not_updated"
    PATCH_CHANGES_STABLE_NAMES = "patch_changes_stable_names"
    PATCH_NONMINIMAL = "nonminimal_patch"

    @property
    def family(self) -> OperatorFamily:
        return _FAMILY_BY_OPERATOR[self]


_FAMILY_BY_OPERATOR = {
    operator: family
    for family, operators in {
        OperatorFamily.LEXICAL: tuple(CorruptionOperator)[:7],
        OperatorFamily.GRAMMAR: tuple(CorruptionOperator)[7:15],
        OperatorFamily.SCHEMA: tuple(CorruptionOperator)[15:22],
        OperatorFamily.REFERENCE_GRAPH: tuple(CorruptionOperator)[22:28],
        OperatorFamily.DATAFLOW: tuple(CorruptionOperator)[28:35],
        OperatorFamily.PATCH: tuple(CorruptionOperator)[35:],
    }.items()
    for operator in operators
}

_ASSIGNMENT_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=")
_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_ROOT_CHILDREN_RE = re.compile(r"(?m)^root\s*=\s*Stack\(\[([^\]]*)\]")
_SCALAR_RE = re.compile(r"\b(?:true|false|-?\d+(?:\.\d+)?)\b")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_TOKEN_RE = re.compile(r'\n|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]')


class CorruptionNotApplicable(ValueError):
    """Raised when a clean program lacks the construct required by an operator."""


@dataclass(frozen=True)
class CorruptionCase:
    clean_ast: dict[str, object]
    clean_openui: str
    broken_openui: str
    operator: CorruptionOperator
    family: OperatorFamily
    location: str
    diagnostics: tuple[str, ...]
    minimal_repair: str
    acceptable_repairs: tuple[str, ...]
    edit_distance: int
    preserved_nodes: tuple[str, ...]
    ast_path: tuple[str | int, ...] = ()
    source_span: tuple[int, int] = (0, 0)
    failure_cone: tuple[str | int, ...] = ()
    contract_mutation: dict[str, Any] | None = None

    @property
    def exact_repair(self) -> bool:
        return len(self.acceptable_repairs) == 1

    def to_record(
        self, spec: ProgramSpec, *, record_id: str | None = None
    ) -> ExampleRecord:
        """Project this corruption to a split-safe repair training row."""
        if _canonical(spec.canonical_openui) != self.clean_openui:
            raise ValueError("corruption clean target does not match ProgramSpec")
        repair_meta = {
            "family": "repair_taxonomy",
            "operator_family": self.family.value,
            "operator": self.operator.value,
            "location": self.location,
            "diagnostics": list(self.diagnostics),
            "clean_ast": self.clean_ast,
            "minimal_repair": self.minimal_repair,
            "acceptable_repairs": list(self.acceptable_repairs),
            "exact_repair": self.exact_repair,
            "edit_distance": self.edit_distance,
            "preserved_nodes": list(self.preserved_nodes),
            "ast_path": list(self.ast_path),
            "source_span": list(self.source_span),
            "failure_cone": list(self.failure_cone),
            "contract_mutation": self.contract_mutation,
        }
        return emit_record(
            spec,
            record_id=record_id,
            prompt=f"Repair this OpenUI program.\n---BROKEN---\n{self.broken_openui}",
            task="repair",
            openui=self.clean_openui,
            source="repair_taxonomy",
            meta={"repair": repair_meta},
        )


def build_corruption(
    clean_openui: str,
    operator: CorruptionOperator,
    *,
    acceptable_repairs: Iterable[str] | None = None,
) -> CorruptionCase:
    """Apply one operator and fail closed unless the layered verifier rejects it."""
    clean_program = validate(clean_openui)
    clean = clean_program.serialized or clean_openui.strip()
    broken, location = _apply(clean, operator)
    if broken == clean:
        raise CorruptionNotApplicable(f"{operator.value} did not change the program")

    official_error = ""
    try:
        validate(broken)
    except (ParseError, RuntimeError, ValueError) as exc:
        official_error = str(exc).splitlines()[0][:240]

    broken_record = ExampleRecord(
        id="corruption-probe",
        prompt="Repair probe",
        openui=broken,
        source="repair_taxonomy",
    )
    report = verify_record(
        broken_record, VerificationContext(source_kind="deterministic")
    )
    if report.ok:
        raise CorruptionNotApplicable(
            f"{operator.value} was accepted by every deterministic verifier gate"
        )
    failed = next(result for result in report.results if not result.ok)
    diagnostics = (f"{failed.gate.value}:{failed.detail or failed.gate.name.lower()}",)
    if official_error:
        diagnostics += (f"lang-core:{official_error}",)

    repairs = tuple(acceptable_repairs or (clean,))
    if not repairs:
        raise ValueError("acceptable_repairs must not be empty")
    canonical_repairs = tuple(_canonical(repair) for repair in repairs)
    clean_nodes = set(_binders(clean))
    broken_nodes = set(_binders(broken))
    return CorruptionCase(
        clean_ast=dict(clean_program.root or {}),
        clean_openui=clean,
        broken_openui=broken,
        operator=operator,
        family=operator.family,
        location=location,
        diagnostics=diagnostics,
        minimal_repair=clean,
        acceptable_repairs=canonical_repairs,
        edit_distance=_edit_distance(broken, clean),
        preserved_nodes=tuple(sorted(clean_nodes & broken_nodes)),
        source_span=(0, len(broken)),
    )


@dataclass(frozen=True)
class ScopedCorruption:
    """One verified sub-document corruption (statement / expression / lexical)."""

    kind: str  # validate_output kind of the clean fragment
    clean_text: str
    broken_text: str
    operator: CorruptionOperator
    family: OperatorFamily
    location: str


# Text-local operators that make sense on a fragment (no document invariants).
_SCOPED_OPERATORS = (
    CorruptionOperator.MISSING_QUOTE,
    CorruptionOperator.EXTRA_QUOTE,
    CorruptionOperator.INVALID_ESCAPE,
    CorruptionOperator.BROKEN_NUMBER_BOOL,
    CorruptionOperator.WRONG_CAPITALIZATION,
    CorruptionOperator.MISSING_ASSIGNMENT,
    CorruptionOperator.TRUNCATED_LINE,
)


def _lexical_typos(text: str) -> list[tuple[str, CorruptionOperator, str]]:
    """Deterministic meaningful typos for a single lexical token."""
    typos: list[tuple[str, CorruptionOperator, str]] = []
    if text and text[0] == text[-1] and text[0] in {'"', "'"} and len(text) >= 2:
        typos.append((text[:-1], CorruptionOperator.MISSING_QUOTE, "closing_quote"))
    elif len(text) >= 3:
        # Transpose the two middle-most characters: true -> ture, 3.14 -> 31.4.
        mid = len(text) // 2
        swapped = text[: mid - 1] + text[mid] + text[mid - 1] + text[mid + 1 :]
        if swapped != text:
            typos.append(
                (swapped, CorruptionOperator.BROKEN_NUMBER_BOOL, "transposed_chars")
            )
    if len(text) >= 2 and not text.startswith(('"', "'")):
        typos.append((text[:-1], CorruptionOperator.TRUNCATED_LINE, "dropped_char"))
    return typos


def build_scoped_corruptions(
    text: str,
    kind: str,
    *,
    category: str | None = None,
    limit: int = 2,
) -> tuple[ScopedCorruption, ...]:
    """Verified corruptions of one sub-document fragment, fail-closed.

    The clean fragment must pass ``validate_output`` for its kind and every
    emitted corruption must be *rejected* by the same validator — mirroring
    ``build_corruption``'s contract at fragment granularity.
    """
    from slm_training.dsl.parser import validate_output

    clean = validate_output(text, kind, category)  # raises if not valid
    candidates: list[tuple[str, CorruptionOperator, str]] = []
    if kind == "lexical":
        candidates.extend(_lexical_typos(clean))
    else:
        for operator in _SCOPED_OPERATORS:
            try:
                broken, location = _apply(text, operator)
            except (CorruptionNotApplicable, ValueError, IndexError):
                continue
            candidates.append((broken, operator, location))

    cases: list[ScopedCorruption] = []
    seen: set[str] = set()
    for broken, operator, location in candidates:
        if len(cases) >= limit:
            break
        if broken == text or broken in seen or not broken.strip():
            continue
        try:
            validate_output(broken, kind, category)
        except ParseError:
            seen.add(broken)
            cases.append(
                ScopedCorruption(
                    kind=kind,
                    clean_text=text,
                    broken_text=broken,
                    operator=operator,
                    family=operator.family,
                    location=location,
                )
            )
    return tuple(cases)


def generate_corruptions(clean_openui: str) -> tuple[CorruptionCase, ...]:
    """Generate every applicable catalog corruption in stable enum order."""
    cases = []
    for operator in CorruptionOperator:
        try:
            cases.append(build_corruption(clean_openui, operator))
        except CorruptionNotApplicable:
            continue
    return tuple(cases)


def _canonical(source: str) -> str:
    program = validate(source)
    return program.serialized or source.strip()


def _binders(source: str) -> tuple[str, ...]:
    return tuple(_ASSIGNMENT_RE.findall(source))


def _edit_distance(left: str, right: str) -> int:
    """Token edit distance used by the reversible minimal-repair contract."""
    a, b = _TOKEN_RE.findall(left), _TOKEN_RE.findall(right)
    previous = list(range(len(b) + 1))
    for i, lhs in enumerate(a, start=1):
        current = [i]
        for j, rhs in enumerate(b, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lhs != rhs))
            )
        previous = current
    return previous[-1]


def _replace(source: str, old: str, new: str, operator: CorruptionOperator) -> str:
    if old not in source:
        raise CorruptionNotApplicable(f"{operator.value} requires {old!r}")
    return source.replace(old, new, 1)


def _remove_line(source: str, prefix: str, operator: CorruptionOperator) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return "\n".join(lines[:index] + lines[index + 1 :])
    raise CorruptionNotApplicable(f"{operator.value} requires a {prefix!r} statement")


def _join_lines(
    source: str, first: str, second: str, operator: CorruptionOperator
) -> str:
    return _replace(source, f"{first}\n{second}", f"{first} {second}", operator)


def _line(source: str, *, non_root: bool = False) -> str:
    for line in source.splitlines():
        if line.strip() and (not non_root or not line.startswith("root =")):
            return line
    raise CorruptionNotApplicable("program lacks an applicable statement")


def _binder_line(source: str, name: str) -> str:
    prefix = f"{name} ="
    return next(
        (line for line in source.splitlines() if line.startswith(prefix)),
        "",
    )


def _first_child(source: str) -> str:
    match = _ROOT_CHILDREN_RE.search(source)
    if not match:
        raise CorruptionNotApplicable("operator requires root Stack children")
    refs = re.findall(r"\b[a-z_][A-Za-z0-9_]*\b", match.group(1))
    if not refs:
        raise CorruptionNotApplicable("operator requires a referenced root child")
    return refs[0]


def _component(source: str) -> tuple[str, str]:
    statement = _line(source, non_root=True)
    match = _COMPONENT_RE.search(statement)
    if not match:
        statement = _line(source)
        match = _COMPONENT_RE.search(statement)
    if not match:
        raise CorruptionNotApplicable("operator requires a component call")
    return statement, match.group(1)


def _truncate(source: str, line: str, operator: CorruptionOperator) -> str:
    if not line:
        raise CorruptionNotApplicable(f"{operator.value} requires a statement")
    return _replace(source, line, line[:-1], operator)


def _apply(source: str, operator: CorruptionOperator) -> tuple[str, str]:
    """Apply one local deterministic mutation, skipping unsupported shapes."""
    statement, component = _component(source)
    first_string = _STRING_RE.search(source)
    if operator is CorruptionOperator.MISSING_QUOTE:
        if not first_string:
            raise CorruptionNotApplicable("missing_quote requires a string")
        value = first_string.group()
        return _replace(source, value, value[:-1], operator), "string"
    if operator is CorruptionOperator.EXTRA_QUOTE:
        if not first_string:
            raise CorruptionNotApplicable("extra_quote requires a string")
        value = first_string.group()
        return _replace(source, value, f'{value}"', operator), "string"
    if operator is CorruptionOperator.INVALID_ESCAPE:
        if not first_string:
            raise CorruptionNotApplicable("invalid_escape requires a string")
        return _replace(source, first_string.group(), '"\\uZZZZ"', operator), "string"
    if operator is CorruptionOperator.MISSING_DELIMITER:
        return _truncate(source, statement, operator), "component"
    if operator is CorruptionOperator.BROKEN_NUMBER_BOOL:
        scalar = _SCALAR_RE.search(source)
        if not scalar:
            raise CorruptionNotApplicable(
                "broken_number_bool requires a number or bool"
            )
        return _replace(
            source, scalar.group(), f"{scalar.group()}x", operator
        ), "scalar"
    if operator is CorruptionOperator.TRUNCATED_LINE:
        target = source.splitlines()[-1]
        return _truncate(source, target, operator), "last_statement"
    if operator is CorruptionOperator.CONCATENATED_STATEMENTS:
        lines = source.splitlines()
        if len(lines) < 2:
            raise CorruptionNotApplicable(
                "concatenated_statements requires two statements"
            )
        return _join_lines(source, lines[0], lines[1], operator), "statements[0:2]"

    if operator is CorruptionOperator.MISSING_ASSIGNMENT:
        target = _line(source, non_root=True)
        return _replace(
            source, target, target.replace(" = ", " ", 1), operator
        ), "assignment"
    if operator is CorruptionOperator.TWO_STATEMENTS_PER_LINE:
        lines = source.splitlines()
        if len(lines) < 2:
            raise CorruptionNotApplicable(
                "two_statements_per_line requires two statements"
            )
        return _join_lines(source, lines[-2], lines[-1], operator), "statements[-2:]"
    if operator is CorruptionOperator.NAMED_WHERE_POSITIONAL:
        return _replace(
            source, f"{component}(", f"{component}(value=", operator
        ), "component.arg[0]"
    if operator is CorruptionOperator.MALFORMED_ARRAY:
        return _replace(source, "[", "[}", operator), "array"
    if operator is CorruptionOperator.MALFORMED_OBJECT:
        return _replace(source, "[", "{", operator), "object"
    if operator is CorruptionOperator.INVALID_NESTING:
        child = _first_child(source)
        return _replace(
            source, f"[{child}", f"[{{{child}}}", operator
        ), "root.children[0]"
    if operator is CorruptionOperator.MISSING_ROOT:
        return _remove_line(source, "root =", operator), "root"
    if operator is CorruptionOperator.DUPLICATE_ROOT:
        root = source.splitlines()[0]
        return f"{root}\n{source}", "root"

    if operator is CorruptionOperator.UNKNOWN_COMPONENT:
        return _replace(
            source, f"{component}(", "MissingComponent(", operator
        ), "component.type"
    if operator is CorruptionOperator.WRONG_CAPITALIZATION:
        return _replace(
            source, f"{component}(", f"{component.lower()}(", operator
        ), "component.type"
    if operator is CorruptionOperator.WRONG_ARG_COUNT:
        start = statement.index(f"{component}(") + len(component) + 1
        end = statement.rfind(")")
        if end < start:
            raise CorruptionNotApplicable("wrong_arg_count requires a complete call")
        return _replace(
            source, statement, statement[:start] + statement[end:], operator
        ), "component.args"
    if operator is CorruptionOperator.SWAPPED_POSITIONAL_ARGS:
        arguments = re.search(r'Input\(("[^"]+"),\s*("[^"]+")', source)
        if not arguments:
            raise CorruptionNotApplicable("swapped_positional_args requires an Input")
        old = arguments.group(0)
        return _replace(
            source,
            old,
            f"Input({arguments.group(2)}, {arguments.group(1)}",
            operator,
        ), "input.args"
    if operator is CorruptionOperator.WRONG_PROP_TYPE:
        match = _ROOT_CHILDREN_RE.search(source)
        if not match:
            raise CorruptionNotApplicable("wrong_prop_type requires root children")
        return _replace(
            source, f"[{match.group(1)}]", '":bad.children"', operator
        ), "root.children"
    if operator is CorruptionOperator.INVALID_ENUM:
        enum = next((value for value in ('"column"', '"row"') if value in source), "")
        if not enum:
            raise CorruptionNotApplicable("invalid_enum requires a known enum")
        return _replace(source, enum, "diagonal", operator), "enum"
    if operator is CorruptionOperator.INVALID_CHILD_TYPE:
        child = _first_child(source)
        return _replace(
            source, f"[{child}", '[":bad.child"', operator
        ), "root.children[0]"

    if operator is CorruptionOperator.UNDEFINED_REF:
        child = _first_child(source)
        return _replace(source, f"[{child}", "[missing", operator), "root.children[0]"
    if operator is CorruptionOperator.DUPLICATE_NAME:
        return _replace(
            source, statement, f"{statement}\n{statement}", operator
        ), "binder"
    if operator is CorruptionOperator.RENAMED_WITHOUT_REFS:
        child = _first_child(source)
        target = _binder_line(source, child)
        if not target:
            raise CorruptionNotApplicable(
                "renamed_without_refs requires a child binder"
            )
        return _replace(
            source, target, target.replace(f"{child} =", "renamed =", 1), operator
        ), child
    if operator is CorruptionOperator.REFERENCE_CYCLE:
        child = _first_child(source)
        target = _binder_line(source, child)
        if not target:
            raise CorruptionNotApplicable("reference_cycle requires a child binder")
        return _replace(source, target, f"{child} = Stack([root])", operator), child
    if operator is CorruptionOperator.UNREACHABLE_NODE:
        return f'{source}\norphan = TextContent(":orphan.text")', "orphan"
    if operator is CorruptionOperator.ROOT_TO_MISSING:
        child = _first_child(source)
        return _replace(source, f"[{child}", "[missing", operator), "root.children"

    dataflow = {
        CorruptionOperator.MUTATION_WITHOUT_TRIGGER: "$count = 1",
        CorruptionOperator.INVALID_RUN_TARGET: "@Run(missing)",
        CorruptionOperator.INVALID_SET_TARGET: "@Set(missing, 1)",
        CorruptionOperator.INVALID_RESET_TARGET: "@Reset(missing)",
        CorruptionOperator.QUERY_BAD_DEFAULT: 'query = Query("items", missing)',
        CorruptionOperator.INVALID_MEMBER_ACCESS: "value = state.missing",
        CorruptionOperator.UNSUPPORTED_TOOL_ARG: 'tool = Tool("missing", {bad: true})',
    }
    if operator in dataflow:
        return f"{dataflow[operator]}\n{source}", "dataflow"

    if operator is CorruptionOperator.PATCH_UNRELATED_STATEMENTS:
        broken, _ = _apply(source, CorruptionOperator.WRONG_ARG_COUNT)
        return _truncate(
            broken, broken.splitlines()[-1], operator
        ), "unrelated_statements"
    if operator is CorruptionOperator.PATCH_NONEXISTENT_OLD:
        return f"{source}\nghost = MissingComponent()", "ghost"
    if operator is CorruptionOperator.PATCH_REMOVAL_DISCONNECTS:
        child = _first_child(source)
        return _remove_line(source, f"{child} =", operator), child
    if operator is CorruptionOperator.PATCH_PARENT_NOT_UPDATED:
        child = _first_child(source)
        return _remove_line(source, f"{child} =", operator), "root.children"
    if operator is CorruptionOperator.PATCH_CHANGES_STABLE_NAMES:
        child = _first_child(source)
        target = _binder_line(source, child)
        if not target:
            raise CorruptionNotApplicable(
                "patch_changes_stable_names requires a child binder"
            )
        return _replace(
            source, target, target.replace(f"{child} =", "renamed =", 1), operator
        ), child
    if operator is CorruptionOperator.PATCH_NONMINIMAL:
        broken, _ = _apply(source, CorruptionOperator.WRONG_ARG_COUNT)
        broken, _ = _apply(broken, CorruptionOperator.INVALID_ENUM)
        return broken, "component/enum"
    raise AssertionError(f"unhandled corruption operator: {operator}")


__all__ = [
    "CorruptionCase",
    "CorruptionNotApplicable",
    "CorruptionOperator",
    "OperatorFamily",
    "ScopedCorruption",
    "build_corruption",
    "build_scoped_corruptions",
    "generate_corruptions",
]
