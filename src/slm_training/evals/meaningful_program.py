"""Versioned, binding-aware meaningful-program evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from slm_training.data.quality import (
    _prompt_component_requirements,
    _schema_semantic_reasons,
    _semantic_request,
    schema_placeholder_role_matches,
)
from slm_training.data.contract import GenerationRequest
from slm_training.data.verify.stack import Gate, GateStatus, evaluate_gate
from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.parser import parse, validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

METRIC_NAME = "binding_aware_meaningful_v2"
METRIC_VERSION = "2.13.0"
_ASSIGNMENT_RE = re.compile(
    r"(?m)^\s*(\$?[A-Za-z_][A-Za-z0-9_]*)\s*="
)
_INVENTORY_SECTION_RE = re.compile(
    r"(?im)(?:placeholders?|slot(?:\s+inventory)?|inventory)\s*:\s*([^\n]*)"
)
_TEXT_COMPONENTS = frozenset(
    {"TextContent", "CardHeader", "Callout", "TextCallout", "FormControl"}
)
_ACTION_SLOT_TERMS = frozenset({"action", "cta", "save", "submit", "cancel", "delete"})
_TEXT_SLOT_TERMS = frozenset({"title", "subtitle", "body", "text", "copy", "kicker", "description"})
_FORM_SLOT_TERMS = frozenset({"email", "name", "phone", "password", "value"})


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SemanticEvidenceV2:
    ast_path: str
    detail: str
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class SemanticCheckV2:
    name: str
    status: CheckStatus
    reason_codes: tuple[str, ...] = ()
    evidence: tuple[SemanticEvidenceV2, ...] = ()


@dataclass(frozen=True)
class PromptContractV2:
    required_placeholders: tuple[str, ...]
    required_components: tuple[str, ...]
    provenance: tuple[dict[str, str], ...]
    placeholder_coverage_known: bool
    component_coverage_known: bool

    @property
    def coverage_known(self) -> bool:
        return self.placeholder_coverage_known and self.component_coverage_known


@dataclass(frozen=True)
class SemanticMeaningReportV2:
    verdict: bool
    checks: tuple[SemanticCheckV2, ...]
    reason_codes: tuple[str, ...]
    prompt_contract: PromptContractV2
    component_inventory: tuple[dict[str, Any], ...]
    binding_inventory: tuple[dict[str, Any], ...]
    placeholder_inventory: tuple[dict[str, Any], ...]
    metric_name: str = METRIC_NAME
    metric_version: str = METRIC_VERSION
    metric_implementation_hash: str = ""

    @property
    def coverage_known(self) -> bool:
        return not any(check.status is CheckStatus.UNKNOWN for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [
            {
                **asdict(check),
                "status": check.status.value,
            }
            for check in self.checks
        ]
        payload["prompt_contract"] = {
            **asdict(self.prompt_contract),
            "coverage_known": self.prompt_contract.coverage_known,
        }
        payload["coverage_known"] = self.coverage_known
        return payload


@lru_cache(maxsize=1)
def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    dependencies = (
        Path(__file__),
        root / "data" / "quality.py",
        root / "data" / "verify" / "stack.py",
        root / "dsl" / "parser.py",
        root / "dsl" / "placeholders.py",
        root / "dsl" / "language_contract.py",
    )
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(path.relative_to(root.parent).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _evidence(path: str, detail: str, span: tuple[int, int] | None = None) -> SemanticEvidenceV2:
    return SemanticEvidenceV2(path, detail, span)


def _walk(
    value: Any,
    path: str = "root",
    owner_component: str | None = None,
    owner_prop: str | None = None,
) -> Iterable[tuple[str, Any, str | None, str | None]]:
    """Yield AST values with their owning component/property."""
    yield path, value, owner_component, owner_prop
    if isinstance(value, dict):
        component = value.get("typeName") if value.get("type") == "element" else None
        props = value.get("props") if component else None
        if isinstance(props, dict):
            for name, child in props.items():
                child_path = f"{path}.props.{name}"
                yield from _walk(child, child_path, component, name)
        else:
            for name, child in value.items():
                yield from _walk(
                    child,
                    f"{path}.{name}",
                    owner_component,
                    owner_prop,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(
                child,
                f"{path}[{index}]",
                owner_component,
                owner_prop,
            )


def _component_inventory(root: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    if root is None:
        return ()
    return tuple(
        {"type": value["typeName"], "ast_path": path}
        for path, value, _component, _prop in _walk(root)
        if isinstance(value, dict)
        and value.get("type") == "element"
        and isinstance(value.get("typeName"), str)
    )


def _placeholder_inventory(
    source: str, root: dict[str, Any] | None, program: Any | None
) -> tuple[dict[str, Any], ...]:
    if root is None:
        return ()
    spans: dict[str, list[tuple[int, int]]] = {}
    for placeholder in extract_placeholders(source):
        spans[placeholder] = [
            (match.start(), match.end())
            for match in re.finditer(
                rf"{re.escape(placeholder)}(?![A-Za-z0-9_.-])", source
            )
        ]
    used: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    state = program.state_declarations if program is not None else {}

    def resolved_placeholder(value: Any, seen: frozenset[str] = frozenset()) -> str | None:
        if isinstance(value, str) and value.startswith(":"):
            return value
        if (
            isinstance(value, dict)
            and value.get("k") == "StateRef"
            and isinstance(value.get("n"), str)
            and value["n"] not in seen
        ):
            return resolved_placeholder(state.get(value["n"]), seen | {value["n"]})
        return None

    for path, value, component, prop in _walk(root):
        value = resolved_placeholder(value)
        if not isinstance(value, str) or not value.startswith(":"):
            continue
        index = used[value]
        used[value] += 1
        value_spans = spans.get(value) or ()
        rows.append(
            {
                "placeholder": value,
                "ast_path": path,
                "component": component,
                "property": prop,
                "source_span": (
                    list(value_spans[index]) if index < len(value_spans) else None
                ),
            }
        )
    return tuple(rows)


def _binding_inventory(source: str, program: Any | None) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for match in _ASSIGNMENT_RE.finditer(source):
        rows.append(
            {
                "symbol": match.group(1),
                "role": "definition",
                "source_span": [match.start(1), match.end(1)],
            }
        )
    if program is not None:
        for path, value, _component, _prop in _walk(program.root):
            if not isinstance(value, dict):
                continue
            kind = value.get("k")
            if kind in {"RuntimeRef", "StateRef", "Ref"} and isinstance(
                value.get("n"), str
            ):
                rows.append(
                    {
                        "symbol": value["n"],
                        "role": str(value.get("refType") or kind),
                        "ast_path": path,
                    }
                )
    return tuple(rows)


def _prompt_contract(
    record: ExampleRecord, request: GenerationRequest | None
) -> PromptContractV2:
    # Only model-visible inputs may define evaluator coverage. DESIGN.md and
    # required_facts are deliberately excluded unless represented in the
    # effective GenerationRequest.
    visible_prompt = tuple(
        dict.fromkeys(
            placeholder
            for match in _INVENTORY_SECTION_RE.finditer(record.prompt)
            for placeholder in extract_placeholders(match.group(1))
        )
    )
    visible_request = tuple(request.slot_contract) if request is not None else ()
    semantic_request = _INVENTORY_SECTION_RE.sub("", _semantic_request(record))
    prompt_components = _prompt_component_requirements(semantic_request)
    placeholders = tuple(dict.fromkeys((*visible_prompt, *visible_request)))
    components = prompt_components
    provenance: list[dict[str, str]] = []
    if visible_prompt:
        provenance.append(
            {
                "kind": "prompt_visible_placeholder_inventory",
                "version": "explicit_prompt_inventory_v2",
            }
        )
    if visible_request:
        provenance.append(
            {"kind": "generation_request_slot_contract", "version": "request_v1"}
        )
    if prompt_components:
        provenance.append(
            {
                "kind": "schema_component_mention",
                "version": "prompt_component_mentions_v1",
            }
        )
    return PromptContractV2(
        required_placeholders=placeholders,
        required_components=components,
        provenance=tuple(provenance),
        placeholder_coverage_known=bool(placeholders),
        component_coverage_known=bool(components),
    )


def _parse_checks(source: str) -> tuple[Any | None, SemanticCheckV2, SemanticCheckV2]:
    try:
        raw = parse(source)
        validated = validate(source)
        first = (validated.serialized or source).strip()
        second = (validate(first).serialized or first).strip()
    except (ParseError, RuntimeError, ValueError) as exc:
        reason = _evidence("root", str(exc).splitlines()[0][:240])
        failed = SemanticCheckV2(
            "official_parse", CheckStatus.FAIL, ("official_parse_failed",), (reason,)
        )
        return None, failed, SemanticCheckV2(
            "canonical_roundtrip",
            CheckStatus.FAIL,
            ("canonical_roundtrip_unavailable",),
            (reason,),
        )
    parse_check = SemanticCheckV2(
        "official_parse",
        CheckStatus.PASS,
        evidence=(_evidence("root", "official parser and schema accepted raw source"),),
    )
    if first != second:
        canonical = SemanticCheckV2(
            "canonical_roundtrip",
            CheckStatus.FAIL,
            ("canonical_roundtrip_not_idempotent",),
            (_evidence("root", "serialize(validate(x)) changed on second pass"),),
        )
    else:
        canonical = SemanticCheckV2(
            "canonical_roundtrip",
            CheckStatus.PASS,
            evidence=(_evidence("root", "official canonical form is idempotent"),),
        )
    return raw, parse_check, canonical


def _semantic_content_check(
    components: tuple[dict[str, Any], ...],
    contract: PromptContractV2,
) -> SemanticCheckV2:
    content = [row for row in components if row["type"] != "Stack"]
    if not content:
        return SemanticCheckV2(
            "prompt_relevant_semantic_content",
            CheckStatus.FAIL,
            ("no_nontrivial_content",),
            tuple(_evidence(row["ast_path"], row["type"]) for row in components),
        )
    present = Counter(str(row["type"]) for row in components)
    required = Counter(contract.required_components)
    missing = sorted(
        name
        for name, count in required.items()
        for _ in range(max(0, count - present[name]))
    )
    if missing:
        return SemanticCheckV2(
            "prompt_relevant_semantic_content",
            CheckStatus.FAIL,
            ("prompt_component_missing",),
            tuple(_evidence("prompt", name) for name in missing),
        )
    if not contract.component_coverage_known:
        return SemanticCheckV2(
            "prompt_relevant_semantic_content",
            CheckStatus.UNKNOWN,
            ("prompt_contract_unknown",),
            (_evidence("prompt", "no deterministic prompt inventory or component fact"),),
        )
    return SemanticCheckV2(
        "prompt_relevant_semantic_content",
        CheckStatus.PASS,
        evidence=tuple(_evidence(row["ast_path"], row["type"]) for row in content),
    )


def _inventory_check(
    placeholders: tuple[dict[str, Any], ...],
    components: tuple[dict[str, Any], ...],
    contract: PromptContractV2,
) -> SemanticCheckV2:
    present_slots = {str(row["placeholder"]) for row in placeholders}
    slot_counts = Counter(str(row["placeholder"]) for row in placeholders)
    duplicate_slots = (
        sorted(
            slot
            for slot, count in slot_counts.items()
            for _ in range(max(0, count - 1))
        )
        if contract.placeholder_coverage_known
        else []
    )
    present_components = Counter(str(row["type"]) for row in components)
    missing_slots = sorted(set(contract.required_placeholders) - present_slots)
    unexpected_slots = (
        sorted(present_slots - set(contract.required_placeholders))
        if contract.placeholder_coverage_known
        else []
    )
    required_components = Counter(contract.required_components)
    missing_components = sorted(
        name
        for name, count in required_components.items()
        for _ in range(max(0, count - present_components[name]))
    )
    role_mismatches: list[str] = []
    for row in placeholders:
        term = str(row["placeholder"]).rsplit(".", 1)[-1].lstrip(":").lower()
        owner = str(row.get("component") or "")
        prop = str(row.get("property") or "")
        recognized_role = term in (
            _ACTION_SLOT_TERMS | _TEXT_SLOT_TERMS | _FORM_SLOT_TERMS
        )
        if owner == "Input" and prop == "name":
            role_mismatches.append(f"{row['placeholder']}->{owner}.{prop}")
        elif term in _FORM_SLOT_TERMS and owner in {"Input", "TextArea", "Select"}:
            continue
        elif recognized_role and not schema_placeholder_role_matches(
            str(row["placeholder"]), owner, prop
        ):
            role_mismatches.append(f"{row['placeholder']}->{owner or 'unknown'}")
    missing_slots = missing_slots if contract.placeholder_coverage_known else []
    missing_components = (
        missing_components if contract.component_coverage_known else []
    )
    if (
        missing_slots
        or unexpected_slots
        or duplicate_slots
        or missing_components
        or role_mismatches
    ):
        evidence = tuple(
            _evidence("prompt.placeholders", value) for value in missing_slots
        ) + tuple(
            _evidence("root.placeholders", value) for value in unexpected_slots
        ) + tuple(
            _evidence("root.placeholders", value) for value in duplicate_slots
        ) + tuple(
            _evidence("prompt.components", value) for value in missing_components
        ) + tuple(
            _evidence("root.placeholders", value) for value in role_mismatches
        )
        return SemanticCheckV2(
            "required_inventory_coverage",
            CheckStatus.FAIL,
            tuple(
                code
                for code, values in (
                    ("required_placeholder_missing", missing_slots),
                    ("unexpected_placeholder_identity", unexpected_slots),
                    ("duplicate_placeholder_identity", duplicate_slots),
                    ("required_component_missing", missing_components),
                    ("placeholder_semantic_role_mismatch", role_mismatches),
                )
                if values
            ),
            evidence,
        )
    if not contract.coverage_known:
        return SemanticCheckV2(
            "required_inventory_coverage",
            CheckStatus.UNKNOWN,
            ("required_inventory_unknown",),
            (_evidence("prompt", "one or more prompt inventory dimensions are unknown"),),
        )
    return SemanticCheckV2(
        "required_inventory_coverage",
        CheckStatus.PASS,
        evidence=tuple(
            _evidence(row["ast_path"], str(row["placeholder"])) for row in placeholders
        ),
    )


def _binding_check(
    source: str, program: Any | None, record: ExampleRecord
) -> SemanticCheckV2:
    if program is None:
        return SemanticCheckV2(
            "binding_correctness",
            CheckStatus.FAIL,
            ("binding_analysis_unavailable",),
        )
    names = [match.group(1) for match in _ASSIGNMENT_RE.finditer(source)]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    unresolved = tuple(str(value) for value in program.meta.get("unresolved") or ())
    orphaned = tuple(str(value) for value in program.meta.get("orphaned") or ())
    reasons: list[str] = []
    evidence: list[SemanticEvidenceV2] = []
    if duplicates:
        reasons.append("duplicate_binding")
        evidence.extend(_evidence("bindings", value) for value in duplicates)
    if unresolved:
        reasons.append("unresolved_binding")
        evidence.extend(_evidence("bindings", value) for value in unresolved)
    if orphaned:
        reasons.append("unreachable_binding")
        evidence.extend(_evidence("bindings", value) for value in orphaned)
    # `unresolved`/`orphaned` above already come from the official parser's
    # own analysis and are authoritative for both structural (placeholder
    # scaffold) and runtime ($state/Query/Mutation) sources. This second pass
    # additionally walks the $state/Query/Mutation dependency graph to catch
    # *dead* runtime bindings the parser doesn't flag as orphaned (declared,
    # reachable only through other dead code). It runs unconditionally: for
    # structural-only sources, state_declarations/query_statements/
    # mutation_statements are empty, so `dependencies` is empty and this is a
    # safe no-op.
    #
    # This replaces a former regex-based Gate.REFERENCES fallback that ran
    # instead of this check for any source without $/Query/Mutation/@ syntax.
    # That fallback treated bare object-literal property keys (e.g. `src:` in
    # a typed-array item like `{src: ..., alt: ...}`) as unresolved variable
    # references, so it permanently failed binding_correctness -- and with it
    # binding_aware_meaningful_v2's strict verdict -- for any correctly
    # produced typed-array-of-objects prediction, independent of model
    # quality. Confirmed present in eval evidence back through E612-E617
    # (`reference_graph_invalid` in docs/design/iter-e612..e617*.json) before
    # E618 diagnosed and removed it; see
    # docs/design/iter-e618-strict-v2-reference-graph-false-positive-20260720.md.
    definitions = set(names)

    def refs(value: Any) -> set[str]:
        found: set[str] = set()
        for _path, child, _component, _prop in _walk(value):
            if (
                isinstance(child, dict)
                and child.get("k") in {"RuntimeRef", "StateRef", "Ref"}
                and isinstance(child.get("n"), str)
            ):
                found.add(child["n"])
        return found

    dependencies: dict[str, set[str]] = {}
    for name, value in program.state_declarations.items():
        dependencies[name] = refs(value)
    for statement in (*program.query_statements, *program.mutation_statements):
        dependencies[str(statement.get("statementId"))] = refs(statement)
    reachable = refs(program.root)
    frontier = list(reachable)
    while frontier:
        name = frontier.pop()
        for dependency in dependencies.get(name, set()):
            if dependency not in reachable:
                reachable.add(dependency)
                frontier.append(dependency)
    dead_runtime = sorted(definitions.intersection(dependencies) - reachable)
    if dead_runtime:
        reasons.append("unreachable_binding")
        evidence.extend(_evidence("bindings", value) for value in dead_runtime)
    if reasons:
        return SemanticCheckV2(
            "binding_correctness",
            CheckStatus.FAIL,
            tuple(dict.fromkeys(reasons)),
            tuple(evidence),
        )
    return SemanticCheckV2(
        "binding_correctness",
        CheckStatus.PASS,
        evidence=(_evidence("bindings", "definitions and references are scope-clean"),),
    )


def _schema_role_check(source: str) -> SemanticCheckV2:
    reasons = tuple(_schema_semantic_reasons(source))
    if reasons:
        return SemanticCheckV2(
            "schema_value_role_correctness",
            CheckStatus.FAIL,
            reasons,
            tuple(_evidence("root", reason) for reason in reasons),
        )
    return SemanticCheckV2(
        "schema_value_role_correctness",
        CheckStatus.PASS,
        evidence=(_evidence("root", "resolved values match official schema roles"),),
    )


def _verifier_check(source: str, record: ExampleRecord) -> SemanticCheckV2:
    if not source.strip():
        return SemanticCheckV2(
            "whole_program_verifier",
            CheckStatus.FAIL,
            ("verifier_input_empty",),
        )
    candidate = ExampleRecord.from_dict(
        {**record.to_dict(), "openui": source, "placeholders": []}
    )
    gates = (Gate.LEXICAL, Gate.GRAMMAR, Gate.CANONICAL)
    results = tuple(evaluate_gate(gate, candidate) for gate in gates)
    failed = tuple(result for result in results if result.status is GateStatus.FAIL)
    if failed:
        return SemanticCheckV2(
            "whole_program_verifier",
            CheckStatus.FAIL,
            tuple(f"verifier_{result.gate.value.lower()}_failed" for result in failed),
            tuple(
                _evidence(f"verifier.{result.gate.value}", result.detail)
                for result in failed
            ),
        )
    return SemanticCheckV2(
        "whole_program_verifier",
        CheckStatus.PASS,
        evidence=tuple(
            _evidence(f"verifier.{result.gate.value}", result.status.value)
            for result in results
        ),
    )


_FINGERPRINT_EXCLUDED_KEYS = frozenset({"statementId", "partial", "hasDynamicProps"})


def _subtree_hashes(root: Any) -> dict[int, str]:
    """Structural hash per dict node, computed bottom-up in one pass.

    Hashes agree exactly when the ``statementId``/``partial``/``hasDynamicProps``-
    stripped normal forms agree, which is the only property ``_gaming_check``
    relies on; composing child digests avoids re-serializing every subtree per
    node (previously O(n^2) in AST size).
    """
    by_id: dict[int, str] = {}

    def visit(value: Any) -> str:
        if isinstance(value, dict):
            items: list[tuple[str, str]] = []
            for key, child in sorted(value.items()):
                # Visit every child so nested dicts are registered even under
                # excluded keys; only non-excluded keys feed the parent hash.
                child_digest = visit(child)
                if key not in _FINGERPRINT_EXCLUDED_KEYS:
                    items.append((key, child_digest))
            digest = hashlib.sha256(("D" + repr(items)).encode()).hexdigest()
            by_id[id(value)] = digest
            return digest
        if isinstance(value, list):
            return hashlib.sha256(
                ("L" + "".join(visit(child) for child in value)).encode()
            ).hexdigest()
        return hashlib.sha256(
            ("S" + json.dumps(value, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()

    visit(root)
    return by_id


def _gaming_check(
    program: Any | None,
    components: tuple[dict[str, Any], ...],
    placeholders: tuple[dict[str, Any], ...],
    contract: PromptContractV2,
) -> SemanticCheckV2:
    if program is None or program.root is None:
        return SemanticCheckV2(
            "anti_gaming",
            CheckStatus.FAIL,
            ("gaming_analysis_unavailable",),
        )
    reasons: list[str] = []
    evidence: list[SemanticEvidenceV2] = []
    fingerprints: dict[str, list[str]] = {}
    subtree_hashes = _subtree_hashes(program.root)
    for path, value, _component, _prop in _walk(program.root):
        if isinstance(value, dict) and value.get("type") == "element":
            fingerprints.setdefault(subtree_hashes[id(value)], []).append(path)
    repeated = [paths for paths in fingerprints.values() if len(paths) >= 3]
    if repeated:
        reasons.append("duplicate_subtree_spam")
        evidence.extend(_evidence(path, "repeated subtree") for path in repeated[0])
    slot_counts = Counter(str(row["placeholder"]) for row in placeholders)
    spammed = sorted(slot for slot, count in slot_counts.items() if count >= 3)
    if spammed:
        reasons.append("placeholder_spam")
        evidence.extend(_evidence("placeholders", slot) for slot in spammed)
    component_types = [str(row["type"]) for row in components]
    scale = max(
        1,
        len(contract.required_placeholders) + len(contract.required_components),
    )
    if len(component_types) > max(12, 4 * scale) and len(set(component_types)) <= 2:
        reasons.append("low_diversity_filler")
        evidence.append(
            _evidence(
                "root",
                f"{len(component_types)} components across {len(set(component_types))} types",
            )
        )
    owners = {
        str(row["component"])
        for row in placeholders
        if row.get("component") is not None
        and str(row.get("placeholder")) in contract.required_placeholders
    }
    requested_non_text = set(contract.required_components) - _TEXT_COMPONENTS - {
        "Stack",
        "Card",
    }
    from slm_training.data.quality import semantic_role_properties
    from slm_training.dsl.lang_core import library_schema

    required_role_properties = {
        prop
        for properties in semantic_role_properties(
            list(contract.required_placeholders)
        ).values()
        for prop in properties
    }
    definitions = library_schema().get("$defs", {})
    requested_direct_owners = {
        family
        for family in requested_non_text
        if required_role_properties.intersection(
            {
                name
                for name, schema in (
                    definitions.get(family, {}).get("properties", {}) or {}
                ).items()
                if isinstance(schema, dict) and schema.get("type") == "string"
            }
        )
    }
    if (
        len(contract.required_placeholders) >= 2
        and requested_direct_owners
        and owners
        and owners <= _TEXT_COMPONENTS
    ):
        reasons.append("mechanical_inventory_coverage")
        evidence.append(
            _evidence("root", "all required slots were routed through text-only owners")
        )
    if reasons:
        return SemanticCheckV2(
            "anti_gaming",
            CheckStatus.FAIL,
            tuple(dict.fromkeys(reasons)),
            tuple(evidence),
        )
    return SemanticCheckV2(
        "anti_gaming",
        CheckStatus.PASS,
        evidence=(_evidence("root", "no deterministic gaming signature matched"),),
    )


def binding_aware_meaningful_v2(
    pred: str, *, record: ExampleRecord, request: GenerationRequest | None = None
) -> SemanticMeaningReportV2:
    """Return a fail-closed, replayable v2 semantic-meaning report."""
    program, parse_check, canonical_check = _parse_checks(pred)
    contract = _prompt_contract(record, request)
    components = _component_inventory(program.root if program is not None else None)
    placeholders = _placeholder_inventory(
        pred, program.root if program is not None else None, program
    )
    bindings = _binding_inventory(pred, program)
    from slm_training.dsl.language_contract import output_contract_violations

    free_form = output_contract_violations(pred) if program is not None else ()
    output_contract_check = SemanticCheckV2(
        "symbol_only_output",
        CheckStatus.FAIL if free_form else CheckStatus.PASS,
        ("free_form_output_string",) if free_form else (),
        tuple(_evidence("root", repr(value)) for value in free_form),
    )
    checks = (
        parse_check,
        canonical_check,
        output_contract_check,
        _semantic_content_check(components, contract),
        _inventory_check(placeholders, components, contract),
        _binding_check(pred, program, record),
        _schema_role_check(pred),
        _verifier_check(pred, record),
        _gaming_check(program, components, placeholders, contract),
    )
    reasons = tuple(
        dict.fromkeys(
            reason
            for check in checks
            if check.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN}
            for reason in check.reason_codes
        )
    )
    verdict = all(
        check.status in {CheckStatus.PASS, CheckStatus.NOT_APPLICABLE}
        for check in checks
    )
    return SemanticMeaningReportV2(
        verdict=verdict,
        checks=checks,
        reason_codes=reasons,
        prompt_contract=contract,
        component_inventory=components,
        binding_inventory=bindings,
        placeholder_inventory=placeholders,
        metric_implementation_hash=_implementation_hash(),
    )


def aggregate_meaning_reports_v2(
    reports: Iterable[SemanticMeaningReportV2],
) -> dict[str, Any]:
    rows = tuple(reports)
    covered = tuple(report for report in rows if report.coverage_known)
    positives = sum(report.verdict for report in rows)
    covered_positives = sum(report.verdict for report in covered)
    reasons = Counter(reason for report in rows for reason in report.reason_codes)
    return {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "metric_implementation_hash": (
            rows[0].metric_implementation_hash if rows else _implementation_hash()
        ),
        "n": len(rows),
        "covered_n": len(covered),
        "strict_rate": positives / len(rows) if rows else 0.0,
        "coverage_conditioned_rate": (
            covered_positives / len(covered) if covered else 0.0
        ),
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "reason_prevalence": dict(sorted(reasons.items())),
    }


__all__ = [
    "CheckStatus",
    "METRIC_NAME",
    "METRIC_VERSION",
    "PromptContractV2",
    "SemanticCheckV2",
    "SemanticEvidenceV2",
    "SemanticMeaningReportV2",
    "aggregate_meaning_reports_v2",
    "binding_aware_meaningful_v2",
]
