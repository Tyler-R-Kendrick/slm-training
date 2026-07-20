"""Deterministic quality gates for training ExampleRecords."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_ROOT_RE = re.compile(r"(?m)^root\s*=")
_ASSIGNMENT_RE = re.compile(
    r"(?m)^\s*([a-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)
_DECLARATION_COMPONENT_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9]*)\s*\(")
_IDENTIFIER_RE = re.compile(r"\b[a-z_][A-Za-z0-9_]*\b")
_QUOTED_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_COMPONENT_PROMPT_RE = re.compile(r"\b(?:the|a|an)\s+([A-Z][A-Za-z0-9]*)\s+component\b", re.I)
_CONSTRUCT_PROMPT_RE = re.compile(r"construct:\s+(?:a|an|the)?\s*([^.]*)", re.I)

# Preferred structural vocabulary for high-signal layouts.
PREFERRED_COMPONENTS = frozenset(
    {
        "Stack",
        "Card",
        "TextContent",
        "Button",
        "Input",
        "Form",
        "FormControl",
        "ImageBlock",
        "Tabs",
        "TabItem",
        "Modal",
        "Callout",
        "TextCallout",
        "CardHeader",
        "Separator",
        "Slider",
        "CheckBoxItem",
        "RadioItem",
        "SwitchItem",
        "DatePicker",
        "Buttons",
    }
)


@lru_cache(maxsize=1)
def _official_schema() -> dict[str, Any]:
    """Return the generated OpenUI schema, with an offline empty fallback."""
    try:
        from slm_training.dsl import lang_core

        schema = lang_core.library_schema()
        if isinstance(schema, dict):
            return schema
    except Exception:  # noqa: BLE001
        pass
    return {}


@lru_cache(maxsize=1)
def _official_component_names() -> frozenset[str]:
    """Return the generated OpenUI component inventory, with an offline fallback."""
    schema = _official_schema()
    names = set(schema.get("properties") or schema.get("components") or ())
    return frozenset(str(name) for name in names) or PREFERRED_COMPONENTS


@lru_cache(maxsize=1)
def _component_phrases() -> tuple[tuple[str, str, "re.Pattern[str]"], ...]:
    """Longest-first (name, phrase, matcher) rows.

    Derived only from the cached component inventory, so it is invariant for
    the process; the per-record judges previously rebuilt (and re-sorted) it
    on every call.
    """
    names = _official_component_names()
    rows: list[tuple[str, str]] = []
    for name in names:
        # In ordinary prose, "buttons" means plural Button. Explicit requests such
        # as "the Buttons component" are resolved separately by the exact matcher.
        if name.endswith("s") and name[:-1] in names:
            continue
        rows.append((name, re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()))
    rows.sort(key=lambda item: len(item[1]), reverse=True)
    return tuple(
        (name, phrase, re.compile(rf"\b{re.escape(phrase)}s?\b"))
        for name, phrase in rows
    )


def _prompt_component_mentions(prompt: str) -> frozenset[str]:
    """Find maximal schema component names mentioned in ordinary prompt prose."""
    prose = re.sub(r":[A-Za-z0-9_.-]+", " ", prompt)
    normalized = re.sub(r"[^a-z0-9]+", " ", prose.lower()).strip()
    occupied: list[tuple[int, int]] = []
    found: set[str] = set()
    for name, _phrase, matcher in _component_phrases():
        for match in matcher.finditer(normalized):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            found.add(name)
    return frozenset(found)


def semantic_role_candidates(
    placeholders: list[str], component_names: list[str]
) -> dict[str, tuple[str, ...]]:
    """Map visible slots to compatible visible component types."""
    from slm_training.dsl.lang_core import library_schema

    definitions = library_schema().get("$defs", {})
    property_aliases = {
        "body": {"text"},
        "copy": {"text"},
        "description": {"text"},
        "confirm": {"label", "action"},
        "create": {"label", "action"},
        "save": {"label", "action"},
        "submit": {"label", "action"},
        "continue": {"label", "action"},
        "cta": {"label", "action"},
    }
    result: dict[str, tuple[str, ...]] = {}
    for placeholder in sorted(set(placeholders)):
        role = placeholder.removeprefix(":").split(".")[-1]
        compatible_properties = {role, *property_aliases.get(role, ())}
        result[placeholder] = tuple(
            name
            for name in sorted(set(component_names))
            if compatible_properties.intersection(
                definitions.get(name, {}).get("properties") or {}
            )
        )
    return result


def semantic_role_contract(
    placeholders: list[str], component_names: list[str]
) -> str:
    """Describe visible slot roles using only visible schema component mentions."""
    candidates_by_slot = semantic_role_candidates(placeholders, component_names)
    groups: dict[str, dict[str, tuple[str, ...]]] = {}
    for placeholder in sorted(set(placeholders)):
        parts = placeholder.removeprefix(":").split(".")
        namespace = ".".join(parts[:-1]) or parts[0]
        role = parts[-1] if len(parts) > 1 else "value"
        groups.setdefault(namespace, {})[role] = candidates_by_slot[placeholder]
    return "; ".join(
        f"{namespace}("
        + ", ".join(
            f"{role} -> {'|'.join(candidates)}" if candidates else role
            for role, candidates in sorted(roles.items())
        )
        + ")"
        for namespace, roles in sorted(groups.items())
    )


def _prompt_component_requirements(
    prompt: str,
    *,
    preserve_repeated_mentions: bool = False,
) -> tuple[str, ...]:
    """Find positive component requirements, retaining explicit multiplicity.

    Unlike :func:`_prompt_component_mentions` (which returns a de-duplicated set
    for the data-quality judge), this drops negated/replaced mentions and keeps
    explicit counts. It is consumed by the binding-aware meaningful-program eval.
    """
    normalized = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
    occupied: list[tuple[int, int]] = []
    required: dict[str, int] = {}
    for name, _phrase, matcher in _component_phrases():
        for match in matcher.finditer(normalized):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            before = normalized[max(0, span[0] - 48) : span[0]]
            after = normalized[span[1] : span[1] + 24]
            if re.search(
                r"(?:\bnot|\bno|\bwithout|\bexclude|\bomit|\bavoid|"
                r"\bdo not (?:use|include|add)|\binstead of)\s+(?:a |an |the |any )?$",
                before,
            ) or re.match(r"\s+free\b", after):
                continue
            if re.search(
                r"(?:\breplace|\bswap|\bchange)\s+(?:a |an |the )?$", before
            ) and re.match(r"\s+(?:with|for|to)\b", after):
                continue
            count_match = re.search(
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+$",
                before,
            )
            count = 1
            if count_match:
                count = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }.get(
                    count_match.group(1),
                    int(count_match.group(1))
                    if count_match.group(1).isdigit()
                    else 1,
                )
            if preserve_repeated_mentions:
                required[name] = required.get(name, 0) + count
            else:
                required[name] = max(required.get(name, 0), count)
    return tuple(name for name in sorted(required) for _ in range(required[name]))


def _semantic_request(record: ExampleRecord) -> str:
    """Return the authored request, excluding embedded edit-program context."""
    if isinstance(record.meta.get("semantic_contract"), dict):
        return record.prompt
    edit = record.meta.get("edit")
    if isinstance(edit, dict) and isinstance(edit.get("instruction"), str):
        return edit["instruction"]
    if isinstance(record.meta.get("repair"), dict):
        return ""
    # Scope-graded rows embed DSL source, not prose intent; their contract
    # (echo / canonicalization equality) is judged separately.
    if isinstance(record.meta.get("scope_slice"), dict):
        return ""
    return record.prompt


def semantic_contract_for_openui(openui: str) -> dict[str, Any]:
    """Build a deterministic semantic scaffold from an OpenUI document."""
    assignments = {
        name: expression
        for name, expression in _ASSIGNMENT_RE.findall(openui.strip())
    }
    if "root" not in assignments:
        raise ValueError("semantic contract requires a root declaration")
    names = set(assignments)
    declarations: dict[str, str] = {}
    references: dict[str, list[str]] = {}
    for name, expression in assignments.items():
        component = _DECLARATION_COMPONENT_RE.match(expression)
        if component:
            declarations[name] = component.group(1)
        unquoted = _QUOTED_RE.sub("", expression)
        refs = sorted(
            {
                identifier
                for identifier in _IDENTIFIER_RE.findall(unquoted)
                if identifier in names and identifier != name
            }
        )
        if refs:
            references[name] = refs
    return {
        "version": 1,
        "component_counts": dict(sorted(component_counts(openui).items())),
        "declarations": dict(sorted(declarations.items())),
        "references": dict(sorted(references.items())),
        "placeholders": sorted(extract_placeholders(openui)),
    }


def render_semantic_contract_prompt(contract: dict[str, Any]) -> str:
    """Render the canonical prompt whose claims are independently judgeable."""
    components = ", ".join(
        f"{name} x{count}"
        for name, count in sorted(
            dict(contract.get("component_counts") or {}).items()
        )
    )
    declarations = "; ".join(
        f"{name} as {component}"
        for name, component in sorted(
            dict(contract.get("declarations") or {}).items()
        )
    )
    references = "; ".join(
        f"{name} references {', '.join(str(ref) for ref in refs)}"
        for name, refs in sorted(dict(contract.get("references") or {}).items())
    )
    placeholders = ", ".join(
        str(value) for value in contract.get("placeholders") or ()
    )
    return (
        "Create an OpenUI program. "
        f"Component inventory: {components or 'none'}. "
        f"Declarations: {declarations or 'none'}. "
        f"Reference graph: {references or 'none'}. "
        f"Placeholders: {placeholders or 'none'}."
    )


def _semantic_contract_reasons(record: ExampleRecord) -> list[str]:
    contract = record.meta.get("semantic_contract")
    if not isinstance(contract, dict):
        return []
    if contract.get("version") != 1:
        return ["semantic_contract_version_unsupported"]
    try:
        actual = semantic_contract_for_openui(record.openui)
    except ValueError:
        return ["semantic_contract_output_invalid"]
    reasons: list[str] = []
    if actual != contract:
        reasons.append("semantic_contract_output_mismatch")
    if record.prompt.strip() != render_semantic_contract_prompt(contract):
        reasons.append("semantic_contract_prompt_mismatch")
    return reasons


def _ast_component_names(value: Any) -> frozenset[str]:
    """Collect component types from a generated OpenUI AST."""
    found: set[str] = set()
    if isinstance(value, dict):
        type_name = value.get("typeName")
        if isinstance(type_name, str):
            found.add(type_name)
        for child in value.values():
            found.update(_ast_component_names(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_ast_component_names(child))
    return frozenset(found)


def _matches_schema_value(
    value: Any, spec: dict[str, Any], definitions: dict[str, Any]
) -> bool:
    if not spec:
        return True
    if isinstance(value, dict) and isinstance(value.get("k"), str):
        kind = value["k"]
        if kind in {"Str", "Num", "Bool"}:
            return _matches_schema_value(value.get("v"), spec, definitions)
        if kind == "Arr":
            elements = value.get("els")
            return isinstance(elements, list) and _matches_schema_value(
                elements, spec, definitions
            )
        if kind == "Obj":
            entries = value.get("entries")
            if not isinstance(entries, list) or not all(
                isinstance(entry, list) and len(entry) == 2 for entry in entries
            ):
                return False
            return _matches_schema_value(
                {str(entry[0]): entry[1] for entry in entries}, spec, definitions
            )
        inferred: str | None = None
        if kind == "Comp":
            inferred = (
                "number"
                if value.get("name")
                in {"Count", "Sum", "Avg", "Min", "Max", "Round", "Abs", "Ceil", "Floor"}
                else None
            )
        elif kind == "BinOp" and value.get("op") == "+":
            left = value.get("left")
            inferred = "string" if isinstance(left, dict) and left.get("k") == "Str" else None
        if "anyOf" in spec:
            return any(
                _matches_schema_value(value, branch, definitions)
                for branch in spec["anyOf"]
                if isinstance(branch, dict)
            )
        reference = spec.get("$ref")
        if isinstance(reference, str):
            return inferred == "object"
        expected_type = spec.get("type")
        if expected_type == "integer":
            return inferred == "integer"
        if expected_type == "number":
            return inferred in {"integer", "number"}
        # An unknown runtime result is not evidence that a static schema role
        # was satisfied. Fail closed until lang-core exposes a result type.
        return inferred is not None and inferred == expected_type
    enum = spec.get("enum")
    if isinstance(enum, list):
        return value in enum
    if "anyOf" in spec:
        return any(
            _matches_schema_value(value, branch, definitions)
            for branch in spec["anyOf"]
            if isinstance(branch, dict)
        )
    reference = spec.get("$ref")
    if isinstance(reference, str):
        expected = reference.rsplit("/", 1)[-1]
        return isinstance(value, dict) and value.get("typeName") == expected
    expected_type = spec.get("type")
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_spec = spec.get("items")
        return not isinstance(item_spec, dict) or all(
            _matches_schema_value(item, item_spec, definitions) for item in value
        )
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _schema_semantic_reasons(openui: str) -> list[str]:
    """Judge resolved AST property roles against the generated component schema."""
    schema = _official_schema()
    definitions = schema.get("$defs") or {}
    if not definitions:
        return []
    try:
        from slm_training.dsl.parser import parse

        program = parse(openui)
    except Exception:  # noqa: BLE001
        return ["judge_schema_parse_failed"]
    reasons: set[str] = set()
    for error in program.meta.get("errors") or ():
        if not isinstance(error, dict):
            continue
        code = str(error.get("code") or "unknown")
        component = str(error.get("component") or "unknown")
        path = str(error.get("path") or "").strip("/").replace("/", ".")
        reasons.add(f"schema_parser_error:{code}:{component}.{path}".rstrip("."))
    root = program.root
    if root is None:
        return sorted(reasons) or ["judge_schema_parse_failed"]

    state = program.state_declarations

    def resolve(value: Any, seen: frozenset[str] = frozenset()) -> Any:
        if (
            isinstance(value, dict)
            and value.get("k") == "StateRef"
            and isinstance(value.get("n"), str)
            and value["n"] not in seen
        ):
            return resolve(state.get(value["n"]), seen | {value["n"]})
        return value

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            component = value.get("typeName")
            props = value.get("props")
            definition = definitions.get(component) if isinstance(component, str) else None
            if isinstance(definition, dict) and isinstance(props, dict):
                property_specs = definition.get("properties") or {}
                required = set(definition.get("required") or ())
                for name in required:
                    if props.get(name) is None:
                        reasons.add(f"schema_required_value_missing:{component}.{name}")
                for name, prop_value in props.items():
                    # Positional null is the language's omission sentinel for an
                    # optional prop that precedes a later required prop.
                    if prop_value is None and name not in required:
                        continue
                    spec = property_specs.get(name)
                    if isinstance(spec, dict) and not _matches_schema_value(
                        resolve(prop_value), spec, definitions
                    ):
                        reasons.add(f"schema_value_role_mismatch:{component}.{name}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(root)
    return sorted(reasons)


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    score: float
    reasons: tuple[str, ...]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "score": self.score,
            "reasons": list(self.reasons),
            **self.meta,
        }


def independent_judge(record: ExampleRecord) -> dict[str, Any]:
    """Deterministic prompt/output contract judge before training admission."""
    prompt = (record.prompt or "").strip()
    openui = (record.openui or "").strip()
    components = component_counts(openui)
    request = _semantic_request(record)
    lowered_prompt = request.lower()
    reasons: list[str] = []
    if record.target_kind != "document":
        from slm_training.dsl.parser import ParseError, validate_output

        try:
            validate_output(openui, record.target_kind, record.target_category)
            for target_output in record.accepted_outputs:
                validate_output(
                    target_output.text,
                    target_output.kind,
                    target_output.category,
                )
        except (ParseError, ValueError, RuntimeError) as exc:
            reasons.append(f"invalid_output_contract:{exc}")
    repair = record.meta.get("repair")
    repair_ast = repair.get("clean_ast") if isinstance(repair, dict) else None
    targets = set(_ast_component_names(repair_ast))
    targets.update(_prompt_component_mentions(request))
    target = None
    component_match = _COMPONENT_PROMPT_RE.search(request)
    if component_match:
        target = component_match.group(1)
    else:
        construct_match = _CONSTRUCT_PROMPT_RE.search(request)
        if construct_match and construct_match.group(1).strip():
            target = construct_match.group(1).strip().split()[0]
    if (target and target[:1].isupper() and target not in components) or (
        targets - set(components)
    ):
        reasons.append("prompt_component_missing_from_output")
    if "boolean literal" in lowered_prompt:
        if not re.search(r"\b(?:true|false)\b", openui):
            reasons.append("prompt_boolean_missing_from_output")
        if len(components) > 1:
            reasons.append("prompt_lexical_target_wrapped_in_unrelated_layout")
    if (
        str((record.meta or {}).get("source_family") or "") == "language_contract"
        and lowered_prompt.startswith("emit the openui construct:")
    ):
        reasons.append("prompt_under_specified_for_layout")
    if (
        isinstance(record.meta.get("scope_slice"), dict)
        and record.meta.get("task") == "identity"
    ):
        # Identity anchors must echo the embedded source byte-for-byte.
        _, marker, embedded = prompt.partition("---INPUT---\n")
        if not marker or embedded != openui:
            reasons.append("identity_echo_mismatch")
    if not prompt or not openui:
        reasons.append("judge_missing_prompt_or_output")
    if (
        "prompt_under_specified_for_layout" in reasons
        and str(record.meta.get("task") or "generation") == "generation"
        and not isinstance(record.meta.get("semantic_contract"), dict)
    ):
        # An under-specified generation prompt is only independently judgeable
        # when a semantic contract makes its claims explicit; flag its absence.
        # Well-specified prompts (which name their components) and non-generation
        # records remain admissible without a contract.
        reasons.append("generation_semantic_contract_missing")
    reasons.extend(_semantic_contract_reasons(record))
    if record.target_kind == "document":
        reasons.extend(_schema_semantic_reasons(openui))
    return {"ok": not reasons, "score": round(max(0.0, 1.0 - 0.5 * len(reasons)), 4), "reasons": reasons}


def component_counts(openui: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _COMPONENT_RE.findall(openui):
        counts[name] = counts.get(name, 0) + 1
    return counts


def assess_record(
    record: ExampleRecord,
    *,
    min_placeholders: int = 1,
    min_components: int = 2,
    require_design_md: bool = True,
    min_design_lint: float = 0.7,
    max_openui_chars: int | None = None,
    max_components: int | None = None,
) -> QualityReport:
    """
    Score a record for training usefulness.

    Deterministic: pure function of record fields (no wall-clock / RNG).
    """
    reasons: list[str] = []
    score = 1.0
    openui = (record.openui or "").strip()
    prompt = (record.prompt or "").strip()
    judge = independent_judge(record)
    if not judge["ok"]:
        reasons.extend(str(reason) for reason in judge["reasons"])
        score -= 0.45

    if record.target_kind != "document":
        from slm_training.dsl.parser import lexical_tokens

        if len(prompt) < 12:
            reasons.append("prompt_too_short")
            score -= 0.25
        score = max(0.0, min(1.0, round(score, 4)))
        counts = component_counts(openui)
        return QualityReport(
            ok=score >= 0.55 and judge["ok"],
            score=score,
            reasons=tuple(reasons),
            meta={
                "n_components": sum(counts.values()),
                "n_placeholders": len(
                    list(record.placeholders) or extract_placeholders(openui)
                ),
                "component_diversity": len(counts),
                "openui_chars": len(openui),
                "output_symbols": len(lexical_tokens(openui)),
                "target_kind": record.target_kind,
                "accepted_outputs": len(record.accepted_outputs),
                "independent_judge": judge,
            },
        )

    if not _ROOT_RE.search(openui):
        reasons.append("missing_root")
        score -= 0.5
    if len(prompt) < 12:
        reasons.append("prompt_too_short")
        score -= 0.25

    placeholders = list(record.placeholders) or extract_placeholders(openui)
    if len(placeholders) < min_placeholders:
        reasons.append("too_few_placeholders")
        score -= 0.35

    # Free-form quoted strings that are not placeholders are a smell
    # (except enum-like tokens handled by the bridge policy already).
    for match in re.finditer(r'"([^"]+)"', openui):
        val = match.group(1)
        if val.startswith(":"):
            continue
        if re.fullmatch(r"[a-z0-9_\-]+", val):
            # enum / field name
            continue
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", val):
            continue
        reasons.append("non_placeholder_string")
        score -= 0.2
        break

    counts = component_counts(openui)
    n_comp = sum(counts.values())
    if n_comp < min_components:
        reasons.append("too_few_components")
        score -= 0.3
    unknown = sorted(set(counts) - PREFERRED_COMPONENTS)
    if unknown:
        reasons.append("unknown_components")
        score -= 0.15

    diversity = len(set(counts))
    if diversity >= 3:
        score += 0.05
    if diversity >= 4:
        score += 0.05

    if max_openui_chars is not None and len(openui) > max_openui_chars:
        reasons.append("openui_too_long")
        score -= 0.4
    if max_components is not None and n_comp > max_components:
        reasons.append("too_many_components")
        score -= 0.35
    # Soft preference for compact layouts (easier for small models).
    if len(openui) > 900:
        score -= 0.05
    if n_comp > 14:
        score -= 0.05

    design_score = None
    if require_design_md:
        if not record.design_md:
            reasons.append("missing_design_md")
            score -= 0.3
        else:
            lint_meta = (record.meta or {}).get("design_lint") or {}
            design_score = lint_meta.get("score")
            errors = int((lint_meta.get("summary") or {}).get("errors") or 0)
            if design_score is None:
                # Offline / not linted yet — small penalty, not a hard fail.
                score -= 0.05
            elif errors > 0 and float(design_score) < min_design_lint:
                # Only error-level DESIGN.md issues can soft-penalize.
                # Style warnings must not reject structure-only records.
                reasons.append("low_design_lint")
                score -= 0.25

    score = max(0.0, min(1.0, round(score, 4)))
    hard = {
        "missing_root",
        "too_few_placeholders",
        "too_few_components",
        "missing_design_md",
        "non_placeholder_string",
        "openui_too_long",
        "too_many_components",
        "prompt_component_missing_from_output",
        "prompt_boolean_missing_from_output",
        "prompt_lexical_target_wrapped_in_unrelated_layout",
        "judge_missing_prompt_or_output",
        "prompt_under_specified_for_layout",
    }
    ok = score >= 0.55 and not hard.intersection(reasons)
    return QualityReport(
        ok=ok,
        score=score,
        reasons=tuple(reasons),
        meta={
            "n_components": n_comp,
            "n_placeholders": len(placeholders),
            "component_diversity": diversity,
            "openui_chars": len(openui),
            "design_lint_score": design_score,
            "components": counts,
            "independent_judge": judge,
        },
    )


def filter_quality(
    records: list[ExampleRecord],
    *,
    min_score: float = 0.55,
    require_design_md: bool = True,
    max_openui_chars: int | None = None,
    max_components: int | None = None,
) -> tuple[list[ExampleRecord], list[dict[str, Any]]]:
    """Keep high-quality records; return (kept, rejection reports)."""
    kept: list[ExampleRecord] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        report = assess_record(
            record,
            require_design_md=require_design_md,
            max_openui_chars=max_openui_chars,
            max_components=max_components,
        )
        if report.ok and report.score >= min_score:
            meta = dict(record.meta or {})
            meta["quality"] = report.to_dict()
            record.meta = meta
            kept.append(record)
        else:
            rejected.append({"id": record.id, **report.to_dict()})
    # Stable order for determinism.
    kept.sort(key=lambda r: r.id)
    rejected.sort(key=lambda r: r["id"])
    return kept, rejected
