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


def _prompt_component_mentions(prompt: str) -> frozenset[str]:
    """Find maximal schema component names mentioned in ordinary prompt prose."""
    normalized = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
    names = _official_component_names()
    occupied: list[tuple[int, int]] = []
    found: set[str] = set()
    phrases = []
    for name in names:
        # In ordinary prose, "buttons" means plural Button. Explicit requests such
        # as "the Buttons component" are resolved separately by the exact matcher.
        if name.endswith("s") and name[:-1] in names:
            continue
        phrase = re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()
        phrases.append((name, phrase))
    for name, phrase in sorted(phrases, key=lambda item: len(item[1]), reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}s?\b", normalized):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            found.add(name)
    return frozenset(found)


def _semantic_request(record: ExampleRecord) -> str:
    """Return the authored request, excluding embedded edit-program context."""
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
                        prop_value, spec, definitions
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
        return QualityReport(
            ok=score >= 0.55 and judge["ok"],
            score=score,
            reasons=tuple(reasons),
            meta={
                "n_components": sum(component_counts(openui).values()),
                "n_placeholders": len(
                    list(record.placeholders) or extract_placeholders(openui)
                ),
                "component_diversity": len(component_counts(openui)),
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
