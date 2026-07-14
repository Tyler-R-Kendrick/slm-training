"""Language-contract coverage corpus generator (deterministic, offline-friendly).

Emits, for the pinned OpenUI v0.2.x contract:

* **positives** — one minimal valid, *meaningful* program per grammar production,
  lexical form, and component (exercising each component's positional props), each
  routed through the F1 ``emit_record`` projection so it carries ``contract_id`` /
  family / lineage; and
* **negatives** — targeted corruptions, each annotated with the single verifier
  gate it is designed to trip (G0 lexical · G1 grammar · G2 schema · G3 references
  · G4 dataflow).

Component minimal-instances are derived from the authoritative bridge
``library_schema`` (required props + types + enums) so the corpus tracks the
component library rather than a hand-maintained list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from slm_training.bridge_utils import repo_root
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.dsl.language_contract import contract_id as current_contract_id
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.schema import ExampleRecord

LANGUAGE_CONTRACT_FAMILY = "language_contract"
_SOURCE = "language_contract"
_PROP_ORDER_PATH = repo_root() / "grammars" / "openui_prop_order.json"

# Gate ids mirror ``slm_training.data.verify.stack.Gate`` values.
GATE_LEXICAL = "G0"
GATE_GRAMMAR = "G1"
GATE_SCHEMA = "G2"
GATE_REFERENCES = "G3"
GATE_DATAFLOW = "G4"
NEGATIVE_GATES = (
    GATE_LEXICAL,
    GATE_GRAMMAR,
    GATE_SCHEMA,
    GATE_REFERENCES,
    GATE_DATAFLOW,
)

# String props that carry a literal token, not a user-facing placeholder.
_LITERAL_STRING_PROPS = frozenset({"language", "value", "category", "name"})


@lru_cache(maxsize=1)
def _defs() -> dict[str, Any]:
    return dict(library_schema().get("$defs", {}))


@lru_cache(maxsize=1)
def _prop_order() -> dict[str, list[str]]:
    return json.loads(_PROP_ORDER_PATH.read_text(encoding="utf-8"))


def component_names() -> list[str]:
    """All components in the pinned contract (sorted)."""
    return sorted(_prop_order())


# --------------------------------------------------------------------------- #
# Schema-driven minimal valid instance builder
# --------------------------------------------------------------------------- #


class _Builder:
    """Accumulates the binder statements for one component instance."""

    def __init__(self) -> None:
        self.counter = 0
        self.lines: list[str] = []

    def fresh(self, base: str) -> str:
        self.counter += 1
        return f"{base.lower()}{self.counter}"

    def _child_any(self, prefix: str) -> str:
        binder = self.fresh("txt")
        self.lines.append(f'{binder} = TextContent(":{prefix}.child")')
        return binder

    def _arg(self, prop: str, schema: dict, prefix: str) -> str:
        if not isinstance(schema, dict):
            schema = {}
        if "enum" in schema:
            return f'"{schema["enum"][0]}"'
        kind = schema.get("type")
        if kind == "string":
            if prop in _LITERAL_STRING_PROPS:
                return '"item"'
            return f'":{prefix}.{prop}"'
        if kind == "boolean":
            return "true"
        if kind == "number":
            return "1"
        if kind == "object":
            return "null"
        if kind == "array":
            item = schema.get("items")
            item = item if isinstance(item, dict) else {}
            if "$ref" in item:
                return f"[{self.build(item['$ref'].split('/')[-1], prefix)}]"
            if item.get("type") == "array":
                return f"[[{self._child_any(prefix)}]]"
            if item.get("type") == "object":
                return "[]"
            if "enum" in item:
                return f'["{item["enum"][0]}"]'
            if item.get("type") == "number":
                return "[1]"
            if item.get("type") == "string":
                return f'[":{prefix}.item"]'
            return f"[{self._child_any(prefix)}]"
        if "$ref" in schema:
            return self.build(schema["$ref"].split("/")[-1], prefix)
        return f'":{prefix}.{prop}"'

    def build(self, name: str, prefix: str) -> str:
        definition = _defs()[name]
        props = definition.get("properties", {})
        required = set(definition.get("required", ()))
        order = _prop_order().get(name, list(props.keys()))
        required_indices = [i for i, prop in enumerate(order) if prop in required]
        upto = max(required_indices) if required_indices else -1
        args: list[str] = []
        for index in range(upto + 1):
            prop = order[index]
            if prop in required:
                args.append(self._arg(prop, props.get(prop, {}), prefix))
            else:
                # Optional prop preceding a required one: null keeps positions aligned.
                args.append("null")
        binder = self.fresh(name)
        self.lines.append(f'{binder} = {name}({", ".join(args)})')
        return binder


def component_program(name: str) -> str:
    """A minimal, meaningful program that exercises ``name`` (plus a caption)."""
    builder = _Builder()
    instance = builder.build(name, "cov")
    caption = builder.fresh("cap")
    builder.lines.append(f'{caption} = TextContent(":cov.caption")')
    return f"root = Stack([{instance}, {caption}])\n" + "\n".join(builder.lines)


# --------------------------------------------------------------------------- #
# Grammar / lexical production positives (beyond per-component coverage)
# --------------------------------------------------------------------------- #

_PRODUCTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "assignment",
        "grammar",
        "root = Stack([hero])\nhero = TextContent(\":hero.title\")",
        "a single binder assignment",
    ),
    (
        "nested_list",
        "grammar",
        (
            "root = Stack([card])\n"
            "card = Card([hero, body])\n"
            "hero = TextContent(\":hero.title\")\n"
            "body = TextContent(\":hero.body\")"
        ),
        "a nested list of references",
    ),
    (
        "multi_child",
        "grammar",
        (
            "root = Stack([a, b, c])\n"
            "a = TextContent(\":a\")\n"
            "b = TextContent(\":b\")\n"
            "c = TextContent(\":c\")"
        ),
        "a multi-child list",
    ),
    (
        "forward_reference",
        "grammar",
        (
            "root = Stack([panel])\n"
            "panel = Card([leaf])\n"
            "leaf = TextContent(\":leaf\")"
        ),
        "a forward reference from the root",
    ),
    (
        "number_literal",
        "lexical",
        (
            "root = Stack([vol, cap])\n"
            "vol = Slider(\"volume\", \"continuous\", 0, 100)\n"
            "cap = TextContent(\":cap\")"
        ),
        "integer number literals",
    ),
    (
        "negative_number",
        "lexical",
        (
            "root = Stack([temp, cap])\n"
            "temp = Slider(\"temp\", \"continuous\", -10, 40)\n"
            "cap = TextContent(\":cap\")"
        ),
        "a negative number literal",
    ),
    (
        "boolean_literal",
        "lexical",
        (
            "root = Stack([sep, cap])\n"
            "sep = Separator(\"horizontal\", true)\n"
            "cap = TextContent(\":cap\")"
        ),
        "a boolean literal",
    ),
    (
        "comment_line",
        "lexical",
        (
            "// layout header\n"
            "root = Stack([hero])\n"
            "hero = TextContent(\":hero.title\")"
        ),
        "a line comment",
    ),
    (
        "enum_prop",
        "expression",
        (
            "root = Stack([cta, cap], \"row\")\n"
            "cta = Button(\":cta.label\", null, \"primary\")\n"
            "cap = TextContent(\":cap\")"
        ),
        "enum-valued props",
    ),
)


# --------------------------------------------------------------------------- #
# Negatives: each corruption targets exactly one verifier gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Negative:
    unit_id: str
    gate: str
    operator: str
    openui: str


_NEGATIVES: tuple[Negative, ...] = (
    # G0 — lexical
    Negative(
        "lex_unterminated_string",
        GATE_LEXICAL,
        "unterminated_string",
        "root = Stack([hero])\nhero = TextContent(\":hero.title)",
    ),
    Negative(
        "lex_control_char",
        GATE_LEXICAL,
        "forbidden_control_char",
        "root = Stack([hero])\nhero = TextContent(\":hero.\x01title\")",
    ),
    # G1 — grammar
    Negative(
        "gram_missing_assignment",
        GATE_GRAMMAR,
        "missing_assignment",
        "root = Stack([hero])\nhero TextContent(\":hero.title\")",
    ),
    Negative(
        "gram_unclosed_paren",
        GATE_GRAMMAR,
        "unclosed_paren",
        "root = Stack([hero])\nhero = TextContent(\":hero.title\"",
    ),
    Negative(
        "gram_missing_comma",
        GATE_GRAMMAR,
        "missing_list_comma",
        "root = Stack([hero cta])\nhero = TextContent(\":a\")\ncta = Button(\":b\")",
    ),
    # G2 — schema
    Negative(
        "schema_unknown_component",
        GATE_SCHEMA,
        "unknown_component",
        "root = Stack([hero])\nhero = Bogus(\":hero.title\")",
    ),
    Negative(
        "schema_wrong_arg_count",
        GATE_SCHEMA,
        "too_many_positional_args",
        "root = Stack([hero])\nhero = TextContent(\":a\", \"small\", \"extra\")",
    ),
    Negative(
        "schema_missing_required",
        GATE_SCHEMA,
        "missing_required_prop",
        "root = Stack([hero])\nhero = TextContent()",
    ),
    Negative(
        "schema_literal_content",
        GATE_SCHEMA,
        "literal_content_prop",
        "root = Stack([cta])\ncta = Button(\"Click me\")",
    ),
    # G3 — reference graph
    Negative(
        "ref_undefined",
        GATE_REFERENCES,
        "undefined_reference",
        "root = Stack([ghost, hero])\nhero = TextContent(\":a\")",
    ),
    Negative(
        "ref_duplicate_binder",
        GATE_REFERENCES,
        "duplicate_binder",
        "root = Stack([hero])\nhero = TextContent(\":a\")\nhero = TextContent(\":b\")",
    ),
    Negative(
        "ref_missing_root",
        GATE_REFERENCES,
        "missing_root",
        "main = Stack([hero])\nhero = TextContent(\":a\")",
    ),
    Negative(
        "ref_unreachable",
        GATE_REFERENCES,
        "unreachable_binder",
        "root = Stack([hero])\nhero = TextContent(\":a\")\norphan = Button(\":o\")",
    ),
    # G4 — dataflow (v0.5 syntax outside the pinned 0.2.x contract)
    Negative(
        "flow_query",
        GATE_DATAFLOW,
        "v05_query",
        "root = Stack([hero])\nhero = TextContent(\":a\")\nq = Query(\":src\")",
    ),
    Negative(
        "flow_mutation",
        GATE_DATAFLOW,
        "v05_mutation",
        "root = Stack([hero])\nhero = TextContent(\":a\")\nm = Mutation(\":t\")",
    ),
    Negative(
        "flow_action",
        GATE_DATAFLOW,
        "v05_action_binding",
        "root = Stack([cta])\ncta = Button(\":label\", @Run)",
    ),
    Negative(
        "flow_state",
        GATE_DATAFLOW,
        "v05_state_declaration",
        "root = Stack([hero])\nhero = TextContent(\":a\")\n$count = 0",
    ),
)


# --------------------------------------------------------------------------- #
# Record projection
# --------------------------------------------------------------------------- #


def _positive_record(
    unit_id: str,
    category: str,
    openui: str,
    description: str,
    split: str,
) -> ExampleRecord:
    spec = ProgramSpec.from_openui(
        id=f"lc_{unit_id}",
        openui=openui,
        facts={"contract_target": unit_id, "category": category},
        program_family_id=f"{LANGUAGE_CONTRACT_FAMILY}:{unit_id}",
        lineage_id=f"lc_{unit_id}",
        split_group_id=f"lc_{unit_id}",
        split=split,
    )
    return emit_record(
        spec,
        prompt=f"Emit the OpenUI construct: {description}.",
        task="generation",
        source=_SOURCE,
        determinacy="deterministic",
        tier="Silver",
        meta={
            "category": category,
            "contract_target": unit_id,
            "polarity": "positive",
        },
    )


def _negative_record(negative: Negative, split: str) -> ExampleRecord:
    return ExampleRecord(
        id=f"lc_{negative.unit_id}",
        prompt=(
            f"Invalid OpenUI ({negative.gate} / {negative.operator}): "
            "the verifier must reject this program."
        ),
        openui=negative.openui,
        split=split,
        source=_SOURCE,
        meta={
            "contract_id": current_contract_id(),
            "task": "adversarial",
            "family": LANGUAGE_CONTRACT_FAMILY,
            "category": "negative",
            "polarity": "negative",
            "expected_gate": negative.gate,
            "operator": negative.operator,
            "determinacy": "deterministic",
            "tier": "Quarantine",
            "split_group_id": f"lc_{negative.unit_id}",
        },
    )


def iter_positives(split: str = "train") -> Iterator[ExampleRecord]:
    """Grammar/lexical production positives, then one per component."""
    for unit_id, category, openui, description in _PRODUCTIONS:
        yield _positive_record(unit_id, category, openui, description, split)
    for name in component_names():
        yield _positive_record(
            f"component_{name}",
            "component",
            component_program(name),
            f"the {name} component",
            split,
        )


def iter_negatives(split: str = "train") -> Iterator[ExampleRecord]:
    for negative in _NEGATIVES:
        yield _negative_record(negative, split)


def build_corpus(split: str = "train") -> list[ExampleRecord]:
    """The full language-contract corpus (positives + negatives)."""
    return [*iter_positives(split), *iter_negatives(split)]


def coverage_report() -> dict[str, Any]:
    """Which productions / components / prop positions / gates are covered."""
    components = component_names()
    covered_components = set(components)  # one positive per component
    production_ids = [unit_id for unit_id, *_ in _PRODUCTIONS]

    required_positions = 0
    covered_positions = 0
    per_component: dict[str, dict[str, int]] = {}
    defs = _defs()
    for name in components:
        order = _prop_order().get(name, [])
        required = set(defs.get(name, {}).get("required", ()))
        # The minimal instance fills every required positional prop.
        covered = [prop for prop in order if prop in required]
        per_component[name] = {"props": len(order), "covered": len(covered)}
        required_positions += len(required)
        covered_positions += len(covered)

    return {
        "components": {
            "total": len(components),
            "covered": len(covered_components),
            "uncovered": sorted(set(components) - covered_components),
        },
        "productions": {"count": len(production_ids), "ids": production_ids},
        "required_prop_positions": {
            "total": required_positions,
            "covered": covered_positions,
        },
        "per_component_props": per_component,
        "gates": {
            "covered": sorted({negative.gate for negative in _NEGATIVES}),
            "expected": list(NEGATIVE_GATES),
        },
        "negatives": len(_NEGATIVES),
    }
