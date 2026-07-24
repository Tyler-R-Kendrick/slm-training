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
from slm_training.dsl.language_contract import contract_id as current_contract_id
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.renderability import STRUCTURAL_ROOT_CONTAINERS
from slm_training.dsl.schema import ExampleRecord, OutputKind, OutputTarget

LANGUAGE_CONTRACT_FAMILY = "language_contract"
_SOURCE = "language_contract"
_PROP_ORDER_PATH = (
    repo_root()
    / "src"
    / "slm_training"
    / "dsl"
    / "grammars"
    / "openui_prop_order.json"
)

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
    """Build one self-contained component expression."""

    def _child_any(self, prefix: str) -> str:
        return f'TextContent(":{prefix}.child")'

    def _arg(self, prop: str, schema: dict, prefix: str) -> str:
        if not isinstance(schema, dict):
            schema = {}
        branches = schema.get("anyOf")
        if isinstance(branches, list):
            branch = next((item for item in branches if isinstance(item, dict)), {})
            return self._arg(prop, branch, prefix)
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
        return f'{name}({", ".join(args)})'


def component_expression(name: str) -> str:
    """A minimal expression that exercises every required prop of ``name``."""
    return _Builder().build(name, "cov")


def component_program(name: str) -> str:
    """A minimal complete document that exercises ``name``."""
    return _renderable_component_program(name, component_expression(name))


# --------------------------------------------------------------------------- #
# Grammar / lexical production positives (beyond per-component coverage)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Production:
    unit_id: str
    category: str
    primary: str
    kind: OutputKind
    description: str
    accepted: tuple[OutputTarget, ...]
    target_category: str | None = None


_PRODUCTIONS: tuple[Production, ...] = (
    Production(
        "assignment",
        "grammar",
        "x = true",
        "statement",
        "a single binder assignment",
        (OutputTarget('root = Separator("horizontal", true)'),),
    ),
    Production(
        "nested_list",
        "grammar",
        "[[true]]",
        "expression",
        "a nested list expression",
        (OutputTarget("root = Stack([[TextContent(\":item\")]])"),),
    ),
    Production(
        "multi_child",
        "grammar",
        "[true, false, null]",
        "expression",
        "a multi-child list expression",
        (OutputTarget("root = Stack([TextContent(\":a\"), TextContent(\":b\")])"),),
    ),
    Production(
        "forward_reference",
        "grammar",
        "root = Stack([panel])\npanel = Card([leaf])\nleaf = TextContent(\":leaf\")",
        "document",
        "a forward reference from the root",
        (),
    ),
    Production(
        "number_literal",
        "lexical",
        "0",
        "lexical",
        "a number literal",
        (
            OutputTarget("1", "lexical", "number"),
            OutputTarget('root = Slider("value", "continuous", 0, 1)'),
        ),
        "number",
    ),
    Production(
        "negative_number",
        "lexical",
        "-1",
        "lexical",
        "a negative number literal",
        (OutputTarget('root = Slider("value", "continuous", -1, 1)'),),
        "number",
    ),
    Production(
        "boolean_literal",
        "lexical",
        "true",
        "lexical",
        "a boolean literal",
        (
            OutputTarget("false", "lexical", "boolean"),
            OutputTarget('root = Separator("horizontal", true)'),
            OutputTarget('root = Separator("horizontal", false)'),
        ),
        "boolean",
    ),
    Production(
        "enum_prop",
        "lexical",
        '"row"',
        "lexical",
        "an enum-valued prop",
        (
            OutputTarget('"column"', "lexical", "enum"),
            OutputTarget('root = Stack([], "row")'),
            OutputTarget('root = Stack([], "column")'),
        ),
        "enum",
    ),
)

# Comments are ignored by the output lexer and remain validator coverage only.
_VALIDATOR_ONLY_PRODUCTIONS = ("comment_line",)

# --------------------------------------------------------------------------- #
# Negatives: each corruption targets exactly one verifier gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Negative:
    unit_id: str
    gate: str
    operator: str
    openui: str


@dataclass(frozen=True)
class RootRenderabilityPair:
    """A parseable but blank root paired with its visible repair."""

    component: str
    container: str
    prompt: str
    chosen: str
    rejected: str


def _renderable_component_program(name: str, expression: str) -> str:
    """Place one structural declaration in its smallest visible document."""
    expression = expression.replace('"item"', '":cov.name"')
    if name == "Col":
        program = f"root = Table([{expression}])"
    elif name == "Series":
        program = f'root = BarChart([":cov.item"], [{expression}])'
    elif name == "Slice":
        # PieChart consumes parallel labels and values rather than Slice nodes.
        program = 'root = PieChart([":cov.item"], [1])'
    elif name == "ScatterSeries":
        program = f"root = ScatterChart([{expression}])"
    elif name == "Point":
        program = f'root = ScatterChart([ScatterSeries(":cov.name", [{expression}])])'
    elif name == "SelectItem":
        program = f'root = Select(":cov.name", [{expression}])'
    elif name == "CheckBoxItem":
        program = f'root = CheckBoxGroup(":cov.name", [{expression}])'
    elif name == "RadioItem":
        program = f'root = RadioGroup(":cov.name", [{expression}])'
    elif name == "SwitchItem":
        program = f'root = SwitchGroup(":cov.name", [{expression}])'
    elif name == "TabItem":
        program = f"root = Tabs([{expression}])"
    elif name == "AccordionItem":
        program = f"root = Accordion([{expression}])"
    elif name == "StepsItem":
        program = f"root = Steps([{expression}])"
    else:
        program = f"root = {expression}"
    if ":cov." in program:
        return program
    # Meaningful-program admission requires a visible slot. Keep data-only
    # declarations useful without changing their repaired container semantics.
    return (
        "root = Stack([content, context])\n"
        f"content = {program.removeprefix('root = ')}\n"
        'context = TextContent(":cov.context")'
    )


def iter_root_renderability_pairs() -> Iterator[RootRenderabilityPair]:
    """Emit one explicit preferred repair for every banned standalone root."""
    for name, container in STRUCTURAL_ROOT_CONTAINERS.items():
        expression = component_expression(name)
        rejected = f"root = {expression}"
        prompt = (
            "Repair this OpenUI document so it renders visibly. "
            f"The root {name} is structural-only; integrate it in {container}(...).\n"
            f"---BROKEN---\n{rejected}"
        )
        yield RootRenderabilityPair(
            component=name,
            container=container,
            prompt=prompt,
            chosen=_renderable_component_program(name, expression),
            rejected=rejected,
        )


def iter_root_renderability_repairs(split: str = "train") -> Iterator[ExampleRecord]:
    """Valid SFT repairs; the banned form is prompt context, never a target."""
    for pair in iter_root_renderability_pairs():
        unit_id = f"root_renderability_{pair.component.lower()}"
        yield ExampleRecord(
            id=f"lc_{unit_id}",
            prompt=pair.prompt,
            openui=pair.chosen,
            split=split,
            source=_SOURCE,
            target_kind="document",
            meta={
                "category": "root_renderability",
                "contract_target": unit_id,
                "polarity": "positive",
                "contract_id": current_contract_id(),
                "program_family_id": f"{LANGUAGE_CONTRACT_FAMILY}:{unit_id}",
                "lineage_id": f"lc_{unit_id}",
                "split_group_id": f"lc_{unit_id}",
                "task": "repair",
                "determinacy": "deterministic",
                "tier": "Silver",
                "source_kind": "deterministic",
                "root_renderability": {
                    "root_type": pair.component,
                    "container": pair.container,
                    "rejected": pair.rejected,
                },
            },
        )


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
    *,
    target_kind: OutputKind,
    target_category: str | None = None,
    accepted_outputs: tuple[OutputTarget, ...] = (),
) -> ExampleRecord:
    return ExampleRecord(
        id=f"lc_{unit_id}",
        prompt=(
            f"Return only the shortest valid OpenUI {target_kind} for: {description}."
        ),
        openui=openui,
        split=split,
        source=_SOURCE,
        target_kind=target_kind,
        target_category=target_category,
        accepted_outputs=list(accepted_outputs),
        meta={
            "category": category,
            "contract_target": unit_id,
            "polarity": "positive",
            "contract_id": current_contract_id(),
            "program_family_id": f"{LANGUAGE_CONTRACT_FAMILY}:{unit_id}",
            "lineage_id": f"lc_{unit_id}",
            "split_group_id": f"lc_{unit_id}",
            "task": "generation",
            "determinacy": "deterministic",
            "tier": "Silver",
            "source_kind": "deterministic",
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
    for production in _PRODUCTIONS:
        yield _positive_record(
            production.unit_id,
            production.category,
            production.primary,
            production.description,
            split,
            target_kind=production.kind,
            target_category=production.target_category,
            accepted_outputs=production.accepted,
        )
    for name in component_names():
        expression = component_expression(name)
        yield _positive_record(
            f"component_{name}",
            "component",
            expression,
            f"the {name} component",
            split,
            target_kind="expression",
            target_category="component",
            accepted_outputs=(OutputTarget(component_program(name)),),
        )
    yield from iter_root_renderability_repairs(split)


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
    production_ids = [item.unit_id for item in _PRODUCTIONS]

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
        "productions": {
            "count": len(production_ids),
            "ids": production_ids,
            "validator_only": list(_VALIDATOR_ONLY_PRODUCTIONS),
        },
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
