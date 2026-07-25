"""Coverage-guided typed ProgramSpec generation for the pinned OpenUI contract."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from typing import Any

from slm_training.bridge_utils import repo_root
from slm_training.data.language_contract.corpus import component_names
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.verify import Tier, VerificationContext, verify_record
from slm_training.dsl.lang_core import library_schema

GENERATOR_VERSION = 1
PROGRAM_FAMILY = "programspec_generated"
_PROP_ORDER_PATH = (
    repo_root() / "src" / "slm_training" / "dsl" / "grammars" / "openui_prop_order.json"
)
_BINDER_RE = re.compile(r"[^a-z0-9_]+")
_LITERAL_STRING_PROPS = frozenset({"language", "value", "category", "name"})
_DEFERRED_PATTERNS = ("state", "query", "mutation", "action", "tool")
_FORWARD_REFERENCE_PATTERNS = frozenset(
    {"root_distinct_slot_inputs", "root_shared_slot_free_input"}
)


@dataclass(frozen=True, order=True)
class CoverageCell:
    axis: str
    key: str

    def label(self) -> str:
        return f"{self.axis}:{self.key}"


@dataclass
class CoverageTracker:
    """Tracks target-grid hits and verifier outcomes without hiding unsupported cells."""

    targets: frozenset[CoverageCell]
    unsupported: frozenset[CoverageCell] = frozenset()
    hits: Counter[CoverageCell] = field(default_factory=Counter)
    passed: int = 0
    failed: int = 0

    @property
    def coverable(self) -> frozenset[CoverageCell]:
        return self.targets - self.unsupported

    @property
    def uncovered(self) -> tuple[CoverageCell, ...]:
        return tuple(sorted(cell for cell in self.coverable if not self.hits[cell]))

    @property
    def complete(self) -> bool:
        return not self.uncovered

    def score(self, cells: Iterable[CoverageCell]) -> float:
        return sum(
            2.0
            if cell in self.coverable and not self.hits[cell]
            else 1 / (1 + self.hits[cell])
            for cell in set(cells)
            if cell not in self.unsupported
        )

    def record(self, cells: Iterable[CoverageCell], *, verifier_passed: bool) -> None:
        self.hits.update(set(cells))
        if verifier_passed:
            self.passed += 1
        else:
            self.failed += 1

    def report(self) -> dict[str, Any]:
        axes = sorted({cell.axis for cell in self.targets | set(self.hits)})
        per_axis: dict[str, dict[str, Any]] = {}
        for axis in axes:
            targets = sorted(cell for cell in self.targets if cell.axis == axis)
            unsupported = sorted(cell for cell in self.unsupported if cell.axis == axis)
            uncovered = [
                cell
                for cell in targets
                if cell not in self.unsupported and not self.hits[cell]
            ]
            per_axis[axis] = {
                "total": len(targets),
                "covered": sum(bool(self.hits[cell]) for cell in targets),
                "uncovered": [cell.key for cell in uncovered],
                "unsupported": [cell.key for cell in unsupported],
            }
        return {
            "complete": self.complete,
            "targets": len(self.targets),
            "covered": sum(bool(self.hits[cell]) for cell in self.targets),
            "uncovered": [cell.label() for cell in self.uncovered],
            "unsupported": [cell.label() for cell in sorted(self.unsupported)],
            "verifier": {"passed": self.passed, "failed": self.failed},
            "axes": per_axis,
        }


@dataclass(frozen=True)
class Reference:
    binder: str


TypedValue = str | int | float | bool | None | Reference | tuple["TypedValue", ...]


def _serialize_value(value: TypedValue) -> str:
    if isinstance(value, Reference):
        return value.binder
    if isinstance(value, ComponentCall):
        return value.serialize()
    if isinstance(value, tuple):
        return f"[{', '.join(_serialize_value(item) for item in value)}]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


@dataclass(frozen=True)
class ComponentCall:
    type_name: str
    args: tuple[TypedValue, ...] = ()

    def serialize(self) -> str:
        return (
            f"{self.type_name}({', '.join(_serialize_value(arg) for arg in self.args)})"
        )


@dataclass(frozen=True)
class TypedStatement:
    binder: str
    call: ComponentCall

    def serialize(self) -> str:
        return f"{self.binder} = {self.call.serialize()}"


@dataclass(frozen=True)
class TypedProgram:
    root: ComponentCall
    statements: tuple[TypedStatement, ...]

    def serialize(self) -> str:
        lines = [f"root = {self.root.serialize()}"]
        lines.extend(statement.serialize() for statement in self.statements)
        return "\n".join(lines)


@dataclass(frozen=True)
class GeneratorConfig:
    components: tuple[str, ...] | None = None
    max_depth: int = 3
    max_width: int = 3
    viewports: tuple[str, ...] = ("mobile", "desktop")
    render_states: tuple[str, ...] = ("empty", "loading", "success", "error")
    content_classes: tuple[str, ...] = ("plain", "escaped", "dsl_like")
    selected_triples: tuple[tuple[str, str, str], ...] = ()
    required_components: tuple[str, ...] = ()
    required_content_properties: tuple[tuple[str, str], ...] = ()
    forward_reference_patterns: tuple[str, ...] = ()
    split: str = "train"
    # DSL-pack seams (F1): default None resolves to the pinned OpenUI
    # library schema / prop-order file, byte-identical to the old behavior.
    schema: Mapping[str, Any] | None = None
    prop_order: Mapping[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.max_width < 1:
            raise ValueError("max_depth and max_width must be positive")
        if not self.viewports or not self.render_states or not self.content_classes:
            raise ValueError(
                "viewports, render_states, and content_classes must be non-empty"
            )
        if not set(self.required_components) <= set(
            self.components or component_names()
        ):
            raise ValueError("required_components must be drawn from components")
        configured_components = {
            component for component, _ in self.required_content_properties
        }
        if not configured_components <= set(self.components or component_names()):
            raise ValueError(
                "required_content_properties must be drawn from components"
            )
        unknown_patterns = set(self.forward_reference_patterns) - _FORWARD_REFERENCE_PATTERNS
        if unknown_patterns:
            raise ValueError(
                "unknown forward_reference_patterns: "
                + ", ".join(sorted(unknown_patterns))
            )


@dataclass(frozen=True)
class _PropTarget:
    component: str
    prop: str
    variant: str


@dataclass(frozen=True)
class _Candidate:
    components: tuple[str, ...]
    prop_target: _PropTarget | None
    depth: int
    width: int
    viewport: str
    render_state: str
    content_class: str
    hints: frozenset[CoverageCell]
    forward_reference_pattern: str | None = None

    def key(self) -> tuple[Any, ...]:
        return (
            self.components,
            self.prop_target,
            self.depth,
            self.width,
            self.viewport,
            self.render_state,
            self.content_class,
            self.forward_reference_pattern,
        )


@dataclass(frozen=True)
class GenerationResult:
    programs: tuple[ProgramSpec, ...]
    coverage: dict[str, Any]


def _value_class(schema: Mapping[str, Any], variant: str) -> str:
    if "enum" in schema:
        return f"enum_{variant}"
    kind = schema.get("type")
    if kind == "string":
        return "literal" if variant == "literal" else "placeholder"
    if kind == "boolean":
        return variant
    if kind == "number":
        return variant
    if kind == "array":
        return variant
    if kind == "object":
        return "null"
    if "$ref" in schema:
        return "reference"
    return "deferred"


def _variants(schema: Mapping[str, Any], prop: str) -> tuple[str, ...]:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return ("first", "last") if len(enum) > 1 else ("first",)
    kind = schema.get("type")
    if kind == "string":
        return ("literal",) if prop in _LITERAL_STRING_PROPS else ("placeholder",)
    if kind == "boolean":
        return ("false", "true")
    if kind == "number":
        return ("minimum", "zero", "maximum")
    if kind == "array":
        items = schema.get("items")
        if isinstance(items, Mapping) and items.get("type") == "object":
            return ("empty",)
        return ("empty", "nonempty")
    if kind == "object":
        return ("null",)
    if "$ref" in schema:
        return ("reference",)
    return ()


class _TypedBuilder:
    def __init__(
        self,
        definitions: Mapping[str, Any],
        prop_order: Mapping[str, Sequence[str]],
        target: _PropTarget | None,
        *,
        force_content_components: frozenset[str] = frozenset(),
        force_content_properties: frozenset[tuple[str, str]] = frozenset(),
        reference_array_count: int = 1,
    ) -> None:
        self.definitions = definitions
        self.prop_order = prop_order
        self.target = target
        self.force_content_components = force_content_components
        self.force_content_properties = force_content_properties
        self.reference_array_count = reference_array_count
        self.statements: list[TypedStatement] = []
        self.covered: set[CoverageCell] = {
            CoverageCell("production", "assignment"),
            CoverageCell("production", "component_call"),
            CoverageCell("production", "reference"),
            CoverageCell("production", "list"),
        }
        self._counter = 0
        self._structural_id_counter = 0

    def _binder(self, name: str) -> str:
        self._counter += 1
        stem = _BINDER_RE.sub("", name.lower()) or "node"
        return f"{stem}{self._counter}"

    def _leaf(self, prefix: str) -> Reference:
        binder = self._binder("text")
        self.statements.append(
            TypedStatement(
                binder,
                ComponentCall("TextContent", (f":{prefix}.child",)),
            )
        )
        self.covered.update(
            {
                CoverageCell("component", "TextContent"),
                CoverageCell("prop", "TextContent.text"),
                CoverageCell("prop_value_class", "TextContent.text=placeholder"),
                CoverageCell("production", "string"),
            }
        )
        return Reference(binder)

    def _number(self, schema: Mapping[str, Any], variant: str) -> int | float:
        minimum = schema.get("minimum", -1)
        maximum = schema.get("maximum", 1)
        if variant == "minimum":
            return minimum
        if variant == "maximum":
            return maximum
        return 0

    def _value(
        self,
        prop: str,
        schema: Mapping[str, Any],
        prefix: str,
        variant: str,
        stack: tuple[str, ...],
    ) -> TypedValue:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return str(enum[-1] if variant == "last" else enum[0])
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            for option in any_of:
                if isinstance(option, Mapping) and "$ref" in option:
                    return self._value(prop, option, prefix, variant, stack)
        kind = schema.get("type")
        if kind == "string":
            if prop in _LITERAL_STRING_PROPS:
                value = f"${self._structural_id_counter}"
                self._structural_id_counter += 1
                return value
            return f":{prefix}.{prop}"
        if kind == "boolean":
            return variant == "true"
        if kind == "number":
            return self._number(schema, variant)
        if kind == "object":
            return None
        if kind == "array":
            if variant == "empty":
                return ()
            item = schema.get("items")
            item = item if isinstance(item, Mapping) else {}
            if "$ref" in item:
                name = str(item["$ref"]).split("/")[-1]
                count = (
                    self.reference_array_count if self.force_content_components else 1
                )
                return tuple(
                    self.build(name, f"{prefix}_{index}", stack)
                    for index in range(count)
                )
            if item.get("type") == "array":
                return ((self._leaf(prefix),),)
            if item.get("type") == "object":
                return ()
            if "enum" in item:
                values = item["enum"]
                return (str(values[0]),)
            if item.get("type") == "number":
                return (0, 1)
            if item.get("type") == "string":
                return (f":{prefix}.item",)
            return (self._leaf(prefix),)
        if "$ref" in schema:
            name = str(schema["$ref"]).split("/")[-1]
            return self.build(name, prefix, stack)
        # ponytail: Col.data is untyped in 0.2.x but rejects null; keep the
        # smallest valid array until a contract bump publishes its real type.
        return ()

    def build(self, name: str, prefix: str, stack: tuple[str, ...] = ()) -> Reference:
        if name in stack:
            return self._leaf(prefix)
        definition = self.definitions[name]
        properties = definition.get("properties", {})
        required = set(definition.get("required", ()))
        order = list(self.prop_order.get(name, properties))
        selected = (
            self.target if self.target and self.target.component == name else None
        )
        used = set(required)
        forced_properties = {
            prop
            for component, prop in self.force_content_properties
            if component == name
        }
        if forced_properties:
            used.update(forced_properties)
        elif name in self.force_content_components:
            content_property_added = False
            for prop, schema in properties.items():
                schema = schema if isinstance(schema, Mapping) else {}
                items = schema.get("items")
                if (
                    schema.get("type") == "string"
                    and prop not in _LITERAL_STRING_PROPS
                    and "enum" not in schema
                    and not content_property_added
                ):
                    used.add(prop)
                    content_property_added = True
                elif (
                    schema.get("type") == "array"
                    and isinstance(items, Mapping)
                    and "$ref" in items
                ):
                    used.add(prop)
        if selected is not None:
            used.add(selected.prop)
        positions = [index for index, prop in enumerate(order) if prop in used]
        upto = max(positions) if positions else -1
        args: list[TypedValue] = []
        for index in range(upto + 1):
            prop = order[index]
            if prop not in used:
                args.append(None)
                continue
            schema = properties.get(prop, {})
            schema = schema if isinstance(schema, Mapping) else {}
            variant = (
                selected.variant
                if selected is not None and selected.prop == prop
                else (
                    "placeholder"
                    if (
                        (name, prop) in self.force_content_properties
                        or (
                            name in self.force_content_components
                            and "enum" not in schema
                        )
                        and schema.get("type") == "string"
                        and prop not in _LITERAL_STRING_PROPS
                    )
                    else (
                        "nonempty"
                        if (
                            name in self.force_content_components
                            and schema.get("type") == "array"
                            and isinstance(schema.get("items"), Mapping)
                            and "$ref" in schema["items"]
                        )
                        else (_variants(schema, prop) or ("base",))[0]
                    )
                )
            )
            args.append(self._value(prop, schema, prefix, variant, stack + (name,)))
            value_class = _value_class(schema, variant)
            self.covered.add(CoverageCell("prop", f"{name}.{prop}"))
            self.covered.add(
                CoverageCell("prop_value_class", f"{name}.{prop}={value_class}")
            )
            kind = schema.get("type")
            if kind in {"string", "number", "boolean"}:
                self.covered.add(CoverageCell("production", str(kind)))
            elif kind == "array":
                self.covered.add(CoverageCell("production", "list"))
            elif "$ref" in schema:
                self.covered.add(CoverageCell("production", "reference"))
        binder = self._binder(name)
        self.statements.append(TypedStatement(binder, ComponentCall(name, tuple(args))))
        self.covered.add(CoverageCell("component", name))
        return Reference(binder)


class ProgramGenerator:
    """Seeded adaptive sampler over pairwise and selected three-way targets."""

    def __init__(self, config: GeneratorConfig = GeneratorConfig(), *, seed: int = 0):
        self.config = config
        self.seed = seed
        schema = dict(config.schema) if config.schema is not None else library_schema()
        self.definitions: dict[str, Any] = dict(schema.get("$defs", {}))
        self.prop_order: dict[str, list[str]] = (
            {name: list(order) for name, order in config.prop_order.items()}
            if config.prop_order is not None
            else json.loads(_PROP_ORDER_PATH.read_text(encoding="utf-8"))
        )
        requested = config.components or tuple(component_names())
        unknown = sorted(set(requested) - self.definitions.keys())
        if unknown:
            raise ValueError(f"unknown component: {unknown[0]}")
        self.components = tuple(dict.fromkeys(requested))
        if not self.components:
            raise ValueError("components must be non-empty")
        self.required_components = frozenset(
            (
                *self.config.required_components,
                *(
                    component
                    for component, _ in self.config.required_content_properties
                ),
            )
        )
        if self.config.required_content_properties:
            invalid_content_properties = [
                f"{component}.{prop}"
                for component, prop in self.config.required_content_properties
                if not self._is_content_property(component, prop)
            ]
            if invalid_content_properties:
                raise ValueError(
                    "required_content_properties must name free-form string properties: "
                    + ", ".join(sorted(invalid_content_properties))
                )
        self.triples = config.selected_triples or self._default_triples()
        for triple in self.triples:
            if (
                len(triple) != 3
                or self.config.max_width < 3
                or not set(triple) <= set(self.components)
            ):
                raise ValueError(f"invalid selected triple: {triple}")
        targets, unsupported = self._target_grid()
        self.tracker = CoverageTracker(targets, unsupported)
        self._candidates = self._build_candidates()
        self._used: set[tuple[Any, ...]] = set()
        self._rng = random.Random(seed)

    def _default_triples(self) -> tuple[tuple[str, str, str], ...]:
        if len(self.components) < 3 or self.config.max_width < 3:
            return ()
        return tuple(
            tuple(self.components[index : index + 3])  # type: ignore[misc]
            for index in range(0, len(self.components) - 2, 3)
        )

    def _is_content_property(self, component: str, prop: str) -> bool:
        schema = self.definitions[component].get("properties", {}).get(prop, {})
        return (
            isinstance(schema, Mapping)
            and schema.get("type") == "string"
            and prop not in _LITERAL_STRING_PROPS
            and "enum" not in schema
        )

    def _target_grid(self) -> tuple[frozenset[CoverageCell], frozenset[CoverageCell]]:
        targets: set[CoverageCell] = {
            CoverageCell("production", value)
            for value in (
                "assignment",
                "component_call",
                "reference",
                "list",
                "string",
            )
        }
        targets.update(CoverageCell("component", name) for name in self.components)
        unsupported: set[CoverageCell] = set()
        for name in self.components:
            definition = self.definitions[name]
            for prop, schema in definition.get("properties", {}).items():
                schema = schema if isinstance(schema, Mapping) else {}
                if schema.get("type") in {"number", "boolean"}:
                    targets.add(CoverageCell("production", str(schema["type"])))
                variants = _variants(schema, prop)
                prop_cell = CoverageCell("prop", f"{name}.{prop}")
                if not variants:
                    unsupported.add(prop_cell)
                    continue
                targets.add(prop_cell)
                items = schema.get("items")
                if (
                    schema.get("type") == "array"
                    and isinstance(items, Mapping)
                    and items.get("type") == "object"
                ):
                    cell = CoverageCell("prop_value_class", f"{name}.{prop}=nonempty")
                    targets.add(cell)
                    unsupported.add(cell)
                targets.update(
                    CoverageCell(
                        "prop_value_class",
                        f"{name}.{prop}={_value_class(schema, variant)}",
                    )
                    for variant in variants
                )
        pairs = combinations(self.components, 2) if self.config.max_width >= 2 else ()
        targets.update(CoverageCell("component_pair", "|".join(pair)) for pair in pairs)
        targets.update(
            CoverageCell("component_triple", "|".join(triple))
            for triple in self.triples
        )
        targets.update(
            CoverageCell("depth", str(value))
            for value in range(1, self.config.max_depth + 1)
        )
        targets.update(
            CoverageCell("width", str(value))
            for value in range(1, self.config.max_width + 1)
        )
        targets.update(
            CoverageCell("reference_topology", self._topology(depth))
            for depth in range(1, self.config.max_depth + 1)
        )
        targets.update(
            CoverageCell("length", bucket) for bucket in ("short", "medium", "long")
        )
        targets.update(
            CoverageCell("viewport_state", f"{viewport}|{state}")
            for viewport in self.config.viewports
            for state in self.config.render_states
        )
        targets.update(
            CoverageCell("content_class", value)
            for value in self.config.content_classes
        )
        targets.update(
            CoverageCell("forward_reference_pattern", pattern)
            for pattern in self.config.forward_reference_patterns
        )
        deferred = {CoverageCell("dataflow", pattern) for pattern in _DEFERRED_PATTERNS}
        targets.update(deferred)
        unsupported.update(deferred)
        targets.update(unsupported)
        return frozenset(targets), frozenset(unsupported)

    def _hints(
        self,
        components: tuple[str, ...],
        prop_target: _PropTarget | None,
        depth: int,
        width: int,
        viewport: str,
        state: str,
        content_class: str,
        forward_reference_pattern: str | None = None,
    ) -> frozenset[CoverageCell]:
        cells = {CoverageCell("component", name) for name in components}
        cells.update(
            CoverageCell("component_pair", "|".join(pair))
            for pair in combinations(components, 2)
        )
        if len(components) == 3:
            cells.add(CoverageCell("component_triple", "|".join(components)))
        cells.update(
            {
                CoverageCell("depth", str(depth)),
                CoverageCell("width", str(width)),
                CoverageCell("reference_topology", self._topology(depth)),
                CoverageCell("viewport_state", f"{viewport}|{state}"),
                CoverageCell("content_class", content_class),
            }
        )
        if prop_target is not None:
            schema = self.definitions[prop_target.component]["properties"][
                prop_target.prop
            ]
            cells.add(
                CoverageCell("prop", f"{prop_target.component}.{prop_target.prop}")
            )
            cells.add(
                CoverageCell(
                    "prop_value_class",
                    f"{prop_target.component}.{prop_target.prop}="
                    f"{_value_class(schema, prop_target.variant)}",
                )
            )
        if forward_reference_pattern is not None:
            cells.add(
                CoverageCell("forward_reference_pattern", forward_reference_pattern)
            )
        return frozenset(cells)

    def _candidate(
        self,
        components: Sequence[str],
        index: int,
        *,
        prop_target: _PropTarget | None = None,
        depth: int | None = None,
        width: int | None = None,
        viewport: str | None = None,
        render_state: str | None = None,
        content_class: str | None = None,
        forward_reference_pattern: str | None = None,
    ) -> _Candidate:
        selected = tuple(components[: self.config.max_width])
        depth = depth or 1 + index % self.config.max_depth
        width = width or min(self.config.max_width, max(1, len(selected)))
        viewport = viewport or self.config.viewports[index % len(self.config.viewports)]
        state = (
            render_state
            or self.config.render_states[index % len(self.config.render_states)]
        )
        content_class = (
            content_class
            or self.config.content_classes[index % len(self.config.content_classes)]
        )
        return _Candidate(
            selected,
            prop_target,
            depth,
            width,
            viewport,
            state,
            content_class,
            self._hints(
                selected,
                prop_target,
                depth,
                width,
                viewport,
                state,
                content_class,
                forward_reference_pattern,
            ),
            forward_reference_pattern,
        )

    def _build_candidates(self) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        groups: list[tuple[str, ...]] = [(name,) for name in self.components]
        if self.config.max_width >= 2:
            groups.extend(combinations(self.components, 2))
        groups.extend(self.triples)
        if self.required_components:
            required = set(self.required_components)
            groups.extend(
                group
                for width in range(
                    len(required), min(self.config.max_width, len(self.components)) + 1
                )
                for group in combinations(self.components, width)
            )
            groups = [group for group in groups if required <= set(group)]
            groups.extend(
                order
                for group in tuple(groups)
                if len(group) > 1
                for order in permutations(group)
            )
        for index, group in enumerate(groups):
            candidates.append(self._candidate(group, index))
        if self.required_components:
            for group in groups:
                for component in group:
                    properties = self.definitions[component].get("properties", {})
                    for prop, schema in properties.items():
                        schema = schema if isinstance(schema, Mapping) else {}
                        kind = schema.get("type")
                        items = schema.get("items")
                        if kind == "string" and prop not in _LITERAL_STRING_PROPS:
                            candidates.append(
                                self._candidate(
                                    group,
                                    len(candidates),
                                    prop_target=_PropTarget(
                                        component, prop, "placeholder"
                                    ),
                                )
                            )
                        elif (
                            kind == "array"
                            and isinstance(items, Mapping)
                            and "$ref" in items
                        ):
                            candidates.append(
                                self._candidate(
                                    group,
                                    len(candidates),
                                    prop_target=_PropTarget(
                                        component, prop, "nonempty"
                                    ),
                                )
                            )
            for group, viewport, state, content_class in product(
                groups,
                self.config.viewports,
                self.config.render_states,
                self.config.content_classes,
            ):
                candidates.append(
                    self._candidate(
                        group,
                        len(candidates),
                        viewport=viewport,
                        render_state=state,
                        content_class=content_class,
                    )
                )
        offset = len(candidates)
        for name in self.components:
            properties = self.definitions[name].get("properties", {})
            for prop, schema in properties.items():
                schema = schema if isinstance(schema, Mapping) else {}
                for variant in _variants(schema, prop):
                    target = _PropTarget(name, prop, variant)
                    candidates.append(
                        self._candidate(
                            (name,), offset + len(candidates), prop_target=target
                        )
                    )
        for depth in range(1, self.config.max_depth + 1):
            candidates.append(
                self._candidate((self.components[0],), len(candidates), depth=depth)
            )
        for width in range(1, self.config.max_width + 1):
            selected = tuple(
                self.components[index % len(self.components)] for index in range(width)
            )
            candidates.append(
                self._candidate(selected, len(candidates), depth=1, width=width)
            )
        for viewport in self.config.viewports:
            for state in self.config.render_states:
                base = self._candidate((self.components[0],), len(candidates))
                candidates.append(
                    _Candidate(
                        base.components,
                        None,
                        base.depth,
                        base.width,
                        viewport,
                        state,
                        base.content_class,
                        self._hints(
                            base.components,
                            None,
                            base.depth,
                            base.width,
                            viewport,
                            state,
                            base.content_class,
                        ),
                    )
                )
        for content_class in self.config.content_classes:
            base = self._candidate((self.components[0],), len(candidates))
            candidates.append(
                _Candidate(
                    base.components,
                    None,
                    base.depth,
                    base.width,
                    base.viewport,
                    base.render_state,
                    content_class,
                    self._hints(
                        base.components,
                        None,
                        base.depth,
                        base.width,
                        base.viewport,
                        base.render_state,
                        content_class,
                    ),
                )
            )
        for pattern in self.config.forward_reference_patterns:
            components = (
                ("Stack", "Input", "TextContent")
                if pattern == "root_shared_slot_free_input"
                else ("Stack", "Input")
            )
            required = set(components)
            if not required <= set(self.components):
                raise ValueError(
                    f"{pattern} requires components: "
                    + ", ".join(sorted(required))
                )
            for viewport, state, content_class in product(
                self.config.viewports,
                self.config.render_states,
                self.config.content_classes,
            ):
                candidates.append(
                    self._candidate(
                        components,
                        len(candidates),
                        depth=1,
                        width=2,
                        viewport=viewport,
                        render_state=state,
                        content_class=content_class,
                        forward_reference_pattern=pattern,
                    )
                )
        if self.required_components:
            required = set(self.required_components)
            candidates = [
                candidate
                for candidate in candidates
                if required <= set(candidate.components)
            ]
        unique = {candidate.key(): candidate for candidate in candidates}
        return tuple(unique.values())

    def _choose(self) -> _Candidate:
        available = [
            candidate
            for candidate in self._candidates
            if candidate.key() not in self._used
        ]
        if not available:
            raise ValueError("candidate grid exhausted")
        jitter = {candidate.key(): self._rng.random() for candidate in available}
        chosen = max(
            available,
            key=lambda candidate: (
                self.tracker.score(candidate.hints),
                jitter[candidate.key()],
            ),
        )
        self._used.add(chosen.key())
        return chosen

    @staticmethod
    def _length_cell(source: str) -> CoverageCell:
        # ponytail: character buckets are deterministic and dependency-free;
        # replace with tokenizer quantiles when the target length grid is frozen.
        length = len(source)
        bucket = "short" if length < 100 else "medium" if length < 160 else "long"
        return CoverageCell("length", bucket)

    @staticmethod
    def _topology(depth: int) -> str:
        return "star" if depth == 1 else "nested" if depth == 2 else "chain"

    def _build_program(self, candidate: _Candidate) -> tuple[str, set[CoverageCell]]:
        if candidate.forward_reference_pattern is not None:
            return self._build_forward_reference_program(candidate)
        forced_components = set(candidate.components)
        if "Form" in forced_components and "Button" in forced_components:
            forced_components.add("Buttons")
        builder = _TypedBuilder(
            self.definitions,
            self.prop_order,
            candidate.prop_target,
            force_content_components=(
                frozenset(forced_components)
                if self.required_components
                else frozenset()
            ),
            force_content_properties=frozenset(self.config.required_content_properties),
            reference_array_count=1 + (candidate.depth - 1) % 3,
        )
        selected = list(candidate.components)
        while len(selected) < candidate.width:
            selected.append(self.components[len(selected) % len(self.components)])
        selected = selected[: candidate.width]
        card_selected = "Card" in selected
        form_selected = "Form" in selected
        leaves = [
            name
            for name in selected
            if name != "Card"
            and not (name == "SwitchItem" and "SwitchGroup" in selected)
            and not (
                form_selected and name in {"FormControl", "Input", "Button", "Buttons"}
            )
        ]
        built = [
            (
                name,
                builder.build(
                    name,
                    f"gen{index}_{name.lower()}_{candidate.content_class}",
                ),
            )
            for index, name in enumerate(leaves)
        ]
        refs = [ref for _, ref in built]
        if card_selected:
            input_refs = [ref for name, ref in built if name == "Input"]
            direct_refs = [ref for name, ref in built if name != "Input"]
            if candidate.depth % 2:
                content_binder = builder._binder("cardcontent")
                builder.statements.append(
                    TypedStatement(
                        content_binder,
                        ComponentCall("Stack", (tuple(refs), "column")),
                    )
                )
                card_children = (Reference(content_binder),)
            else:
                input_binder = builder._binder("cardinput")
                builder.statements.append(
                    TypedStatement(
                        input_binder,
                        ComponentCall("Stack", (tuple(input_refs), "column")),
                    )
                )
                card_children = (*direct_refs, Reference(input_binder))
            card_binder = builder._binder("card")
            builder.statements.append(
                TypedStatement(card_binder, ComponentCall("Card", (card_children,)))
            )
            builder.covered.update(
                {
                    CoverageCell("component", "Card"),
                    CoverageCell("component", "Stack"),
                    CoverageCell("prop", "Card.children"),
                    CoverageCell("prop_value_class", "Card.children=nonempty"),
                    CoverageCell("production", "list"),
                }
            )
            refs = [Reference(card_binder)]
        for layer in range(1, candidate.depth):
            binder = builder._binder("layer")
            builder.statements.append(
                TypedStatement(binder, ComponentCall("Stack", (tuple(refs), "column")))
            )
            builder.covered.add(CoverageCell("component", "Stack"))
            refs = [Reference(binder)]
        direction = "column" if candidate.viewport == "mobile" else "row"
        program = TypedProgram(
            ComponentCall("Stack", (tuple(refs), direction)),
            tuple(builder.statements),
        )
        source = program.serialize()
        cells = set(builder.covered) | set(candidate.hints)
        cells.add(self._length_cell(source))
        return source, cells

    def _build_forward_reference_program(
        self, candidate: _Candidate
    ) -> tuple[str, set[CoverageCell]]:
        """Build paired root-level forward-reference supervision for E1421."""
        suffix = f"{candidate.render_state}_{candidate.content_class}"
        direction = "column" if candidate.viewport == "mobile" else "row"
        gap = {"empty": "none", "loading": "xs", "success": "m", "error": "l"}.get(
            candidate.render_state, "m"
        )
        input_type = {"plain": "text", "escaped": "email", "dsl_like": "password"}.get(
            candidate.content_class, "text"
        )
        if candidate.forward_reference_pattern == "root_distinct_slot_inputs":
            statements = (
                TypedStatement(
                    "input1",
                    ComponentCall(
                        "Input",
                        ("$0", f":forward_{suffix}_input_0", input_type),
                    ),
                ),
                TypedStatement(
                    "input2",
                    ComponentCall(
                        "Input",
                        ("$1", f":forward_{suffix}_input_1", input_type),
                    ),
                ),
            )
            root_refs = (Reference("input1"), Reference("input2"))
        else:
            statements = (
                TypedStatement(
                    "input1", ComponentCall("Input", ("$0", None, input_type))
                ),
                TypedStatement(
                    "textcontent2",
                    ComponentCall("TextContent", (f":forward_{suffix}_share_0",)),
                ),
                TypedStatement(
                    "textcontent3",
                    ComponentCall("TextContent", (f":forward_{suffix}_share_1",)),
                ),
            )
            root_refs = (
                Reference("input1"),
                Reference("input1"),
                Reference("textcontent2"),
                Reference("textcontent3"),
            )
        source = TypedProgram(
            ComponentCall("Stack", (root_refs, direction, gap)), statements
        ).serialize()
        cells = set(candidate.hints)
        cells.update(
            {
                CoverageCell("component", "Stack"),
                CoverageCell("component", "Input"),
                CoverageCell("production", "assignment"),
                CoverageCell("production", "component_call"),
                CoverageCell("production", "reference"),
                CoverageCell("production", "list"),
                CoverageCell("production", "string"),
                self._length_cell(source),
            }
        )
        if candidate.forward_reference_pattern == "root_shared_slot_free_input":
            cells.add(CoverageCell("component", "TextContent"))
        return source, cells

    def generate_one(self) -> ProgramSpec:
        candidate = self._choose()
        openui, cells = self._build_program(candidate)
        identity = json.dumps(
            [
                openui,
                candidate.viewport,
                candidate.render_state,
                candidate.depth,
                candidate.width,
                candidate.prop_target,
                candidate.content_class,
                candidate.forward_reference_pattern,
            ],
            default=str,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        family_identity = json.dumps(
            [candidate.components, candidate.depth, candidate.width],
            separators=(",", ":"),
        )
        family_digest = hashlib.sha256(family_identity.encode("utf-8")).hexdigest()[:12]
        components = sorted(cell.key for cell in cells if cell.axis == "component")
        facts = {
            "components": components,
            "coverage_cells": [cell.label() for cell in sorted(cells)],
            "depth": candidate.depth,
            "width": candidate.width,
            "reference_topology": self._topology(candidate.depth),
            "viewport": candidate.viewport,
            "render_state": candidate.render_state,
            "content_class": candidate.content_class,
        }
        if candidate.forward_reference_pattern is not None:
            facts["forward_reference_pattern"] = candidate.forward_reference_pattern
        spec = ProgramSpec.from_openui(
            id=f"program_{digest}",
            openui=openui,
            facts=facts,
            program_family_id=f"generated_{family_digest}",
            lineage_id=f"lineage_{digest}",
            split_group_id=f"group_{family_digest}",
            split=self.config.split,
            provenance={
                "generator": "typed_ast",
                "generator_version": GENERATOR_VERSION,
                "seed": self.seed,
            },
        )
        record = emit_record(
            spec,
            prompt="Generate this typed OpenUI program.",
            task="generation",
            source=PROGRAM_FAMILY,
        )
        report = verify_record(record, VerificationContext(source_kind="program"))
        passed = report.ok and report.tier is Tier.SILVER
        self.tracker.record(cells, verifier_passed=passed)
        if not passed:
            gate = report.failing_gate.value if report.failing_gate else "tier"
            raise ValueError(f"generated ProgramSpec failed F2 at {gate}")
        return spec

    def generate(self, count: int) -> GenerationResult:
        if count < 0:
            raise ValueError("count must be non-negative")
        programs = tuple(self.generate_one() for _ in range(count))
        return GenerationResult(programs, self.tracker.report())

    def generate_until_covered(
        self, *, max_programs: int | None = None
    ) -> GenerationResult:
        limit = max_programs if max_programs is not None else len(self._candidates)
        programs: list[ProgramSpec] = []
        while not self.tracker.complete and len(programs) < limit:
            try:
                programs.append(self.generate_one())
            except ValueError as exc:
                if str(exc) == "candidate grid exhausted":
                    break
                raise
        return GenerationResult(tuple(programs), self.tracker.report())


def generate_program_specs(
    count: int,
    *,
    config: GeneratorConfig = GeneratorConfig(),
    seed: int = 0,
) -> GenerationResult:
    return ProgramGenerator(config, seed=seed).generate(count)


__all__ = [
    "GENERATOR_VERSION",
    "PROGRAM_FAMILY",
    "ComponentCall",
    "CoverageCell",
    "CoverageTracker",
    "GenerationResult",
    "GeneratorConfig",
    "ProgramGenerator",
    "Reference",
    "TypedProgram",
    "TypedStatement",
    "generate_program_specs",
]
