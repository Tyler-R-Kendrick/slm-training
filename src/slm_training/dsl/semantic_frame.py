"""SemanticFrameV1: deterministic semantic facts from canonical AST + schema.

DSH2-02 (SLM-363): a natural-language prompt for a DSL target states some
facts, may optionally state others, must never invent facts the target
contradicts, and may leave some facts unspecified. This module derives that
partition deterministically from the canonical OpenUI AST plus a small,
explicit structured schema, so an LLM in the training loop is a paraphraser
of already-fixed facts rather than a semantic oracle guessing them.

**SLM-343 interpretation.** SLM-343 ("CAP1-GEN-01: define the generic
NL<->DSL grounding contract and capability certificate") has not merged any
code — it is a design-doc-only backlog issue. This module does not import
from it. It instead derives its own minimal, explicit
:class:`CAP1SchemaV1` from artifacts that *are* merged: the same
:class:`~slm_training.dsl.pack.DslPack` prop-order/backend slots the DSH1-01
:class:`~slm_training.dsl.grammar_capabilities.GrammarCapabilityAdapterV1`
fingerprints. The interpretation adopted here of SLM-343's invariant
("prompt/context may contain natural language; the DSL target contains only
symbolic structure") is: every prop whose value is free natural-language text
is declared a *content prop*; a placeholder token there is a required,
exactly-provenanced fact, while the natural-language text the placeholder
will eventually be filled with is explicitly ``unspecified`` — this module
never guesses it. Everything else in the AST (component types, structural
prop values, order, cardinality, closed-value literals, state/query/mutation
relations and effects) is symbolic and must be derivable exactly.

**Stop rule.** Any AST shape, component, or prop this schema does not
declare raises :class:`UnsupportedSemanticFactError` instead of being
guessed. There is no free-form/LLM-guessed fallback anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.lang_core import Program
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.dsl.parser import validate as validate_source
from slm_training.dsl.placeholders import is_placeholder

AstPath = tuple[str | int, ...]
FactCategory = Literal["required", "optional", "forbidden", "unspecified"]

# Grammar-intrinsic closed domains (openui.lark): the exact set of literal
# operator tokens the grammar can ever produce, not a schema-configured guess.
_BINOP_DOMAIN: tuple[str, ...] = (
    "||",
    "&&",
    "==",
    "!=",
    ">=",
    "<=",
    ">",
    "<",
    "+",
    "-",
    "*",
    "/",
    "%",
)
_UNARYOP_DOMAIN: tuple[str, ...] = ("!", "-")

_SCALAR = (str, int, float)
_MISSING = object()


class UnsupportedSemanticFactError(ValueError):
    """A fact could not be derived from AST + schema alone.

    Fail-closed per the DSH2-02 stop rule: raised instead of guessing a role,
    value, or provenance for a construct the declared schema does not cover.
    """


class CounterfactualError(ValueError):
    """A one-fact counterfactual could not be produced or proven."""


# --------------------------------------------------------------------------
# Structured schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CAP1SchemaV1:
    """Minimal, explicit CAP1 structured schema (see module docstring).

    ``component_names`` / ``component_prop_order`` are the closed component
    vocabulary and each component's declared positional-prop order, read from
    the pack's own ``prop_order`` slot — no parallel schema is invented.
    ``content_props`` names free-natural-language props. ``closed_value_domains``
    is an *optional* caller-declared table of additional closed domains beyond
    the ones this module always treats as closed by grammar construction
    (``bool`` literals, binary/unary operator tokens); declaring one here is a
    schema-level policy choice, not a re-derivation from the grammar, and an
    observed value outside a declared domain fails closed rather than being
    accepted. ``default_values`` optionally marks a ``(component, prop)``
    value as the schema default, downgrading a matching fact from
    ``required`` to ``optional``; empty by default because SLM-343 defines no
    default table yet and guessing one would violate the stop rule.
    """

    dsl: str
    component_names: frozenset[str]
    component_prop_order: Mapping[str, tuple[str, ...]]
    content_props: frozenset[str]
    closed_value_domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    default_values: Mapping[tuple[str, str], Any] = field(default_factory=dict)

    @classmethod
    def from_pack(
        cls, dsl: str | None = None, *, pack: DslPack | None = None
    ) -> "CAP1SchemaV1":
        pack = pack or get_pack(dsl)
        prop_order = {
            str(name): tuple(str(p) for p in props)
            for name, props in dict(pack.require("prop_order")()).items()
        }
        return cls(
            dsl=pack.pack_id,
            component_names=frozenset(pack.backend.component_names())
            | frozenset(prop_order),
            component_prop_order=prop_order,
            content_props=frozenset(pack.backend.content_props()),
        )

    def fingerprint(self) -> str:
        payload = {
            "dsl": self.dsl,
            "component_names": sorted(self.component_names),
            "component_prop_order": {
                name: list(props)
                for name, props in sorted(self.component_prop_order.items())
            },
            "content_props": sorted(self.content_props),
            "closed_value_domains": {
                name: list(domain)
                for name, domain in sorted(self.closed_value_domains.items())
            },
            "default_values": sorted(
                f"{comp}.{prop}={value!r}"
                for (comp, prop), value in self.default_values.items()
            ),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------
# Frame entities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticFrameNodeV1:
    """One AST node (element, ref, or expression combinator)."""

    node_id: str
    ast_path: AstPath
    kind: str
    type_name: str


@dataclass(frozen=True)
class SemanticRoleV1:
    """A parent -> child edge: which prop/slot of the parent the child fills."""

    parent_id: str
    child_id: str
    role: str
    order: int | None = None


@dataclass(frozen=True)
class SemanticRelationV1:
    """A non-tree semantic binding: state reads, query reads, mutation calls."""

    kind: str
    node_id: str
    target: str
    ast_path: AstPath
    schema_field: str


@dataclass(frozen=True)
class SemanticEffectV1:
    """A side-effecting DSL construct (action/mutation invocation)."""

    node_id: str
    ast_path: AstPath
    effect_kind: str
    schema_field: str


@dataclass(frozen=True)
class SemanticFactV1:
    """One atomic fact, always carrying exact AST-path + schema-field provenance."""

    category: FactCategory
    predicate: str
    ast_path: AstPath
    schema_field: str
    value: Any = None


@dataclass(frozen=True)
class SemanticFrameV1:
    """The full deterministic semantic frame for one canonical AST."""

    dsl: str
    schema_fingerprint: str
    canonical_source: str
    nodes: tuple[SemanticFrameNodeV1, ...]
    roles: tuple[SemanticRoleV1, ...]
    relations: tuple[SemanticRelationV1, ...]
    effects: tuple[SemanticEffectV1, ...]
    facts: tuple[SemanticFactV1, ...]

    def facts_by_category(self, category: FactCategory) -> tuple[SemanticFactV1, ...]:
        return tuple(f for f in self.facts if f.category == category)

    def equivalent_to(self, other: "SemanticFrameV1") -> bool:
        """Semantic equivalence: same schema, same fact/role/relation/effect content.

        Deliberately ignores ``canonical_source`` text identity — two frames
        built under the same schema with identical facts are equivalent
        regardless of incidental surface-string bookkeeping.
        """
        return (
            self.dsl == other.dsl
            and self.schema_fingerprint == other.schema_fingerprint
            and frozenset(self.nodes) == frozenset(other.nodes)
            and frozenset(self.roles) == frozenset(other.roles)
            and frozenset(self.relations) == frozenset(other.relations)
            and frozenset(self.effects) == frozenset(other.effects)
            and frozenset(self.facts) == frozenset(other.facts)
        )


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------

# node["k"] -> ordered field specs. Each spec is (field_name, field_kind, domain).
# field_kind in {"scalar_enum", "scalar", "node_or_scalar", "list"}.
_EXPR_SHAPES: dict[str, tuple[tuple[str, str, tuple[Any, ...] | None], ...]] = {
    "BinOp": (
        ("op", "scalar_enum", _BINOP_DOMAIN),
        ("left", "node_or_scalar", None),
        ("right", "node_or_scalar", None),
    ),
    "UnaryOp": (
        ("op", "scalar_enum", _UNARYOP_DOMAIN),
        ("operand", "node_or_scalar", None),
    ),
    "Ternary": (
        ("cond", "node_or_scalar", None),
        ("then", "node_or_scalar", None),
        ("else", "node_or_scalar", None),
    ),
    "Member": (
        ("obj", "node_or_scalar", None),
        ("field", "scalar", None),
    ),
    "Index": (
        ("obj", "node_or_scalar", None),
        ("index", "node_or_scalar", None),
    ),
    "Arr": (("els", "list", None),),
}


class _DeriveContext:
    def __init__(self, schema: CAP1SchemaV1) -> None:
        self.schema = schema
        self.nodes: list[SemanticFrameNodeV1] = []
        self.roles: list[SemanticRoleV1] = []
        self.relations: list[SemanticRelationV1] = []
        self.effects: list[SemanticEffectV1] = []
        self.facts: list[SemanticFactV1] = []
        self._next_id = 0

    def _new_node(self, path: AstPath, kind: str, type_name: str) -> str:
        node_id = f"n{self._next_id}"
        self._next_id += 1
        self.nodes.append(
            SemanticFrameNodeV1(
                node_id=node_id, ast_path=path, kind=kind, type_name=str(type_name)
            )
        )
        return node_id

    def _fact(
        self,
        category: FactCategory,
        predicate: str,
        path: AstPath,
        schema_field: str,
        value: Any = None,
    ) -> None:
        self.facts.append(
            SemanticFactV1(
                category=category,
                predicate=predicate,
                ast_path=path,
                schema_field=schema_field,
                value=value,
            )
        )

    def _prop_category(
        self, type_name: str, prop_name: str, value: Any
    ) -> FactCategory:
        key = (type_name, prop_name)
        if (
            key in self.schema.default_values
            and self.schema.default_values[key] == value
        ):
            return "optional"
        return "required"

    def walk(self, value: Any, path: AstPath) -> str:
        """Walk one AST node position, returning its assigned node id.

        Only ever called for dict-shaped (node-like) values; scalar leaves are
        handled directly by callers because the caller alone knows the
        enclosing role/schema-field needed to classify the leaf's fact.
        """
        if isinstance(value, dict):
            return self._walk_dict(value, path)
        raise UnsupportedSemanticFactError(
            f"unsupported AST value at {path!r}: {type(value).__name__}"
        )

    def _walk_dict(self, node: dict[str, Any], path: AstPath) -> str:
        if node.get("type") == "element":
            return self._walk_element(node, path)
        if node.get("type") == "ref":
            return self._walk_unresolved_ref(node, path)
        kind = node.get("k")
        if kind == "StateRef":
            return self._walk_state_ref(node, path)
        if kind == "RuntimeRef":
            return self._walk_runtime_ref(node, path)
        if kind == "Comp":
            return self._walk_comp(node, path)
        if kind == "Obj":
            return self._walk_obj(node, path)
        if isinstance(kind, str) and kind in _EXPR_SHAPES:
            return self._walk_expr(kind, node, path)
        raise UnsupportedSemanticFactError(
            f"unrecognized AST node shape at {path!r}: keys={sorted(node)!r}"
        )

    # -- elements -----------------------------------------------------

    def _walk_element(self, node: dict[str, Any], path: AstPath) -> str:
        type_name = node.get("typeName")
        if (
            not isinstance(type_name, str)
            or type_name not in self.schema.component_names
        ):
            raise UnsupportedSemanticFactError(
                f"component {type_name!r} at {path!r} is not in the declared "
                "CAP1 schema component vocabulary"
            )
        node_id = self._new_node(path, "element", type_name)
        self._fact(
            "required",
            "component_type",
            path + ("typeName",),
            "component_registry",
            value=type_name,
        )
        props = node.get("props")
        props = props if isinstance(props, dict) else {}
        declared = self.schema.component_prop_order.get(type_name, ())
        for prop_name, prop_value in props.items():
            prop_path = path + ("props", prop_name)
            if prop_name not in declared:
                # Covers both unknown names and the Lark backend's "_args"
                # catch-all for positional args beyond the declared order —
                # neither has a schema-declared role to attach a fact to.
                raise UnsupportedSemanticFactError(
                    f"prop {prop_name!r} on {type_name!r} at {prop_path!r} is not "
                    "declared in the CAP1 schema's component prop order"
                )
            self._walk_prop(node_id, type_name, prop_name, prop_value, prop_path)
        return node_id

    def _walk_prop(
        self,
        parent_id: str,
        type_name: str,
        prop_name: str,
        value: Any,
        path: AstPath,
    ) -> None:
        if isinstance(value, list):
            self._walk_list(
                parent_id, value, path, role=prop_name, schema_field="cardinality"
            )
            return
        if prop_name in self.schema.content_props:
            self._walk_content_prop(value, path)
            return
        if isinstance(value, bool):
            self._walk_closed_value(
                type_name,
                prop_name,
                value,
                path,
                domain=(True, False),
                schema_field="grammar.bool_literal",
            )
            return
        if isinstance(value, _SCALAR) and prop_name in self.schema.closed_value_domains:
            domain = self.schema.closed_value_domains[prop_name]
            if value not in domain:
                raise UnsupportedSemanticFactError(
                    f"{type_name}.{prop_name} value {value!r} at {path!r} is outside "
                    f"the declared closed domain {domain!r}"
                )
            self._walk_closed_value(
                type_name,
                prop_name,
                value,
                path,
                domain=domain,
                schema_field=f"closed_value_domains.{prop_name}",
            )
            return
        if isinstance(value, _SCALAR) or value is None:
            category = self._prop_category(type_name, prop_name, value)
            self._fact(
                category, "prop_value", path, "component_prop_order", value=value
            )
            return
        child_id = self.walk(value, path)
        self.roles.append(SemanticRoleV1(parent_id, child_id, prop_name))

    def _walk_closed_value(
        self,
        type_name: str,
        prop_name: str,
        value: Any,
        path: AstPath,
        *,
        domain: tuple[Any, ...],
        schema_field: str,
    ) -> None:
        category = self._prop_category(type_name, prop_name, value)
        self._fact(category, "closed_value", path, schema_field, value=value)
        for alt in domain:
            if alt != value:
                self._fact("forbidden", "excluded_value", path, schema_field, value=alt)

    def _walk_content_prop(self, value: Any, path: AstPath) -> None:
        if isinstance(value, str) and is_placeholder(value):
            self._fact(
                "required", "content_placeholder_id", path, "content_props", value=value
            )
            self._fact("unspecified", "content_text", path, "content_props", value=None)
            return
        if isinstance(value, _SCALAR) or value is None:
            self._fact(
                "required", "content_literal_value", path, "content_props", value=value
            )
            return
        raise UnsupportedSemanticFactError(
            f"unsupported content-prop value at {path!r}: {value!r}"
        )

    def _walk_list(
        self,
        parent_id: str,
        items: list[Any],
        path: AstPath,
        *,
        role: str,
        schema_field: str,
    ) -> None:
        self._fact("required", "cardinality", path, schema_field, value=len(items))
        for index, item in enumerate(items):
            item_path = path + (index,)
            if isinstance(item, dict):
                child_id = self.walk(item, item_path)
                self.roles.append(
                    SemanticRoleV1(parent_id, child_id, role, order=index)
                )
            elif isinstance(item, _SCALAR) or item is None:
                self._fact(
                    "required", "list_item_value", item_path, schema_field, value=item
                )
            else:
                raise UnsupportedSemanticFactError(
                    f"unsupported list item at {item_path!r}: {item!r}"
                )

    # -- refs / relations / effects -----------------------------------

    def _walk_unresolved_ref(self, node: dict[str, Any], path: AstPath) -> str:
        name = node.get("name")
        node_id = self._new_node(path, "ref", str(name))
        self._fact(
            "required",
            "unresolved_reference",
            path + ("name",),
            "ast_shape",
            value=str(name),
        )
        return node_id

    def _walk_state_ref(self, node: dict[str, Any], path: AstPath) -> str:
        name = str(node.get("n"))
        node_id = self._new_node(path, "state_ref", name)
        self._fact(
            "required",
            "state_ref_name",
            path + ("n",),
            "state_declarations",
            value=name,
        )
        self.relations.append(
            SemanticRelationV1(
                kind="reads_state",
                node_id=node_id,
                target=name,
                ast_path=path,
                schema_field="state_declarations",
            )
        )
        return node_id

    def _walk_runtime_ref(self, node: dict[str, Any], path: AstPath) -> str:
        name = str(node.get("n"))
        ref_type = node.get("refType")
        if ref_type not in {"query", "mutation"}:
            raise UnsupportedSemanticFactError(
                f"unrecognized runtime ref type {ref_type!r} at {path!r}"
            )
        node_id = self._new_node(path, "runtime_ref", name)
        schema_field = f"{ref_type}_statements"
        self._fact(
            "required", "runtime_ref_name", path + ("n",), schema_field, value=name
        )
        relation_kind = "reads_query" if ref_type == "query" else "invokes_mutation"
        self.relations.append(
            SemanticRelationV1(
                kind=relation_kind,
                node_id=node_id,
                target=name,
                ast_path=path,
                schema_field=schema_field,
            )
        )
        if ref_type == "mutation":
            self.effects.append(
                SemanticEffectV1(
                    node_id=node_id,
                    ast_path=path,
                    effect_kind="mutation",
                    schema_field=schema_field,
                )
            )
        return node_id

    def _walk_comp(self, node: dict[str, Any], path: AstPath) -> str:
        name = node.get("name")
        if name not in {"Action", "Query", "Mutation"}:
            raise UnsupportedSemanticFactError(
                f"unrecognized inline effect call {name!r} at {path!r}"
            )
        node_id = self._new_node(path, "comp", str(name))
        self._fact(
            "required",
            "inline_effect_kind",
            path + ("name",),
            "ast_shape",
            value=str(name),
        )
        if name in {"Action", "Mutation"}:
            self.effects.append(
                SemanticEffectV1(
                    node_id=node_id,
                    ast_path=path,
                    effect_kind=str(name).lower(),
                    schema_field="ast_shape",
                )
            )
        args = node.get("args")
        args = args if isinstance(args, list) else []
        self._walk_list(
            node_id, args, path + ("args",), role="args", schema_field="cardinality"
        )
        return node_id

    # -- expression combinators ----------------------------------------

    def _walk_obj(self, node: dict[str, Any], path: AstPath) -> str:
        node_id = self._new_node(path, "obj", "Obj")
        entries = node.get("entries")
        entries = entries if isinstance(entries, list) else []
        entries_path = path + ("entries",)
        self._fact(
            "required", "cardinality", entries_path, "ast_shape", value=len(entries)
        )
        for index, entry in enumerate(entries):
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                raise UnsupportedSemanticFactError(
                    f"unsupported object entry at {entries_path + (index,)!r}: {entry!r}"
                )
            key, value = entry
            entry_path = entries_path + (index,)
            self._fact(
                "required",
                "object_key",
                entry_path + ("key",),
                "ast_shape",
                value=str(key),
            )
            if isinstance(value, _SCALAR) or value is None:
                self._fact(
                    "required",
                    "object_value",
                    entry_path + ("value",),
                    "ast_shape",
                    value=value,
                )
            else:
                child_id = self.walk(value, entry_path + ("value",))
                self.roles.append(
                    SemanticRoleV1(node_id, child_id, "entries", order=index)
                )
        return node_id

    def _walk_expr(self, kind: str, node: dict[str, Any], path: AstPath) -> str:
        node_id = self._new_node(path, kind.lower(), kind)
        for field_name, field_kind, domain in _EXPR_SHAPES[kind]:
            fpath = path + (field_name,)
            value = node.get(field_name)
            if field_kind == "scalar_enum":
                assert domain is not None
                if value not in domain:
                    raise UnsupportedSemanticFactError(
                        f"{kind}.{field_name} value {value!r} at {fpath!r} is outside "
                        f"the grammar-declared domain {domain!r}"
                    )
                schema_field = f"grammar.{kind.lower()}_operators"
                self._fact(
                    "required",
                    f"{kind.lower()}_{field_name}",
                    fpath,
                    schema_field,
                    value=value,
                )
                for alt in domain:
                    if alt != value:
                        self._fact(
                            "forbidden",
                            "excluded_value",
                            fpath,
                            schema_field,
                            value=alt,
                        )
            elif field_kind == "scalar":
                if not (isinstance(value, _SCALAR) or value is None):
                    raise UnsupportedSemanticFactError(
                        f"{kind}.{field_name} at {fpath!r} expected a scalar, got {value!r}"
                    )
                self._fact(
                    "required",
                    f"{kind.lower()}_{field_name}",
                    fpath,
                    "expr_shape",
                    value=value,
                )
            elif field_kind == "node_or_scalar":
                if isinstance(value, _SCALAR) or value is None:
                    self._fact(
                        "required",
                        f"{kind.lower()}_{field_name}",
                        fpath,
                        "expr_shape",
                        value=value,
                    )
                else:
                    child_id = self.walk(value, fpath)
                    self.roles.append(SemanticRoleV1(node_id, child_id, field_name))
            elif field_kind == "list":
                items = value if isinstance(value, list) else []
                self._walk_list(
                    node_id, items, fpath, role=field_name, schema_field="expr_shape"
                )
            else:  # pragma: no cover - defensive; every declared kind is handled above
                raise UnsupportedSemanticFactError(
                    f"unhandled field kind {field_kind!r}"
                )
        return node_id


def derive_semantic_frame(
    source: str | Program, schema: CAP1SchemaV1
) -> SemanticFrameV1:
    """Deterministically derive a :class:`SemanticFrameV1` from AST + schema.

    ``source`` may be raw or already-canonical DSL text (canonicalized here
    via :func:`slm_training.dsl.canonicalize.canonicalize`, so two sources
    denoting the same layout always yield byte-identical canonical ASTs and
    therefore equivalent frames), or an already-validated
    :class:`~slm_training.dsl.lang_core.Program` — in which case the caller
    is responsible for having already canonicalized it.

    Fails closed with :class:`UnsupportedSemanticFactError` rather than
    guessing whenever the AST contains a component, prop, or node shape the
    schema does not declare.
    """
    if isinstance(source, Program):
        program = source
        canonical = program.serialized or program.source
    else:
        canonical = canonicalize(source, dsl=schema.dsl)
        program = validate_source(canonical, dsl=schema.dsl)
    root = program.root
    if root is None:
        raise UnsupportedSemanticFactError(
            "program has no root AST to derive a frame from"
        )

    ctx = _DeriveContext(schema)
    ctx.walk(root, ())
    return SemanticFrameV1(
        dsl=schema.dsl,
        schema_fingerprint=schema.fingerprint(),
        canonical_source=canonical,
        nodes=tuple(ctx.nodes),
        roles=tuple(ctx.roles),
        relations=tuple(ctx.relations),
        effects=tuple(ctx.effects),
        facts=tuple(ctx.facts),
    )


# --------------------------------------------------------------------------
# Conflict checks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticFactConflictV1:
    ast_path: AstPath
    schema_field: str
    reason: str
    facts: tuple[SemanticFactV1, ...]


def frame_conflicts(frame: SemanticFrameV1) -> tuple[SemanticFactConflictV1, ...]:
    """Detect internally-contradictory facts (same slot, incompatible categories).

    Empty for any frame produced by :func:`derive_semantic_frame`; exists so
    a mutated or hand-assembled frame can be checked for self-consistency.
    """
    groups: dict[tuple[AstPath, str], list[SemanticFactV1]] = {}
    for fact in frame.facts:
        groups.setdefault((fact.ast_path, fact.schema_field), []).append(fact)

    conflicts: list[SemanticFactConflictV1] = []
    for (ast_path, schema_field), facts in sorted(
        groups.items(), key=lambda kv: repr(kv[0])
    ):
        required_values = {f.value for f in facts if f.category == "required"}
        forbidden_values = {f.value for f in facts if f.category == "forbidden"}
        overlap = required_values & forbidden_values
        if overlap:
            conflicts.append(
                SemanticFactConflictV1(
                    ast_path,
                    schema_field,
                    f"value(s) {sorted(overlap, key=repr)!r} are both required and forbidden",
                    tuple(facts),
                )
            )
        if len(required_values) > 1:
            conflicts.append(
                SemanticFactConflictV1(
                    ast_path,
                    schema_field,
                    f"multiple distinct required values {sorted(required_values, key=repr)!r}",
                    tuple(facts),
                )
            )
    return tuple(conflicts)


# --------------------------------------------------------------------------
# Inverse frame-to-constraint
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameConstraintClauseV1:
    ast_path: AstPath
    schema_field: str
    predicate: str
    polarity: Literal["requires", "forbids"]
    value: Any


@dataclass(frozen=True)
class FrameConstraintV1:
    dsl: str
    schema_fingerprint: str
    clauses: tuple[FrameConstraintClauseV1, ...]


def frame_to_constraint(frame: SemanticFrameV1) -> FrameConstraintV1:
    """Turn a frame's required/forbidden facts into a checkable constraint.

    Optional and unspecified facts impose no constraint: they may or may not
    hold in a satisfying candidate.
    """
    clauses = tuple(
        FrameConstraintClauseV1(
            f.ast_path, f.schema_field, f.predicate, "requires", f.value
        )
        for f in frame.facts
        if f.category == "required"
    ) + tuple(
        FrameConstraintClauseV1(
            f.ast_path, f.schema_field, f.predicate, "forbids", f.value
        )
        for f in frame.facts
        if f.category == "forbidden"
    )
    return FrameConstraintV1(
        dsl=frame.dsl, schema_fingerprint=frame.schema_fingerprint, clauses=clauses
    )


@dataclass(frozen=True)
class ConstraintViolationV1:
    clause: FrameConstraintClauseV1
    observed: Any


@dataclass(frozen=True)
class ConstraintCheckResultV1:
    satisfied: bool
    violations: tuple[ConstraintViolationV1, ...]


def check_constraint(
    constraint: FrameConstraintV1, candidate: str | Program, schema: CAP1SchemaV1
) -> ConstraintCheckResultV1:
    """Check a candidate AST against a frame-derived constraint.

    Never emits DSL source text: the candidate (already-existing DSL text or
    a parsed :class:`Program`) is derived into its own frame and only
    fact-level values are compared — this is the "without generating DSL
    text" inverse constraint check used for counterfactual/ambiguity
    evaluation.
    """
    if schema.fingerprint() != constraint.schema_fingerprint:
        raise UnsupportedSemanticFactError(
            "constraint was derived under a different CAP1 schema fingerprint"
        )
    candidate_frame = derive_semantic_frame(candidate, schema)
    index: dict[tuple[AstPath, str, str], Any] = {
        (f.ast_path, f.schema_field, f.predicate): f.value
        for f in candidate_frame.facts
        if f.category in {"required", "optional"}
    }
    violations: list[ConstraintViolationV1] = []
    for clause in constraint.clauses:
        key = (clause.ast_path, clause.schema_field, clause.predicate)
        observed = index.get(key, _MISSING)
        if clause.polarity == "requires":
            if observed is _MISSING or observed != clause.value:
                violations.append(
                    ConstraintViolationV1(
                        clause, observed if observed is not _MISSING else None
                    )
                )
        else:
            if observed is not _MISSING and observed == clause.value:
                violations.append(ConstraintViolationV1(clause, observed))
    return ConstraintCheckResultV1(
        satisfied=not violations, violations=tuple(violations)
    )


# --------------------------------------------------------------------------
# One-fact counterfactual
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualProofV1:
    changed_ast_path: AstPath
    schema_field: str
    predicate: str
    old_value: Any
    new_value: Any
    single_fact_change: bool


def _asserted_facts(frame: SemanticFrameV1) -> dict[tuple[AstPath, str, str], Any]:
    return {
        (f.ast_path, f.schema_field, f.predicate): f.value
        for f in frame.facts
        if f.category in {"required", "optional"}
    }


def verify_single_fact_change(
    before: SemanticFrameV1, after: SemanticFrameV1
) -> CounterfactualProofV1:
    """Prove exactly one asserted (required/optional) fact differs between frames.

    Forbidden facts are the schema-entailed complement of a closed-domain
    required fact, not independent assertions, so a required flip's
    mechanically-necessary forbidden-set update is not itself counted as a
    second changed fact.
    """
    before_map = _asserted_facts(before)
    after_map = _asserted_facts(after)
    if before_map.keys() != after_map.keys():
        raise CounterfactualError(
            "counterfactual changed which facts are asserted, not just a value"
        )
    changed = [key for key in before_map if before_map[key] != after_map[key]]
    if len(changed) != 1:
        raise CounterfactualError(
            f"expected exactly one changed fact, found {len(changed)}: {changed!r}"
        )
    ast_path, schema_field, predicate = changed[0]
    return CounterfactualProofV1(
        changed_ast_path=ast_path,
        schema_field=schema_field,
        predicate=predicate,
        old_value=before_map[changed[0]],
        new_value=after_map[changed[0]],
        single_fact_change=True,
    )


def mutate_one_fact(
    frame: SemanticFrameV1,
) -> tuple[SemanticFrameV1, CounterfactualProofV1]:
    """Flip exactly one closed-domain required fact to its declared alternative.

    Only ``closed_value`` facts have an exact, schema-declared alternative
    (their sibling ``forbidden`` facts already enumerate it). Flipping any
    other fact — free content text, an open-domain number or string — would
    require guessing a replacement value the schema does not license, which
    the stop rule forbids; :func:`CounterfactualError` is raised instead.
    """
    closed_required = sorted(
        (
            f
            for f in frame.facts
            if f.category == "required" and f.predicate == "closed_value"
        ),
        key=lambda f: repr(f.ast_path),
    )
    if not closed_required:
        raise CounterfactualError(
            "frame has no closed-domain required fact to mutate; refusing to guess"
        )
    target = closed_required[0]
    alternatives = sorted(
        (
            f.value
            for f in frame.facts
            if f.category == "forbidden"
            and f.ast_path == target.ast_path
            and f.schema_field == target.schema_field
        ),
        key=repr,
    )
    if not alternatives:
        raise CounterfactualError(
            f"closed-domain fact at {target.ast_path!r} has no recorded alternative"
        )
    new_value = alternatives[0]

    def _recategorize(fact: SemanticFactV1) -> SemanticFactV1:
        if fact.ast_path != target.ast_path or fact.schema_field != target.schema_field:
            return fact
        if fact.category == "required" and fact.predicate == "closed_value":
            return replace(fact, value=new_value)
        if fact.category == "forbidden" and fact.predicate == "excluded_value":
            if fact.value == new_value:
                return replace(fact, value=target.value)
            return fact
        return fact

    new_frame = replace(frame, facts=tuple(_recategorize(f) for f in frame.facts))
    proof = verify_single_fact_change(frame, new_frame)
    return new_frame, proof


__all__ = [
    "AstPath",
    "CAP1SchemaV1",
    "CounterfactualError",
    "CounterfactualProofV1",
    "ConstraintCheckResultV1",
    "ConstraintViolationV1",
    "FactCategory",
    "FrameConstraintClauseV1",
    "FrameConstraintV1",
    "SemanticEffectV1",
    "SemanticFactConflictV1",
    "SemanticFactV1",
    "SemanticFrameNodeV1",
    "SemanticFrameV1",
    "SemanticRelationV1",
    "SemanticRoleV1",
    "UnsupportedSemanticFactError",
    "check_constraint",
    "derive_semantic_frame",
    "frame_conflicts",
    "frame_to_constraint",
    "mutate_one_fact",
    "verify_single_fact_change",
]
