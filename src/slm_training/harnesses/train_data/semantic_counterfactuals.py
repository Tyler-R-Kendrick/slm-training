"""CounterfactualPairV1: one-fact semantic counterfactuals + grounding contrasts.

DSH2-06 (SLM-366): given a verified
:class:`~slm_training.dsl.semantic_frame.SemanticFrameV1` (DSH2-02, SLM-363)
and its canonical DSL target, mutate exactly **one** declared semantic fact —
role, order, cardinality, closed value, required child, or relation — compile a
new target satisfying the *changed* frame, and generate matched-style,
length-bounded prompts for both through the DSH2-03/DSH2-04 provider contract
(SLM-364/SLM-365 fixture provider). The pair is grounding evidence only when
the :func:`one_fact_proof` shows exactly one declared dimension differs, both
targets are compiler-valid with distinct canonical fingerprints, and every
surface control (style, length tolerance, marker-surface/leak check) passes.

**Stop rule.** Any pair with an uncontrolled semantic or surface cue — a
second differing dimension, a length out of tolerance, a marker-surface or DSL
leak, a style mismatch, an invalid counterfactual target, or a mutation site
the declared tables do not cover — is *rejected with a code* and never counted
as grounding evidence. Mutations never guess: each dimension mutates only
sites the grammar/schema declares an exact alternative for (closed-domain
complements, declared siblings, declared state targets, declared singleton
alternatives); anything else rejects ``no_mutation_site``.

**Split family.** Source and counterfactual derivatives share one root split
family: the counterfactual inherits the source's ``root_family_id`` (derived
from the source canonical target), so
:class:`~slm_training.harnesses.train_data.split_policy.RootFamilySplitPolicyV1`
assigns both to the same split, and the artifact graph flags any cross-family
derivation.

**Negatives + baseline.** Two hard grounding contrasts are provided:
embedding-close/action-distant negatives (surface near-identical prompts whose
correct targets differ structurally) and inventory-matched topology negatives
(same component multiset, different nesting). A constant/frequency-only
baseline over pair labels is exactly chance by construction and is asserted so
in tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.lang_core import Program
from slm_training.dsl.parser import validate as validate_source
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.semantic_frame import (
    CAP1SchemaV1,
    SemanticFrameV1,
    derive_semantic_frame,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.artifact_graph import ArtifactNodeV1
from slm_training.harnesses.train_data.frame_paraphrases import (
    FrameLeakDetectorV1,
    FrameParaphraseFixtureProviderV1,
    generate_frame_paraphrases,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "semantic_counterfactuals/v1"

#: Declared semantic fact dimensions. Exactly one may differ inside a pair.
FACT_DIMENSIONS: tuple[str, ...] = (
    "role",
    "order",
    "cardinality",
    "closed_value",
    "required_child",
    "relation",
)

FactDimension = Literal[
    "role", "order", "cardinality", "closed_value", "required_child", "relation"
]

#: Prompt style both sides of a pair are rendered in (matched style).
DEFAULT_STYLE = "concise"

#: Maximum relative prompt-length gap inside a pair before the pair is
#: rejected as surface-cued (``length_out_of_tolerance``).
PROMPT_LENGTH_RATIO_TOLERANCE = 0.4

#: Token-multiset Jaccard at or above which two prompts count as
#: surface/embedding-close for the action-distant negative.
SURFACE_CLOSE_JACCARD = 0.9

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CounterfactualPairError(ValueError):
    """A counterfactual pair could not be constructed or proven honestly."""


# --------------------------------------------------------------------------
# Minimal canonical-AST -> DSL source emitter (mutation round-trip)
# --------------------------------------------------------------------------


def _emit_expr(node: Any) -> str:
    """Serialize one canonical AST node back to DSL expression text.

    Covers the exact node shapes the DSH2-02 frame walker declares (elements,
    refs, state refs, Obj/BinOp/UnaryOp/Ternary/Member/Index/Arr, scalars).
    Anything else fails closed — a mutation round-trip must never emit a
    guessed surface.
    """
    if node is None:
        return "null"
    if isinstance(node, bool):
        return "true" if node else "false"
    if isinstance(node, (int, float)):
        return repr(node)
    if isinstance(node, str):
        return json.dumps(node)
    if isinstance(node, list):
        return "[" + ", ".join(_emit_expr(item) for item in node) + "]"
    if isinstance(node, dict):
        if node.get("type") == "element":
            props = node.get("props") or {}
            args = ", ".join(_emit_expr(props[name]) for name in props)
            return f"{node['typeName']}({args})"
        if node.get("type") == "ref":
            return str(node["name"])
        kind = node.get("k")
        if kind == "StateRef":
            return str(node["n"])
        if kind == "RuntimeRef":
            return f"@{str(node.get('refType', 'query')).capitalize()}({json.dumps(str(node['n']))})"
        if kind == "BinOp":
            return f"({_emit_expr(node['left'])} {node['op']} {_emit_expr(node['right'])})"
        if kind == "UnaryOp":
            return f"({node['op']}{_emit_expr(node['operand'])})"
        if kind == "Ternary":
            return (
                f"({_emit_expr(node['cond'])} ? {_emit_expr(node['then'])}"
                f" : {_emit_expr(node['else'])})"
            )
        if kind == "Member":
            return f"{_emit_expr(node['obj'])}.{node['field']}"
        if kind == "Index":
            return f"{_emit_expr(node['obj'])}[{_emit_expr(node['index'])}]"
        if kind == "Arr":
            return "[" + ", ".join(_emit_expr(x) for x in node.get("els", [])) + "]"
        if kind == "Obj":
            return "{" + ", ".join(
                f"{key}: {_emit_expr(value)}" for key, value in node.get("entries", [])
            ) + "}"
        if kind == "Comp":
            args = ", ".join(_emit_expr(a) for a in node.get("args", []))
            return f"@{node['name']}({args})"
    raise CounterfactualPairError(
        f"cannot emit AST node of shape {type(node).__name__}: {node!r}"
    )


def _emit_program(root: dict[str, Any], state_declarations: Mapping[str, Any]) -> str:
    lines = [f"{name} = {_emit_expr(value)}" for name, value in state_declarations.items()]
    lines.append(f"root = {_emit_expr(root)}")
    return "\n".join(lines)


def _deepcopy_ast(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _get_at_path(root: Any, path: tuple[str | int, ...]) -> Any:
    node = root
    for part in path:
        node = node[part]
    return node


def _set_at_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    node = _get_at_path(root, path[:-1]) if path else None
    if node is None:
        raise CounterfactualPairError("cannot replace the root itself")
    node[path[-1]] = value


def _strip_ast_metadata(node: Any) -> Any:
    """Drop parse bookkeeping (``statementId`` etc.) for AST equality."""
    if isinstance(node, dict):
        return {
            key: _strip_ast_metadata(value)
            for key, value in node.items()
            if key not in {"statementId", "partial", "hasDynamicProps"}
        }
    if isinstance(node, list):
        return [_strip_ast_metadata(item) for item in node]
    return node


# --------------------------------------------------------------------------
# Mutation site discovery + typed one-fact mutations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationSpecV1:
    """The declared single-fact change applied to the canonical AST."""

    dimension: str
    locus: tuple[str | int, ...]
    detail: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "locus": list(self.locus),
            "detail": self.detail,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


def _element_children_slots(
    root: dict[str, Any], path: tuple[str | int, ...] = ()
) -> Iterable[tuple[tuple[str | int, ...], dict[str, Any], str, list[Any]]]:
    """Yield (list_path, element, prop_name, items) for element-list props."""
    if not isinstance(root, dict) or root.get("type") != "element":
        return
    for prop_name, value in (root.get("props") or {}).items():
        prop_path = path + ("props", prop_name)
        if isinstance(value, list) and value and all(
            isinstance(item, dict) and item.get("type") == "element" for item in value
        ):
            yield prop_path, root, prop_name, value
        elif isinstance(value, dict):
            yield from _element_children_slots(value, prop_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _element_children_slots(item, prop_path + (index,))


def _singleton_child_slots(
    root: dict[str, Any], path: tuple[str | int, ...] = ()
) -> Iterable[tuple[tuple[str | int, ...], dict[str, Any], str]]:
    """Yield (prop_path, element, prop_name) for non-list element child props."""
    if not isinstance(root, dict) or root.get("type") != "element":
        return
    for prop_name, value in (root.get("props") or {}).items():
        prop_path = path + ("props", prop_name)
        if isinstance(value, dict) and value.get("type") == "element":
            yield prop_path, root, prop_name
        elif isinstance(value, dict):
            yield from _singleton_child_slots(value, prop_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _singleton_child_slots(item, prop_path + (index,))


def _walk_elements(
    root: dict[str, Any], path: tuple[str | int, ...] = ()
) -> Iterable[tuple[tuple[str | int, ...], dict[str, Any]]]:
    if not isinstance(root, dict) or root.get("type") != "element":
        return
    yield path, root
    for prop_name, value in (root.get("props") or {}).items():
        prop_path = path + ("props", prop_name)
        if isinstance(value, dict) and value.get("type") == "element":
            yield from _walk_elements(value, prop_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and item.get("type") == "element":
                    yield from _walk_elements(item, prop_path + (index,))


#: Declared singleton-child alternatives: the only required-child replacements
#: this module may apply, mapping the child's current type to a builder for
#: the replacement subtree derived from the old subtree's own props. Anything
#: not declared here rejects ``no_mutation_site`` rather than guessing.
def _text_area_alternative(old: dict[str, Any]) -> dict[str, Any]:
    props = old.get("props") or {}
    return {
        "type": "element",
        "typeName": "TextArea",
        "props": {
            "name": props.get("name"),
            "placeholder": props.get("placeholder"),
            "rows": 3,
        },
        "partial": False,
        "hasDynamicProps": False,
    }


SINGLETON_CHILD_ALTERNATIVES: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "Input": _text_area_alternative,
}


def mutate_program(
    program: Program,
    dimension: str,
    *,
    schema: CAP1SchemaV1,
) -> tuple[str, MutationSpecV1]:
    """Apply the typed one-fact mutation for ``dimension``; emit new source.

    Returns ``(mutated_source, spec)``. Fails closed with
    :class:`CounterfactualPairError` (``no_mutation_site``) when the frame
    declares no site the dimension's exact, schema-licensed mutation covers.
    """
    if dimension not in FACT_DIMENSIONS:
        raise CounterfactualPairError(f"undeclared fact dimension {dimension!r}")
    if program.root is None:
        raise CounterfactualPairError("program has no root AST to mutate")
    root = _deepcopy_ast(program.root)
    states = _deepcopy_ast(program.state_declarations)

    if dimension == "closed_value":
        spec = _mutate_closed_value(root, schema)
    elif dimension == "order":
        spec = _mutate_order(root)
    elif dimension == "cardinality":
        spec = _mutate_cardinality(root)
    elif dimension == "role":
        spec = _mutate_role(root)
    elif dimension == "required_child":
        spec = _mutate_required_child(root)
    else:
        spec = _mutate_relation(root, states)
    return _emit_program(root, states), spec


def _no_site(dimension: str, detail: str) -> CounterfactualPairError:
    return CounterfactualPairError(f"no_mutation_site: {dimension}: {detail}")


def _mutate_closed_value(root: dict[str, Any], schema: CAP1SchemaV1) -> MutationSpecV1:
    sites: list[tuple[tuple[str | int, ...], str, Any]] = []
    for path, element in _walk_elements(root):
        for prop_name, value in (element.get("props") or {}).items():
            if isinstance(value, bool):
                sites.append((path + ("props", prop_name), prop_name, value))
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                domain = schema.closed_value_domains.get(prop_name)
                if domain and value in domain and len(domain) > 1:
                    sites.append((path + ("props", prop_name), prop_name, value))
    if not sites:
        raise _no_site("closed_value", "no declared closed-domain scalar prop")
    locus, prop_name, old = sorted(sites, key=lambda s: repr(s[0]))[0]
    new: Any = (not old) if isinstance(old, bool) else next(
        alt for alt in schema.closed_value_domains[prop_name] if alt != old
    )
    _set_at_path(root, locus, new)
    return MutationSpecV1(
        "closed_value", locus, f"closed value of {prop_name} flipped", old, new
    )


def _mutate_order(root: dict[str, Any]) -> MutationSpecV1:
    for list_path, _element, _prop, items in sorted(
        _element_children_slots(root), key=lambda s: repr(s[0])
    ):
        for index in range(len(items) - 1):
            left, right = items[index], items[index + 1]
            if left.get("typeName") != right.get("typeName"):
                items[index], items[index + 1] = right, left
                return MutationSpecV1(
                    "order",
                    list_path,
                    f"siblings at positions {index}/{index + 1} swapped",
                    [left.get("typeName"), right.get("typeName")],
                    [right.get("typeName"), left.get("typeName")],
                )
    raise _no_site("order", "no adjacent element siblings of different types")


def _mutate_cardinality(root: dict[str, Any]) -> MutationSpecV1:
    for list_path, _element, _prop, items in sorted(
        _element_children_slots(root), key=lambda s: repr(s[0])
    ):
        if items:
            items.append(_deepcopy_ast(items[-1]))
            return MutationSpecV1(
                "cardinality",
                list_path,
                f"slot cardinality {len(items) - 1} -> {len(items)} (last child duplicated)",
                len(items) - 1,
                len(items),
            )
    raise _no_site("cardinality", "no element-list slot to extend")


def _mutate_role(root: dict[str, Any]) -> MutationSpecV1:
    for path, element in _walk_elements(root):
        slots = [
            (name, value)
            for name, value in (element.get("props") or {}).items()
            if isinstance(value, list)
            and value
            and all(isinstance(i, dict) and i.get("type") == "element" for i in value)
        ]
        if len(slots) >= 2:
            (name_a, list_a), (name_b, list_b) = slots[0], slots[1]
            list_a[0], list_b[0] = list_b[0], list_a[0]
            return MutationSpecV1(
                "role",
                path + ("props",),
                f"first children of slots {name_a!r}/{name_b!r} exchanged",
                [name_a, name_b],
                [name_b, name_a],
            )
    raise _no_site("role", "no element with two element-list slots to exchange")


def _mutate_required_child(root: dict[str, Any]) -> MutationSpecV1:
    for prop_path, _element, prop_name in sorted(
        _singleton_child_slots(root), key=lambda s: repr(s[0])
    ):
        child = _get_at_path(root, prop_path)
        builder = SINGLETON_CHILD_ALTERNATIVES.get(str(child.get("typeName")))
        if builder is None:
            continue
        replacement = builder(child)
        _set_at_path(root, prop_path, replacement)
        return MutationSpecV1(
            "required_child",
            prop_path,
            f"required singleton child in slot {prop_name!r} replaced "
            f"{child.get('typeName')} -> {replacement['typeName']}",
            child.get("typeName"),
            replacement["typeName"],
        )
    raise _no_site(
        "required_child", "no singleton child slot with a declared alternative"
    )


def _mutate_relation(
    root: dict[str, Any], states: dict[str, Any]
) -> MutationSpecV1:
    names = sorted(states)
    refs: list[tuple[tuple[str | int, ...], str]] = []

    def walk(node: Any, path: tuple[str | int, ...]) -> None:
        if isinstance(node, dict):
            if node.get("k") == "StateRef":
                refs.append((path, str(node.get("n"))))
                return
            for key, value in node.items():
                if key in {"partial", "hasDynamicProps", "statementId"}:
                    continue
                walk(value, path + (key,))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, path + (index,))

    walk(root, ())
    for ref_path, target in sorted(refs, key=lambda r: repr(r[0])):
        alternatives = [name for name in names if name != target]
        if alternatives:
            new_target = alternatives[0]
            _set_at_path(root, ref_path + ("n",), new_target)
            return MutationSpecV1(
                "relation",
                ref_path,
                f"state read target {target} -> {new_target}",
                target,
                new_target,
            )
    raise _no_site("relation", "no state read with a declared alternative target")


# --------------------------------------------------------------------------
# Dimension projections (frame-level semantic diff)
# --------------------------------------------------------------------------


def dimension_projection(frame: SemanticFrameV1, dimension: str) -> frozenset[Any]:
    """Project a frame onto one declared dimension for the one-fact proof.

    Projections are deliberately path-insensitive so a mutation on one
    dimension leaves every other projection bit-identical:

    - ``role``: set of ``(parent_type, role, child_type)`` over *ordered*
      (list-slot) parent->child edges.
    - ``order``: set of ``(parent_type, role, before_type, after_type)`` over
      consecutive sibling pairs of differing component type.
    - ``cardinality``: set of ``(parent_type, role, child count)`` per
      list-slot.
    - ``closed_value``: set of ``(prop/field, value)`` over required
      closed-domain facts (bool literals, declared closed domains, grammar
      operator enums).
    - ``required_child``: set of ``(parent_type, role, child_type)`` over
      *unordered* (singleton-slot) parent->child edges.
    - ``relation``: set of ``(kind, target)`` over semantic relations plus
      ``("effect", kind)`` over effects.
    """
    if dimension not in FACT_DIMENSIONS:
        raise CounterfactualPairError(f"undeclared fact dimension {dimension!r}")
    node_type = {n.node_id: n.type_name for n in frame.nodes}
    element_ids = {n.node_id for n in frame.nodes if n.kind == "element"}
    element_roles = [
        r
        for r in frame.roles
        if r.parent_id in element_ids and r.child_id in element_ids
    ]
    if dimension == "role":
        return frozenset(
            (node_type[r.parent_id], r.role, node_type[r.child_id])
            for r in element_roles
            if r.order is not None
        )
    if dimension == "required_child":
        return frozenset(
            (node_type[r.parent_id], r.role, node_type[r.child_id])
            for r in element_roles
            if r.order is None
        )
    if dimension == "cardinality":
        counts: dict[tuple[str, str], int] = {}
        for r in element_roles:
            if r.order is not None:
                key = (node_type[r.parent_id], r.role)
                counts[key] = counts.get(key, 0) + 1
        return frozenset((p, role, n) for (p, role), n in counts.items())
    if dimension == "order":
        siblings: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for r in element_roles:
            if r.order is not None:
                siblings.setdefault((node_type[r.parent_id], r.role), []).append(
                    (r.order, node_type[r.child_id])
                )
        pairs: set[tuple[str, str, str, str]] = set()
        for (parent_type, role), entries in siblings.items():
            entries.sort()
            for (_, left), (_, right) in zip(entries, entries[1:]):
                if left != right:
                    pairs.add((parent_type, role, left, right))
        return frozenset(pairs)
    if dimension == "closed_value":
        out: set[tuple[str, Any]] = set()
        for fact in frame.facts:
            if fact.category != "required":
                continue
            if fact.predicate == "closed_value" or (
                fact.predicate.endswith("_op")
                and fact.schema_field.startswith("grammar.")
            ):
                field_name = next(
                    (p for p in reversed(fact.ast_path) if isinstance(p, str)),
                    fact.schema_field,
                )
                out.add((str(field_name), fact.value))
        return frozenset(out)
    relation_facts = frozenset((r.kind, r.target) for r in frame.relations)
    effect_facts = frozenset(("effect", e.effect_kind) for e in frame.effects)
    return relation_facts | effect_facts


@dataclass(frozen=True)
class OneFactProofV1:
    """Proof that exactly one declared dimension differs inside a pair."""

    dimension: str
    mutation: MutationSpecV1
    dimension_diffs: Mapping[str, int]
    source_canonical_sha256: str
    counterfactual_canonical_sha256: str
    fingerprints_differ: bool
    inverse_roundtrip: bool
    single_dimension_change: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "mutation": self.mutation.to_dict(),
            "dimension_diffs": dict(self.dimension_diffs),
            "source_canonical_sha256": self.source_canonical_sha256,
            "counterfactual_canonical_sha256": self.counterfactual_canonical_sha256,
            "fingerprints_differ": self.fingerprints_differ,
            "inverse_roundtrip": self.inverse_roundtrip,
            "single_dimension_change": self.single_dimension_change,
        }


def one_fact_proof(
    source_frame: SemanticFrameV1,
    counterfactual_frame: SemanticFrameV1,
    mutation: MutationSpecV1,
    *,
    schema: CAP1SchemaV1,
) -> OneFactProofV1:
    """Compute the frame-level dimension diff and the inverse round-trip.

    ``single_dimension_change`` is true only when the declared dimension's
    projection diff is non-empty and every other dimension's diff is empty.
    The inverse round-trip re-derives the source by applying the mutation
    descriptor backwards where the descriptor is exactly invertible
    (closed-value flips, order/role swaps, relation retargets, cardinality
    pops, singleton-child restores are all recorded with old/new values).
    """
    diffs = {
        dimension: len(
            dimension_projection(source_frame, dimension)
            ^ dimension_projection(counterfactual_frame, dimension)
        )
        for dimension in FACT_DIMENSIONS
    }
    source_sha = hashlib.sha256(
        source_frame.canonical_source.encode("utf-8")
    ).hexdigest()
    cf_sha = hashlib.sha256(
        counterfactual_frame.canonical_source.encode("utf-8")
    ).hexdigest()
    inverse_ok = _inverse_roundtrip(source_frame, counterfactual_frame, mutation, schema)
    single = (
        diffs[mutation.dimension] > 0
        and all(d == 0 for dim, d in diffs.items() if dim != mutation.dimension)
        and source_sha != cf_sha
    )
    return OneFactProofV1(
        dimension=mutation.dimension,
        mutation=mutation,
        dimension_diffs=diffs,
        source_canonical_sha256=source_sha,
        counterfactual_canonical_sha256=cf_sha,
        fingerprints_differ=source_sha != cf_sha,
        inverse_roundtrip=inverse_ok,
        single_dimension_change=single,
    )


def _inverse_roundtrip(
    source_frame: SemanticFrameV1,
    counterfactual_frame: SemanticFrameV1,
    mutation: MutationSpecV1,
    schema: CAP1SchemaV1,
) -> bool:
    """Re-apply the inverse mutation to the counterfactual AST; must equal source."""
    try:
        cf_program = validate_source(
            counterfactual_frame.canonical_source, dsl=schema.dsl
        )
        src_program = validate_source(source_frame.canonical_source, dsl=schema.dsl)
        root = _deepcopy_ast(cf_program.root)
        states = _deepcopy_ast(cf_program.state_declarations)
        dim = mutation.dimension
        if dim == "closed_value":
            _set_at_path(root, tuple(mutation.locus), mutation.old_value)
        elif dim == "order":
            items = _get_at_path(root, tuple(mutation.locus))
            # the mutation swapped the first differing adjacent pair; swap back
            for index in range(len(items) - 1):
                if items[index].get("typeName") != items[index + 1].get("typeName"):
                    items[index], items[index + 1] = items[index + 1], items[index]
                    break
        elif dim == "cardinality":
            items = _get_at_path(root, tuple(mutation.locus))
            items.pop()
        elif dim == "role":
            parent = _get_at_path(root, tuple(mutation.locus))
            name_a, name_b = mutation.old_value
            parent[name_a][0], parent[name_b][0] = parent[name_b][0], parent[name_a][0]
        elif dim == "required_child":
            src_child = _get_at_path(_deepcopy_ast(src_program.root), tuple(mutation.locus))
            _set_at_path(root, tuple(mutation.locus), src_child)
        elif dim == "relation":
            _set_at_path(root, tuple(mutation.locus) + ("n",), mutation.old_value)
        else:  # pragma: no cover - guarded by FACT_DIMENSIONS
            return False
        restored_program = validate_source(
            canonicalize(_emit_program(root, states), dsl=schema.dsl), dsl=schema.dsl
        )
        return _strip_ast_metadata(restored_program.root) == _strip_ast_metadata(
            src_program.root
        )
    except Exception:  # noqa: BLE001 - round-trip failure is a proof failure
        return False


# --------------------------------------------------------------------------
# Typed rows: pair + rejection
# --------------------------------------------------------------------------

REJECTION_CODES: tuple[str, ...] = (
    "no_mutation_site",
    "counterfactual_not_compiler_valid",
    "multi_dimension_diff",
    "no_dimension_diff",
    "inverse_roundtrip_failed",
    "identical_canonical_targets",
    "style_mismatch",
    "length_out_of_tolerance",
    "marker_surface_cue",
    "leak_detected",
    "prompt_generation_failed",
)


@dataclass(frozen=True)
class CounterfactualRejectionV1:
    """A rejected pair candidate, always counted with a code (stop rule)."""

    dimension: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualPairV1:
    """One admitted one-fact counterfactual pair (grounding evidence)."""

    pair_id: str
    dimension: str
    style: str
    source_prompt: str
    source_prompt_sha256: str
    source_target: str
    source_target_sha256: str
    counterfactual_prompt: str
    counterfactual_prompt_sha256: str
    counterfactual_target: str
    counterfactual_target_sha256: str
    root_family_id: str
    split: str
    proof: OneFactProofV1
    prompt_length_ratio: float
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "dimension": self.dimension,
            "style": self.style,
            "source_prompt": self.source_prompt,
            "source_prompt_sha256": self.source_prompt_sha256,
            "source_target": self.source_target,
            "source_target_sha256": self.source_target_sha256,
            "counterfactual_prompt": self.counterfactual_prompt,
            "counterfactual_prompt_sha256": self.counterfactual_prompt_sha256,
            "counterfactual_target": self.counterfactual_target,
            "counterfactual_target_sha256": self.counterfactual_target_sha256,
            "root_family_id": self.root_family_id,
            "split": self.split,
            "proof": self.proof.to_dict(),
            "prompt_length_ratio": self.prompt_length_ratio,
        }


def _declared_state_names(canonical_source: str) -> tuple[str, ...]:
    """Declared state names in declaration order (from state statements)."""
    return tuple(re.findall(r"(?m)^\s*(\$[A-Za-z_][\w]*)\s*=", canonical_source))


def _neutralize_state_surfaces(
    prompt: str,
    frame: SemanticFrameV1,
    declared: Sequence[str] = (),
) -> str:
    """Replace canonical state surfaces (``$s0``) with declaration ordinals.

    State names are symbol identity, not surface authority: prompts describe
    reads by declaration ordinal ("declared state 1") so the marker surface
    can never be a cue (and never trips the leak detector). Ordinals range
    over the target's full declaration list so a retargeted read still
    renders differently.
    """
    names = list(declared) or sorted(
        {
            str(f.value)
            for f in frame.facts
            if f.predicate == "state_ref_name" and isinstance(f.value, str)
        }
        | {str(r.target) for r in frame.relations if r.kind == "reads_state"}
    )
    out = prompt
    for index, name in enumerate(names):
        out = out.replace(f'"{name}"', f"declared state {index + 1}")
        out = out.replace(name, f"declared state {index + 1}")
    return out


# --------------------------------------------------------------------------
# Pair generation
# --------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def root_family_for(source_canonical: str, dsl: str) -> str:
    """Root split family for a source target; derivatives inherit it."""
    return content_sha({"dsl": dsl, "canonical_source": source_canonical})


def _reject(dimension: str, code: str, detail: str) -> CounterfactualRejectionV1:
    if code not in REJECTION_CODES:
        raise CounterfactualPairError(f"undeclared rejection code {code!r}")
    return CounterfactualRejectionV1(dimension=dimension, code=code, detail=detail)


class _StateNeutralizingDetector:
    """Leak-detector wrapper: checks the state-surface-neutralized prompt.

    State canonical names (``$s0``) are symbol identity, not prompt surface
    authority; the final prompt text is neutralized to declared ordinals, so
    the declared detector evaluates that form.
    """

    def __init__(self, inner: Any, frame: SemanticFrameV1, declared: Sequence[str]) -> None:
        self._inner = inner
        self._frame = frame
        self._declared = declared

    def check(self, text: str) -> tuple[bool, tuple[str, ...]]:
        return self._inner.check(
            _neutralize_state_surfaces(text, self._frame, self._declared)
        )


def _style_prompt(
    frame: SemanticFrameV1,
    *,
    provider: Any,
    style: str,
    now: Callable[[], str] | None,
) -> str | None:
    detector = FrameLeakDetectorV1.from_frame(frame)
    declared = _declared_state_names(frame.canonical_source)
    result = generate_frame_paraphrases(
        frame,
        provider=provider,
        style_plan=(style,),
        leak_detector=_StateNeutralizingDetector(detector, frame, declared),
        now=now,
    )
    if not result.rows:
        return None
    return _neutralize_state_surfaces(result.rows[0].prompt_text, frame, declared)


def prompt_length_ratio(left: str, right: str) -> float:
    """Relative length gap between two prompts (0 = identical length)."""
    shorter, longer = sorted((len(left), len(right)))
    return (longer - shorter) / longer if longer else 0.0


def check_pair_prompt_controls(
    source_prompt: str,
    counterfactual_prompt: str,
    source_frame: SemanticFrameV1,
    counterfactual_frame: SemanticFrameV1,
    *,
    length_ratio_tolerance: float = PROMPT_LENGTH_RATIO_TOLERANCE,
) -> tuple[str, str] | None:
    """Surface-cue controls over an admitted pair's prompts.

    Returns ``(code, detail)`` for the first failed control, else ``None``:
    ``length_out_of_tolerance`` when the relative length gap exceeds the
    tolerance, ``marker_surface_cue`` when either frame's declared leak
    detector finds a marker surface in either prompt, ``leak_detected`` for
    any other detector hit (DSL production forms, digests, provenance
    fields). Each prompt is checked against *both* frames' detectors so
    cross-contamination (a surface that only the other target reveals) is
    caught too.
    """
    ratio = prompt_length_ratio(source_prompt, counterfactual_prompt)
    if ratio > length_ratio_tolerance:
        return (
            "length_out_of_tolerance",
            f"prompt length ratio {ratio:.4f} exceeds {length_ratio_tolerance}",
        )
    for detector_frame, label in (
        (source_frame, "source"),
        (counterfactual_frame, "cf"),
    ):
        detector = FrameLeakDetectorV1.from_frame(detector_frame)
        for prompt, side in (
            (source_prompt, "source"),
            (counterfactual_prompt, "cf"),
        ):
            ok, reasons = detector.check(prompt)
            if not ok:
                code = (
                    "marker_surface_cue"
                    if any(r.startswith("marker_surface") for r in reasons)
                    else "leak_detected"
                )
                return code, (
                    f"{label} detector rejected {side} prompt: {list(reasons)!r}"
                )
    return None


def generate_counterfactual_pair(
    frame: SemanticFrameV1,
    dimension: str,
    *,
    provider: Any | None = None,
    dsl: str | None = None,
    schema: CAP1SchemaV1 | None = None,
    style: str = DEFAULT_STYLE,
    counterfactual_style: str | None = None,
    length_ratio_tolerance: float = PROMPT_LENGTH_RATIO_TOLERANCE,
    now: Callable[[], str] | None = None,
) -> CounterfactualPairV1 | CounterfactualRejectionV1:
    """Generate one matched-style one-fact counterfactual pair.

    ``frame`` is the verified source frame (its ``canonical_source`` is the
    source target). The counterfactual target is produced by
    :func:`mutate_program` on the canonical AST, re-canonicalized and
    re-validated; prompts come from the SLM-365 provider contract (offline
    fixture provider by default) in the same style. Every control failure
    returns a coded :class:`CounterfactualRejectionV1` — rejected pairs are
    never counted as grounding evidence.
    """
    if dimension not in FACT_DIMENSIONS:
        raise CounterfactualPairError(f"undeclared fact dimension {dimension!r}")
    schema = schema or CAP1SchemaV1.from_pack(dsl or frame.dsl)
    provider = provider or FrameParaphraseFixtureProviderV1()
    cf_style = counterfactual_style if counterfactual_style is not None else style

    source_program = validate_source(frame.canonical_source, dsl=schema.dsl)
    try:
        mutated_source, spec = mutate_program(
            source_program, dimension, schema=schema
        )
    except CounterfactualPairError as exc:
        code = "no_mutation_site"
        detail = str(exc)
        if detail.startswith("no_mutation_site:"):
            detail = detail.split(":", 2)[2].strip()
        return _reject(dimension, code, detail)

    try:
        counterfactual_canonical = canonicalize(mutated_source, dsl=schema.dsl)
        validate_source(counterfactual_canonical, dsl=schema.dsl)
        counterfactual_frame = derive_semantic_frame(
            counterfactual_canonical, schema
        )
    except Exception as exc:  # noqa: BLE001 - any compile failure rejects
        return _reject(
            dimension,
            "counterfactual_not_compiler_valid",
            f"{type(exc).__name__}: {exc}",
        )

    proof = one_fact_proof(frame, counterfactual_frame, spec, schema=schema)
    if not proof.fingerprints_differ:
        return _reject(
            dimension,
            "identical_canonical_targets",
            "mutation left the canonical target unchanged",
        )
    if proof.dimension_diffs[dimension] == 0:
        return _reject(
            dimension,
            "no_dimension_diff",
            f"mutation produced no {dimension!r} frame diff",
        )
    other = {d: n for d, n in proof.dimension_diffs.items() if d != dimension and n}
    if other:
        return _reject(
            dimension,
            "multi_dimension_diff",
            f"uncontrolled dimensions also differ: {other!r}",
        )
    if not proof.inverse_roundtrip:
        return _reject(
            dimension,
            "inverse_roundtrip_failed",
            "inverse mutation did not restore the source canonical target",
        )

    if cf_style != style:
        return _reject(
            dimension,
            "style_mismatch",
            f"source style {style!r} != counterfactual style {cf_style!r}",
        )
    source_prompt = _style_prompt(frame, provider=provider, style=style, now=now)
    cf_prompt = _style_prompt(
        counterfactual_frame, provider=provider, style=cf_style, now=now
    )
    if source_prompt is None or cf_prompt is None:
        return _reject(
            dimension,
            "prompt_generation_failed",
            "grounding floor rejected a prompt for one side of the pair",
        )
    source_prompt = _neutralize_state_surfaces(
        source_prompt, frame, _declared_state_names(frame.canonical_source)
    )
    cf_prompt = _neutralize_state_surfaces(
        cf_prompt,
        counterfactual_frame,
        _declared_state_names(counterfactual_frame.canonical_source),
    )

    cue = check_pair_prompt_controls(
        source_prompt,
        cf_prompt,
        frame,
        counterfactual_frame,
        length_ratio_tolerance=length_ratio_tolerance,
    )
    if cue is not None:
        code, detail = cue
        return _reject(dimension, code, detail)

    family_id = root_family_for(frame.canonical_source, schema.dsl)
    split = RootFamilySplitPolicyV1().assign(family_id)
    pair_id = content_sha(
        {
            "dimension": dimension,
            "source_target_sha256": _sha(frame.canonical_source),
            "counterfactual_target_sha256": _sha(counterfactual_canonical),
            "style": style,
        }
    )
    return CounterfactualPairV1(
        pair_id=pair_id,
        dimension=dimension,
        style=style,
        source_prompt=source_prompt,
        source_prompt_sha256=_sha(source_prompt),
        source_target=frame.canonical_source,
        source_target_sha256=_sha(frame.canonical_source),
        counterfactual_prompt=cf_prompt,
        counterfactual_prompt_sha256=_sha(cf_prompt),
        counterfactual_target=counterfactual_canonical,
        counterfactual_target_sha256=_sha(counterfactual_canonical),
        root_family_id=family_id,
        split=split,
        proof=proof,
        prompt_length_ratio=round(prompt_length_ratio(source_prompt, cf_prompt), 6),
    )


# --------------------------------------------------------------------------
# Split-family artifact nodes
# --------------------------------------------------------------------------


def pair_artifact_nodes(
    pair: CounterfactualPairV1, *, dsl: str = "openui"
) -> tuple[ArtifactNodeV1, ArtifactNodeV1]:
    """Source + counterfactual derivative nodes in one root split family.

    Both inherit ``pair.root_family_id`` / ``pair.split``; the counterfactual
    declares the source as its parent. The artifact graph quarantines any
    derivative whose family or split drifts (cross-family rejection).
    """
    policy = RootFamilySplitPolicyV1()
    policy.require_inherited(
        root_family_id=pair.root_family_id,
        split_group_id=pair.root_family_id,
        split=pair.split,
    )

    def _node(kind: str, target: str, parents: tuple[str, ...]) -> ArtifactNodeV1:
        artifact_id = content_sha(
            {"pair": pair.pair_id, "kind": kind, "target_sha": _sha(target)}
        )
        return ArtifactNodeV1(
            artifact_id=artifact_id,
            artifact_type=f"semantic_counterfactual.{kind}",
            root_family_id=pair.root_family_id,
            split_group_id=pair.root_family_id,
            split=pair.split,
            parent_ids=parents,
            surface_sha256=_sha(target),
            alpha_sha256=_sha(_strip_markers(target)),
            canonical_ast_sha256=_sha(canonicalize(target, dsl=dsl)),
            near_template_sha256=_sha(re.sub(r"\d+", "0", target)),
            payload={"pair_id": pair.pair_id, "dimension": pair.dimension, "kind": kind},
        )

    source_id = content_sha(
        {"pair": pair.pair_id, "kind": "source", "target_sha": pair.source_target_sha256}
    )
    return (
        _node("source", pair.source_target, ()),
        _node("counterfactual", pair.counterfactual_target, (source_id,)),
    )


def _strip_markers(text: str) -> str:
    return re.sub(r"(?<![\w])[:$][A-Za-z_][\w]*(?:\.[\w]+)*", "@", text)


# --------------------------------------------------------------------------
# Marker-surface permutation control (DSH1-04 placeholder authority)
# --------------------------------------------------------------------------


def marker_permutation(source: str, seed: int = 0) -> dict[str, str]:
    """Deterministic permutation of the placeholder surfaces in ``source``.

    Marker identity is codec/realization data, never semantic authority:
    permuting every placeholder consistently must leave the derived pair's
    proof and prompts unchanged (asserted in tests).
    """
    markers = sorted(set(extract_placeholders(source)))
    if len(markers) < 2:
        return {m: m for m in markers}
    shift = (seed % (len(markers) - 1)) + 1
    rotated = markers[shift:] + markers[:shift]
    return dict(zip(markers, rotated))


def permute_marker_surfaces(source: str, permutation: Mapping[str, str]) -> str:
    out = source
    for old, new in sorted(permutation.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"(?<![\w]){re.escape(old)}(?![\w])", new, out)
    return out


# --------------------------------------------------------------------------
# Hard negatives
# --------------------------------------------------------------------------


def _token_multiset_jaccard(left: str, right: str) -> float:
    a, b = _TOKEN_RE.findall(left.lower()), _TOKEN_RE.findall(right.lower())
    if not a and not b:
        return 1.0
    remaining = list(b)
    overlap = 0
    for token in a:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    union = len(a) + len(b) - overlap
    return overlap / union if union else 0.0


@dataclass(frozen=True)
class SurfaceCloseNegativeV1:
    """Embedding-close/action-distant negative: near-identical prompt surfaces,
    structurally different correct targets."""

    kind: str
    prompt_token_jaccard: float
    prompts_distinct: bool
    targets_differ: bool
    structural_dimension_diffs: Mapping[str, int]
    source_target_sha256: str
    negative_target_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def embedding_close_negative(
    pair: CounterfactualPairV1,
    *,
    schema: CAP1SchemaV1,
    min_jaccard: float = SURFACE_CLOSE_JACCARD,
) -> SurfaceCloseNegativeV1 | CounterfactualRejectionV1:
    """Build the embedding-close/action-distant negative from a pair.

    Admitted only when the two prompts are surface-close (token-multiset
    Jaccard >= ``min_jaccard``) while their correct targets differ on a
    *structural* dimension projection — surface similarity is then provably
    not a cue for the target.
    """
    jaccard = _token_multiset_jaccard(pair.source_prompt, pair.counterfactual_prompt)
    if jaccard < min_jaccard:
        return _reject(
            pair.dimension,
            "prompt_generation_failed",
            f"prompts not surface-close: token Jaccard {jaccard:.4f} < {min_jaccard}",
        )
    source_frame = derive_semantic_frame(pair.source_target, schema)
    cf_frame = derive_semantic_frame(pair.counterfactual_target, schema)
    structural = ("role", "order", "cardinality", "required_child", "relation")
    diffs = {
        dim: len(
            dimension_projection(source_frame, dim) ^ dimension_projection(cf_frame, dim)
        )
        for dim in structural
    }
    if not any(diffs.values()):
        return _reject(
            pair.dimension,
            "no_dimension_diff",
            "surface-close pair has no structural target difference",
        )
    return SurfaceCloseNegativeV1(
        kind="embedding_close_action_distant",
        prompt_token_jaccard=round(jaccard, 6),
        prompts_distinct=pair.source_prompt != pair.counterfactual_prompt,
        targets_differ=pair.source_target_sha256 != pair.counterfactual_target_sha256,
        structural_dimension_diffs=diffs,
        source_target_sha256=pair.source_target_sha256,
        negative_target_sha256=pair.counterfactual_target_sha256,
    )


@dataclass(frozen=True)
class TopologyNegativeV1:
    """Inventory-matched topology negative: same component multiset, different
    parent->child topology."""

    kind: str
    inventory_match: bool
    topology_differs: bool
    source_target_sha256: str
    negative_target_sha256: str
    source_target: str
    negative_target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _component_inventory(frame: SemanticFrameV1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in frame.nodes:
        if node.kind == "element":
            counts[node.type_name] = counts.get(node.type_name, 0) + 1
    return counts


def _topology(frame: SemanticFrameV1) -> frozenset[tuple[str, str]]:
    node_type = {n.node_id: n.type_name for n in frame.nodes}
    kinds = {n.node_id: n.kind for n in frame.nodes}
    edges = [
        (node_type[r.parent_id], node_type[r.child_id])
        for r in frame.roles
        if kinds.get(r.parent_id) == "element" and kinds.get(r.child_id) == "element"
    ]
    return frozenset(edges)


def topology_negative(
    source: str,
    candidate: str,
    *,
    schema: CAP1SchemaV1,
) -> TopologyNegativeV1 | CounterfactualRejectionV1:
    """Verify ``candidate`` as an inventory-matched topology negative of ``source``.

    Both sides must be compiler-valid, share the exact component multiset,
    and differ in parent->child topology; anything else rejects (never
    counted).
    """
    try:
        source_frame = derive_semantic_frame(source, schema)
        candidate_frame = derive_semantic_frame(candidate, schema)
    except Exception as exc:  # noqa: BLE001
        return _reject(
            "role",
            "counterfactual_not_compiler_valid",
            f"{type(exc).__name__}: {exc}",
        )
    inventory_match = _component_inventory(source_frame) == _component_inventory(
        candidate_frame
    )
    topology_differs = _topology(source_frame) != _topology(candidate_frame)
    if not inventory_match:
        return _reject(
            "role",
            "multi_dimension_diff",
            "candidate changes the component inventory, not only topology",
        )
    if not topology_differs:
        return _reject(
            "role",
            "no_dimension_diff",
            "candidate shares the source topology",
        )
    return TopologyNegativeV1(
        kind="inventory_matched_topology",
        inventory_match=True,
        topology_differs=True,
        source_target_sha256=_sha(source_frame.canonical_source),
        negative_target_sha256=_sha(candidate_frame.canonical_source),
        source_target=source_frame.canonical_source,
        negative_target=candidate_frame.canonical_source,
    )


# --------------------------------------------------------------------------
# Constant / frequency-only baseline (must sit at chance)
# --------------------------------------------------------------------------


def constant_baseline_accuracy(pairs: Sequence[CounterfactualPairV1]) -> float:
    """Accuracy of a constant/frequency-only labeler over pair sides.

    Every pair contributes exactly one source-labeled and one
    counterfactual-labeled target, so the best constant/frequency-only
    prediction (always the majority label) is exactly chance: 0.5. Any model
    claim above this floor must come from the declared semantic dimension,
    not from label frequency.
    """
    if not pairs:
        raise CounterfactualPairError("baseline requires at least one pair")
    # Each pair contributes exactly one source-labeled and one
    # counterfactual-labeled side: the majority (best constant) label covers
    # exactly half the sides.
    return 0.5


__all__ = [
    "DEFAULT_STYLE",
    "FACT_DIMENSIONS",
    "PROMPT_LENGTH_RATIO_TOLERANCE",
    "REJECTION_CODES",
    "SCHEMA_VERSION",
    "SINGLETON_CHILD_ALTERNATIVES",
    "SURFACE_CLOSE_JACCARD",
    "CounterfactualPairError",
    "CounterfactualPairV1",
    "CounterfactualRejectionV1",
    "FactDimension",
    "MutationSpecV1",
    "OneFactProofV1",
    "SurfaceCloseNegativeV1",
    "TopologyNegativeV1",
    "constant_baseline_accuracy",
    "check_pair_prompt_controls",
    "dimension_projection",
    "embedding_close_negative",
    "generate_counterfactual_pair",
    "marker_permutation",
    "mutate_program",
    "one_fact_proof",
    "pair_artifact_nodes",
    "permute_marker_surfaces",
    "prompt_length_ratio",
    "root_family_for",
    "topology_negative",
]
