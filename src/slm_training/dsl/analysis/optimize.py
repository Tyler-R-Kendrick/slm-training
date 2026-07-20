"""Schema-checked semantic optimizer over the statement-binding AST.

Completes the rewrites the D2 canonicalizer explicitly defers
(``dsl/canonicalize.py``: "does not elide schema defaults or flatten
containers — left to a future, schema-checked pass"):

- **elide trailing schema defaults** — drop a trailing positional prop whose
  literal equals the component's documented default (curated
  ``SCHEMA_DEFAULTS``; the schema snapshot carries almost no machine-readable
  ``default`` keys, so the table is pinned by tests against ``library_schema``).
- **drop dead bindings** — remove statements unreachable from ``root``
  (today those hard-fail the G3 reference gate, so this rewrite *rescues*
  otherwise-quarantined candidates).
- **flatten single-child Stacks** — replace a ``Stack`` that carries nothing
  but one child with that child, guarded by schema child-admissibility at
  every insertion site, a root-top-node exclusion, and a prompt-mention
  protection set.

Every rewrite is a pure function of the AST plus the frozen schema snapshot —
no RNG, no wall clock — and the result is re-emitted through the production
codec, so the output is D2-canonical by construction. Elide/dead rewrites are
certified semantics-preserving by ``semantic_fingerprint`` (a render-tree hash
with defaults filled and binder identity erased); flattening intentionally
changes that fingerprint and is certified by the schema admissibility check
plus the caller's parser re-validation instead. ``canonical_equal`` cannot
certify any of these rewrites — D2 equivalence classes deliberately exclude
them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from typing import Any

from slm_training.dsl.lang_core import ParseError, library_schema
from slm_training.dsl.production_codec import (
    emit_statement_bindings,
    parse_statement_bindings,
)

# Documented component-prop defaults. The committed schema snapshot exposes a
# machine-readable "default" only for Form.fields; Stack's defaults live in its
# description prose ('direction: ... (default "column"). gap: ... (default
# "m")'). Test-pinned against library_schema() so schema drift fails loudly.
SCHEMA_DEFAULTS: dict[tuple[str, str], Any] = {
    ("Stack", "direction"): "column",
    ("Stack", "gap"): "m",
    ("Form", "fields"): [],
}

# Quoted literals that encode layout structure, never user content.
STRUCTURAL_LITERALS = frozenset({"column", "row", "horizontal", "vertical"})

_FLATTEN_COMPONENT = "Stack"


class OptimizeError(ParseError):
    """A rewrite produced a program that failed its own safety contract."""


@dataclass(frozen=True)
class OptimizeOptions:
    elide_defaults: bool = True
    drop_dead_bindings: bool = True
    flatten_single_child: bool = True
    # Component type names the prompt explicitly requests; flattening never
    # removes an instance of a protected type (keeps the independent judge's
    # prompt_component_missing_from_output unreachable).
    protected_components: frozenset[str] = frozenset()


@dataclass(frozen=True)
class OptimizeResult:
    source: str
    rewrites: dict[str, int] = field(
        default_factory=lambda: {
            "defaults_elided": 0,
            "dead_bindings_removed": 0,
            "containers_flattened": 0,
        }
    )
    flatten_opportunities: int = 0

    @property
    def changed(self) -> bool:
        return any(self.rewrites.values())


def _schema_defs() -> dict[str, Any]:
    return dict(library_schema().get("$defs") or {})


def _node_refs(node: Any) -> list[str]:
    """Every statement ref inside ``node`` (order-insensitive uses only)."""
    refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind == "ref":
            refs.append(str(value.get("name")))
            return
        if kind == "element":
            for item in (value.get("props") or {}).values():
                walk(item)
            return
        if kind == "call":
            for item in value.get("args") or []:
                walk(item)

    walk(node)
    return refs


def _reachable(bindings: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    stack = ["root"]
    while stack:
        name = stack.pop()
        if name in seen or name not in bindings:
            continue
        seen.add(name)
        stack.extend(_node_refs(bindings[name]))
    return seen


def _effective_type(bindings: dict[str, Any], node: Any) -> str | None:
    """Component type a node renders as, resolving refs through statements."""
    seen: set[str] = set()
    while isinstance(node, dict) and node.get("type") == "ref":
        name = str(node.get("name"))
        if name in seen or name not in bindings:
            return None
        seen.add(name)
        node = bindings[name]
    if isinstance(node, dict) and node.get("type") in {"element", "call"}:
        return str(node.get("typeName") or node.get("name") or "") or None
    return None


def _admissible_child_types(
    defs: dict[str, Any], component: str, prop: str
) -> frozenset[str] | None:
    """Allowed child component types for ``component.prop``; None = any."""
    spec = ((defs.get(component) or {}).get("properties") or {}).get(prop) or {}
    items = spec.get("items")
    if not isinstance(items, dict) or not items:
        return None
    variants = items.get("anyOf")
    if variants is None:
        variants = [items] if "$ref" in items else None
    if variants is None:
        # Typed scalar arrays (labels, values, tags) admit no components.
        return frozenset()
    allowed = {
        str(variant.get("$ref", "")).rsplit("/", 1)[-1]
        for variant in variants
        if isinstance(variant, dict) and variant.get("$ref")
    }
    return frozenset(name for name in allowed if name)


def semantic_fingerprint(source: str, *, dsl: str | None = None) -> str:
    """Render-tree hash: defaults filled, binder identity erased, style kept out.

    Two programs share a fingerprint iff they denote the same rendered tree
    after every ``SCHEMA_DEFAULTS`` entry is made explicit — so eliding a
    default or renaming/reordering statements never changes it, while any
    structural rewrite (including container flattening) does. Dead bindings
    are invisible to it: only the root-reachable tree is hashed. Reads the
    binding AST without the official policy check so it can also fingerprint
    pre-templatization sources.
    """
    bindings = parse_statement_bindings(source, dsl=dsl, validate=False)

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        kind = node.get("type")
        if kind == "ref":
            name = str(node.get("name"))
            if name in seen or name not in bindings:
                raise ParseError(f"unresolvable reference {name!r}")
            return resolve(bindings[name], seen | {name})
        if kind in {"element", "call"}:
            type_name = str(node.get("typeName") or node.get("name") or "")
            props = dict(node.get("props") or {})
            filled = {
                key: resolve(value, seen)
                for key, value in props.items()
                if key != "_args"
            }
            for (component, prop), default in SCHEMA_DEFAULTS.items():
                if component == type_name and prop not in filled:
                    filled[prop] = default
            rendered: dict[str, Any] = {"component": type_name, "props": filled}
            extra = props.get("_args")
            if extra:
                rendered["extra_args"] = resolve(list(extra), seen)
            return rendered
        return node

    tree = resolve(bindings["root"], frozenset({"root"}))
    payload = json.dumps(tree, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _copy(node: Any) -> Any:
    if isinstance(node, list):
        return [_copy(item) for item in node]
    if isinstance(node, dict):
        return {key: _copy(value) for key, value in node.items()}
    return node


def _rewrite_pass(
    bindings: dict[str, Any],
    options: OptimizeOptions,
    defs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], int]:
    """One deterministic rewrite sweep; returns (bindings, counters, opportunities)."""
    counters = {
        "defaults_elided": 0,
        "dead_bindings_removed": 0,
        "containers_flattened": 0,
    }
    work = {name: _copy(node) for name, node in bindings.items()}

    if options.drop_dead_bindings:
        live = _reachable(work)
        dead = [name for name in work if name not in live]
        for name in dead:
            del work[name]
        counters["dead_bindings_removed"] = len(dead)

    def elide_defaults(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                elide_defaults(item)
            return
        if not isinstance(node, dict) or node.get("type") != "element":
            return
        type_name = str(node.get("typeName") or "")
        props = node.get("props") or {}
        for value in props.values():
            elide_defaults(value)
        if props.get("_args"):
            return
        prop_names = [key for key in props if key != "_args"]
        # Trailing-only: eliding a non-final prop would shift positional
        # emission (or force an explicit null placeholder into the surface).
        from slm_training.dsl.production_codec import _prop_order

        order = list(_prop_order(None).get(type_name) or [])
        present = [order.index(key) for key in prop_names if key in order]
        if not present:
            return
        last_prop = order[max(present)]
        if (type_name, last_prop) not in SCHEMA_DEFAULTS:
            return
        spec = ((defs.get(type_name) or {}).get("properties") or {}).get(last_prop)
        if spec is None:
            return
        if props.get(last_prop) == SCHEMA_DEFAULTS[(type_name, last_prop)]:
            del node["props"][last_prop]
            counters["defaults_elided"] += 1
            # The next-outer prop may now be trailing-default too.
            elide_defaults(node)

    if options.elide_defaults:
        for name in list(work):
            elide_defaults(work[name])

    opportunities = 0

    def flatten_shape(node: Any) -> Any | None:
        """The single child if ``node`` is a bare one-child Stack, else None."""
        if not isinstance(node, dict) or node.get("type") != "element":
            return None
        if str(node.get("typeName") or "") != _FLATTEN_COMPONENT:
            return None
        props = {k: v for k, v in (node.get("props") or {}).items() if k != "_args"}
        if (node.get("props") or {}).get("_args"):
            return None
        if set(props) != {"children"}:
            return None
        children = props["children"]
        if not isinstance(children, list) or len(children) != 1:
            return None
        return children[0]

    def flatten(node: Any, parent: tuple[str, str] | None) -> Any:
        """Bottom-up flatten; ``parent`` is the (component, prop) slot context.

        A statement's top node has no slot context (``parent is None``) — its
        flatten decision needs the reference sites and is taken by
        ``flatten_statement`` instead.
        """
        nonlocal opportunities
        if isinstance(node, list):
            return [flatten(item, parent) for item in node]
        if not isinstance(node, dict) or node.get("type") != "element":
            return node
        type_name = str(node.get("typeName") or "")
        props = node.get("props") or {}
        node["props"] = {
            key: (value if key == "_args" else flatten(value, (type_name, key)))
            for key, value in props.items()
        }
        if parent is None:
            return node
        child = flatten_shape(node)
        if child is None:
            return node
        opportunities += 1
        if not options.flatten_single_child:
            return node
        if _FLATTEN_COMPONENT in options.protected_components:
            return node
        child_type = _effective_type(work, child)
        if child_type is None:
            return node
        allowed = _admissible_child_types(defs, *parent)
        if allowed is not None and child_type not in allowed:
            return node
        counters["containers_flattened"] += 1
        return child

    def redirect_refs(node: Any, source_name: str, target_name: str) -> None:
        if isinstance(node, list):
            for item in node:
                redirect_refs(item, source_name, target_name)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "ref":
            if str(node.get("name")) == source_name:
                node["name"] = target_name
            return
        if node.get("type") == "element":
            for item in (node.get("props") or {}).values():
                redirect_refs(item, source_name, target_name)
            return
        if node.get("type") == "call":
            for item in node.get("args") or []:
                redirect_refs(item, source_name, target_name)

    def flatten_statement(name: str) -> None:
        nonlocal opportunities
        work[name] = flatten(work[name], None)
        if name == "root":
            # The document root's own top node always survives.
            return
        child = flatten_shape(work[name])
        if child is None:
            return
        opportunities += 1
        if not options.flatten_single_child:
            return
        if _FLATTEN_COMPONENT in options.protected_components:
            return
        child_type = _effective_type(work, child)
        # Every reference site must admit the promoted child.
        sites = [
            site
            for other, other_node in work.items()
            if other != name
            for site in _ref_sites(other_node, name)
        ]
        if child_type is None or not sites:
            return
        for parent_type, prop in sites:
            allowed = _admissible_child_types(defs, parent_type, prop)
            if allowed is not None and child_type not in allowed:
                return
        counters["containers_flattened"] += 1
        if isinstance(child, dict) and child.get("type") == "ref":
            # Promoting a lone reference would leave an alias statement;
            # redirect the sites to the child's target and drop the wrapper.
            target = str(child.get("name"))
            for other in work:
                if other != name:
                    redirect_refs(work[other], name, target)
            del work[name]
        else:
            work[name] = child

    def _ref_sites(node: Any, target: str) -> list[tuple[str, str]]:
        sites: list[tuple[str, str]] = []

        def walk(value: Any, context: tuple[str, str] | None) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item, context)
                return
            if not isinstance(value, dict):
                return
            if value.get("type") == "ref":
                if str(value.get("name")) == target and context is not None:
                    sites.append(context)
                return
            if value.get("type") == "element":
                type_name = str(value.get("typeName") or "")
                for key, item in (value.get("props") or {}).items():
                    if key == "_args":
                        continue
                    walk(item, (type_name, key))

        walk(node, None)
        return sites

    for name in list(work):
        if name in work:
            flatten_statement(name)

    return work, counters, opportunities


def optimize(
    source: str,
    *,
    options: OptimizeOptions | None = None,
    dsl: str | None = None,
    validate: bool = True,
) -> OptimizeResult:
    """Apply the schema-checked rewrites and return the canonical result.

    Raises :class:`OptimizeError` when a rewrite violates its own contract
    (semantic fingerprint drift without flattening, or a non-idempotent
    sweep); callers treat that as "leave the input unchanged".

    ``validate=False`` skips the official-parser (policy) check on the input
    so the pass can run ahead of literal → placeholder templatization; the
    sanitize orchestration always re-validates its final output officially.
    """
    opts = options or OptimizeOptions()
    defs = _schema_defs()
    bindings = parse_statement_bindings(source, dsl=dsl, validate=validate)
    fingerprint_before = semantic_fingerprint(source, dsl=dsl)

    # Sweep to fixpoint: flattening can orphan wrappers that only the next
    # dead-binding/elision sweep sees. Bounded — each sweep strictly shrinks
    # the AST or terminates.
    counters = {
        "defaults_elided": 0,
        "dead_bindings_removed": 0,
        "containers_flattened": 0,
    }
    opportunities = 0
    rewritten = bindings
    for sweep in range(8):
        rewritten, sweep_counters, sweep_opportunities = _rewrite_pass(
            rewritten, opts, defs
        )
        if sweep == 0:
            opportunities = sweep_opportunities
        if not any(sweep_counters.values()):
            break
        for key, value in sweep_counters.items():
            counters[key] += value
    else:
        raise OptimizeError("optimize failed to reach a fixpoint in 8 sweeps")
    emitted = emit_statement_bindings(rewritten, dsl=dsl)

    # Re-emission stability: emitting the fixpoint AST twice must agree
    # (string-level idempotence is additionally asserted by the tests).
    if emit_statement_bindings(rewritten, dsl=dsl) != emitted:
        raise OptimizeError("optimize re-emission is unstable")

    if counters["containers_flattened"] == 0:
        fingerprint_after = semantic_fingerprint(emitted, dsl=dsl)
        if fingerprint_after != fingerprint_before:
            raise OptimizeError(
                "semantic fingerprint changed without flattening "
                f"({fingerprint_before[:12]} -> {fingerprint_after[:12]})"
            )

    return OptimizeResult(
        source=emitted,
        rewrites=counters,
        flatten_opportunities=opportunities,
    )


__all__ = [
    "SCHEMA_DEFAULTS",
    "STRUCTURAL_LITERALS",
    "OptimizeError",
    "OptimizeOptions",
    "OptimizeResult",
    "optimize",
    "semantic_fingerprint",
]
