"""SLM-299 (LAR1-03): edit-space reachability analyzer for the real X22 space.

Answers one honest question per suite record: can the **actual** X22
``TreeEditSpace`` (STOP / kind-preserving REPLACE / leaf-only ADD bound to the
prompt slot inventory / leaf-only REMOVE — see
``src/slm_training/models/tree_edit_diffusion.py``) connect the standard X22
minimal seed to this record's gold program?

Three verdicts, and only three:

- ``PROVEN_REACHABLE`` — bounded BFS found an edit path; the depth is a
  certified edit lower bound.
- ``PROVEN_UNREACHABLE`` — a structural invariant over the exact action set
  fired (or the frontier provably died), with a machine-readable reason code.
- ``UNKNOWN_BUDGET`` — search stopped on budget with a live frontier. A budget
  stop is **never** evidence of unreachability and must never be counted as
  unreachable downstream.

No logits, no model, no gold-time decode shortcuts: transitions are applied
through the real ``TreeEditSpace.apply`` (re-validated by the DSL parser), so
the analyzed space is the deployed space by construction.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from slm_training.dsl.parser import validate
from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_ADD_CONTAINER,
    ACTION_BIND_PLACEHOLDER,
    ACTION_INSERT_STATEMENT,
    ACTION_INSERT_SUBTREE,
    ACTION_REMOVE,
    ACTION_REMOVE_CONTAINER,
    ACTION_REPLACE,
    ACTION_REPLACE_STATEMENT,
    ACTION_REPLACE_SUBTREE,
    CONTAINER_COMPONENTS,
    CONTAINER_RESTS,
    LEAF_COMPONENTS,
    MAX_SLOTS,
    V05_TEMPLATES,
    Edit,
    Statement,
    TreeEditSpace,
    parse_statements,
    render_statements,
    v05_template_index,
)

__all__ = [
    "DEFAULT_SEED_SOURCE",
    "ExtraAction",
    "ReachabilityCase",
    "Verdict",
    "add_container_action",
    "analyze_reachability",
    "component_widen_action",
    "set_property_action",
]

EXPERIMENT_ID = "slm299-edit-reachability"

# The standard X22 minimal seed — identical to
# ``slm155_factorization_comparison._MINIMAL_SEED_SOURCE`` (and to the
# fallback candidate in ``TreeEditDiffusionModel._seed_state``).
DEFAULT_SEED_SOURCE = 'root = Stack([], "column")'

# Reason codes for PROVEN_UNREACHABLE / UNKNOWN_BUDGET.
REASON_REACHED = "reached"
REASON_NEEDS_CONTAINER_ADD = "needs_container_add"
REASON_NEEDS_CONTAINER_REMOVE = "needs_container_remove"
REASON_NEEDS_SLOT_REBIND = "needs_slot_rebind"
REASON_NEEDS_DIRECTION_CHANGE = "needs_direction_change"
REASON_UNSUPPORTED_PACK_FEATURE = "unsupported_pack_feature"
REASON_UNSUPPORTED_COMPONENT = "unsupported_component"
REASON_INVALID_TARGET = "invalid_target"
REASON_FRONTIER_EXHAUSTED = "frontier_exhausted"
REASON_BUDGET = "budget"

# V0.5 pack statement forms (state/query/mutation/action). These lines can
# match the `name = Comp(...)` shape but are pack features the X22 statement
# space does not model; the marker prefixes mirror topology_adapter.V05_MARKERS
# and are only treated as pack syntax in front of a pack builtin component.
_V05_LINE_RE = re.compile(
    r"^\s*(?:r|\$|q|m|a)\s*=\s*(?:Query|Mutation|Action|State|Resource)\b"
)
_V05_TEXT_MARKERS = ("!v0.5", "!fragment", "<bos>")


class Verdict(str, Enum):
    """Proof status of one seed→target reachability query."""

    PROVEN_REACHABLE = "PROVEN_REACHABLE"
    PROVEN_UNREACHABLE = "PROVEN_UNREACHABLE"
    UNKNOWN_BUDGET = "UNKNOWN_BUDGET"


@dataclass
class ExtraAction:
    """A *hypothetical* transition generator, clearly separated from the real
    X22 action set. Used for what-if fixtures only (e.g. a synthetic
    ``ADD_CONTAINER``); never part of production reachability claims.

    ``capabilities`` names the structural gaps the action closes (e.g.
    ``container_add``); invariants whose gap is closed by an extra action are
    skipped so the what-if search can run.
    """

    name: str
    generate: Callable[[list[Statement], list[str]], list[tuple[list[Statement], dict[str, Any]]]]
    capabilities: frozenset[str] = frozenset()


def add_container_action(
    components: Sequence[str] = CONTAINER_COMPONENTS,
) -> ExtraAction:
    """Synthetic ``ADD_CONTAINER``: append a fresh empty container statement
    and reference it from an existing container. Retired what-if lane: the
    extended (SLM-305) space has the real ``ACTION_ADD_CONTAINER``, so this
    synthetic action is only meaningful in ``mode="v1"`` analyses.
    """

    def _generate(
        statements: list[Statement], inventory: list[str]
    ) -> list[tuple[list[Statement], dict[str, Any]]]:
        del inventory
        space = _shared_space()
        out: list[tuple[list[Statement], dict[str, Any]]] = []
        for parent in statements:
            if not parent.has_list:
                continue
            for comp in components:
                working = [
                    Statement(s.name, s.comp, list(s.children), s.rest, s.has_list)
                    for s in statements
                ]
                name = space.fresh_name(working)
                target_parent = next(s for s in working if s.name == parent.name)
                target_parent.children.append(name)
                working.append(
                    Statement(
                        name=name,
                        comp=comp,
                        children=[],
                        rest=', "column"',
                        has_list=True,
                    )
                )
                rendered = render_statements(working)
                try:
                    validate(rendered)
                except Exception:  # noqa: BLE001
                    continue
                out.append(
                    (
                        working,
                        {"action": "ADD_CONTAINER", "parent": parent.name, "comp": comp},
                    )
                )
        return out

    return ExtraAction(
        name="ADD_CONTAINER", generate=_generate, capabilities=frozenset({"container_add"})
    )


# Candidate rest values for the set_property what-if: the grammar's full
# direction-property domain, wider than CONTAINER_RESTS (which only lists
# what the real container-*minting* actions currently produce — see VAR0-03).
# Both are grammar-legal (checked against the real parser below regardless).
_HYPOTHETICAL_PROPERTY_DOMAIN: tuple[str, ...] = (', "column"', ', "row"', "")


def set_property_action(
    rests: Sequence[str] = _HYPOTHETICAL_PROPERTY_DOMAIN,
) -> ExtraAction:
    """VAR1-01 hypothetical: rebind an existing container's ``rest`` (its
    enum/direction argument) in place, without minting or removing any
    statement. No real action in the deployed edit space edits ``rest`` on an
    existing node — see ``_check_invariants``'s ``needs_direction_change``
    reason. This is a *what-if* probe of a property-mutation action class,
    never a production action; see VAR1-01 (SLM-424).
    """

    def _generate(
        statements: list[Statement], inventory: list[str]
    ) -> list[tuple[list[Statement], dict[str, Any]]]:
        del inventory
        out: list[tuple[list[Statement], dict[str, Any]]] = []
        for idx, stmt in enumerate(statements):
            if not stmt.has_list:
                continue
            for rest in rests:
                if rest == stmt.rest:
                    continue
                working = [
                    Statement(s.name, s.comp, list(s.children), s.rest, s.has_list)
                    for s in statements
                ]
                working[idx] = Statement(
                    stmt.name, stmt.comp, list(stmt.children), rest, stmt.has_list
                )
                rendered = render_statements(working)
                try:
                    validate(rendered)
                except Exception:  # noqa: BLE001
                    continue
                out.append(
                    (working, {"action": "SET_PROPERTY", "stmt": stmt.name, "rest": rest})
                )
        return out

    return ExtraAction(
        name="SET_PROPERTY", generate=_generate, capabilities=frozenset({"set_property"})
    )


def component_widen_action(components: Sequence[str]) -> ExtraAction:
    """VAR1-01 hypothetical: replace an existing statement's component with
    one of ``components`` — names the *analyzed target itself* already uses
    (so they are provably grammar-legal; the target parsed and validated) but
    that ``LEAF_COMPONENTS``/``CONTAINER_COMPONENTS`` excludes. Bounded to the
    specific record under analysis, never an open-ended vocabulary widening,
    and every candidate is re-validated through the real parser exactly like
    a production REPLACE. See VAR1-01 (SLM-424).
    """

    def _generate(
        statements: list[Statement], inventory: list[str]
    ) -> list[tuple[list[Statement], dict[str, Any]]]:
        del inventory
        out: list[tuple[list[Statement], dict[str, Any]]] = []
        for idx, stmt in enumerate(statements):
            for comp in components:
                if comp == stmt.comp:
                    continue
                working = [
                    Statement(s.name, s.comp, list(s.children), s.rest, s.has_list)
                    for s in statements
                ]
                working[idx] = Statement(
                    stmt.name, comp, list(stmt.children), stmt.rest, stmt.has_list
                )
                rendered = render_statements(working)
                try:
                    validate(rendered)
                except Exception:  # noqa: BLE001
                    continue
                out.append(
                    (
                        working,
                        {"action": "COMPONENT_WIDEN", "stmt": stmt.name, "comp": comp},
                    )
                )
        return out

    return ExtraAction(
        name="COMPONENT_WIDEN",
        generate=_generate,
        capabilities=frozenset({"component_widen"}),
    )


@dataclass
class ReachabilityCase:
    """One analyzed seed→target pair."""

    verdict: Verdict
    reason_code: str
    edit_lower_bound: int | None = None
    path: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason_code": self.reason_code,
            "edit_lower_bound": self.edit_lower_bound,
            "path": self.path,
            "details": self.details,
        }


_SHARED_SPACE: TreeEditSpace | None = None


def _shared_space() -> TreeEditSpace:
    global _SHARED_SPACE
    if _SHARED_SPACE is None:
        _SHARED_SPACE = TreeEditSpace()
    return _SHARED_SPACE


def _normalize_inventory(slot_inventory: Sequence[str]) -> list[str]:
    inventory: list[str] = []
    for slot in slot_inventory:
        value = str(slot)
        if not value.startswith(":"):
            value = f":{value}"
        if value not in inventory:
            inventory.append(value)
    return inventory[:MAX_SLOTS]


def _is_unsupported_pack_feature(
    target_source: str, *, extended: bool = False
) -> bool:
    if any(marker in target_source for marker in _V05_TEXT_MARKERS):
        return True
    for line in target_source.splitlines():
        line = line.strip()
        if line and _V05_LINE_RE.match(line):
            if not extended:
                return True
            # SLM-305: the extended space models V0.5 statements through the
            # canonical-template statement actions; a V0.5 line that does not
            # instantiate a canonical template is still unsupported.
            stmt = parse_statements(line)
            if stmt is None or len(stmt) != 1 or v05_template_index(stmt[0]) is None:
                return True
    return False


def _leaf_slot(rest: str) -> str | None:
    """Placeholder string bound by a leaf statement, or None when the arg is
    not a JSON string literal."""
    body = rest.strip()
    if not body.startswith('"'):
        return None
    try:
        value = json.loads(body)
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, str) else None


def _canonical_key(statements: list[Statement]) -> str:
    """Alpha-invariant canonical form: statement names are request-local
    ordinals (the edit space cannot rename), so reachability is judged up to
    renaming. Names are canonicalized by traversal order from ``root``."""
    by_name = {s.name: s for s in statements}
    order: list[str] = []
    seen: set[str] = set()

    def _visit(name: str) -> None:
        if name in seen or name not in by_name:
            return
        seen.add(name)
        order.append(name)
        for child in by_name[name].children:
            _visit(child)

    _visit("root")
    for stmt in statements:  # unreferenced leftovers, in declaration order
        _visit(stmt.name)
    rename = {name: f"k{i}" for i, name in enumerate(order)}
    lines = []
    for name in order:
        stmt = by_name[name]
        children = ",".join(rename.get(c, c) for c in stmt.children)
        lines.append(
            f"{rename[name]}={stmt.comp}({children}|{stmt.rest}|{int(stmt.has_list)})"
        )
    return "\n".join(lines)


def _check_invariants(
    target: list[Statement],
    seed: list[Statement],
    inventory: list[str],
    space: TreeEditSpace,
    capabilities: frozenset[str] = frozenset(),
    *,
    extended: bool = False,
) -> str | None:
    """Structural impossibility proofs over the EXACT action set.

    v1 space: REPLACE preserves container-ness, arity, and the container's
    enum/direction arg; ADD creates leaves only and binds only inventory
    slots; REMOVE deletes leaves only. Extended (SLM-305) space: ADD_CONTAINER
    / INSERT_SUBTREE mint containers carrying one of ``CONTAINER_RESTS`` and
    bind only inventory slots, REPLACE_SUBTREE / BIND_PLACEHOLDER rebind
    leaves only to
    inventory slots, and the V0.5 statement actions mint only canonical
    templates — the invariant reasons fire only when the corresponding REAL
    action of the analyzed mode is absent.
    """
    seed_containers = [s for s in seed if s.has_list]
    target_containers = [s for s in target if s.has_list]
    # V0.5 canonical-template statements are produced by the statement actions
    # in the extended space; they are not UI leaves/containers.
    target_leaves = [
        s
        for s in target
        if not s.has_list and not (extended and v05_template_index(s) is not None)
    ]

    # No action creates a container in the v1 space (ADD is leaf-only); the
    # extended space has the real ADD_CONTAINER / INSERT_SUBTREE.
    if (
        len(target_containers) > len(seed_containers)
        and "container_add" not in capabilities
    ):
        return REASON_NEEDS_CONTAINER_ADD
    # No action removes a container (REMOVE is leaf-only); REMOVE_CONTAINER
    # can only remove non-root containers, and both seed and target keep root,
    # so this invariant holds in every mode.
    if len(target_containers) < len(seed_containers):
        return REASON_NEEDS_CONTAINER_REMOVE

    known = set(space.components)
    if "component_widen" not in capabilities:
        for stmt in target_containers:
            if stmt.comp not in CONTAINER_COMPONENTS or stmt.comp not in known:
                return REASON_UNSUPPORTED_COMPONENT
        for stmt in target_leaves:
            if stmt.comp not in LEAF_COMPONENTS or stmt.comp not in known:
                return REASON_UNSUPPORTED_COMPONENT

    # ADD / INSERT_SUBTREE / REPLACE_SUBTREE / BIND_PLACEHOLDER bind only
    # inventory slots, and the seed carries no leaves. So every target leaf
    # slot must come from the prompt inventory in every mode.
    for stmt in target_leaves:
        slot = _leaf_slot(stmt.rest)
        if slot is None:
            return REASON_NEEDS_SLOT_REBIND
        normalized = slot if slot.startswith(":") else f":{slot}"
        if normalized not in inventory:
            return REASON_NEEDS_SLOT_REBIND

    # REPLACE preserves the container's raw enum/direction arg text (rest);
    # no real action edits it. Containers minted by the container-creating
    # actions carry exactly one of CONTAINER_RESTS. Every target container
    # rest must therefore be a seed rest or (with container_add) a candidate
    # mint rest; without any container-creating action the multisets must
    # match exactly.
    if "set_property" in capabilities:
        # A rest-mutation action can produce any target rest on any
        # container, root included -- skip both rest checks below entirely.
        return None

    seed_rests = sorted(s.rest for s in seed_containers)
    target_rests = sorted(s.rest for s in target_containers)
    # Root can never be removed or re-minted, and REPLACE preserves rest:
    # the target root's rest must equal the seed root's rest in every mode.
    seed_root_rest = next(
        (s.rest for s in seed_containers if s.name == "root"), None
    )
    for stmt in target_containers:
        if stmt.name == "root" and stmt.rest != seed_root_rest:
            return REASON_NEEDS_DIRECTION_CHANGE
    if "container_add" in capabilities:
        allowed = set(seed_rests) | set(CONTAINER_RESTS)
        if any(rest not in allowed for rest in target_rests):
            return REASON_NEEDS_DIRECTION_CHANGE
    elif seed_rests != target_rests:
        return REASON_NEEDS_DIRECTION_CHANGE

    return None


def _enumerate_children(
    space: TreeEditSpace,
    statements: list[Statement],
    inventory: list[str],
    *,
    mode: str = "extended",
    visited: set[str] | None = None,
) -> list[tuple[list[Statement], dict[str, Any]]]:
    """All one-edit successors under the REAL action set of ``mode``, applied
    through ``TreeEditSpace.apply`` so preconditions and parser re-validation
    are the deployed ones by construction. ``v1`` enumerates the original
    REPLACE/ADD/REMOVE set; ``extended`` (SLM-305) adds the container,
    subtree, V0.5-statement, and placeholder-binding actions. Deterministic:
    same state and mode always yield the same enumeration order.

    ``visited`` (canonical-key set) is a pure search-efficiency hook: states
    already keyed are rejected pre-validation inside ``apply``; it never
    changes WHICH distinct states are reachable."""
    children: list[tuple[list[Statement], dict[str, Any]]] = []
    n_comp = len(space.components)
    n_slots = min(len(inventory), MAX_SLOTS)
    leaf_comp_idxs = [
        i for i, c in enumerate(space.components) if c in LEAF_COMPONENTS
    ]
    container_comp_idxs = [
        i for i, c in enumerate(space.components) if c in CONTAINER_COMPONENTS
    ]
    pre = None
    if visited is not None:
        pre = lambda working: _canonical_key(working) not in visited  # noqa: E731
    by_name = {s.name: s for s in statements}

    def _replace_subtree_ok(stmt: Statement) -> bool:
        if not stmt.has_list or len(stmt.children) != 1:
            return False
        leaf = by_name.get(stmt.children[0])
        return leaf is not None and not leaf.has_list

    for stmt_idx in range(len(statements)):
        stmt = statements[stmt_idx]
        for comp_idx in range(n_comp):
            edit = Edit(ACTION_REPLACE, stmt_idx, comp_idx)
            nxt = space.apply(statements, edit, inventory, pre)
            if nxt is not None:
                children.append(
                    (
                        nxt,
                        {
                            "action": "REPLACE",
                            "stmt": stmt_idx,
                            "comp": space.components[comp_idx],
                        },
                    )
                )
            for slot_idx in range(n_slots):
                edit = Edit(ACTION_ADD, stmt_idx, comp_idx, slot_idx)
                nxt = space.apply(statements, edit, inventory, pre)
                if nxt is not None:
                    children.append(
                        (
                            nxt,
                            {
                                "action": "ADD",
                                "stmt": stmt_idx,
                                "comp": space.components[comp_idx],
                                "slot": inventory[slot_idx],
                            },
                        )
                    )
        if mode == "v1":
            edit = Edit(ACTION_REMOVE, stmt_idx)
            nxt = space.apply(statements, edit, inventory, pre)
            if nxt is not None:
                children.append((nxt, {"action": "REMOVE", "stmt": stmt_idx}))
            continue
        # SLM-305 extended real actions.
        if stmt.has_list:
            for comp_idx in container_comp_idxs:
                for rest_idx in range(len(CONTAINER_RESTS)):
                    edit = Edit(ACTION_ADD_CONTAINER, stmt_idx, comp_idx,
                                target=rest_idx)
                    nxt = space.apply(statements, edit, inventory, pre)
                    if nxt is not None:
                        children.append(
                            (
                                nxt,
                                {
                                    "action": "ADD_CONTAINER",
                                    "stmt": stmt_idx,
                                    "comp": space.components[comp_idx],
                                    "rest": CONTAINER_RESTS[rest_idx],
                                },
                            )
                        )
            for comp_idx in container_comp_idxs:
                for slot_idx in range(n_slots):
                    for payload in leaf_comp_idxs:
                        for rest_idx in range(len(CONTAINER_RESTS)):
                            edit = Edit(
                                ACTION_INSERT_SUBTREE, stmt_idx, comp_idx, slot_idx,
                                target=rest_idx, payload=payload,
                            )
                            nxt = space.apply(statements, edit, inventory, pre)
                            if nxt is not None:
                                children.append(
                                    (
                                        nxt,
                                        {
                                            "action": "INSERT_SUBTREE",
                                            "stmt": stmt_idx,
                                            "comp": space.components[comp_idx],
                                            "slot": inventory[slot_idx],
                                            "leaf_comp": space.components[payload],
                                            "rest": CONTAINER_RESTS[rest_idx],
                                        },
                                    )
                                )
            if _replace_subtree_ok(stmt):
                for slot_idx in range(n_slots):
                    for payload in leaf_comp_idxs:
                        edit = Edit(
                            ACTION_REPLACE_SUBTREE, stmt_idx, slot=slot_idx,
                            payload=payload,
                        )
                        nxt = space.apply(statements, edit, inventory, pre)
                        if nxt is not None:
                            children.append(
                                (
                                    nxt,
                                    {
                                        "action": "REPLACE_SUBTREE",
                                        "stmt": stmt_idx,
                                        "slot": inventory[slot_idx],
                                        "leaf_comp": space.components[payload],
                                    },
                                )
                            )
        else:
            for slot_idx in range(n_slots):
                edit = Edit(ACTION_BIND_PLACEHOLDER, stmt_idx, slot=slot_idx)
                nxt = space.apply(statements, edit, inventory, pre)
                if nxt is not None:
                    children.append(
                        (
                            nxt,
                            {
                                "action": "BIND_PLACEHOLDER",
                                "stmt": stmt_idx,
                                "slot": inventory[slot_idx],
                            },
                        )
                    )
        edit = Edit(ACTION_REMOVE_CONTAINER, stmt_idx)
        nxt = space.apply(statements, edit, inventory, pre)
        if nxt is not None:
            children.append(
                (nxt, {"action": "REMOVE_CONTAINER", "stmt": stmt_idx})
            )
        edit = Edit(ACTION_REMOVE, stmt_idx)
        nxt = space.apply(statements, edit, inventory, pre)
        if nxt is not None:
            children.append((nxt, {"action": "REMOVE", "stmt": stmt_idx}))
        for payload in range(len(V05_TEMPLATES)):
            edit = Edit(ACTION_REPLACE_STATEMENT, stmt_idx, payload=payload)
            nxt = space.apply(statements, edit, inventory, pre)
            if nxt is not None:
                children.append(
                    (
                        nxt,
                        {
                            "action": "REPLACE_STATEMENT",
                            "stmt": stmt_idx,
                            "template": payload,
                        },
                    )
                )
    if mode != "v1":
        for payload in range(len(V05_TEMPLATES)):
            edit = Edit(ACTION_INSERT_STATEMENT, payload=payload)
            nxt = space.apply(statements, edit, inventory, pre)
            if nxt is not None:
                children.append(
                    (
                        nxt,
                        {"action": "INSERT_STATEMENT", "template": payload},
                    )
                )
    return children


def analyze_reachability(
    seed_source: str,
    target_source: str,
    *,
    slot_inventory: Sequence[str],
    max_edits: int = 8,
    extra_actions: Sequence[ExtraAction] = (),
    node_budget: int = 800,
    mode: str = "extended",
) -> ReachabilityCase:
    """Prove (or honestly fail to prove) reachability of ``target_source``
    from ``seed_source`` under the real tree-edit space of ``mode``.

    ``mode="v1"`` analyzes the original SLM-299 action set (REPLACE / ADD /
    REMOVE); ``mode="extended"`` (default, SLM-305) analyzes the deployed
    extended edit language. ``extra_actions`` are hypothetical transitions
    (what-if analysis only); when any appear on a found path the case is
    marked in ``details`` so the proof is never confused with the real space.
    """
    if mode not in {"v1", "extended"}:
        raise ValueError(f"unknown reachability mode {mode!r}")
    space = _shared_space()
    inventory = _normalize_inventory(slot_inventory)

    seed = parse_statements(seed_source)
    if seed is None:
        raise ValueError(f"seed source does not parse structurally: {seed_source!r}")
    try:
        validate(seed_source)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"seed source is not a valid program: {exc}") from exc

    details: dict[str, Any] = {
        "max_edits": max_edits,
        "node_budget": node_budget,
        "inventory_size": len(inventory),
        "extra_actions": [a.name for a in extra_actions],
        "mode": mode,
    }

    if _is_unsupported_pack_feature(target_source, extended=(mode == "extended")):
        return ReachabilityCase(
            verdict=Verdict.PROVEN_UNREACHABLE,
            reason_code=REASON_UNSUPPORTED_PACK_FEATURE,
            details=details,
        )
    target = parse_statements(target_source)
    if target is None:
        return ReachabilityCase(
            verdict=Verdict.PROVEN_UNREACHABLE,
            reason_code=REASON_UNSUPPORTED_PACK_FEATURE,
            details={**details, "parse": "non_statement_form"},
        )
    try:
        validate(target_source)
    except Exception:  # noqa: BLE001
        return ReachabilityCase(
            verdict=Verdict.PROVEN_UNREACHABLE,
            reason_code=REASON_INVALID_TARGET,
            details=details,
        )

    capabilities = frozenset().union(
        *(a.capabilities for a in extra_actions)
    ) if extra_actions else frozenset()
    if mode == "extended":
        # ADD_CONTAINER / INSERT_SUBTREE are REAL actions in the extended
        # space, so the container_add invariant must not fire; the synthetic
        # add_container_action what-if lane is retired for this space.
        capabilities = capabilities | frozenset({"container_add"})
    fired = _check_invariants(
        target, seed, inventory, space, capabilities, extended=(mode == "extended")
    )
    if fired is not None:
        return ReachabilityCase(
            verdict=Verdict.PROVEN_UNREACHABLE,
            reason_code=fired,
            details=details,
        )

    # Bounded BFS over exact transitions. Depth = certified edit lower bound.
    # States are keyed by alpha-invariant canonical form (the space cannot
    # rename statements; names are request-local ordinals).
    seed_key = _canonical_key(seed)
    target_key = _canonical_key(target)
    if seed_key == target_key:
        return ReachabilityCase(
            verdict=Verdict.PROVEN_REACHABLE,
            reason_code=REASON_REACHED,
            edit_lower_bound=0,
            path=[],
            details=details,
        )

    visited = {seed_key}
    frontier: deque[tuple[list[Statement], int, list[dict[str, Any]]]] = deque(
        [(seed, 0, [])]
    )
    expanded = 0
    live_at_budget = False
    while frontier:
        statements, depth, path = frontier.popleft()
        if depth >= max_edits:
            live_at_budget = True
            continue
        children = _enumerate_children(
            space, statements, inventory, mode=mode, visited=visited
        )
        for extra in extra_actions:
            for nxt, action in extra.generate(statements, inventory):
                children.append((nxt, {**action, "synthetic": True}))
        for nxt, action in children:
            key = _canonical_key(nxt)
            if key in visited:
                continue
            visited.add(key)
            next_path = [*path, action]
            if key == target_key:
                synthetic = any(step.get("synthetic") for step in next_path)
                return ReachabilityCase(
                    verdict=Verdict.PROVEN_REACHABLE,
                    reason_code=REASON_REACHED,
                    edit_lower_bound=len(next_path),
                    path=next_path,
                    details={
                        **details,
                        "states_visited": len(visited),
                        "uses_synthetic_actions": synthetic,
                    },
                )
            frontier.append((nxt, depth + 1, next_path))
        expanded += 1
        if expanded >= node_budget:
            return ReachabilityCase(
                verdict=Verdict.UNKNOWN_BUDGET,
                reason_code=REASON_BUDGET,
                details={
                    **details,
                    "states_visited": len(visited),
                    "nodes_expanded": expanded,
                    "stop": "node_budget",
                },
            )

    if live_at_budget:
        # Frontier still had unexpanded states when the depth budget cut it.
        return ReachabilityCase(
            verdict=Verdict.UNKNOWN_BUDGET,
            reason_code=REASON_BUDGET,
            details={
                **details,
                "states_visited": len(visited),
                "nodes_expanded": expanded,
                "stop": "depth_budget",
            },
        )
    # Every state within the edit budget was expanded and none matched; the
    # frontier provably died inside MAX_STMTS-bounded space.
    return ReachabilityCase(
        verdict=Verdict.PROVEN_UNREACHABLE,
        reason_code=REASON_FRONTIER_EXHAUSTED,
        details={**details, "states_visited": len(visited), "nodes_expanded": expanded},
    )
