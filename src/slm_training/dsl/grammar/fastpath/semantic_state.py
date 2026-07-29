"""Interned, transition-updated semantic state for OpenUI completion.

Replaces per-build full-prefix scans (``compiler_draft._binder_scope``,
``_binder_dependencies``, ``_active_component_stack``,
``_active_array_direct_references``, ``_active_call``,
``_completed_call_string_values``, ``emitted_component_count``, literal-frame
tracking) with an immutable :class:`SemanticState` advanced one token at a
time.  Every fact that can influence a future constraint decision is part of
the interning key: binder declaration/reference/unresolved masks, the active
declaration, binder→component types, the binder dependency graph (direct
edges plus transitive reachability), the delimiter/call stack with per-call
argument index/count and slot-started flags, array occupancy and direct
references, the pending binder/component, the literal frame kind and body,
completed structural-id string values per role, used content-slot symbols,
the emitted component count, schema-propagated forward requirements, and the
first-appearance declaration/reference orders (the binder inventory admits
the first unresolved reference, so order is a future-relevant fact).
Two prefixes that differ in any of these never merge; prefixes that differ
only in irrelevant tokens (specials, comments, whitespace) return the
identical interned state object.

Token semantics are derived from the live tokenizer (``kind_of``,
``bind_slot_of``, ``sym_slot_of``, ``id_to_token``) — no duplicated vocab
tables.  Bounded pools (binder slots, content slots) are Python ``int``
bitsets so candidate filtering is mask intersection/difference.  Binder
cycle checks are O(1) reachability probes against incrementally maintained
transitive-closure bitsets — never a fresh graph search over the prefix.

This module is observational: it derives exactly the facts the scan-based
reference helpers derive, and ``tests/test_dsl/test_semantic_state.py``
proves prefix-by-prefix parity against them.  ScopeEnv
(``dsl/scope_env.py``) remains the semantic authority; nothing here changes
its serialized form or fingerprints.

Storage policy: production callers own a request-local :class:`SemanticArena`.
Dropping the completion session therefore releases every prefix state at
once.  The compatibility functions use a weak-value arena: live callers still
receive identity convergence, but arbitrary prefixes are never retained by a
process-global strong-reference cache.
"""

from __future__ import annotations

import itertools
import weakref
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "ArrayFrame",
    "BraceFrame",
    "CallFrame",
    "SemanticState",
    "SemanticArena",
    "active_array_direct_references",
    "active_array_is_empty",
    "active_array_position",
    "active_call",
    "active_component_stack",
    "active_declaration_slot",
    "active_list_occupancy",
    "advance",
    "binder_component_types",
    "binder_scope",
    "binder_would_cycle",
    "completed_string_values",
    "SCOPE_COUNTERS",
    "declaration_order",
    "declaration_slots",
    "dependency_edges",
    "emitted_component_count",
    "forward_binder_requirements",
    "initial_state",
    "literal_frame",
    "literal_frame_is_open",
    "reference_order",
    "reference_slots",
    "references_resolved",
    "root_is_declared",
    "schema_array_item_components",
    "schema_call_arity",
    "schema_slot_components",
    "schema_slot_name",
    "schema_slot_type",
    "scope_counters",
    "unresolved_slots",
    "used_sym_mask",
]

class SemanticArena:
    """Request-owned interner for immutable semantic states."""

    def __init__(self) -> None:
        self._states: dict[SemanticState, SemanticState] = {}

    def intern(self, state: SemanticState) -> SemanticState:
        cached = self._states.get(state)
        if cached is not None:
            return cached
        self._states[state] = state
        return state

    def clear(self) -> None:
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)


# Compatibility callers which do not yet own a request session still get
# value interning while a state is live, without retaining arbitrary prefixes.
_WEAK_ARENA: dict[int, list[weakref.ReferenceType[SemanticState]]] = {}


def _weak_intern(state: SemanticState) -> SemanticState:
    state_hash = hash(state)
    bucket = _WEAK_ARENA.get(state_hash, [])
    live: list[weakref.ReferenceType[SemanticState]] = []
    for reference in bucket:
        cached = reference()
        if cached is None:
            continue
        live.append(reference)
        if cached == state:
            if len(live) != len(bucket):
                _WEAK_ARENA[state_hash] = live
            return cached

    def _drop(
        dead: weakref.ReferenceType[SemanticState], *, key: int = state_hash
    ) -> None:
        remaining = [
            reference
            for reference in _WEAK_ARENA.get(key, ())
            if reference is not dead and reference() is not None
        ]
        if remaining:
            _WEAK_ARENA[key] = remaining
        else:
            _WEAK_ARENA.pop(key, None)

    live.append(weakref.ref(state, _drop))
    _WEAK_ARENA[state_hash] = live
    return state

# Process-wide scope instrumentation (Phase 6 aggregates across modules).
# ``scope_bitset_updates``: bitset mutations applied by ``advance`` (binder
# declaration/reference masks, dependency edges, array direct refs, used
# content-slot mask).  ``scope_reference_scans_avoided``: full-prefix rescans
# the interned state made unnecessary (bumped by the forest builder views in
# ``compiler_draft``).
SCOPE_COUNTERS: dict[str, int] = {
    "scope_bitset_updates": 0,
    "scope_reference_scans_avoided": 0,
}


def scope_counters() -> dict[str, int]:
    """Snapshot of the process-wide scope counters."""
    return dict(SCOPE_COUNTERS)

# Request-local tokenizer ordinals keep states from different tokenizers from
# ever merging while staying stable within one build.  Ordinals come from a
# monotonic process counter — never ``len(...)`` — so a GC'd weak entry or a
# flushed strong entry can never cause an ordinal to be reused by a different
# tokenizer (that would false-merge interned states across tokenizers).
_TOKENIZER_ORDINALS: dict[int, tuple[Any, int]] = {}
_TOKENIZER_ORDINALS_CAP = 4096
_TOKENIZER_WEAK: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ORDINAL_COUNTER = itertools.count()


def _next_ordinal() -> int:
    return next(_ORDINAL_COUNTER)


def _tokenizer_key(tokenizer: Any) -> int:
    try:
        ordinal = _TOKENIZER_WEAK.get(tokenizer)
        if ordinal is None:
            ordinal = _next_ordinal()
            _TOKENIZER_WEAK[tokenizer] = ordinal
        return int(ordinal)
    except TypeError:  # not weak-referenceable — fall back to identity
        entry = _TOKENIZER_ORDINALS.get(id(tokenizer))
        if entry is None or entry[0] is not tokenizer:
            if len(_TOKENIZER_ORDINALS) >= _TOKENIZER_ORDINALS_CAP:
                # Eviction drops strong references and loses sharing only;
                # returning tokenizers receive fresh, never-reused ordinals.
                _TOKENIZER_ORDINALS.clear()
            entry = (tokenizer, _next_ordinal())
            _TOKENIZER_ORDINALS[id(tokenizer)] = entry
        return int(entry[1])


@dataclass(frozen=True)
class CallFrame:
    """One open component call: positional progress and string args.

    ``separators`` is the depth-0 COMMA count (= current argument index);
    ``started`` marks whether the current argument has consumed any token, so
    ``arg_count = separators + int(started)`` exactly mirrors
    ``compiler_draft._active_call``.  ``cur_str``/``cur_is_str`` carry the
    in-progress bare-string argument; completed (index, value) pairs collect
    in ``str_args`` until the frame closes.
    """

    component: str | None
    separators: int = 0
    started: bool = False
    str_args: tuple[tuple[int, str], ...] = ()
    cur_str: str | None = None
    cur_is_str: bool = False


@dataclass(frozen=True)
class ArrayFrame:
    """One open ``[`` frame: occupancy, item position, direct binder refs."""

    separators: int = 0
    started: bool = False
    direct_refs: int = 0  # binder-slot bitmask


@dataclass(frozen=True)
class BraceFrame:
    """One open ``{`` frame (object literal); consumes depth-0 commas."""

    started: bool = False


Frame = CallFrame | ArrayFrame | BraceFrame


@dataclass(frozen=True)
class SemanticState:
    """Immutable, interned semantic summary of an OpenUI token prefix."""

    tokenizer_key: int = 0
    declared_mask: int = 0  # binder slots with a completed ``bind =`` head
    referenced_mask: int = 0  # binder slots used as references (deduped)
    active_slot: int = -1  # binder slot of the active declaration, -1 = none
    edges: tuple[int, ...] = ()  # direct declaration → reference edge masks
    reachable: tuple[int, ...] = ()  # transitive closure of ``edges``
    binder_types: tuple[tuple[int, str], ...] = ()  # slot → component name
    stack: tuple[Frame, ...] = ()  # delimiter/call frames, outermost first
    pending_component: str | None = None  # component token awaiting ``(``
    pending_bind: int = -1  # bind token awaiting ``=`` disambiguation
    pending_allowed: tuple[str, ...] | None = None  # use-site components
    expect_decl_component: bool = False  # last tokens were ``bind =``
    emitted_components: int = 0
    used_sym_mask: int = 0  # content-slot bitmask of consumed SYM symbols
    completed_str: tuple[tuple[str, int, str], ...] = ()  # (component, arg, value)
    literal_kind: str = ""  # "", "LIT_STR" or "LIT_NUM"
    literal_body: str = ""
    requirements: tuple[tuple[int, tuple[str, ...]], ...] = ()
    schema_tracked: bool = False
    opaque_bind: bool = False  # a slot-less bind token was seen (conservative)
    decl_order: tuple[int, ...] = ()  # binder slots, first-declaration order
    ref_order: tuple[int, ...] = ()  # binder slots, first-reference order

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.reachable):
            raise ValueError("edges and reachable must have equal length")


def _intern(
    state: SemanticState, arena: SemanticArena | None = None
) -> SemanticState:
    if arena is not None:
        return arena.intern(state)
    return _weak_intern(state)


def initial_state(
    tokenizer: Any, *, arena: SemanticArena | None = None
) -> SemanticState:
    """Return the interned empty-prefix state bound to ``tokenizer``."""
    return _intern(
        SemanticState(tokenizer_key=_tokenizer_key(tokenizer)), arena
    )


def _kind_of(tokenizer: Any, token_id: int) -> str:
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        try:
            return str(getattr(kind_of(token_id), "value", ""))
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _extend(bits: tuple[int, ...], size: int) -> tuple[int, ...]:
    if len(bits) >= size:
        return bits
    return (*bits, *((0,) * (size - len(bits))))


def _add_edge(
    edges: tuple[int, ...], reachable: tuple[int, ...], source: int, target: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Add ``source → target`` and re-close reachability incrementally."""
    size = max(len(edges), source + 1, target + 1)
    edge_list = list(_extend(edges, size))
    reach_list = list(_extend(reachable, size))
    edge_list[source] |= 1 << target
    delta = reach_list[target] | (1 << target)
    if reach_list[source] & (1 << target) and reach_list[source] & delta == delta:
        return tuple(edge_list), tuple(reach_list)
    reach_list[source] |= delta
    # Fixpoint propagation; bounded pools (≤ 64 slots) keep this tiny, and
    # valid decode never adds cycle-closing edges (guarded by would_cycle).
    changed = True
    while changed:
        changed = False
        for slot in range(size):
            closure = reach_list[slot]
            widened = closure
            pending = closure
            while pending:
                low = pending & -pending
                bit = low.bit_length() - 1
                pending ^= low
                widened |= reach_list[bit]
            if widened != closure:
                reach_list[slot] = widened
                changed = True
    return tuple(edge_list), tuple(reach_list)


def _set_binder_type(
    types: tuple[tuple[int, str], ...], slot: int, component: str
) -> tuple[tuple[int, str], ...]:
    kept = tuple((s, name) for s, name in types if s != slot)
    return tuple(sorted((*kept, (slot, component))))


def _intersect_requirement(
    requirements: tuple[tuple[int, tuple[str, ...]], ...],
    slot: int,
    allowed: frozenset[str],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    merged: dict[int, frozenset[str]] = {
        s: frozenset(names) for s, names in requirements
    }
    prior = merged.get(slot)
    merged[slot] = allowed if prior is None else prior & allowed
    return tuple(
        (s, tuple(sorted(names)))
        for s, names in sorted(merged.items())
        if names
    )


def _commit_string_arg(frame: CallFrame) -> CallFrame:
    if frame.cur_is_str and frame.cur_str is not None:
        entry = (frame.separators, frame.cur_str)
        if entry not in frame.str_args:
            frame = replace(frame, str_args=(*frame.str_args, entry))
    return replace(frame, cur_str=None, cur_is_str=False)


def _mark_started(stack: tuple[Frame, ...]) -> tuple[Frame, ...]:
    """Mark the innermost frame's current item as consumed."""
    if not stack:
        return stack
    top = stack[-1]
    if isinstance(top, CallFrame):
        # A second content token in the same argument voids bare-string status.
        top = replace(top, started=True, cur_is_str=top.cur_is_str and not top.started)
    else:
        top = replace(top, started=True)
    return (*stack[:-1], top)


def _context_components(
    stack: tuple[Frame, ...], schema: dict[str, Any]
) -> frozenset[str]:
    """Schema component names admitted at the current binder use-site."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        _schema_component_refs,
    )

    call = next(
        (frame for frame in reversed(stack) if isinstance(frame, CallFrame)),
        None,
    )
    if call is None or call.component is None:
        return frozenset()
    definition = (schema.get("$defs") or {}).get(call.component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if call.separators >= len(names):
        return frozenset()
    value = properties.get(names[call.separators]) or {}
    top = stack[-1] if stack else None
    if isinstance(top, ArrayFrame) and not top.started:
        items = value.get("items") or {}
        return _schema_component_refs(items, schema) if isinstance(items, dict) else frozenset()
    if value.get("type") == "array":
        return frozenset()
    return _schema_component_refs(value, schema)


def advance(
    state: SemanticState,
    token_id: int,
    tokenizer: Any,
    schema: dict[str, Any] | None = None,
    *,
    arena: SemanticArena | None = None,
) -> SemanticState:
    """Advance ``state`` by one token; pure, interned, no-op-stable.

    Tokens with no semantic effect (specials, comments, inline whitespace,
    stray literal bytes outside a frame) return the *identical* state object.
    ``schema`` (optional) enables forward binder-component requirement
    propagation at binder use-sites; pass it consistently within a build.
    """
    tid = int(token_id)
    if state.tokenizer_key != _tokenizer_key(tokenizer):
        raise ValueError("semantic state is bound to a different tokenizer")
    if _kind_of(tokenizer, tid) == "macro":
        expand = getattr(tokenizer, "expand_macros", None)
        expanded = list(expand([tid])) if callable(expand) else []
        # Orphaned/recursive macros are not semantic authority.  The parser
        # transition rejects them; this layer remains a no-op.
        if not expanded or expanded == [tid]:
            return state
        current = state
        for sub_id in expanded:
            current = advance(
                current,
                int(sub_id),
                tokenizer,
                schema=schema,
                arena=arena,
            )
        return current
    specials = {
        int(getattr(tokenizer, name))
        for name in ("pad_id", "bos_id", "eos_id", "mask_id", "unk_id")
        if getattr(tokenizer, name, None) is not None
    }
    if tid in specials:
        return state
    raw = str(getattr(tokenizer, "id_to_token", {}).get(tid, ""))
    if raw in {"", "COMMENT", "WS_INLINE"}:
        return state
    kind = _kind_of(tokenizer, tid)

    declared = state.declared_mask
    referenced = state.referenced_mask
    active = state.active_slot
    edges = state.edges
    reachable = state.reachable
    binder_types = state.binder_types
    stack = state.stack
    pending_component = state.pending_component
    pending_bind = state.pending_bind
    pending_allowed = state.pending_allowed
    expect_decl = state.expect_decl_component
    emitted = state.emitted_components
    used_sym = state.used_sym_mask
    completed_str = state.completed_str
    literal_kind = state.literal_kind
    literal_body = state.literal_body
    requirements = state.requirements
    schema_tracked = state.schema_tracked or schema is not None
    opaque_bind = state.opaque_bind
    decl_order = state.decl_order
    ref_order = state.ref_order

    # Literal-frame bytes are mid-token: they extend the body, and their
    # surface character is parser-visible progress in the current argument.
    if raw.startswith("B:") and literal_kind:
        try:
            literal_body += chr(int(raw[2:], 16))
        except ValueError:
            pass
        stack = _mark_started(stack)
        new = replace(
            state, literal_body=literal_body, stack=stack, schema_tracked=schema_tracked,
            decl_order=decl_order, ref_order=ref_order,
        )
        return state if new == state else _intern(new, arena)

    # Resolve a pending binder: ``bind =`` is a declaration head, anything
    # else makes the pending binder a reference at its use-site.
    if pending_bind >= 0 and raw != "=":
        referenced |= 1 << pending_bind
        SCOPE_COUNTERS["scope_bitset_updates"] += 1
        if pending_bind not in ref_order:
            ref_order = (*ref_order, pending_bind)
        if active >= 0:
            edges, reachable = _add_edge(edges, reachable, active, pending_bind)
            SCOPE_COUNTERS["scope_bitset_updates"] += 1
        if pending_allowed:
            requirements = _intersect_requirement(
                requirements, pending_bind, frozenset(pending_allowed)
            )
        pending_bind = -1
        pending_allowed = None
    if raw == "=":
        if pending_bind >= 0:
            declared |= 1 << pending_bind
            SCOPE_COUNTERS["scope_bitset_updates"] += 1
            if pending_bind not in decl_order:
                decl_order = (*decl_order, pending_bind)
            active = pending_bind
            expect_decl = True
            pending_bind = -1
            pending_allowed = None
        new = replace(
            state,
            declared_mask=declared,
            active_slot=active,
            pending_bind=pending_bind,
            pending_allowed=pending_allowed,
            expect_decl_component=expect_decl,
            schema_tracked=schema_tracked,
            decl_order=decl_order,
            ref_order=ref_order,
        )
        return state if new == state else _intern(new, arena)

    # ``bind = <token>``: a component token certifies the declaration type.
    if expect_decl:
        expect_decl = False
        if kind == "component" and active >= 0:
            binder_types = _set_binder_type(binder_types, active, raw)

    if raw == "NL":
        new = replace(
            state,
            active_slot=-1,
            schema_tracked=schema_tracked,
            decl_order=decl_order,
            ref_order=ref_order,
        )
        return state if new == state else _intern(new, arena)
    if raw == "(":
        stack = _mark_started(stack)
        stack = (*stack, CallFrame(component=pending_component))
        pending_component = None
    elif raw == ")":
        if stack and isinstance(stack[-1], CallFrame):
            frame = _commit_string_arg(stack[-1])
            stack = stack[:-1]
            if frame.component is not None and frame.str_args:
                additions = tuple(
                    (frame.component, index, value)
                    for index, value in frame.str_args
                    if (frame.component, index, value) not in completed_str
                )
                if additions:
                    completed_str = tuple(sorted((*completed_str, *additions)))
            stack = _mark_started(stack)
        else:
            stack = _mark_started(stack)
    elif raw == "[":
        stack = _mark_started(stack)
        stack = (*stack, ArrayFrame())
    elif raw == "]":
        if stack and isinstance(stack[-1], ArrayFrame):
            stack = stack[:-1]
        stack = _mark_started(stack)
    elif raw == "{":
        stack = _mark_started(stack)
        stack = (*stack, BraceFrame())
    elif raw == "}":
        if stack and isinstance(stack[-1], BraceFrame):
            stack = stack[:-1]
        stack = _mark_started(stack)
    elif raw == ",":
        if stack:
            top = stack[-1]
            if isinstance(top, CallFrame):
                top = _commit_string_arg(top)
                top = replace(top, separators=top.separators + 1, started=False)
            elif isinstance(top, ArrayFrame):
                top = replace(top, separators=top.separators + 1, started=False)
            else:
                top = replace(top, started=False)
            stack = (*stack[:-1], top)
    elif raw in {"LIT_STR", "LIT_NUM"} and not literal_kind:
        literal_kind = raw
        literal_body = ""
        if raw == "LIT_STR" and stack and isinstance(stack[-1], CallFrame):
            top = stack[-1]
            if not top.started:
                top = replace(top, cur_is_str=True, cur_str="")
                stack = (*stack[:-1], top)
        # LIT_STR renders as a quote (parser-visible progress); LIT_NUM
        # renders as nothing until its first byte arrives.
        from slm_training.dsl.grammar.fastpath.token_map import (
            token_surface_piece,
        )

        if token_surface_piece(tokenizer, tid):
            stack = _mark_started(stack)
    elif raw == "LIT_END" and literal_kind:
        if literal_kind == "LIT_STR" and stack and isinstance(stack[-1], CallFrame):
            top = stack[-1]
            if top.cur_is_str:
                top = replace(top, cur_str=literal_body)
                stack = (*stack[:-1], top)
        literal_kind = ""
        literal_body = ""
    elif kind == "component":
        pending_component = raw
        emitted += 1
        stack = _mark_started(stack)
    elif kind == "bind":
        slot_of = getattr(tokenizer, "bind_slot_of", None)
        slot = slot_of(tid) if callable(slot_of) else None
        if slot is None:
            opaque_bind = True
        else:
            slot = int(slot)
            if stack and isinstance(stack[-1], ArrayFrame):
                top = stack[-1]
                top = replace(top, direct_refs=top.direct_refs | (1 << slot))
                SCOPE_COUNTERS["scope_bitset_updates"] += 1
                stack = (*stack[:-1], top)
            if schema is not None:
                allowed = _context_components(stack, schema)
                pending_allowed = tuple(sorted(allowed)) if allowed else None
            pending_bind = slot
        stack = _mark_started(stack)
    elif kind == "sym":
        slot_of = getattr(tokenizer, "sym_slot_of", None)
        slot = slot_of(tid) if callable(slot_of) else None
        if slot is not None:
            used_sym |= 1 << int(slot)
            SCOPE_COUNTERS["scope_bitset_updates"] += 1
        if stack and isinstance(stack[-1], CallFrame) and not stack[-1].started:
            from slm_training.dsl.grammar.fastpath.token_map import (
                token_surface_piece,
            )

            piece = token_surface_piece(tokenizer, tid)
            value = (
                piece[1:-1]
                if len(piece) >= 2 and piece.startswith('"') and piece.endswith('"')
                else piece
            )
            top = replace(stack[-1], cur_str=value, cur_is_str=True)
            stack = (*stack[:-1], top)
        stack = _mark_started(stack)
    elif kind == "lit" and raw.startswith("STR:"):
        if stack and isinstance(stack[-1], CallFrame) and not stack[-1].started:
            top = replace(stack[-1], cur_str=raw[4:], cur_is_str=True)
            stack = (*stack[:-1], top)
        stack = _mark_started(stack)
    elif raw.startswith("B:"):
        return state  # stray literal byte outside a frame: no semantic effect
    else:
        stack = _mark_started(stack)

    new = replace(
        state,
        declared_mask=declared,
        referenced_mask=referenced,
        active_slot=active,
        edges=edges,
        reachable=reachable,
        binder_types=binder_types,
        stack=stack,
        pending_component=pending_component,
        pending_bind=pending_bind,
        pending_allowed=pending_allowed,
        expect_decl_component=expect_decl,
        emitted_components=emitted,
        used_sym_mask=used_sym,
        completed_str=completed_str,
        literal_kind=literal_kind,
        literal_body=literal_body,
        requirements=requirements,
        schema_tracked=schema_tracked,
        opaque_bind=opaque_bind,
        decl_order=decl_order,
        ref_order=ref_order,
    )
    return state if new == state else _intern(new, arena)


# ---------------------------------------------------------------------------
# Derived views (reference-helper-shaped facts, all O(1)-ish over the state)
# ---------------------------------------------------------------------------


def declaration_slots(state: SemanticState) -> frozenset[int]:
    """Binder slots with a completed declaration head."""
    return frozenset(_bits(state.declared_mask))


def declaration_order(state: SemanticState) -> tuple[int, ...]:
    """Binder slots in first-declaration order (order-sensitive fact)."""
    return state.decl_order


def reference_order(state: SemanticState) -> tuple[int, ...]:
    """Binder slots in first-reference order.

    A trailing pending binder is a reference per the scan semantics, so it
    appears (last) here exactly as the prefix scan reports it.  Order is a
    future-relevant fact — the binder inventory admits the *first* unresolved
    reference — so it is part of the interning key.
    """
    order = state.ref_order
    pending = state.pending_bind
    if pending >= 0 and pending not in order:
        order = (*order, pending)
    return order


def reference_slots(state: SemanticState) -> frozenset[int]:
    """Binder slots used as references, including a trailing pending binder."""
    mask = state.referenced_mask
    if state.pending_bind >= 0:
        mask |= 1 << state.pending_bind
    return frozenset(_bits(mask))


def unresolved_slots(state: SemanticState) -> frozenset[int]:
    """Referenced binder slots with no declaration yet."""
    return frozenset(_bits((state.referenced_mask | _pending_bit(state)) & ~state.declared_mask))


def references_resolved(state: SemanticState) -> bool:
    """Whether every binder reference has a declaration (fail closed)."""
    if state.opaque_bind:
        return False
    return not unresolved_slots(state)


def active_declaration_slot(state: SemanticState) -> int | None:
    """Binder slot owning the active declaration statement, if any."""
    return None if state.active_slot < 0 else state.active_slot


def root_is_declared(state: SemanticState) -> bool:
    """Whether the root binder (slot 0) has a completed declaration head."""
    return bool(state.declared_mask & 1)


def binder_scope(state: SemanticState) -> tuple[frozenset[int], frozenset[int], int | None]:
    """Reference-shaped (declarations, references, active) slot triple."""
    return declaration_slots(state), reference_slots(state), active_declaration_slot(state)


def binder_component_types(state: SemanticState) -> dict[int, str]:
    """Binder slot → component type certified by a declaration value."""
    return dict(state.binder_types)


def dependency_edges(state: SemanticState) -> dict[int, frozenset[int]]:
    """Direct declaration → reference edges (a trailing pending binder counts
    as a reference of the active declaration, matching the prefix scan)."""
    edges = {slot: frozenset(_bits(mask)) for slot, mask in enumerate(state.edges) if mask}
    for slot in _bits(state.declared_mask):
        edges.setdefault(slot, frozenset())
    if state.pending_bind >= 0 and state.active_slot >= 0:
        prior = edges.get(state.active_slot, frozenset())
        edges[state.active_slot] = prior | {state.pending_bind}
    return edges


def binder_would_cycle(state: SemanticState, source: int, target: int) -> bool:
    """Whether adding ``source → target`` closes a declaration cycle (O(1)).

    A trailing pending binder counts as a reference of the active declaration
    (matching the prefix scan), so its edge is folded into the closure here.
    """
    if source == target:
        return True
    reachable = state.reachable
    if state.pending_bind >= 0 and state.active_slot >= 0:
        _edges, reachable = _add_edge(
            state.edges, state.reachable, state.active_slot, state.pending_bind
        )
    if target >= len(reachable):
        return False
    return bool((reachable[target] >> source) & 1)


def active_component_stack(state: SemanticState) -> tuple[str, ...]:
    """Currently open component calls, outermost first."""
    return tuple(
        frame.component
        for frame in state.stack
        if isinstance(frame, CallFrame) and frame.component is not None
    )


def active_call(state: SemanticState) -> tuple[str, int, int] | None:
    """Innermost open call as (component, arg_index, arg_count)."""
    call = next(
        (frame for frame in reversed(state.stack) if isinstance(frame, CallFrame)),
        None,
    )
    if call is None or call.component is None:
        return None
    return call.component, call.separators, call.separators + int(call.started)


def _innermost_array(state: SemanticState) -> ArrayFrame | None:
    return next(
        (frame for frame in reversed(state.stack) if isinstance(frame, ArrayFrame)),
        None,
    )


def active_array_direct_references(state: SemanticState) -> frozenset[int]:
    """Binder slots used as direct items of the innermost live array."""
    array = _innermost_array(state)
    return frozenset() if array is None else frozenset(_bits(array.direct_refs))


def active_array_is_empty(state: SemanticState) -> bool:
    """Whether the innermost open frame is a just-opened array."""
    return bool(
        state.stack
        and isinstance(state.stack[-1], ArrayFrame)
        and not state.stack[-1].started
        and state.stack[-1].separators == 0
    )


def active_array_position(state: SemanticState) -> str | None:
    """Item position of the innermost in-call array (item_start/item_end).

    Mirrors the reference: only arrays opened *after* the innermost open
    call's LPAR count — an array beneath a nested open call is not active.
    """
    above: list[Frame] = []
    for frame in reversed(state.stack):
        if isinstance(frame, CallFrame):
            break
        above.append(frame)
    else:
        return None
    array = next((f for f in above if isinstance(f, ArrayFrame)), None)
    if array is None:
        return None
    return "item_start" if not array.started else "item_end"


def active_list_occupancy(state: SemanticState) -> str | None:
    """Empty/populated state of the innermost open list frame."""
    array = _innermost_array(state)
    if array is None:
        return None
    return "empty" if not array.started and array.separators == 0 else "populated"


def emitted_component_count(state: SemanticState) -> int:
    """Component instantiations emitted so far (min-content measure)."""
    return state.emitted_components


def used_sym_mask(state: SemanticState) -> int:
    """Content-slot bitmask of consumed placeholder symbols."""
    return state.used_sym_mask


def literal_frame(state: SemanticState) -> tuple[str | None, str]:
    """Open literal frame as (kind, body); kind is None when closed."""
    return (state.literal_kind or None), state.literal_body


def literal_frame_is_open(state: SemanticState) -> bool:
    """Whether the prefix ends inside a framed literal body."""
    return bool(state.literal_kind)


def completed_string_values(
    state: SemanticState, component: str, index: int
) -> frozenset[str]:
    """Completed string values already used for one (component, arg) role."""
    return frozenset(
        value
        for name, arg_index, value in state.completed_str
        if name == component and arg_index == index
    )


def forward_binder_requirements(state: SemanticState) -> dict[int, frozenset[str]]:
    """Schema-propagated component-type requirements per referenced binder.

    A trailing pending binder is a reference per the scan semantics, so its
    use-site constraint is merged into the view exactly like a resolved one.
    """
    merged: dict[int, frozenset[str]] = {
        slot: frozenset(names) for slot, names in state.requirements
    }
    if state.pending_bind >= 0 and state.pending_allowed:
        allowed = frozenset(state.pending_allowed)
        prior = merged.get(state.pending_bind)
        merged[state.pending_bind] = allowed if prior is None else prior & allowed
    return {slot: names for slot, names in merged.items() if names}


def schema_slot_name(state: SemanticState, schema: dict[str, Any]) -> str | None:
    """Schema property name for the active positional argument."""
    call = active_call(state)
    if call is None:
        return None
    component, index, _count = call
    definition = (schema.get("$defs") or {}).get(component) or {}
    names = list(definition.get("properties") or {})
    return str(names[index]) if index < len(names) else None


def schema_slot_type(state: SemanticState, schema: dict[str, Any]) -> str | None:
    """Schema value type for the active positional argument."""
    call = active_call(state)
    if call is None:
        return None
    component, index, _count = call
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return None
    value = properties.get(names[index]) or {}
    return str(value.get("type")) if value.get("type") else None


def schema_call_arity(state: SemanticState, schema: dict[str, Any]) -> tuple[int, int, int, bool] | None:
    """(required, maximum, supplied, current-slot started) for the live call."""
    call = active_call(state)
    if call is None:
        return None
    component, index, arg_count = call
    definition = (schema.get("$defs") or {}).get(component) or {}
    names = list(definition.get("properties") or {})
    if not names:
        return None
    required = set(definition.get("required") or ())
    minimum = max((names.index(name) + 1 for name in required if name in names), default=0)
    return minimum, len(names), arg_count, arg_count > index


def schema_slot_components(state: SemanticState, schema: dict[str, Any]) -> frozenset[str]:
    """Component types admitted directly by the active positional property."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        _schema_component_refs,
    )

    call = active_call(state)
    if call is None:
        return frozenset()
    component, index, _count = call
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return frozenset()
    value = properties.get(names[index]) or {}
    if value.get("type") == "array":
        return frozenset()
    return _schema_component_refs(value, schema)


def schema_array_item_components(state: SemanticState, schema: dict[str, Any]) -> frozenset[str]:
    """Component types admitted by the active array property's items."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        _schema_component_refs,
    )

    call = active_call(state)
    if call is None:
        return frozenset()
    component, index, _count = call
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return frozenset()
    items = (properties.get(names[index]) or {}).get("items") or {}
    if not isinstance(items, dict):
        return frozenset()
    return _schema_component_refs(items, schema)


def _pending_bit(state: SemanticState) -> int:
    return (1 << state.pending_bind) if state.pending_bind >= 0 else 0


def _bits(mask: int) -> tuple[int, ...]:
    out: list[int] = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return tuple(out)
