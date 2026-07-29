"""Differential parity: incremental SemanticState vs scan-based reference helpers.

For every prefix of representative OpenUI programs, the interned,
transition-updated :class:`SemanticState` must expose exactly the facts the
executable reference semantics in ``compiler_draft`` (and ``token_map``)
derive by rescanning the whole prefix.  Reference helpers are imported and
called directly on ``prefix_ids``; Lark parser-state helpers are called on a
fresh ``OpenUIIncrementalEngine`` synced to the decoded prefix text.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath import semantic_state as ss
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    _active_array_direct_references,
    _active_array_direct_references_reference,
    _active_array_is_empty,
    _active_array_position,
    _active_call,
    _active_component_stack,
    _active_component_stack_reference,
    _active_list_occupancy,
    _binder_component_types,
    _binder_component_types_reference,
    _binder_dependencies,
    _binder_dependencies_reference,
    _binder_reference_would_cycle,
    _binder_scope,
    _binder_scope_reference,
    _completed_call_string_values,
    _forward_binder_component_requirements,
    _forward_binder_component_requirements_reference,
    _literal_frame_is_open,
    _generated_ast_is_complete,
    _literal_frame_is_open_reference,
    _references_resolved,
    _references_resolved_reference,
    _schema_array_item_components,
    _schema_call_arity,
    _schema_slot_components,
    _schema_slot_name,
    _schema_slot_type,
    emitted_component_count,
    emitted_component_count_reference,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import (
    _NUMBER_COMPLETE,
    _NUMBER_PREFIX,
    apply_literal_frame,
    decode_prefix,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, TokenKind


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


@pytest.fixture(scope="module")
def schema() -> dict:
    from slm_training.dsl import lang_core

    return lang_core.library_schema()


# Representative programs: root + multiple declarations, forward references,
# repeated references, cycles and near-cycles, nested component/list
# expressions, required + optional schema fields, typed arrays, enum values,
# newline boundaries.  The two hard cases appear both as standalone prefixes
# and as prefixes of the longer programs.
_PROGRAMS = (
    'root = TextContent(":slot_0")\n',
    "root = Card([])\n",
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\n',
    'root = Card([b1, b1])\nb1 = TextContent(":slot_0")\n',
    'root = Card([b1])\nb1 = Card([b2])\nb2 = TextContent(":slot_0")\n',
    "root = Card([b1])\nb1 = Card([b1])\n",
    "root = Card([b1])\nb1 = Card([b2])\nb2 = Card([b1])\n",
    'root = Callout("info", ":slot_0", ":slot_1", true)\n',
    'root = Card([TextContent(":slot_0"), b1])\nb1 = TextContent(":slot_1")\n',
    # Redeclaration of an already-declared binder slot.
    'root = Card([b1])\nb1 = TextContent(":slot_0")\nb1 = TextContent(":slot_1")\n',
    # Optional schema fields consumed (wrap boolean, enum variant).
    'root = Card([b1], "primary", "row", "md", "center", "between", true)\nb1 = TextContent(":slot_0", "lg")\n',
    # Nested component/list/object: object-typed rules prop, typed number arrays.
    'root = Slider(":slot_0", "range", 0, 100, 5, [1, 2], ":slot_1", {"info": 1}, [3])\n',
    # Typed array of component items (Table columns) and array of objects
    # (ImageGallery images) — brace frames inside arrays.
    'root = Table([Col(":slot_0", ":slot_1", "info")])\n',
    'root = ImageGallery([{"info": 1}, {"info": 2}])\n',
    # Near-cycle chain that stays acyclic, then one referencing the root body.
    'root = Card([b1])\nb1 = Stack([b2], "row")\nb2 = Stack([b3])\nb3 = TextContent(":slot_0")\n',
    "root = Card([b1",
    "root = Card([b1,",
    "root =",
    "root",
    "",
)


def _fold(tok: DSLNativeTokenizer, ids: list[int], schema: dict) -> ss.SemanticState:
    state = ss.initial_state(tok)
    for tid in ids:
        state = ss.advance(state, tid, tok, schema=schema)
    return state


def _slots_to_ids(tok: DSLNativeTokenizer, slots) -> set[int]:
    return {int(tok.bind_id(slot)) for slot in slots}


def _check_prefix(tok, schema, prefix: list[int], label: str) -> None:
    state = _fold(tok, prefix, schema)

    # --- binder scope: declarations / references / active declaration ---
    ref_decl, ref_refs, ref_active = _binder_scope_reference(tok, list(prefix))
    assert _slots_to_ids(tok, ss.declaration_slots(state)) == set(ref_decl), label
    assert _slots_to_ids(tok, ss.reference_slots(state)) == set(ref_refs), label
    # Order is a future-relevant fact: first-declaration order (deduped — the
    # scan appends a redeclaration twice) and first-reference order must match
    # exactly, including a trailing pending binder as the last reference.
    ref_decl_ordered = list(dict.fromkeys(int(token_id) for token_id in ref_decl))
    assert [int(tok.bind_id(s)) for s in ss.declaration_order(state)] == (
        ref_decl_ordered
    ), label
    assert [int(tok.bind_id(s)) for s in ss.reference_order(state)] == [
        int(token_id) for token_id in ref_refs
    ], label
    expected_active = None if ref_active is None else int(ref_active)
    got_active = ss.active_declaration_slot(state)
    assert (None if got_active is None else int(tok.bind_id(got_active))) == (
        expected_active
    ), label
    assert ss.references_resolved(state) == _references_resolved_reference(
        tok, list(prefix)
    ), label
    unresolved = set(ref_refs) - set(ref_decl)
    assert _slots_to_ids(tok, ss.unresolved_slots(state)) == unresolved, label
    assert ss.root_is_declared(state) == bool(int(tok.bind_id(0)) in set(ref_decl)), (
        label
    )

    # --- binder -> component types ---
    ref_types = {
        int(k): v
        for k, v in _binder_component_types_reference(tok, list(prefix)).items()
    }
    got_types = {
        int(tok.bind_id(s)): n for s, n in ss.binder_component_types(state).items()
    }
    assert got_types == ref_types, label

    # --- dependency graph + cycle relation ---
    ref_deps = {
        int(k): {int(v) for v in vs}
        for k, vs in _binder_dependencies_reference(tok, list(prefix)).items()
    }
    got_deps = {
        int(tok.bind_id(s)): {int(tok.bind_id(t)) for t in ts}
        for s, ts in ss.dependency_edges(state).items()
    }
    assert got_deps == ref_deps, label
    involved = set(ref_deps) | {v for vs in ref_deps.values() for v in vs}
    for source in involved:
        for target in involved:
            assert ss.binder_would_cycle(
                state, tok.bind_slot_of(source), tok.bind_slot_of(target)
            ) == _binder_reference_would_cycle(source, target, ref_deps), (
                f"{label} cycle {source}->{target}"
            )

    # --- component call stack / array frames / emitted count ---
    assert ss.active_component_stack(state) == _active_component_stack_reference(
        tok, list(prefix)
    ), label
    ref_array_refs = {
        int(v) for v in _active_array_direct_references_reference(tok, list(prefix))
    }
    assert _slots_to_ids(tok, ss.active_array_direct_references(state)) == (
        ref_array_refs
    ), label
    assert ss.emitted_component_count(state) == emitted_component_count_reference(
        tok, list(prefix)
    ), label

    # --- literal frame (kind, body) via the token_map candidate restriction ---
    assert ss.literal_frame_is_open(state) == _literal_frame_is_open_reference(
        tok, list(prefix)
    ), label
    kind, body = ss.literal_frame(state)
    byte_ids = set(tok.kind_ids(TokenKind.BYTE))
    closer = tok.token_to_id.get("LIT_END")
    candidates = set(tok.token_to_id.values())
    if kind == "LIT_NUM":
        numeric = {
            tid
            for tid in byte_ids
            if (raw := str(tok.id_to_token.get(tid, ""))).startswith("B:")
            and _NUMBER_PREFIX.fullmatch(body + chr(int(raw[2:], 16)))
        }
        expected = numeric | (
            {int(closer)} if _NUMBER_COMPLETE.fullmatch(body) else set()
        )
    elif kind is not None:
        expected = byte_ids | ({int(closer)} if body else set())
    else:
        expected = set(candidates) - byte_ids - {int(closer)}
    assert apply_literal_frame(tok, list(prefix), set(candidates)) == expected, label

    # --- schema-propagated forward binder requirements ---
    ref_req = {
        int(k): frozenset(v)
        for k, v in _forward_binder_component_requirements_reference(
            tok, list(prefix), schema
        ).items()
    }
    got_req = {
        int(tok.bind_id(s)): frozenset(v)
        for s, v in ss.forward_binder_requirements(state).items()
    }
    assert got_req == ref_req, label

    # --- Lark parser-state facts (skipped when the engine rejects the prefix) ---
    engine = OpenUIIncrementalEngine()
    if not engine.set_prefix(decode_prefix(tok, list(prefix))):
        return
    # Lark reduces a closed call into its value stack only with lookahead, so
    # at prefixes ending in ``)`` parser-state helpers can lag the token-exact
    # state by the trailing run of unreduced RPARs.
    trimmed = list(prefix)
    while trimmed and int(trimmed[-1]) == int(tok.token_to_id[")"]):
        trimmed.pop()
    trimmed_state = _fold(tok, trimmed, schema)
    assert ss.active_call(state) == _active_call(engine), label
    assert ss.active_array_is_empty(state) == _active_array_is_empty(engine), label
    # ``_active_array_position`` searches for brackets after the *raw last
    # LPAR* on Lark's value stack; a just-closed call's LPAR still occupies
    # the stack until the next token forces reduction, so at RPAR-final
    # prefixes the reference lags by exactly the trailing unreduced RPAR run.
    position_state = trimmed_state if len(trimmed) != len(prefix) else state
    assert ss.active_array_position(position_state) == _active_array_position(engine), (
        label
    )
    assert ss.active_list_occupancy(state) == _active_list_occupancy(engine), label
    assert ss.schema_slot_name(state, schema) == _schema_slot_name(engine, schema), (
        label
    )
    assert ss.schema_slot_type(state, schema) == _schema_slot_type(engine, schema), (
        label
    )
    assert ss.schema_call_arity(state, schema) == _schema_call_arity(engine, schema), (
        label
    )
    assert ss.schema_slot_components(state, schema) == _schema_slot_components(
        engine, schema
    ), label
    assert ss.schema_array_item_components(state, schema) == (
        _schema_array_item_components(engine, schema)
    ), label
    cursor = ss.schema_cursor(state)
    assert cursor.active_call == _active_call(engine), label
    assert cursor.active_array_empty == _active_array_is_empty(engine), label
    assert cursor.active_array_position == _active_array_position(engine), label
    assert cursor.active_list_occupancy == _active_list_occupancy(engine), label
    if "$END" in engine.next_terminals():
        assert ss.generated_ast_is_complete(state) == _generated_ast_is_complete(
            decode_prefix(tok, list(prefix))
        ), label
    # Prove the exact completed-string relationship: at RPAR-final prefixes
    # reference(P) == state(P minus trailing RPAR run) ⊆ state(P).
    for component in {name for name, _i, _v in state.completed_str} | {
        "Card",
        "TextContent",
        "Callout",
    }:
        for index in range(4):
            reference = _completed_call_string_values(engine, component, index)
            assert reference == ss.completed_string_values(
                trimmed_state, component, index
            ), f"{label} strings {component}[{index}]"
            assert reference <= ss.completed_string_values(state, component, index), (
                f"{label} strings {component}[{index}]"
            )


@pytest.mark.parametrize("program", _PROGRAMS, ids=lambda p: repr(p[-24:]))
def test_every_prefix_matches_reference(tok, schema, program: str) -> None:
    ids = [int(tid) for tid in tok.encode(program, add_special=False)]
    for end in range(len(ids) + 1):
        _check_prefix(tok, schema, ids[:end], f"{program!r}[:{end}]")


def test_framed_literal_prefixes_match_reference(tok, schema) -> None:
    """Hand-built framed-literal prefixes (encode forbids free-form strings)."""

    def byte(ch: str) -> int:
        return tok.token_to_id[f"B:{ord(ch):02x}"]

    lit_end = tok.token_to_id["LIT_END"]
    num_head = tok.encode("root = Slider(", add_special=False)
    sequences = []
    if "LIT_STR" in tok.token_to_id:
        lit_str = tok.token_to_id["LIT_STR"]
        head = tok.encode("root = TextContent(", add_special=False)
        sequences += [
            list(head) + [lit_str],
            list(head) + [lit_str, byte("a")],
            list(head) + [lit_str, byte("a"), byte("b"), lit_end],
            list(head) + [lit_str, byte("a"), lit_end, tok.token_to_id[")"]],
        ]
    if "LIT_NUM" in tok.token_to_id:
        lit_num = tok.token_to_id["LIT_NUM"]
        sequences += [
            list(num_head) + [lit_num],
            list(num_head) + [lit_num, byte("3")],
            list(num_head) + [lit_num, byte("3"), byte("."), byte("5"), lit_end],
        ]
    assert sequences
    for ids in sequences:
        for end in range(len(ids) + 1):
            _check_prefix(tok, schema, [int(t) for t in ids[:end]], f"literal[:{end}]")


def test_irrelevant_tokens_return_identical_state(tok, schema) -> None:
    ids = list(tok.encode("root = Card([b1,", add_special=False))
    state = _fold(tok, ids, schema)
    for special in (tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id, tok.unk_id):
        assert ss.advance(state, special, tok, schema=schema) is state
    unknown = max(tok.token_to_id.values()) + 10_000
    assert ss.advance(state, unknown, tok, schema=schema) is state
    # Stray literal byte outside a frame is semantically inert.
    assert ss.advance(state, tok.token_to_id["B:41"], tok, schema=schema) is state


def test_whitespace_equivalent_prefixes_converge(tok, schema) -> None:
    """Prefixes differing only in irrelevant tokens intern to one state."""
    ids = list(tok.encode("root = Card([b1])\nb1 = TextContent(", add_special=False))
    plain = _fold(tok, ids, schema)
    noisy = ss.initial_state(tok)
    for index, tid in enumerate(ids):
        noisy = ss.advance(noisy, tid, tok, schema=schema)
        if index % 3 == 0:
            noisy = ss.advance(noisy, tok.bos_id, tok, schema=schema)
            noisy = ss.advance(noisy, tok.pad_id, tok, schema=schema)
    assert noisy is plain


def test_equivalence_key_separates_semantic_differences(tok, schema) -> None:
    """States differing in authority-relevant facts must never merge."""
    open_item = _fold(
        tok, list(tok.encode("root = Card([b1", add_special=False)), schema
    )
    after_comma = _fold(
        tok, list(tok.encode("root = Card([b1,", add_special=False)), schema
    )
    assert open_item is not after_comma
    assert open_item != after_comma

    flat = _fold(
        tok,
        list(tok.encode("root = Card([b1])\nb1 = Card([b2])", add_special=False)),
        schema,
    )
    cycle = _fold(
        tok,
        list(
            tok.encode(
                "root = Card([b1])\nb1 = Card([b2])\nb2 = Card([b1]", add_special=False
            )
        ),
        schema,
    )
    assert flat is not cycle
    assert ss.binder_would_cycle(cycle, 2, 1)
    # flat already carries the edge 1→2 (b1's body references b2), so 2→1
    # would close a cycle; 0→2 would not (b2 reaches nothing yet).
    assert ss.binder_would_cycle(flat, 2, 1)
    assert not ss.binder_would_cycle(flat, 0, 2)

    opener = tok.token_to_id.get("LIT_STR") or tok.token_to_id["LIT_NUM"]
    literal = _fold(
        tok,
        list(tok.encode("root = TextContent(", add_special=False))
        + [opener, tok.token_to_id["B:61"]],
        schema,
    )
    literal2 = _fold(
        tok,
        list(tok.encode("root = TextContent(", add_special=False))
        + [opener, tok.token_to_id["B:62"]],
        schema,
    )
    assert literal is not literal2  # literal body is part of the key


def test_replaying_same_prefix_converges_to_one_interned_state(tok, schema) -> None:
    ids = list(tok.encode(_PROGRAMS[2], add_special=False))
    first = _fold(tok, ids, schema)
    second = _fold(tok, ids, schema)
    assert first is second


def test_reference_aliases_are_the_scan_oracle() -> None:
    """The ``<name>_reference`` surface IS the scan helper (one oracle copy)."""
    assert _binder_scope_reference is _binder_scope
    assert _references_resolved_reference is _references_resolved
    assert _binder_component_types_reference is _binder_component_types
    assert _binder_dependencies_reference is _binder_dependencies
    assert _forward_binder_component_requirements_reference is (
        _forward_binder_component_requirements
    )
    assert _active_component_stack_reference is _active_component_stack
    assert _active_array_direct_references_reference is _active_array_direct_references
    assert _literal_frame_is_open_reference is _literal_frame_is_open
    assert emitted_component_count_reference is emitted_component_count


def test_all_binder_slots_consumed(tok, schema) -> None:
    """Parity holds when every one of the bounded binder slots is used."""
    statement = [
        int(tid)
        for tid in tok.encode('root = TextContent(":slot_0")\n', add_special=False)
    ]
    root_id = int(tok.bind_id(0))
    assert statement[0] == root_id
    ids: list[int] = []
    for slot in range(tok.bind_slots):
        stmt = [int(tok.bind_id(slot)), *statement[1:]]
        ids.extend(stmt)
        if slot < 3 or slot >= tok.bind_slots - 2:
            # Spot-check early prefixes and the saturated tail; the full
            # per-prefix sweep over 64 statements is covered structurally by
            # the parametric programs above.
            _check_prefix(tok, schema, ids, f"all-slots[:{slot}]")
    state = _fold(tok, ids, schema)
    assert ss.declaration_slots(state) == frozenset(range(tok.bind_slots))
    assert ss.references_resolved(state)
    assert not ss.unresolved_slots(state)
    assert [int(tok.bind_id(s)) for s in ss.declaration_order(state)] == [
        int(tok.bind_id(s)) for s in range(tok.bind_slots)
    ]


def test_scope_bitset_updates_counter(tok, schema) -> None:
    before = ss.SCOPE_COUNTERS["scope_bitset_updates"]
    ids = list(
        tok.encode(
            'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\n'
            'b2 = TextContent(":slot_1")\n',
            add_special=False,
        )
    )
    state = _fold(tok, ids, schema)
    updates = ss.SCOPE_COUNTERS["scope_bitset_updates"] - before
    # 2 declarations + 2 references + 2 dependency edges + 2 used SYM slots
    # + 2 array direct refs.
    assert updates >= 8
    # Semantically inert tokens never touch a bitset.
    again = ss.SCOPE_COUNTERS["scope_bitset_updates"]
    ss.advance(state, tok.bos_id, tok, schema=schema)
    assert ss.SCOPE_COUNTERS["scope_bitset_updates"] == again


def test_production_forest_build_never_rescans_prefix(tok, schema, monkeypatch) -> None:
    """The forest builder reads SemanticState facts, not the scan oracle.

    Every prefix-rescan helper is replaced with a raising stub; a full
    production build (stateless memo front AND an explicit
    ``semantic_state`` kernel-style call) must still succeed and must record
    the avoided scans in both the request-local and module-level counters.
    """
    import slm_training.dsl.grammar.fastpath.compiler_draft as cd

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("production path rescanned the prefix")

    for name in (
        "_binder_scope",
        "_references_resolved",
        "_binder_component_types",
        "_binder_dependencies",
        "_forward_binder_component_requirements",
        "_active_component_stack",
        "_active_array_direct_references",
        "_literal_frame_is_open",
        "emitted_component_count",
    ):
        monkeypatch.setattr(cd, name, _explode)

    program = (
        'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\n'
        'b2 = TextContent(":slot_1")\n'
    )
    ids = [int(tid) for tid in tok.encode(program, add_special=False)]
    avoided_before = ss.SCOPE_COUNTERS["scope_reference_scans_avoided"]

    # 1) Stateless memo front: folds the interned state internally.
    forest = cd._build_openui_completion_forest(tok, ids[:-1])
    assert forest.paths

    # 2) Kernel-style call: caller-supplied state + request-local counter.
    state = _fold(tok, ids[:-1], schema)
    scan_counter: dict[str, int] = {"avoided": 0}
    forest2 = cd._build_openui_completion_forest_impl(
        tok,
        ids[:-1],
        semantic_state=state,
        scan_counter=scan_counter,
    )
    assert [path.token_ids for path in forest2.paths] == [
        path.token_ids for path in forest.paths
    ]
    assert scan_counter["avoided"] > 0
    assert (
        ss.SCOPE_COUNTERS["scope_reference_scans_avoided"] - avoided_before
        >= scan_counter["avoided"]
    )


def test_completion_session_threads_one_state_per_request(tok, schema) -> None:
    """pack.py's kernel path seeds once, then advances per committed token."""
    from slm_training.dsl.grammar.fastpath.completion_kernel import (
        CompletionSession,
    )

    program = 'root = Card([b1])\nb1 = TextContent(":slot_0")\n'
    ids = [int(tid) for tid in tok.encode(program, add_special=False)]
    session = CompletionSession(tok)
    seed = session.seed(ids[:-3])
    seeded = session.semantic_of(seed)
    assert seeded is _fold(tok, ids[:-3], schema)
    # Each committed token advances the SAME interned-state chain — no refold.
    current = seed
    for tid in ids[-3:]:
        nxt = session.advance(current, tid)
        assert nxt is not None
        expected = ss.advance(session.semantic_of(current), tid, tok, schema=schema)
        assert session.semantic_of(nxt) is expected
        current = nxt
    stats = session.stats()
    assert stats["unique_states"] >= 4
