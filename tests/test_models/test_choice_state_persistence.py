"""ChoiceDecodeState hot-path guards: persistent frames, schema interning,
memoized bounded completion distance.

Pins the copy-on-write frame discipline (``clone()`` shares frames; transitions
never mutate shared frames), the schema-interning identity used by the cheap
``signature()``, and proves the memoized bounded-distance
``minimal_completion_length`` equal to a brute-force completion oracle and to
the previous greedy probe behavior on reachable states.
"""

from __future__ import annotations

import random

import pytest

from slm_training.dsl.production_codec import CLOSE, LIST_CLOSE, OBJ_CLOSE
from slm_training.models.choice_tokenizer import (
    LIT_NUM,
    ChoiceDecodeState,
    ChoiceTokenizer,
    _ChoiceFrame,
    _DISTANCE_CACHE_MAX_ENTRIES,
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


@pytest.fixture
def tok() -> ChoiceTokenizer:
    return ChoiceTokenizer.build()


def _reachable_states(tok: ChoiceTokenizer, text: str, slot_count: int = 4):
    """Every prefix state of the encoded stream (post-BOS, through EOS)."""
    state = ChoiceDecodeState(tok, slot_count=slot_count)
    states = [state]
    for token_id in tok.encode(text):
        if token_id == tok.bos_id:
            continue
        nxt = states[-1].clone()
        if not nxt.advance_id(token_id):
            break
        states.append(nxt)
    return states


def _synthetic_states(tok: ChoiceTokenizer) -> list[ChoiceDecodeState]:
    """Small hand-built configurations covering every frame kind."""
    typed_array = ChoiceDecodeState(tok, slot_count=1)
    typed_array.frames.append(
        _ChoiceFrame(
            "variadic",
            "array",
            close=LIST_CLOSE,
            schemas=({"type": "string"},),
            schema_ids=(tok.intern_schema({"type": "string"}),),
        )
    )
    typed_object = ChoiceDecodeState(tok, slot_count=1)
    typed_object.frames.append(
        _ChoiceFrame(
            "object",
            "object",
            close=OBJ_CLOSE,
            phase="key",
            schemas=({"type": "string"}, {"type": "number"}),
            schema_ids=(
                tok.intern_schema({"type": "string"}),
                tok.intern_schema({"type": "number"}),
            ),
            arg_index=-1,
            property_names=("src", "alt"),
            required_properties=("src",),
            additional_properties=False,
        )
    )
    component = ChoiceDecodeState(tok, slot_count=1)
    component.frames.append(
        _ChoiceFrame(
            "component",
            "element:Synthetic",
            close=CLOSE,
            schemas=({"type": "string"}, {"type": "number"}),
            schema_ids=(
                tok.intern_schema({"type": "string"}),
                tok.intern_schema({"type": "number"}),
            ),
            required_args=1,
        )
    )
    fixed = ChoiceDecodeState(tok, slot_count=1)
    fixed.frames.append(_ChoiceFrame("fixed", "other", remaining=2))
    literal = ChoiceDecodeState(tok, slot_count=1)
    assert literal.advance_id(tok.token_to_id[LIT_NUM])
    root = ChoiceDecodeState(tok, slot_count=1)
    return [root, typed_array, typed_object, component, fixed, literal]


def _brute_force_distance(
    state: ChoiceDecodeState, room: int, memo: dict | None = None
) -> int:
    """Independent bounded-minimum oracle (EOS included).

    Plain memoized recursion over admitted actions — no greedy seed, no lower
    bound, no child ordering — so it validates the production DFS rather than
    mirroring it. ``room + 1`` when no completion fits within ``room``. The
    caller may share one ``memo`` across calls; without lower-bound pruning
    the explored graph grows exponentially in ``room``, so keep rooms small.
    """
    if memo is None:
        memo = {}

    def distance(st: ChoiceDecodeState, budget: int) -> int:
        if st.can_end():
            return 1 if budget >= 1 else budget + 1
        if budget < 1:
            return budget + 1
        key = (st.signature(), budget)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best = budget + 1
        for token_id in st._candidate_ids():
            if token_id == st.tokenizer.eos_id:
                continue
            probe = st.clone()
            if probe.advance_id(token_id):
                value = distance(probe, budget - 1)
                if 1 + value < best:
                    best = 1 + value
        memo[key] = best
        return best

    return distance(state, room)


def _greedy_completion_length(state: ChoiceDecodeState, cap: int = 1024) -> int:
    """The previous probe-clone greedy completion loop (pre-optimization).

    ``cap`` bounds the simulated steps: values above ``cap`` collapse to the
    same 1025 sentinel, which is all the feasibility comparison consumes.
    """
    probe = state.clone()
    result = 1025
    for count in range(1, min(cap, 1024) + 1):
        token_id = probe._completion_id()
        if not probe.advance_id(token_id):
            break
        if token_id == probe.tokenizer.eos_id:
            result = count
            break
    return result


def _greedy_allowed_ids(state: ChoiceDecodeState, remaining: int) -> set[int]:
    """``allowed_ids`` as computed by the previous greedy feasibility check."""
    allowed: set[int] = set()
    for token_id in state._candidate_ids():
        probe = state.clone()
        if not probe.advance_id(token_id):
            continue
        completion = (
            0
            if token_id == state.tokenizer.eos_id
            else _greedy_completion_length(probe, cap=remaining - 1)
        )
        if completion <= remaining - 1:
            allowed.add(token_id)
    return allowed


# --- persistent frames / clone -------------------------------------------


def test_clone_shares_untouched_frames(tok: ChoiceTokenizer) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    assert state.advance_id(tok.token_to_id["+Card"])
    clone = state.clone()
    assert clone.frames is not state.frames
    assert clone.frames[-1] is state.frames[-1]

    top_before = state.frames[-1]
    # A transition on the clone must copy-on-write, never touch the parent.
    assert clone.advance_id(tok.token_to_id["["])
    assert clone.frames[-1] is not top_before or len(clone.frames) != len(
        state.frames
    )
    assert state.frames[-1] is top_before
    assert state.frames[-1].arg_index == top_before.arg_index


def test_clone_advance_matches_replay_reference(tok: ChoiceTokenizer) -> None:
    """COW transitions equal from-scratch replay semantics on random walks."""
    rng = random.Random(20260728)
    for _ in range(6):
        state = ChoiceDecodeState(tok, slot_count=2)
        history: list[int] = []
        for _step in range(40):
            candidates = sorted(state._candidate_ids())
            rng.shuffle(candidates)
            advanced = False
            for token_id in candidates[:24]:
                probe = state.clone()
                if probe.advance_id(token_id):
                    state = probe
                    history.append(token_id)
                    advanced = True
                    break
            if not advanced:
                break
        replay = ChoiceDecodeState(tok, slot_count=2)
        for token_id in history:
            assert replay.advance_id(token_id)
        assert replay.signature() == state.signature()
        assert replay.allowed_ids(6) == state.allowed_ids(6)


def test_cow_counters_track_clones_and_frame_swaps(tok: ChoiceTokenizer) -> None:
    tok.clone_count = 0
    tok.frame_cow_copies = 0
    state = ChoiceDecodeState(tok, slot_count=1)
    assert state.advance_id(tok.token_to_id["+Input"])
    clone = state.clone()
    assert tok.clone_count == 1
    before = tok.frame_cow_copies
    # Completing a component argument swaps the top frame (arg_index += 1).
    assert clone.advance_id(tok.sym_id(0))
    assert tok.frame_cow_copies > before
    assert state.frames[-1].arg_index == 0


# --- schema interning ------------------------------------------------------


def test_schema_interning_never_aliases_distinct_schemas(
    tok: ChoiceTokenizer,
) -> None:
    a = tok.intern_schema({"type": "string"})
    b = tok.intern_schema({"type": "number"})
    c = tok.intern_schema({"type": "object", "properties": {"x": {"type": "string"}}})
    d = tok.intern_schema({"type": "object", "properties": {"x": {"type": "number"}}})
    e = tok.intern_schema({"enum": ["a", 1, None, True]})
    f = tok.intern_schema({"enum": ["a", 1, None, False]})
    assert len({a, b, c, d, e, f}) == 6
    # Same canonical content, fresh object: identical id.
    assert tok.intern_schema(dict({"type": "string"})) == a
    assert tok.intern_schema({"enum": ["a", 1, None, True]}) == e


def test_schema_intern_counters(tok: ChoiceTokenizer) -> None:
    tok.schema_intern.clear()
    tok.schema_intern_hits = 0
    tok.schema_intern_misses = 0
    tok.intern_schema({"type": "boolean"})
    tok.intern_schema({"type": "boolean"})
    assert tok.schema_intern_misses == 1
    assert tok.schema_intern_hits == 1


def test_signature_uses_interned_schema_identity(tok: ChoiceTokenizer) -> None:
    def with_schema(schema: dict) -> ChoiceDecodeState:
        state = ChoiceDecodeState(tok, slot_count=1)
        # Deliberately no schema_ids: exercises the signature fallback that
        # interns on demand (synthetic frames built outside advance paths).
        state.frames.append(
            _ChoiceFrame(
                "component",
                "element:Synthetic",
                close=CLOSE,
                schemas=(schema,),
                required_args=1,
            )
        )
        return state

    assert with_schema({"type": "string"}).signature() == with_schema(
        dict({"type": "string"})
    ).signature()
    assert with_schema({"type": "string"}).signature() != with_schema(
        {"type": "number"}
    ).signature()
    # Interned-at-construction and interned-on-demand agree.
    interned = ChoiceDecodeState(tok, slot_count=1)
    interned.frames.append(
        _ChoiceFrame(
            "component",
            "element:Synthetic",
            close=CLOSE,
            schemas=({"type": "string"},),
            required_args=1,
            schema_ids=(tok.intern_schema({"type": "string"}),),
        )
    )
    assert interned.signature() == with_schema({"type": "string"}).signature()


# --- bounded completion distance -------------------------------------------


def test_completion_distance_equals_brute_force(tok: ChoiceTokenizer) -> None:
    # The unpruned oracle is exponential in room over wide candidate sets
    # (every expression position branches over the full alphabet), so rooms
    # stay shallow here; feasibility at larger budgets is covered
    # end-to-end by the exhaustive-allowed oracle below.
    synthetic = _synthetic_states(tok)
    narrow = list(synthetic)
    narrow.extend(_reachable_states(tok, HERO)[1::3])
    memo: dict = {}
    for state in narrow:
        for room in (0, 1, 2):
            expected = _brute_force_distance(state.clone(), room, memo)
            assert state._completion_distance(room) == expected, (
                room,
                [(f.kind, f.expr_type, f.phase) for f in state.frames],
                state.literal_frame,
            )
            if expected <= room:
                # The pruning bound never overestimates a real completion.
                assert state._completion_lower_bound() <= expected


def test_minimal_completion_never_exceeds_greedy_probe_on_reachable_states(
    tok: ChoiceTokenizer,
) -> None:
    """The memoized true minimum is bounded above by the previous greedy
    probe loop (greedy picks the alphabetically-first open, not the cheapest,
    so it could overestimate; the new distance is never larger)."""
    states = _reachable_states(tok, HERO)[1:]
    states.extend(_reachable_states(tok, 'root = TextContent(":a")')[1:])
    for state in states:
        distance = state.minimal_completion_length()
        greedy = _greedy_completion_length(state)
        assert distance <= greedy
        assert distance == state._completion_distance(1024)


def test_allowed_ids_match_exhaustive_oracle_and_preserve_greedy_admissions(
    tok: ChoiceTokenizer,
) -> None:
    states = _synthetic_states(tok)
    states.extend(_reachable_states(tok, HERO)[1::2])
    for state in states:
        for remaining in (1, 2, 3, 4, 6, 8):
            expected = state.exhaustive_allowed_ids(remaining)
            assert state.allowed_ids(remaining) == expected
            # Result sets only grow vs the previous greedy feasibility check:
            # every token the greedy check admitted stays admitted.
            assert _greedy_allowed_ids(state, remaining) <= expected


def test_exact_distance_handles_an_active_nested_expression(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    assert state.advance_id(tok.token_to_id["+Card"])
    assert state.advance_id(tok.token_to_id["["])

    # The active list already fills Card's required first argument. Counting
    # the schema again would report five tokens instead of the exact three:
    # close list, close Card, EOS.
    assert _brute_force_distance(state.clone(), 3) == 3
    assert state._completion_lower_bound() == 3
    assert state._completion_distance(3) == 3


def test_exact_distance_corrects_legacy_greedy_false_pruning(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=0)
    root_marker = tok.token_to_id["r="]
    assert root_marker not in _greedy_allowed_ids(state, 4)
    assert root_marker in state.allowed_ids(4)

    # A four-token witness exists; the old greedy probe happened to choose a
    # more expensive component and incorrectly removed the root marker.
    witness = (
        root_marker,
        tok.token_to_id["+CardHeader"],
        tok.token_to_id[CLOSE],
        tok.eos_id,
    )
    replay = state.clone()
    assert all(replay.advance_id(token_id) for token_id in witness)


def test_distance_cache_counters(tok: ChoiceTokenizer) -> None:
    tok.distance_cache.clear()
    tok.distance_cache_hits = 0
    tok.distance_cache_misses = 0
    state = ChoiceDecodeState(tok, slot_count=1)
    assert state.advance_id(tok.token_to_id["+Input"])
    first = state._completion_distance(8)
    again = state._completion_distance(8)
    assert again == first
    assert tok.distance_cache_misses >= 1
    misses = tok.distance_cache_misses
    state._completion_distance(8)
    assert tok.distance_cache_misses == misses
    assert tok.distance_cache_hits >= 1


def test_distance_cache_is_bounded_and_counts_evictions(
    tok: ChoiceTokenizer,
) -> None:
    tok.distance_cache.clear()
    tok.distance_cache_evictions = 0
    tok.distance_cache_peak_entries = 0
    tok.distance_cache.update(
        ((("synthetic", index), 1), 2)
        for index in range(_DISTANCE_CACHE_MAX_ENTRIES)
    )
    state = ChoiceDecodeState(tok, slot_count=1)
    assert state._completion_distance(8) <= 8
    assert len(tok.distance_cache) == 1
    assert tok.distance_cache_evictions == _DISTANCE_CACHE_MAX_ENTRIES
    assert tok.distance_cache_peak_entries == 1


def test_completion_cache_still_collapses_equivalent_states(
    tok: ChoiceTokenizer,
) -> None:
    tok.completion_cache.clear()
    tok.completion_cache_hits = 0
    tok.completion_cache_misses = 0
    one = ChoiceDecodeState(tok, slot_count=1)
    two = ChoiceDecodeState(tok, slot_count=1)
    assert one.advance_id(tok.token_to_id[LIT_NUM])
    assert two.advance_id(tok.token_to_id[LIT_NUM])
    length = one.minimal_completion_length()
    assert two.minimal_completion_length() == length
    assert tok.completion_cache_misses == 1
    assert tok.completion_cache_hits == 1
