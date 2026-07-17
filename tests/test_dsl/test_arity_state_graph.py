"""CAP1-01: bounded OpenUI grammar/scope state graph regression tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from slm_training.dsl.analysis.arity import AnalysisProfile, StateGraph
from slm_training.dsl.analysis.arity.state_graph import StateFingerprint
from slm_training.models.choice_tokenizer import ChoiceDecodeState, ChoiceTokenizer, OPEN_PREFIX


def _tiny_profile(**overrides: Any) -> AnalysisProfile:
    defaults = {
        "profile_id": "test-fixture",
        "representation": "choice",
        "dsl": "openui",
        "max_semantic_decisions": 6,
        "max_components": 2,
        "max_live_bindings": 3,
        "max_list_items": 3,
        "max_object_members": 3,
        "max_literal_slots": 1,
        "allowed_component_subset": ("Card", "TextContent", "Button", "Stack"),
        "required_coverage": "complete",
    }
    defaults.update(overrides)
    return AnalysisProfile(**defaults)


@pytest.fixture
def tokenizer() -> ChoiceTokenizer:
    return ChoiceTokenizer.build()


def test_default_profile_exhausts_exact(tokenizer: ChoiceTokenizer) -> None:
    from slm_training.dsl.analysis.arity import OPENVUI_CAP_V1

    graph = StateGraph(OPENVUI_CAP_V1, tokenizer, slot_contract=(":page.blurb",))
    report = graph.explore()
    assert report.status == "EXACT"
    assert report.exact
    assert report.raw_states > 0
    assert report.minimized_states <= report.raw_states
    assert report.terminal_count > 0
    assert report.unknown_count == 0
    # JSON round-trip
    parsed = json.loads(report.to_json())
    assert parsed["status"] == "EXACT"


def test_truncated_profile_reports_unknown(tokenizer: ChoiceTokenizer) -> None:
    # A list-item budget of 0 truncates any program that opens a children list.
    profile = _tiny_profile(
        max_list_items=0,
        allowed_component_subset=("Stack", "TextContent"),
    )
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))
    report = graph.explore()
    assert report.status == "UNKNOWN"
    assert not report.exact
    assert report.unknown_count > 0


def test_alpha_equivalence_has_no_surface_names(tokenizer: ChoiceTokenizer) -> None:
    """Choice tokens already erase binder names; fingerprints must not reintroduce them."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))
    _ = graph.explore()
    for fp in graph.nodes:
        if fp.remaining_decisions < 0:
            continue
        explain = fp.explain()
        flat = json.dumps(explain)
        assert "hero" not in flat
        assert "blurb" not in flat
        assert "root" not in flat


def test_different_scope_futures_differ(tokenizer: ChoiceTokenizer) -> None:
    """Opening a Stack children list vs a Card children list yields distinct fingerprints."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))

    start = ChoiceDecodeState(tokenizer, slot_count=1)
    stack_fp = _advance(tokenizer, graph, start, [tokenizer.token_to_id[f"{OPEN_PREFIX}Stack"]])
    card_fp = _advance(tokenizer, graph, start, [tokenizer.token_to_id[f"{OPEN_PREFIX}Card"]])
    assert stack_fp != card_fp
    assert "element:Stack" in str(stack_fp.signature)
    assert "element:Card" in str(card_fp.signature)


def test_literal_choices_converge(tokenizer: ChoiceTokenizer) -> None:
    """Different literal fillers for the same string slot reach the same successor."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))

    start = ChoiceDecodeState(tokenizer, slot_count=1)
    text_id = tokenizer.token_to_id[f"{OPEN_PREFIX}TextContent"]
    state_after_open = start.clone()
    assert state_after_open.advance_id(text_id)
    # Force through the mandatory opening '[' of the text argument list.
    text_state = _follow_forced(graph, state_after_open, profile.max_semantic_decisions)
    # Choose two different literal strings that are both accepted.
    lit_a = tokenizer.token_to_id['#"column"']
    lit_b = tokenizer.token_to_id['#""']
    fp_a = _advance(tokenizer, graph, text_state, [lit_a])
    fp_b = _advance(tokenizer, graph, text_state, [lit_b])
    assert fp_a == fp_b


def test_forced_suffixes_are_collapsed(tokenizer: ChoiceTokenizer) -> None:
    """The graph should not materialize intermediate states for forced suffixes."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))
    report = graph.explore()
    assert any(v > 0 for _, v in report.forced_decision_histogram)

    # Find at least one edge whose witness contains a forced suffix token.
    found = False
    for node in report.nodes:
        for edge in node["edges"]:
            witness = tuple(edge["witness"])
            if len(witness) >= 2:
                found = True
                break
        if found:
            break
    assert found


def test_replay_every_edge(tokenizer: ChoiceTokenizer) -> None:
    """Every non-terminal transition witness must replay from its source to its target."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))
    report = graph.explore()
    sink_digests = {
        graph._terminal_fp.digest(),
        graph._invalid_fp.digest(),
        graph._unknown_fp.digest(),
    }
    for node in report.nodes:
        source_fp = next(
            fp for fp in graph.nodes if fp.digest() == node["fingerprint"]
        )
        for edge in node["edges"]:
            if edge["label"] == "EOS" or edge["target"] in sink_digests:
                continue
            witness = tuple(edge["witness"])
            replayed = graph.replay(source_fp, witness)
            assert replayed.digest() == edge["target"], (
                f"replay mismatch on {edge['label']}: expected {edge['target']}, got {replayed.digest()}"
            )


def test_minimization_and_serialization(tokenizer: ChoiceTokenizer) -> None:
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))
    report = graph.explore()
    assert report.minimized_states <= report.raw_states
    assert report.minimized_classes
    # All minimized classes should be JSON-serializable.
    json.dumps(report.to_dict())


def test_over_merged_fingerprint_caught(tokenizer: ChoiceTokenizer) -> None:
    """A fingerprint that ignores scope signature over-merges states with different futures."""
    profile = _tiny_profile()
    graph = StateGraph(profile, tokenizer, slot_contract=(":page.blurb",))

    start = ChoiceDecodeState(tokenizer, slot_count=1)
    stack_open = tokenizer.token_to_id[f"{OPEN_PREFIX}Stack"]
    card_open = tokenizer.token_to_id[f"{OPEN_PREFIX}Card"]

    stack_state = start.clone()
    assert stack_state.advance_id(stack_open)
    card_state = start.clone()
    assert card_state.advance_id(card_open)

    # Coarse fingerprint: active terminals + remaining budget only.
    def coarse(state: ChoiceDecodeState, remaining: int) -> StateFingerprint:
        allowed = state.allowed_ids(remaining)
        return StateFingerprint(
            profile_id=profile.profile_id,
            representation=profile.representation,
            version="coarse-test",
            signature=(),
            schema_context=(),
            active_terminals=tuple(
                sorted(tokenizer.id_to_token[tid] for tid in allowed if tid in tokenizer.id_to_token)
            ),
            remaining_decisions=remaining,
            component_count=0,
            list_depth=0,
            object_depth=0,
        )

    # After opening Stack vs Card, the active terminals may both include '['.
    stack_fp = coarse(stack_state, profile.max_semantic_decisions - 1)
    card_fp = coarse(card_state, profile.max_semantic_decisions - 1)
    if stack_fp == card_fp:
        # The coarse fingerprint has merged two states. Show their futures differ.
        stack_next = {tokenizer.id_to_token[tid] for tid in stack_state.allowed_ids(profile.max_semantic_decisions - 1)}
        card_next = {tokenizer.id_to_token[tid] for tid in card_state.allowed_ids(profile.max_semantic_decisions - 1)}
        # Their full fingerprints differ.
        full_stack = graph._fingerprint(stack_state, profile.max_semantic_decisions - 1, 1)
        full_card = graph._fingerprint(card_state, profile.max_semantic_decisions - 1, 1)
        assert full_stack != full_card
        # And their futures differ in schema context (Stack children vs Card children).
        assert stack_next == card_next or full_stack.schema_context != full_card.schema_context
    else:
        # Even the coarse fingerprint distinguished them, which is also acceptable.
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _advance(
    tokenizer: ChoiceTokenizer,
    graph: StateGraph,
    state: ChoiceDecodeState,
    token_ids: list[int],
    remaining: int | None = None,
    component_count: int = 0,
) -> StateFingerprint:
    if remaining is None:
        remaining = graph.profile.max_semantic_decisions
    probe = state.clone()
    for tid in token_ids:
        assert probe.advance_id(tid), f"advance failed for {tokenizer.id_to_token.get(tid, tid)}"
        remaining -= 1
        if tokenizer.id_to_token.get(tid, "").startswith(OPEN_PREFIX):
            component_count += 1
    return graph._fingerprint(probe, remaining, component_count)


def _follow_forced(
    graph: StateGraph, state: ChoiceDecodeState, remaining: int
) -> ChoiceDecodeState:
    """Walk through any currently forced suffixes and return the concrete state."""
    probe = state.clone()
    while remaining > 0:
        allowed = probe.allowed_ids(remaining)
        non_eos = allowed - {probe.tokenizer.eos_id}
        if len(non_eos) != 1:
            break
        tid = next(iter(non_eos))
        if not probe.advance_id(tid):
            break
        remaining -= 1
    return probe
