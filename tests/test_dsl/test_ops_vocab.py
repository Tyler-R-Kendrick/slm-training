"""Decode invariant I13: one reserved ops vocabulary, shared by both towers.

The requirement is not "each tower has ops" — it is that they have *the same*
ops at *the same* ids, with grammar symbols layered above and natural language
optional. These tests pin all three halves of that.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.openui_tokens import (
    TOKEN_ID_NAMESPACE_RANGES,
    logical_token_id,
)
from slm_training.dsl.ops_vocab import (
    OPS_NAMESPACE,
    OPS_VOCAB,
    OpFamily,
    OpsVocabError,
    assert_layering,
    assert_shared_across_towers,
    build_ops_vocab,
    fingerprint,
    load_registry,
    manifest,
    ops_by_family,
    shared_token_ids,
)


def test_every_op_comes_from_a_live_operator_registry() -> None:
    """Derived, not authored: no op exists here without an implementation."""
    from slm_training.dsl.operators import local, topology
    from slm_training.dsl.operators.conversation import ConversationOperation

    implemented = {
        str(declaration.operator_id)
        for module in (local, topology)
        for declaration in module._declarations()
    }
    implemented |= {f"conversation.{item.value}" for item in ConversationOperation}
    assert {symbol.op_id for symbol in OPS_VOCAB} == implemented


def test_all_four_named_families_are_populated() -> None:
    """The goal names AST / graph / set / topology; history carries copy/undo/redo."""
    for family in OpFamily:
        assert ops_by_family(family), f"{family.value} family is empty"
    history = {symbol.op_id for symbol in ops_by_family(OpFamily.HISTORY)}
    assert {"conversation.copy_state", "conversation.undo", "conversation.redo"} <= history


def test_ids_live_in_the_reserved_namespace() -> None:
    span = TOKEN_ID_NAMESPACE_RANGES[OPS_NAMESPACE]
    for symbol in OPS_VOCAB:
        assert symbol.token_id in span


def test_reserved_range_sits_above_every_codec_range() -> None:
    """Reserving ops must move no stored embedding row."""
    span = TOKEN_ID_NAMESPACE_RANGES[OPS_NAMESPACE]
    others = [
        value
        for name, value in TOKEN_ID_NAMESPACE_RANGES.items()
        if name != OPS_NAMESPACE
    ]
    assert span.start >= max(other.stop for other in others)


def test_the_mapping_is_shared_not_duplicated() -> None:
    """Both towers call one function; there is no second copy to drift."""
    encoder = shared_token_ids()
    decoder = shared_token_ids()
    assert encoder == decoder
    assert_shared_across_towers(encoder, decoder)


def test_a_tower_missing_reserved_ops_is_rejected() -> None:
    full = shared_token_ids()
    partial = dict(list(full.items())[:-1])
    with pytest.raises(OpsVocabError, match="missing reserved ops"):
        assert_shared_across_towers(partial, full)


def test_a_tower_renumbering_reserved_ops_is_rejected() -> None:
    full = shared_token_ids()
    shifted = {token: token_id + 1 for token, token_id in full.items()}
    with pytest.raises(OpsVocabError, match="different ids"):
        assert_shared_across_towers(shifted, full)


def test_grammar_symbols_layer_above_ops_without_colliding() -> None:
    grammar = {f"openui:{index}": logical_token_id("openui", index) for index in range(64)}
    assert_layering(grammar)


def test_a_grammar_symbol_inside_the_reserved_range_is_rejected() -> None:
    intruder = {"squatter": OPS_VOCAB[0].token_id}
    with pytest.raises(OpsVocabError, match="collide"):
        assert_layering(intruder)


def test_no_natural_language_vocabulary_is_load_bearing() -> None:
    """NL is optional fluff by design — it has no place in the reserved base."""
    for symbol in OPS_VOCAB:
        assert symbol.op_id.startswith(("openui.", "conversation."))


def test_ids_are_dense_stable_and_unique() -> None:
    local_ids = [symbol.local_id for symbol in OPS_VOCAB]
    assert local_ids == list(range(len(OPS_VOCAB)))
    assert len({symbol.token for symbol in OPS_VOCAB}) == len(OPS_VOCAB)
    assert len({symbol.token_id for symbol in OPS_VOCAB}) == len(OPS_VOCAB)


def test_build_is_deterministic() -> None:
    assert build_ops_vocab() == OPS_VOCAB
    assert fingerprint() == fingerprint()


def test_committed_registry_matches_the_derived_vocabulary() -> None:
    """The gate future regressions trip over."""
    assert load_registry() == manifest()
    assert load_registry()["fingerprint"] == fingerprint()


def test_an_unclassified_operator_fails_the_build(monkeypatch) -> None:
    """Adding an operator without deciding its family is an error, not a default."""
    import slm_training.dsl.ops_vocab as module

    monkeypatch.setattr(
        module, "_live_operator_ids", lambda: ("openui.add_child", "openui.brand_new")
    )
    with pytest.raises(OpsVocabError, match="missing an OpFamily"):
        module.build_ops_vocab()


def test_a_reserved_op_with_no_implementation_fails_the_build(monkeypatch) -> None:
    import slm_training.dsl.ops_vocab as module

    monkeypatch.setattr(module, "_live_operator_ids", lambda: ("openui.add_child",))
    with pytest.raises(OpsVocabError, match="no longer exist"):
        module.build_ops_vocab()
