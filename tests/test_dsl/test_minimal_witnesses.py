"""DSH1-02: reachable/productive alternative witness conformance."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    GrammarWitnessCandidateV1,
    lark_authority,
    production_id,
)
from slm_training.dsl.language_contract import SymbolicSurfacePolicyV1
from slm_training.dsl.minimal_witnesses import (
    UnexplainedAlternativeGap,
    generate_minimal_witness_basis,
)
from slm_training.dsl.pack import DslPack, PlaceholderPolicy, get_pack


@pytest.fixture
def mini_pack(tmp_path: Path) -> DslPack:
    import slm_training.dsl.pack as pack_mod

    grammar_path = tmp_path / "witness-mini.lark"
    grammar_path.write_text(
        (
            "start: pair | atom\n"
            'pair: atom "," atom\n'
            "atom: WORD\n"
            "WORD: /[a-z]+/\n"
            "%import common.WS\n"
            "%ignore WS\n"
        ),
        encoding="utf-8",
    )
    backend = LarkFileBackend(
        dsl_id="witness-mini",
        grammar_path=grammar_path,
        start="start",
        call_as_component=False,
    )
    marker = RuntimeSymbol(surface="word", role="alpha_binder")
    authority = lark_authority(
        grammar_path=grammar_path,
        start_symbols=("start",),
        canonical_serialize=lambda source: source.strip(),
        static_validate=backend.parse_tree,
        scope_policy=lambda source: (("document", source.strip()),),
        completion_frontier=lambda _prefix: frozenset({"WORD"}),
        witness_candidates=lambda: (
            GrammarWitnessCandidateV1("word", (marker,)),
            GrammarWitnessCandidateV1("word,word", (marker,)),
        ),
    )
    pack = DslPack(
        pack_id="witness-mini",
        backend=backend,
        placeholder_policy=PlaceholderPolicy(
            placeholder_re=re.compile(r":[a-z.]+"),
            content_props=frozenset(),
            slot_contract=lambda _source: (),
        ),
        reward_label="syntax_only",
        grammar_capability_authority=authority,
    )
    pack_mod.register_pack(pack)
    try:
        yield pack
    finally:
        pack_mod._PACKS.pop(pack.pack_id, None)


def test_openui_basis_covers_every_reachable_productive_alternative() -> None:
    pack = get_pack("openui")
    adapter = GrammarCapabilityAdapterV1(pack)
    basis = generate_minimal_witness_basis(pack)
    expected = {production_id(item) for item in adapter.production_alternatives}
    observed = {item.alternative_id for item in basis.witnesses} | {
        item.alternative_id for item in basis.unsupported
    }

    assert observed == expected
    assert len(basis.witnesses) == 71
    assert len(basis.unsupported) == 5
    assert {item.reason for item in basis.unsupported} == {
        "STATIC_SEMANTICS_REQUIRES_ROOT",
        "LEXER_COLLAPSES_REPEATED_NEWLINES",
        "SYMBOLIC_SURFACE_FORBIDS_OPEN_NUMBER",
        "SYMBOLIC_SURFACE_HAS_NO_DECLARED_BUILTIN",
    }


def test_every_witness_replays_exact_focus_and_admission() -> None:
    pack = get_pack("openui")
    adapter = GrammarCapabilityAdapterV1(pack)
    authority = adapter.authority
    assert authority is not None and authority.production_trace is not None

    for witness in generate_minimal_witness_basis(pack).witnesses:
        assert adapter.canonical_serialize(witness.canonical_source) == (
            witness.canonical_source
        )
        assert adapter.static_validate(witness.source) is not None
        assert adapter.scope_policy(witness.source)
        assert SymbolicSurfacePolicyV1("openui").evaluate(
            witness.source,
            runtime_symbols=witness.runtime_symbols,
        ).admitted
        assert any(
            occurrence.production == witness.production
            and occurrence.ast_path == witness.focus_ast_path
            for occurrence in authority.production_trace(
                witness.start_symbol, witness.source
            )
        )


def test_basis_is_deterministic_for_identical_snapshot_and_seed() -> None:
    pack = get_pack("openui")

    first = generate_minimal_witness_basis(pack, seed=17)
    second = generate_minimal_witness_basis(pack, seed=17)

    assert first == second
    assert first.identity == second.identity
    assert first.authority_fingerprint == GrammarCapabilityAdapterV1(
        pack
    ).authority_fingerprints["combined"]


def test_complete_mini_dsl_uses_same_minimum_cost_search(
    mini_pack: DslPack,
) -> None:
    basis = generate_minimal_witness_basis(mini_pack)
    by_lhs = {item.production.lhs: item for item in basis.witnesses}

    assert not basis.unsupported
    assert len(basis.witnesses) == 4
    assert by_lhs["atom"].source == "word"
    assert by_lhs["pair"].source == "word,word"
    assert by_lhs["atom"].cost < by_lhs["pair"].cost


def test_unexplained_gap_blocks_basis(mini_pack: DslPack) -> None:
    authority = mini_pack.grammar_capability_authority
    assert authority is not None
    incomplete = replace(
        mini_pack,
        grammar_capability_authority=replace(
            authority,
            witness_candidates=lambda: (
                GrammarWitnessCandidateV1(
                    "word",
                    (RuntimeSymbol(surface="word", role="alpha_binder"),),
                ),
            ),
        ),
    )

    with pytest.raises(UnexplainedAlternativeGap, match="no witness or typed"):
        generate_minimal_witness_basis(incomplete)
