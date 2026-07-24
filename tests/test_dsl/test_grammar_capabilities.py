"""DSH1-01: declared grammar capability adapter conformance."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar_capabilities import (
    CompletionDomainCandidateV1,
    CompletionDomainRequestV1,
    CompletionDomainV1,
    GrammarCapabilityAdapterV1,
    GrammarSymbolV1,
    ProductionAlternativeV1,
    UnsupportedCapabilityV1,
    lark_authority,
)
from slm_training.dsl.pack import DslPack, PlaceholderPolicy, get_pack

OPENUI = 'root = TextContent(":hero.title")'


@pytest.fixture
def mini_pack(tmp_path: Path) -> DslPack:
    grammar_path = tmp_path / "mini.lark"
    grammar_path.write_text(
        "start: item\nitem: WORD\nWORD: /[a-z]+/\n%import common.WS\n%ignore WS\n",
        encoding="utf-8",
    )
    backend = LarkFileBackend(
        dsl_id="test-mini",
        grammar_path=grammar_path,
        start="start",
        call_as_component=False,
    )
    def completion_domain(request: CompletionDomainRequestV1) -> CompletionDomainV1:
        declared = (7,) if not request.prefix_ids else (8,)
        return CompletionDomainV1(
            status="complete",
            candidates=(
                CompletionDomainCandidateV1(
                    token_ids=declared,
                    kind="word",
                    terminal_witness=(*declared, 9),
                ),
            ),
            scope_fingerprint=f"scope:{request.prefix_ids!r}",
        )

    authority = lark_authority(
        grammar_path=grammar_path,
        start_symbols=("start",),
        canonical_serialize=lambda source: source.strip(),
        static_validate=backend.parse_tree,
        scope_policy=lambda source: (("document", source.strip()),),
        completion_frontier=lambda prefix: (
            frozenset({"WORD"}) if not prefix.strip() else frozenset({"$END"})
        ),
        completion_domain=completion_domain,
    )
    return DslPack(
        pack_id="test-mini",
        backend=backend,
        placeholder_policy=PlaceholderPolicy(
            placeholder_re=re.compile(r":[a-z.]+"),
            content_props=frozenset({"text"}),
            slot_contract=lambda source: (),
        ),
        reward_label="syntax_only",
        prop_order=lambda: {"word": ("value",)},
        grammar_capability_authority=authority,
    )


@pytest.mark.parametrize(
    ("pack_fixture", "source"),
    (("openui", OPENUI), ("mini_pack", "hello")),
)
def test_complete_packs_pass_one_conformance_suite(
    request: pytest.FixtureRequest, pack_fixture: str, source: str
) -> None:
    pack = (
        get_pack("openui")
        if pack_fixture == "openui"
        else request.getfixturevalue(pack_fixture)
    )
    adapter = GrammarCapabilityAdapterV1(pack)

    assert adapter.is_complete
    assert adapter.start_symbols == ("start",)
    assert adapter.production_alternatives
    assert adapter.terminal_categories
    assert adapter.nonterminal_analysis
    assert adapter.fragment_parse("start", source) is not None
    assert isinstance(adapter.canonical_serialize(source), str)
    assert adapter.static_validate(source) is not None
    assert adapter.scope_policy(source) is not None
    assert isinstance(adapter.completion_frontier(""), frozenset)
    if pack_fixture == "mini_pack":
        domain = adapter.completion_domain(
            CompletionDomainRequestV1(
                prefix_ids=(), tokenizer=object(), remaining_tokens=3
            )
        )
        assert isinstance(domain, CompletionDomainV1)
        assert domain.candidates[0].terminal_witness == (7, 9)
    assert set(adapter.authority_fingerprints) == {
        "grammar",
        "backend",
        "schema",
        "property_order",
        "placeholder",
        "combined",
    }


def test_partial_pack_is_typed_unsupported_and_never_complete() -> None:
    adapter = GrammarCapabilityAdapterV1(get_pack("toy-layout"))

    assert not adapter.is_complete
    for value in (
        adapter.start_symbols,
        adapter.production_alternatives,
        adapter.terminal_categories,
        adapter.nonterminal_analysis,
        adapter.fragment_parse("start", ""),
        adapter.canonical_serialize(""),
        adapter.static_validate(""),
        adapter.scope_policy(""),
        adapter.completion_frontier(""),
        adapter.completion_domain(
            CompletionDomainRequestV1(prefix_ids=(), tokenizer=object())
        ),
    ):
        assert isinstance(value, UnsupportedCapabilityV1)
        assert value.status == "UNSUPPORTED"


def test_partially_declared_authority_never_appears_complete(
    mini_pack: DslPack,
) -> None:
    authority = mini_pack.grammar_capability_authority
    assert authority is not None
    adapter = GrammarCapabilityAdapterV1(
        replace(
            mini_pack,
            grammar_capability_authority=replace(
                authority, canonical_serialize=None
            ),
        )
    )

    assert not adapter.is_complete
    assert isinstance(adapter.canonical_serialize("hello"), UnsupportedCapabilityV1)


def test_declared_productions_ignore_examples_when_authority_exists(
    mini_pack: DslPack,
) -> None:
    with_examples = replace(
        mini_pack,
        corpus_generator=lambda: (
            "this example contains fake -> productions",
            "start: invented",
        ),
    )

    assert (
        GrammarCapabilityAdapterV1(with_examples).production_alternatives
        == GrammarCapabilityAdapterV1(mini_pack).production_alternatives
    )


def test_authority_change_alters_fingerprint(mini_pack: DslPack) -> None:
    authority = mini_pack.grammar_capability_authority
    assert authority is not None
    changed = replace(
        authority,
        productions=authority.productions
        + (
            ProductionAlternativeV1(
                lhs="invented",
                rhs=(GrammarSymbolV1(name="WORD", kind="terminal"),),
            ),
        ),
    )
    before = GrammarCapabilityAdapterV1(mini_pack).authority_fingerprints
    after = GrammarCapabilityAdapterV1(
        replace(mini_pack, grammar_capability_authority=changed)
    ).authority_fingerprints

    assert before["grammar"] != after["grammar"]
    assert before["combined"] != after["combined"]


def test_analysis_reports_nullable_recursive_productive_and_reachable(
    mini_pack: DslPack,
) -> None:
    analysis = {
        row.name: row for row in GrammarCapabilityAdapterV1(mini_pack).nonterminal_analysis
    }

    assert analysis["start"].reachable
    assert analysis["start"].productive
    assert not analysis["start"].nullable
    assert not analysis["start"].recursive


def test_fragment_parse_rejects_undeclared_start(mini_pack: DslPack) -> None:
    with pytest.raises(ValueError, match="undeclared start symbol"):
        GrammarCapabilityAdapterV1(mini_pack).fragment_parse("item", "hello")


def test_completion_domain_is_scope_local_and_witnessed(mini_pack: DslPack) -> None:
    adapter = GrammarCapabilityAdapterV1(mini_pack)
    root = adapter.completion_domain(
        CompletionDomainRequestV1(prefix_ids=(), tokenizer=object(), remaining_tokens=3)
    )
    child = adapter.completion_domain(
        CompletionDomainRequestV1(prefix_ids=(7,), tokenizer=object(), remaining_tokens=2)
    )

    assert isinstance(root, CompletionDomainV1)
    assert isinstance(child, CompletionDomainV1)
    assert root.candidates[0].token_ids == (7,)
    assert child.candidates[0].token_ids == (8,)
    assert root.scope_fingerprint != child.scope_fingerprint
    assert root.candidates[0].terminal_witness[:1] == root.candidates[0].token_ids
