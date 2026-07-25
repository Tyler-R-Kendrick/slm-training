"""DSH1-07 (SLM-359): partial-state classification contract tests."""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    lark_authority,
)
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.train_data.partial_states import (
    CompletionCertificateV1,
    PartialStateClassV1,
    PartialStatePruneError,
    authorize_pruning,
    classify_partial_state,
    family_valid,
)

COMPLETE_DOC = 'root = TextContent(":w.text")'
PREFIX_UNCLOSED_CALL = 'root = TextContent(":w.text"'
PREFIX_AFTER_EQUALS = "root = "
PREFIX_EXPR_TAIL = "$s = true && "
CUT_INSIDE_STRING = 'root = TextContent(":hero.tit'
CUT_INSIDE_MARKER = '$s = ":a.b" :w.'
UNREACHABLE = "TextConten"
PARTIAL_NO_WITNESS = 'root = Stack([item], "column"'


@pytest.fixture(scope="module")
def adapter() -> GrammarCapabilityAdapterV1:
    return GrammarCapabilityAdapterV1(get_pack("openui"))


@pytest.fixture(scope="module")
def multi_start_adapter(tmp_path_factory: pytest.TempPathFactory) -> GrammarCapabilityAdapterV1:
    """Declared multi-start authority: ``document`` requires root, ``fragment``
    accepts a bare item — so named fragments are reachable."""
    import re
    from pathlib import Path

    from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
    from slm_training.dsl.pack import DslPack, PlaceholderPolicy

    grammar_path = tmp_path_factory.mktemp("slm359") / "mini.lark"
    grammar_path.write_text(
        "document: ROOT item\n"
        "fragment: item\n"
        "item: WORD\n"
        'ROOT: "root"\n'
        "WORD: /[a-z]+/\n"
        "%import common.WS\n"
        "%ignore WS\n",
        encoding="utf-8",
    )
    backend = LarkFileBackend(
        dsl_id="test-slm359",
        grammar_path=Path(grammar_path),
        start="document",
        call_as_component=False,
    )
    authority = lark_authority(
        grammar_path=Path(grammar_path),
        start_symbols=("document", "fragment"),
        canonical_serialize=lambda source: source.strip(),
        static_validate=backend.parse_tree,
        scope_policy=lambda source: (("document", source.strip()),),
        completion_frontier=lambda prefix: (
            frozenset({"ROOT", "WORD"})
            if not prefix.strip()
            else frozenset({"$END"})
        ),
    )
    return GrammarCapabilityAdapterV1(
        DslPack(
            pack_id="test-slm359",
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
    )


def test_complete_document(adapter: GrammarCapabilityAdapterV1) -> None:
    report = classify_partial_state(COMPLETE_DOC, adapter=adapter)
    assert report.state_class is PartialStateClassV1.COMPLETE
    assert report.support_coverage == "full"
    assert report.min_completion_cost == 0
    assert report.start_symbol == "start"


def test_fragment_parses_under_declared_start(
    multi_start_adapter: GrammarCapabilityAdapterV1,
) -> None:
    source = "hello"
    report = classify_partial_state(source, adapter=multi_start_adapter)
    assert report.state_class is PartialStateClassV1.FRAGMENT
    assert report.start_symbol == "fragment"
    assert report.support_coverage == "full"
    # Acceptance: every FRAGMENT parses under its declared start rule.
    multi_start_adapter.fragment_parse(report.start_symbol, source)


def test_prefix_replays_to_verified_terminal(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(PREFIX_UNCLOSED_CALL, adapter=adapter)
    assert report.state_class is PartialStateClassV1.PREFIX
    assert report.support_coverage == "full"
    assert report.frontier
    assert report.open_nonterminals
    certificate = report.certificate
    assert isinstance(certificate, CompletionCertificateV1)
    assert certificate.terminal_verified
    assert certificate.completion_cost == report.min_completion_cost == 1
    assert certificate.completion_suffix == ")"
    # Replay acceptance: the certified continuation validates as a terminal.
    adapter.static_validate(certificate.witness_source)
    assert certificate.witness_source == COMPLETE_DOC


def test_prefix_records_cost_and_singletons(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(PREFIX_EXPR_TAIL, adapter=adapter)
    assert report.state_class is PartialStateClassV1.PREFIX
    assert report.min_completion_cost is not None and report.min_completion_cost > 0
    assert report.exact_singleton_positions
    assert all(
        len(PREFIX_EXPR_TAIL) < position for position in report.exact_singleton_positions
    )


def test_cut_never_splits_token_or_marker(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    string_cut = classify_partial_state(CUT_INSIDE_STRING, adapter=adapter)
    assert string_cut.state_class is PartialStateClassV1.INVALID
    assert string_cut.reason == "cut_inside_token:string"
    marker_cut = classify_partial_state(CUT_INSIDE_MARKER, adapter=adapter)
    assert marker_cut.state_class is PartialStateClassV1.INVALID
    assert marker_cut.reason == "cut_inside_marker:placeholder"


def test_unreachable_partial_unknown_coverage(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(UNREACHABLE, adapter=adapter)
    assert report.state_class is PartialStateClassV1.INVALID
    assert report.reason == "unreachable_from_compiler"
    assert report.support_coverage == "unknown"


def test_uncertified_frontier_is_partial_not_admitted(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(PARTIAL_NO_WITNESS, adapter=adapter)
    assert report.state_class is PartialStateClassV1.INVALID
    assert report.reason == "no_certified_completion"
    assert report.support_coverage == "partial"
    assert report.frontier  # frontier observed, but no certificate


def test_unknown_coverage_never_authorizes_pruning(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    for source in (UNREACHABLE, PARTIAL_NO_WITNESS, CUT_INSIDE_STRING):
        report = classify_partial_state(source, adapter=adapter)
        with pytest.raises(PartialStatePruneError):
            authorize_pruning(report)


def test_admitted_states_authorize_pruning(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    for source in (COMPLETE_DOC, PREFIX_UNCLOSED_CALL):
        report = classify_partial_state(source, adapter=adapter)
        assert authorize_pruning(report) is report


def test_family_stop_rule(adapter: GrammarCapabilityAdapterV1) -> None:
    good = classify_partial_state(PREFIX_UNCLOSED_CALL, adapter=adapter)
    bad = classify_partial_state(UNREACHABLE, adapter=adapter)
    partial_only = classify_partial_state(PARTIAL_NO_WITNESS, adapter=adapter)
    assert family_valid([good])
    # Unreachable partial invalidates its family.
    assert not family_valid([good, bad])
    # Partial (no certificate) coverage alone does not trip the stop rule.
    assert family_valid([good, partial_only])


def test_required_roles_and_binders_recorded(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(PREFIX_EXPR_TAIL, adapter=adapter)
    assert "state" in report.required_roles
    assert report.required_binders  # alpha-renamable binder surfaces recorded


def test_max_completion_edits_bounds_certificate(
    adapter: GrammarCapabilityAdapterV1,
) -> None:
    report = classify_partial_state(
        PREFIX_AFTER_EQUALS, adapter=adapter, max_completion_edits=1
    )
    # Cheapest accepting continuation costs 3 edits — over budget: no cert.
    assert report.state_class is PartialStateClassV1.INVALID
    assert report.reason == "no_certified_completion"
    admitted = classify_partial_state(
        PREFIX_AFTER_EQUALS, adapter=adapter, max_completion_edits=8
    )
    assert admitted.state_class is PartialStateClassV1.PREFIX
    assert admitted.min_completion_cost == 3


def test_determinism(adapter: GrammarCapabilityAdapterV1) -> None:
    for source in (COMPLETE_DOC, PREFIX_UNCLOSED_CALL, UNREACHABLE, CUT_INSIDE_STRING):
        first = classify_partial_state(source, adapter=adapter)
        second = classify_partial_state(source, adapter=adapter)
        assert first == second
        assert first.to_dict() == second.to_dict()
