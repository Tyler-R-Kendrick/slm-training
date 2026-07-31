"""E1 / E4 / E8 / E9: certified static control + forced macros + classification.

Drives the shipped ``static_control_domain`` entry points — not a parallel
reimplementation of domain building.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.completion_kernel import CompletionSession
from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
from slm_training.dsl.grammar.fastpath.static_control_domain import (
    E9_WARM_HARD_PREFIX_CLAIM_CLASS,
    E9_WARM_HARD_PREFIX_LABEL,
    E9_WARM_HARD_PREFIX_ROW_ID,
    HARD_PREFIX_SURFACE,
    assert_gamma_only_tightens,
    compare_domain_parity,
    dynamic_forced_closure,
    e9_warm_hard_prefix_classification,
    extract_static_forced_macro,
    forest_expansion_count,
    oracle_lark_domain_snapshot,
    production_domain_snapshot,
    require_certified_static_projection,
    static_forced_macro_edge,
    walk_static_forced_macro,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import GrammarDecodeState, exact_forced_token_id
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


_CORPUS = (
    'root = TextContent(":slot_0")\n',
    'root = Card([])\n',
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\n',
    "root = Card([b1",
    HARD_PREFIX_SURFACE,
    "root =",
    "root",
    "",
)


def test_require_certified_static_projection(tok: DSLNativeTokenizer) -> None:
    direct = require_certified_static_projection(tok)
    assert direct is not None
    assert "punct" in direct or "kind_terminals" in direct


def test_e1_hard_prefix_multiset_parity(tok: DSLNativeTokenizer) -> None:
    prefix = [tok.bos_id, *tok.encode(HARD_PREFIX_SURFACE, add_special=False)]
    production = production_domain_snapshot(prefix, tok, budget=32)
    oracle = oracle_lark_domain_snapshot(prefix, tok, budget=32)
    report = compare_domain_parity(production, oracle)
    assert report.equal, report.reasons
    assert report.multiset_match
    assert report.coverage_match
    assert report.no_unknown_collapse
    assert production.certified_projection is True
    assert production.authority == "certified_static_plus_gamma"
    # Hard prefix yields 12 complete paths under the kernel corpus contract.
    assert production.status == "complete"
    assert len(production.candidates) == 12


@pytest.mark.parametrize("program", _CORPUS, ids=lambda p: repr(p[-24:]))
def test_e1_kernel_corpus_parity(tok: DSLNativeTokenizer, program: str) -> None:
    ids = [int(t) for t in tok.encode(program, add_special=False)]
    for end in range(0, len(ids) + 1):
        prefix = [tok.bos_id, *ids[:end]]
        production = production_domain_snapshot(prefix, tok, budget=32)
        oracle = oracle_lark_domain_snapshot(prefix, tok, budget=32)
        report = compare_domain_parity(production, oracle)
        assert report.equal, (program, end, report.reasons)
        assert report.no_unknown_collapse


def test_e1_gamma_only_tightens_subset() -> None:
    base = [(1, 2), (3,), (4, 5, 6)]
    filtered = [(1, 2), (4, 5, 6)]
    assert assert_gamma_only_tightens(base, filtered)
    assert not assert_gamma_only_tightens(base, [(1, 2), (9,)])


def test_e4_static_extractor_is_independent_of_forced_closure(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static walk must not call session.forced_closure (distinct implementation)."""
    session = CompletionSession(tok, slot_contract=(":slot_0", ":slot_1"))
    seed = session.seed([tok.bos_id, *tok.encode("root", add_special=False)])

    def _ban(*_a, **_k):
        raise AssertionError("static walk must not call forced_closure")

    monkeypatch.setattr(session, "forced_closure", _ban)
    token_ids, terminal_id, coverage, stopped_on = walk_static_forced_macro(
        session, seed, room=32
    )
    # Independence proven: pure walk completed without forced_closure.
    assert terminal_id >= 0
    assert isinstance(coverage, str) and coverage
    assert isinstance(token_ids, tuple)
    assert stopped_on in {
        "empty",
        "branch",
        "eos",
        "literal",
        "incomplete",
        "budget",
        "rejected",
    }


def test_e4_static_forced_macro_matches_dynamic(tok: DSLNativeTokenizer) -> None:
    session = CompletionSession(tok, slot_contract=(":slot_0", ":slot_1"))
    seed = session.seed([tok.bos_id, *tok.encode("root", add_special=False)])
    # Clear any forced cache so both paths start cold on this key.
    session._forced.clear()
    static_edge = extract_static_forced_macro(session, seed, room=32)
    dyn = dynamic_forced_closure(session, seed, room=32)
    assert static_edge.matched_dynamic is True
    assert static_edge.token_ids == dyn[0]
    assert static_edge.terminal_state_id == dyn[1]
    assert static_edge.coverage == dyn[2]
    # Public alias must agree.
    via_api = static_forced_macro_edge(session, seed, room=32)
    assert via_api.token_ids == static_edge.token_ids
    equals = tok.token_to_id["="]
    assert equals in static_edge.token_ids or static_edge.token_ids == ()


def test_e4_never_compresses_across_literals(tok: DSLNativeTokenizer) -> None:
    """Literal boundary tokens stop the static walk (E4 fail criterion)."""
    session = CompletionSession(tok, slot_contract=(":slot_0",))
    seed = session.seed(
        [tok.bos_id, *tok.encode("root = TextContent(", add_special=False)]
    )
    edge = extract_static_forced_macro(session, seed, room=64)
    assert edge.matched_dynamic is True
    lit_ids = {
        tid
        for tid, name in tok.id_to_token.items()
        if name in {"LIT_STR", "LIT_NUM", "LIT_END"}
    }
    assert not any(t in lit_ids for t in edge.token_ids)
    # If the next forest edge is a literal open, static must stop with 'literal'.
    forest = session.outgoing(edge.terminal_state_id)
    if (
        forest.coverage == "complete"
        and len(forest.paths) == 1
        and any(
            str(tok.id_to_token.get(int(t), "")) in {"LIT_STR", "LIT_NUM", "LIT_END"}
            for t in forest.paths[0].token_ids
        )
    ):
        assert edge.stopped_on == "literal"


def test_e4_static_macro_does_not_invent_edges(tok: DSLNativeTokenizer) -> None:
    session = CompletionSession(tok)
    seed = session.seed([tok.bos_id, *tok.encode("root =", add_special=False)])
    before = forest_expansion_count(session)
    edge = extract_static_forced_macro(session, seed, room=8)
    after = forest_expansion_count(session)
    assert edge.matched_dynamic
    assert after >= before
    # Tokens must be a real prefix of the dynamic chain (no invented symbols).
    dyn_tokens, _, _ = dynamic_forced_closure(session, seed, room=8)
    assert edge.token_ids == dyn_tokens


def test_e8_singleton_greedy_ltr_zero_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """E8: real greedy LTR path with collect_decode_stats — no neural forward.

    Drives TwoTowerModel._greedy_ltr_decode_batch under complete singleton
    proofs (exact_forced_token_id). Would fail if decode still forwarded.
    """
    sample = 'root = TextContent(":slot_0")\n'
    records = [
        ExampleRecord(
            id="e8-singleton",
            prompt="hero card",
            openui=sample,
            design_md="# Design\n",
            split="train",
            source="fixture",
        )
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            context_backend="scratch",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            grammar_ltr_stages=(2,),
            max_target_len=8,
            max_prompt_len=16,
        ),
        device="cpu",
    )
    # Complete singleton proofs for every row position: force EOS (legal after
    # BOS under the fixture length), and ban denoiser so any forward fails hard.
    monkeypatch.setattr(
        "slm_training.models.twotower.force_emit_token_id",
        lambda *_a, **_k: model.tokenizer.eos_id,
    )
    monkeypatch.setattr(
        "slm_training.models.twotower.exact_forced_token_id",
        lambda *_a, **_k: model.tokenizer.eos_id,
    )

    def _ban_forward(*_a, **_k):
        raise AssertionError("neural forward on complete singleton (E8)")

    monkeypatch.setattr(model, "_denoiser_forward", _ban_forward)
    ctx, ctx_pad = model._encode_context(["a", "b"])
    with collect_decode_stats() as stats:
        ids = model._greedy_ltr_decode_batch(ctx, ctx_pad, 2)
    assert ids[:, 1].tolist() == [model.tokenizer.eos_id] * 2
    assert stats.forwards_count == 0
    assert stats.forced_row_tokens_without_forward == 2
    assert stats.all_forced_steps_without_forward == 1


def test_e8_ambiguous_domain_refuses_singleton_bypass(tok: DSLNativeTokenizer) -> None:
    prefix = [tok.bos_id, *tok.encode("root = ", add_special=False)]
    state = GrammarDecodeState(engine=engine_for_dsl("openui"))
    state.remaining_tokens = 32
    production = production_domain_snapshot(prefix, tok, budget=32)
    if production.status == "complete" and len(set(production.candidates)) > 1:
        assert exact_forced_token_id(tok, prefix, state=state) is None


def test_e9_classification_is_request_local_memo_reuse() -> None:
    labels = e9_warm_hard_prefix_classification()
    assert labels["label"] == E9_WARM_HARD_PREFIX_LABEL
    assert labels["label"] == "request_local_memo_reuse"
    assert labels["claim_class"] == E9_WARM_HARD_PREFIX_CLAIM_CLASS
    assert labels["not"] == "aot_cold"
    assert labels["row_id"] == E9_WARM_HARD_PREFIX_ROW_ID
    assert labels["prefix"] == HARD_PREFIX_SURFACE


def test_e9_perf_results_row_documents_memo_reuse() -> None:
    """Durable measurement surface must not market warm 89.8× as AOT cold."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/design/completion-kernel-perf-results.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claim_class"] == "fixture_or_scratch"
    warm = next(r for r in payload["rows"] if r["id"] == E9_WARM_HARD_PREFIX_ROW_ID)
    assert warm["mode"] == "persistent_row_session_no_domain_cache"
    assert warm["classification"] == "request_local_memo_reuse"
    assert warm.get("not_aot_cold") is True
    assert warm["speedup_v1_over_v2"] > 10.0
    # Cold hard prefix remains a regression (must not be sold as AOT win).
    cold = next(r for r in payload["rows"] if r["id"] == "card_open_comma")
    assert cold["speedup_v1_over_v2"] < 1.0
