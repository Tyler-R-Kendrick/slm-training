"""Reason-coded constraint evidence for the completion forest (VSS0-02 / SLM-58).

Two invariants are fixed here:

1. ``explain=True`` never changes ``paths`` / ``candidate_ids`` / ``coverage`` /
   ``terminals``; the default path stays byte-for-byte and carries no evidence.
2. every considered-but-excluded candidate is reason-coded at the stage that
   dropped it, and the coverage verdict makes exhaustiveness explicit — prefix
   legality is *not* a support proof (see docs/design/verified-scope-solver.md).
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar.fastpath.constraint_evidence import (
    ConstraintEvidence,
    ConstraintEvidenceRecorder,
    ConstraintStage,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def _tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def _prefixes(tok: DSLNativeTokenizer) -> dict[str, list[int]]:
    """One representative prefix per constraint region, lexer-native driven."""
    return {
        "empty": [],
        "bos": [tok.bos_id],
        "children_open": [tok.bos_id, *tok.encode("root=Stack([", add_special=False)],
        "root_value": [tok.bos_id, tok.bind_id(0), tok.token_to_id["="]],
        "complete_doc": [
            tok.bos_id,
            *tok.encode('root=TextContent(":hero.title")', add_special=False),
        ],
    }


def _assert_output_parity(tok, prefix, **kwargs):
    """explain=True must not change any output field; return the explained forest."""
    off = build_completion_forest(tok, list(prefix), **kwargs)
    on = build_completion_forest(tok, list(prefix), explain=True, **kwargs)
    assert off.paths == on.paths
    assert off.candidate_ids == on.candidate_ids
    assert off.coverage == on.coverage
    assert off.terminals == on.terminals
    assert off.evidence == ()  # default-off allocates no evidence
    return on


def test_default_off_matches_default_on_for_lexer_native() -> None:
    tok = _tok()
    for name, prefix in _prefixes(tok).items():
        on = _assert_output_parity(tok, prefix, slot_contract=[":hero.title"])
        assert on.evidence, f"{name}: explain must produce evidence"


def test_explanations_do_not_change_candidate_ids() -> None:
    # The acceptance criterion called out explicitly.
    tok = _tok()
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    plain = build_completion_forest(tok, prefix)
    explained = build_completion_forest(tok, prefix, explain=True)
    assert explained.candidate_ids == plain.candidate_ids
    assert explained.paths == plain.paths


def test_choice_codec_forest_is_unsupported_and_explain_is_inert() -> None:
    """The completion forest is lexer-native by construction. A ``ChoiceTokenizer``
    has its own constrained-decode path (``ChoiceDecodeState``) and is rejected by
    the token map before any explanation logic runs, so toggling ``explain`` must
    not change that boundary (the instrumentation is inert on the choice path)."""
    from slm_training.dsl.lang_core import ParseError
    from slm_training.models.choice_tokenizer import ChoiceTokenizer

    ctok = ChoiceTokenizer.build()
    with pytest.raises(ParseError):
        build_completion_forest(ctok, [])
    with pytest.raises(ParseError):
        build_completion_forest(ctok, [], explain=True)


def test_binding_stage_reason_codes_illegal_binders() -> None:
    tok = _tok()
    # At a fresh child list only the next binder slot is legal; the root binder
    # and forward binders beyond the next slot are excluded by binder scope.
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tok, prefix, explain=True)
    assert tok.bind_id(1) in forest.candidate_ids
    binding_rejections = {
        e.candidate_id
        for e in forest.evidence
        if not e.admitted and e.stage == ConstraintStage.BINDING
    }
    assert tok.bind_id(0) in binding_rejections  # root binder is not legal here
    assert tok.bind_id(2) in binding_rejections  # forward binder beyond next slot
    assert tok.bind_id(1) not in binding_rejections


def test_min_content_eos_withholding_is_distinct_from_grammar() -> None:
    tok = _tok()
    contract = [":hero.title"]
    prefix = [
        tok.bos_id,
        *tok.encode('root=TextContent(":hero.title")', add_special=False),
    ]
    forest = build_completion_forest(
        tok, prefix, slot_contract=contract, min_content=2, explain=True
    )
    assert tok.eos_id not in forest.candidate_ids  # withheld
    eos_rejections = [
        e for e in forest.evidence if e.candidate_id == tok.eos_id and not e.admitted
    ]
    assert len(eos_rejections) == 1
    ev = eos_rejections[0]
    # The distinguishing invariant: min-content withholding is MIN_CONTENT, never
    # conflated with a grammar rejection.
    assert ev.stage == ConstraintStage.MIN_CONTENT
    assert ev.reason_code == "eos_below_min_content"
    assert ("min_content", "2") in ev.details

    # Floor met: EOS is admitted, no withholding record.
    met = build_completion_forest(
        tok, prefix, slot_contract=contract, min_content=1, explain=True
    )
    assert tok.eos_id in met.candidate_ids
    assert not any(
        e.candidate_id == tok.eos_id and not e.admitted for e in met.evidence
    )


def test_schema_enum_exclusion_is_reason_coded(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tok = _tok()
    schema = {
        "properties": {"Stack": {}},
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "direction": {"enum": ["row", "column"]},
                }
            }
        },
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = tok.encode("root=Stack([],", add_special=False)
    plain = build_completion_forest(tok, prefix)
    explained = build_completion_forest(tok, prefix, explain=True)
    assert explained.candidate_ids == plain.candidate_ids
    assert set(explained.candidate_ids) == {
        tok.token_to_id["STR:row"],
        tok.token_to_id["STR:column"],
    }
    schema_rejections = [
        e for e in explained.evidence if e.stage == ConstraintStage.SCHEMA and not e.admitted
    ]
    assert schema_rejections  # enum takeover excludes non-enum candidates


def test_schema_type_and_slot_contract_stages_are_distinct(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tok = _tok()
    monkeypatch.setattr(
        compiler_draft,
        "_official_schema",
        lambda: {
            "properties": {"TextContent": {}},
            "$defs": {"TextContent": {"properties": {"text": {"type": "string"}}}},
        },
    )
    prefix = tok.encode("root=TextContent(", add_special=False)
    plain = build_completion_forest(tok, prefix, slot_contract=[":hero.title"])
    explained = build_completion_forest(
        tok, prefix, slot_contract=[":hero.title"], explain=True
    )
    assert explained.candidate_ids == plain.candidate_ids
    assert set(explained.candidate_ids) == {tok.sym_id(0)}
    stages = {e.stage for e in explained.evidence}
    # The slot contract narrows separately from the schema value-type filter.
    assert ConstraintStage.SLOT_CONTRACT in stages
    assert ConstraintStage.SCHEMA in stages


def test_partial_coverage_is_explicit_and_not_a_proof(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tok = _tok()
    # Force the coverage classifier to report incomplete terminal coverage (the
    # unavailable-capability style state): a forest that still has legal paths
    # must be recorded as partial and therefore not exhaustive.
    monkeypatch.setattr(
        compiler_draft, "_known_terminal_coverage", lambda *a, **k: False
    )
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tok, prefix, explain=True)
    assert forest.paths  # legal candidates remain live
    assert forest.coverage == "partial"
    assert forest.is_exhaustive is False
    coverage_records = [
        e for e in forest.evidence if e.stage == ConstraintStage.COVERAGE
    ]
    assert len(coverage_records) == 1
    cov = coverage_records[0]
    assert cov.admitted is False
    assert cov.reason_code == "coverage_partial"


def test_unparseable_prefix_records_a_grammar_verdict() -> None:
    tok = _tok()
    forest = build_completion_forest(tok, list(tok.encode(")", add_special=False)), explain=True)
    assert forest.coverage == "none"
    assert forest.paths == ()
    grammar = [
        e
        for e in forest.evidence
        if e.stage == ConstraintStage.GRAMMAR and not e.admitted and e.candidate_id is None
    ]
    assert grammar
    assert grammar[0].reason_code == "prefix_not_parseable"


def test_coverage_record_marks_exhaustiveness_for_every_prefix() -> None:
    tok = _tok()
    for name, prefix in _prefixes(tok).items():
        forest = build_completion_forest(tok, list(prefix), explain=True)
        cov = [e for e in forest.evidence if e.stage == ConstraintStage.COVERAGE]
        assert len(cov) == 1, name
        rec = cov[0]
        assert rec.reason_code == f"coverage_{forest.coverage}"
        assert rec.admitted == (forest.coverage == "complete")
        assert rec.admitted == forest.is_exhaustive


def test_every_excluded_candidate_is_reason_coded_and_paths_are_admitted() -> None:
    tok = _tok()
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tok, prefix, explain=True)
    for e in forest.evidence:
        if not e.admitted and e.stage != ConstraintStage.COVERAGE:
            assert e.reason_code, "every rejection names a reason"
            assert isinstance(e.stage, ConstraintStage)
    admitted_ids = {
        e.candidate_id for e in forest.evidence if e.admitted and e.candidate_id is not None
    }
    for path in forest.paths:
        assert path.token_ids
        assert path.token_ids[0] in admitted_ids


def test_stage_summary_matches_evidence_and_is_deterministically_ordered() -> None:
    tok = _tok()
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tok, prefix, explain=True)
    admitted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for e in forest.evidence:
        (admitted if e.admitted else rejected)[e.stage.value] += 1
    for stage, passed, failed in forest.stage_summary():
        assert admitted[stage] == passed
        assert rejected[stage] == failed
    order = [s.value for s in ConstraintStage]
    seen = [row[0] for row in forest.stage_summary()]
    assert seen == sorted(seen, key=order.index)


def test_evidence_json_round_trip_and_rebuild_are_deterministic() -> None:
    tok = _tok()
    prefix = [tok.bos_id, *tok.encode("root=Stack([", add_special=False)]
    forest = build_completion_forest(tok, prefix, explain=True)
    dumped = json.dumps([e.to_dict() for e in forest.evidence])
    restored = tuple(ConstraintEvidence.from_dict(d) for d in json.loads(dumped))
    assert restored == forest.evidence
    again = build_completion_forest(tok, prefix, explain=True)
    assert again.evidence == forest.evidence  # logit-independent, deterministic


def test_evidence_carries_no_prompt_literal_leakage() -> None:
    tok = _tok()
    prefix = [
        tok.bos_id,
        *tok.encode('root=TextContent(":hero.title")', add_special=False),
    ]
    forest = build_completion_forest(
        tok, prefix, slot_contract=[":hero.title"], min_content=2, explain=True
    )
    for e in forest.evidence:
        assert e.candidate_id is None or isinstance(e.candidate_id, int)
        assert all(isinstance(t, int) for t in e.path_token_ids)
        for key, value in e.details:
            assert isinstance(key, str) and isinstance(value, str)
            assert ":hero.title" not in value  # bounded safe metadata only


def test_recorder_partial_and_none_coverage_are_not_exhaustive() -> None:
    # Direct, deterministic check of the coverage verdict semantics.
    for coverage in ("partial", "none"):
        recorder = ConstraintEvidenceRecorder()
        recorder.seed([1, 2, 3])
        recorder.narrow(ConstraintStage.SCHEMA, "x", [1, 2])
        recorder.reject_unreachable(2)
        recorder.admit_path(1, [1], "component")
        evidence = recorder.finalize(coverage)
        cov = evidence[-1]
        assert cov.stage == ConstraintStage.COVERAGE
        assert cov.candidate_id is None
        assert cov.admitted is False
        assert cov.reason_code == f"coverage_{coverage}"
        # the excluded candidate (3, dropped at schema) is reason-coded
        assert any(
            e.candidate_id == 3 and e.stage == ConstraintStage.SCHEMA and not e.admitted
            for e in evidence
        )
