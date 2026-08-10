"""RSP-005 (SLM-480) regression tests: the closed-macro mining/coverage
mechanism and the exp-sr-9 fixture campaign wiring."""

from __future__ import annotations

import hashlib

import pytest

from slm_training.dsl.symbolic_expr_corpus import CorpusOracleV1, CorpusRecordV1
from slm_training.dsl.symbolic_expr_ir import OpApply, VarRef, serialize_expr
from slm_training.dsl.symbolic_regression_pack import (
    SYMBOLIC_REGRESSION_PACK_ID,
    SYMBOLIC_REGRESSION_PROBLEM_SCHEMA,
    SymbolicRegressionProblemV1,
)
from slm_training.harnesses.experiments.symbolic_macro_library import (
    ARM_IDS,
    CATALOGUE_ID,
    MacroCandidateV1,
    MacroEntryV1,
    MacroLibraryV1,
    Rsp005CampaignV1,
    build_macro_library,
    empty_macro_library,
    expand_macro,
    instrumented_node_count,
    macro_library_size_reduction_rate,
    mine_macro_candidates,
    run_campaign,
    select_macros_by_frequency,
    select_macros_by_mdl,
    verify_expansion_equivalence,
)


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


def _problem(*, num_variables: int) -> SymbolicRegressionProblemV1:
    variables = tuple(f"x{i}" for i in range(num_variables))
    return SymbolicRegressionProblemV1(
        variables=variables,
        table=((0.0,) * num_variables, (1.0,) * num_variables),
        targets=(0.0, 1.0),
        allowed_operators=frozenset({"+", "-", "*", "/", "neg", "sin", "cos", "exp", "log", "sqrt", "abs"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=64,
        numeric_domain_policy="allow_nonfinite",
        train_indices=(0, 1),
        validation_indices=(),
        extrapolation_indices=(),
        pack_id=SYMBOLIC_REGRESSION_PACK_ID,
        schema=SYMBOLIC_REGRESSION_PROBLEM_SCHEMA,
    )


def _record(*, corpus_split: str, canonical_expr: str, num_variables: int) -> CorpusRecordV1:
    family_id = hashlib.sha256(canonical_expr.encode("utf-8")).hexdigest()
    oracle = CorpusOracleV1(
        schema_version="v1",
        canonical_expr=canonical_expr,
        true_coefficients=(),
        family_id=family_id,
        operator_signature=frozenset(),
    )
    return CorpusRecordV1(
        schema_version="v1",
        family_id=family_id,
        corpus_split=corpus_split,
        problem=_problem(num_variables=num_variables),
        oracle=oracle,
    )


# --- MacroEntryV1 / MacroLibraryV1 self-consistency -------------------------


def test_macro_id_is_content_addressed_and_independent_of_library_membership():
    subtree = _op("sin", _v(0))
    key = serialize_expr(subtree)
    candidate = MacroCandidateV1(canonical_key=key, node_count=2, frequency=3, net_gain_nodes=3 * 1 - 2)
    library_one = select_macros_by_mdl([candidate], min_frequency=1, min_gain_nodes=1, max_macros=1)
    # Rebuilding a library from the same macros (e.g. after retiring an
    # unrelated sibling macro elsewhere) must never renumber this one.
    library_two = build_macro_library(library_one.macros, selection_policy="mdl_net_gain")
    assert library_one.macros[0].macro_id == library_two.macros[0].macro_id


def test_macro_entry_rejects_inconsistent_net_gain():
    key = serialize_expr(_op("sin", _v(0)))
    with pytest.raises(ValueError, match="net_gain_nodes"):
        MacroEntryV1(macro_id="macro_deadbeef", expansion_expr=key, node_count=2, frequency=3, net_gain_nodes=999)


def test_macro_entry_rejects_macro_id_not_matching_content_address():
    key = serialize_expr(_op("sin", _v(0)))
    with pytest.raises(ValueError, match="content-addressed"):
        MacroEntryV1(
            macro_id="macro_0000000000000000",
            expansion_expr=key,
            node_count=2,
            frequency=3,
            net_gain_nodes=3 * 1 - 2,
        )


def test_macro_library_digest_is_tamper_evident():
    library = empty_macro_library()
    with pytest.raises(ValueError, match="library_digest"):
        MacroLibraryV1(
            schema_version=library.schema_version,
            selection_policy=library.selection_policy,
            macros=library.macros,
            library_digest="0" * 64,
        )


def test_empty_macro_library_yields_zero_reduction_no_op_control():
    library = empty_macro_library()
    tree = _op("+", _op("sin", _v(0)), _v(1))
    assert macro_library_size_reduction_rate([tree], library) == 0.0


# --- Mining is train-only by construction (leakage control) ----------------


def test_mine_macro_candidates_only_reflects_records_it_is_given():
    train_only_pattern = serialize_expr(_op("sin", _op("+", _v(0), _v(1))))
    test_only_pattern = serialize_expr(_op("cos", _op("*", _v(0), _v(1))))

    train_records = [
        _record(corpus_split="train", canonical_expr=train_only_pattern, num_variables=2),
        _record(corpus_split="train", canonical_expr=train_only_pattern, num_variables=2),
    ]
    test_records = [
        _record(corpus_split="test", canonical_expr=test_only_pattern, num_variables=2),
        _record(corpus_split="test", canonical_expr=test_only_pattern, num_variables=2),
    ]

    # A caller that (correctly) mines only from train_records never sees the
    # test-only recurring pattern as a candidate -- this is the entire
    # leakage-control surface: run_campaign must always call this with
    # corpus_split=='train' records only.
    train_candidates = {c.canonical_key for c in mine_macro_candidates(train_records)}
    assert train_only_pattern in train_candidates
    assert test_only_pattern not in train_candidates

    # And the reverse: mining the test set alone never surfaces the
    # train-only pattern (sanity check that the function is a pure
    # function of its input, not some shared/global corpus state).
    test_candidates = {c.canonical_key for c in mine_macro_candidates(test_records)}
    assert test_only_pattern in test_candidates
    assert train_only_pattern not in test_candidates


# --- Coverage counting is nesting-safe --------------------------------------


def test_instrumented_node_count_never_double_counts_a_nested_macro():
    inner = _op("sin", _v(0))  # node_count == 2
    outer = _op("cos", inner)  # node_count == 3, contains inner as a child
    outer_key = serialize_expr(outer)
    inner_key = serialize_expr(inner)

    outer_candidate = MacroCandidateV1(canonical_key=outer_key, node_count=3, frequency=2, net_gain_nodes=2 * 2 - 3)
    inner_candidate = MacroCandidateV1(canonical_key=inner_key, node_count=2, frequency=2, net_gain_nodes=2 * 1 - 2)
    library = select_macros_by_mdl(
        [outer_candidate, inner_candidate], min_frequency=1, min_gain_nodes=0, max_macros=2
    )
    assert {m.expansion_expr for m in library.macros} == {outer_key, inner_key}

    # The whole tree collapses to exactly one call-site node (the outer
    # macro matches at the root, so its inner macro is never separately
    # counted even though it is also independently admitted).
    assert instrumented_node_count(outer, library) == 1


def test_instrumented_node_count_covers_a_macro_nested_beneath_non_macro_structure():
    inner = _op("sin", _v(0))
    tree = _op("+", inner, _v(1))  # inner is a macro, '+' node is not
    library = select_macros_by_mdl(
        [MacroCandidateV1(canonical_key=serialize_expr(inner), node_count=2, frequency=2, net_gain_nodes=0)],
        min_frequency=1,
        min_gain_nodes=0,
        max_macros=1,
    )
    # '+' (1) + macro call-site (1, replacing sin(x0)) + VarRef(1) (1) == 3
    assert instrumented_node_count(tree, library) == 3


# --- Expansion is deterministic and legality-preserving ---------------------


def test_expand_macro_reconstructs_the_exact_stored_expansion():
    subtree = _op("sin", _op("+", _v(0), _v(1)))
    key = serialize_expr(subtree)
    library = select_macros_by_mdl(
        [MacroCandidateV1(canonical_key=key, node_count=4, frequency=2, net_gain_nodes=2 * 3 - 4)],
        min_frequency=1,
        min_gain_nodes=1,
        max_macros=1,
    )
    entry = library.macros[0]
    reconstructed = expand_macro(entry, num_variables=2)
    assert serialize_expr(reconstructed) == key


def test_verify_expansion_equivalence_passes_for_a_real_occurrence():
    pattern = serialize_expr(_op("sin", _op("+", _v(0), _v(1))))
    records = [
        _record(corpus_split="train", canonical_expr=pattern, num_variables=2),
        _record(corpus_split="train", canonical_expr=pattern, num_variables=2),
    ]
    candidates = mine_macro_candidates(records, min_node_count=2)
    library = select_macros_by_mdl(candidates, min_frequency=2, min_gain_nodes=1, max_macros=4)
    assert library.macros, "expected at least one admitted macro for this fixture"
    reports = verify_expansion_equivalence(library, records, rng_seed=0)
    assert reports
    assert all(report["verified"] for report in reports)


def test_verify_expansion_equivalence_reports_a_macro_with_no_occurrence():
    orphan_key = serialize_expr(_op("cos", _op("*", _v(0), _v(1))))
    orphan = MacroCandidateV1(canonical_key=orphan_key, node_count=4, frequency=2, net_gain_nodes=2 * 3 - 4)
    library = select_macros_by_mdl([orphan], min_frequency=1, min_gain_nodes=1, max_macros=1)

    unrelated_pattern = serialize_expr(_op("sin", _v(0)))
    records = [_record(corpus_split="train", canonical_expr=unrelated_pattern, num_variables=1)]

    reports = verify_expansion_equivalence(library, records, rng_seed=0)
    assert len(reports) == 1
    assert reports[0]["verified"] is False
    assert reports[0]["reason"] == "no occurrence found"


# --- Frequency-only vs MDL selection differ under the right corpus ---------


def test_mdl_selection_prefers_higher_net_gain_over_raw_frequency():
    # A cheap (2-node) subtree occurring often vs. an expensive (5-node)
    # subtree occurring less often but with a larger net gain.
    cheap_key = serialize_expr(_op("neg", _v(0)))
    expensive = _op("sin", _op("+", _op("*", _v(0), _v(1)), _v(0)))
    expensive_key = serialize_expr(expensive)
    cheap = MacroCandidateV1(canonical_key=cheap_key, node_count=2, frequency=5, net_gain_nodes=5 * 1 - 2)
    costly = MacroCandidateV1(
        canonical_key=expensive_key,
        node_count=5,
        frequency=3,
        net_gain_nodes=3 * 4 - 5,
    )
    assert costly.net_gain_nodes > cheap.net_gain_nodes
    assert cheap.frequency > costly.frequency

    mdl_library = select_macros_by_mdl([cheap, costly], min_frequency=1, min_gain_nodes=1, max_macros=1)
    frequency_library = select_macros_by_frequency([cheap, costly], min_frequency=1, max_macros=1)

    assert mdl_library.macros[0].expansion_expr == expensive_key
    assert frequency_library.macros[0].expansion_expr == cheap_key


# --- Campaign wiring (exp-sr-9, fixture-only) -------------------------------


def test_rsp005_campaign_rejects_non_fixture_claim_class():
    with pytest.raises(ValueError, match="fixture-only"):
        Rsp005CampaignV1(claim_class="promotion_candidate")


def test_rsp005_campaign_manifest_matches_the_locked_exp_sr_9_catalogue_entry():
    campaign = Rsp005CampaignV1()
    manifest = campaign.manifest()
    assert manifest.claim_class == "fixture"
    assert manifest.experiment_id == CATALOGUE_ID
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].minimum_effect == campaign.minimum_effect()


def test_run_campaign_produces_fixture_evidence_and_never_promotes(tmp_path):
    campaign = Rsp005CampaignV1(campaign_id="test-rsp005-macro-library-fixture")
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert set(result["arms"]) == set(ARM_IDS)
    # The no-macros control arm must always show exactly zero reduction.
    assert result["arms"]["no_macros"]["reduction_rate"] == 0.0
    # Every admitted mdl_macros entry must have passed independent
    # expansion-equivalence re-verification for the result to be authorized.
    assert result["expansion_equivalence_all_verified"] or not result["authorized"]
    assert result["n_train_records"] > 0
    assert result["n_test_records"] > 0

    # CampaignStore actually persisted a lock + result artifact.
    assert (tmp_path / campaign.campaign_id / "campaign.json").is_file()
