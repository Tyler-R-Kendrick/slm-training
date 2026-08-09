"""SRP-008 (SLM-471) regression tests: deterministic symbolic-regression
corpus generation with topology-family leakage controls and OOD slices."""

from __future__ import annotations

import numpy as np
import pytest

from slm_training.dsl.symbolic_expr_corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusGenerationConfigV1,
    audit_corpus_leakage,
    generate_corpus,
)
from slm_training.dsl.symbolic_expr_fit import fit_least_squares
from slm_training.dsl.symbolic_expr_ir import parse_expr
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

_BALANCED_SPLIT_POLICY = RootFamilySplitPolicyV1(modulus=4, validation_buckets=(2,), test_buckets=(3,))


def _config(**overrides: object) -> CorpusGenerationConfigV1:
    fields = dict(
        schema_version=CORPUS_SCHEMA_VERSION,
        seed=2026,
        num_problems=30,
        num_variables_range=(1, 3),
        allowed_operators=frozenset({"+", "-", "*", "sin", "cos"}),
        max_basis_depth=3,
        max_coefficient_terms=2,
        coefficient_probability=0.7,
        numeric_domain_policy="reject_nonfinite",
        complexity_budget=64,
        train_rows=16,
        validation_rows=6,
        extrapolation_rows=6,
        train_domain=(-3.0, 3.0),
        extrapolation_domain=(4.0, 7.0),
        coefficient_range=(-5.0, 5.0),
        split_policy=_BALANCED_SPLIT_POLICY,
    )
    fields.update(overrides)
    return CorpusGenerationConfigV1(**fields)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_rebuild_from_same_seed_is_content_identical():
    corpus_a = generate_corpus(_config())
    corpus_b = generate_corpus(_config())
    assert corpus_a.corpus_fingerprint == corpus_b.corpus_fingerprint
    assert corpus_a == corpus_b


def test_different_seed_produces_a_different_fingerprint():
    corpus_a = generate_corpus(_config(seed=1))
    corpus_b = generate_corpus(_config(seed=2))
    assert corpus_a.corpus_fingerprint != corpus_b.corpus_fingerprint


# ---------------------------------------------------------------------------
# Canonical/equivalent equations never cross train/eval lineage
# ---------------------------------------------------------------------------


def test_no_family_is_duplicated_or_spans_more_than_one_split():
    corpus = generate_corpus(_config())
    audit = audit_corpus_leakage(corpus)
    assert audit.is_clean
    assert audit.cross_split_family_violations == ()
    assert audit.duplicate_family_ids == ()


def test_corpus_construction_itself_rejects_a_duplicated_family_id():
    corpus = generate_corpus(_config())
    tampered_records = corpus.records + (corpus.records[0],)
    with pytest.raises(ValueError, match="must not repeat a family_id"):
        type(corpus)(
            schema_version=corpus.schema_version,
            seed=corpus.seed,
            records=tampered_records,
            rejections=corpus.rejections,
            ood_slices=corpus.ood_slices,
            corpus_fingerprint=corpus.corpus_fingerprint,
        )


# ---------------------------------------------------------------------------
# OOD slices are frozen, named, and reproducible
# ---------------------------------------------------------------------------


def test_ood_slices_are_named_reproducible_and_disjoint_from_train():
    corpus = generate_corpus(_config())
    train_family_ids = {r.family_id for r in corpus.records if r.corpus_split == "train"}
    slices = corpus.ood_slices

    assert len(slices.unseen_topology) > 0  # the balanced split policy guarantees non-train families
    for family_id in slices.unseen_topology:
        assert family_id not in train_family_ids
    for slice_name in (
        "unseen_topology",
        "unseen_operator_composition",
        "unseen_variable_dimensionality",
        "unseen_coefficient_regime",
    ):
        values = getattr(slices, slice_name)
        assert tuple(sorted(values)) == values  # deterministic ordering

    corpus_again = generate_corpus(_config())
    assert corpus.ood_slices == corpus_again.ood_slices


# ---------------------------------------------------------------------------
# Degenerate/constant/invalid datasets are filtered with recorded reasons
# ---------------------------------------------------------------------------


def test_degenerate_candidates_are_filtered_with_a_recorded_reason_not_silently():
    # A single variable, depth-1, addition-only, coefficient-free config has
    # very few distinct canonical families -- generating even a couple of
    # problems is very likely to hit repeats along the way.
    config = _config(
        seed=0,
        num_problems=2,
        num_variables_range=(1, 1),
        allowed_operators=frozenset({"+"}),
        max_basis_depth=1,
        coefficient_probability=0.0,
    )
    corpus = generate_corpus(config)
    assert corpus.rejections  # some candidate was filtered, not silently dropped
    for rejection in corpus.rejections:
        assert rejection.reason in {"duplicate_family", "constant_expression", "over_budget"}


def test_unreachable_problem_count_fails_closed_instead_of_returning_fewer():
    config = _config(
        seed=7,
        num_problems=10,
        num_variables_range=(1, 1),
        allowed_operators=frozenset({"+"}),
        max_basis_depth=1,
        coefficient_probability=0.0,
    )
    with pytest.raises(ValueError, match="generated only"):
        generate_corpus(config)


# ---------------------------------------------------------------------------
# Target equation/coefficients never leak into the model-visible problem
# ---------------------------------------------------------------------------


def test_oracle_truth_never_appears_on_the_model_visible_problem():
    # provenance is the only free-form field SymbolicRegressionProblemV1 has
    # through which the equation/coefficients could leak; every other field
    # is a fixed-vocabulary enum, a variable-name list, or numeric data.
    corpus = generate_corpus(_config())
    for record in corpus.records:
        assert record.problem.provenance == {}


# ---------------------------------------------------------------------------
# Known generated examples with exact oracle recovery
# ---------------------------------------------------------------------------


def test_generated_coefficients_are_exactly_recoverable_by_fitting():
    config = _config(seed=11, num_problems=10, coefficient_probability=1.0)
    corpus = generate_corpus(config)
    coefficient_bearing = [r for r in corpus.records if r.oracle.true_coefficients]
    assert coefficient_bearing

    for record in coefficient_bearing:
        node = parse_expr(
            record.oracle.canonical_expr,
            num_variables=len(record.problem.variables),
            allowed_operators=record.problem.allowed_operators,
        )
        fit = fit_least_squares(
            node,
            variables=np.array(record.problem.table, dtype=np.float64),
            targets=np.array(record.problem.targets, dtype=np.float64),
            train_indices=np.array(record.problem.train_indices, dtype=np.intp),
            domain_policy=record.problem.numeric_domain_policy,
        )
        assert fit.fitted_coefficients == pytest.approx(record.oracle.true_coefficients, abs=1e-6)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_problems": 0}, "num_problems"),
        ({"num_variables_range": (0, 1)}, "num_variables_range"),
        ({"num_variables_range": (3, 1)}, "num_variables_range"),
        ({"allowed_operators": frozenset()}, "allowed_operators"),
        ({"allowed_operators": frozenset({"???"})}, "allowed_operators"),
        ({"max_basis_depth": 0}, "max_basis_depth"),
        ({"max_coefficient_terms": 0}, "max_coefficient_terms"),
        ({"coefficient_probability": 1.5}, "coefficient_probability"),
        ({"numeric_domain_policy": "nope"}, "numeric_domain_policy"),
        ({"complexity_budget": 0}, "complexity_budget"),
        ({"train_rows": 0}, "train_rows"),
        ({"extrapolation_rows": -1}, "train_rows"),
        ({"train_domain": (3.0, -3.0)}, "train_domain"),
        ({"extrapolation_domain": (1.0, -1.0)}, "extrapolation_domain"),
        ({"extrapolation_domain": (-1.0, 1.0)}, "extrapolation_domain"),
        ({"coefficient_range": (5.0, -5.0)}, "coefficient_range"),
    ],
)
def test_config_rejects_malformed_input(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)
