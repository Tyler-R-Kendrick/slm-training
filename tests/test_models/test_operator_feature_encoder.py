"""Tests for the DSH3-23 typed operator-feature encoder and matched arms."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from slm_training.models.operator_feature_encoder import (
    FeatureArm,
    FixtureOperatorDecisionV1,
    OperatorFeatureVocabularyV1,
    OperatorFeatureEncoder,
    make_fixture_operator_decisions,
    permute_fixture_decision,
    row_identity_bucket,
    run_matched_arms_fixture,
    train_fixture_arm,
)
from slm_training.models.operator_policy_view import tag_recently_touched_rows


def test_fixture_decisions_distinguish_gold_by_content_not_position() -> None:
    decisions = make_fixture_operator_decisions(20, seed=1, n_candidates=5)
    positions = {decision.gold_row for decision in decisions}
    assert len(positions) > 1, "fixture should vary the gold row's position"
    for decision in decisions:
        gold_view = decision.view.reference_rows[decision.gold_row]
        assert gold_view.value_type == "openui.number"
        others = [
            row
            for row in decision.view.reference_rows
            if row.row != decision.gold_row
        ]
        assert all(row.value_type == "openui.string" for row in others)


def test_every_arm_encodes_to_finite_matched_shapes() -> None:
    decisions = make_fixture_operator_decisions(3, seed=2, n_candidates=4)
    vocabulary = OperatorFeatureVocabularyV1.from_inputs([d.view for d in decisions])
    for arm in FeatureArm:
        torch.manual_seed(0)
        encoder = OperatorFeatureEncoder(vocabulary, dim=6, arm=arm)
        for decision in decisions:
            action_embeddings = encoder.encode(decision.view)
            reference_embeddings = encoder.encode_reference_rows(decision.view)
            assert action_embeddings.shape == (1, 6)
            assert reference_embeddings.shape == (4, 6)
            assert torch.isfinite(action_embeddings).all()
            assert torch.isfinite(reference_embeddings).all()


def test_typed_and_hash_scalar_arms_are_permutation_invariant() -> None:
    decision = make_fixture_operator_decisions(1, seed=3, n_candidates=5)[0]
    permuted = permute_fixture_decision(decision, seed=99)
    assert permuted.view.reference_rows != decision.view.reference_rows

    vocabulary = OperatorFeatureVocabularyV1.from_inputs([decision.view])
    for arm in (FeatureArm.TYPED, FeatureArm.HASH_SCALAR):
        torch.manual_seed(0)
        encoder = OperatorFeatureEncoder(vocabulary, dim=6, arm=arm)

        def embedding_by_content(view):
            embeddings = encoder.encode_reference_rows(view)
            return sorted(
                (row.ref_kind.value, row.value_type, tuple(embeddings[row.row].tolist()))
                for row in view.reference_rows
            )

        original = embedding_by_content(decision.view)
        reshuffled = embedding_by_content(permuted.view)
        assert len(original) == len(reshuffled)
        for (kind_a, type_a, vector_a), (kind_b, type_b, vector_b) in zip(
            original, reshuffled
        ):
            assert kind_a == kind_b
            assert type_a == type_b
            assert vector_a == pytest.approx(vector_b, abs=1e-5)


def test_identity_bucket_arm_is_not_guaranteed_permutation_invariant() -> None:
    """The labeled anti-generalization control can see row position — unlike
    TYPED/HASH_SCALAR, its per-row embedding is allowed to change under a
    permutation that preserves every allowed content field."""
    decision = make_fixture_operator_decisions(1, seed=4, n_candidates=6)[0]
    permuted = permute_fixture_decision(decision, seed=17)

    vocabulary = OperatorFeatureVocabularyV1.from_inputs([decision.view])
    torch.manual_seed(0)
    encoder = OperatorFeatureEncoder(
        vocabulary, dim=6, arm=FeatureArm.TYPED_IDENTITY_BUCKET, n_identity_buckets=6
    )
    gold_original = encoder.encode_reference_rows(decision.view)[decision.gold_row]
    gold_permuted = encoder.encode_reference_rows(permuted.view)[permuted.gold_row]
    assert not torch.allclose(gold_original, gold_permuted)


# --------------------------------------------------------------------------- #
# recently_touched (SLM-418 ninth slice): a real content field both arms read.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", [FeatureArm.TYPED, FeatureArm.HASH_SCALAR])
def test_recently_touched_changes_the_reference_embedding_in_every_arm(arm) -> None:
    """Both matched arms must actually consume the field.

    The DSH3-23 arms differ only in *how* fields are encoded, never in which
    fields are visible — a typed-only field would confound the comparison.
    """
    decision = make_fixture_operator_decisions(1, seed=11, n_candidates=4)[0]
    tagged = replace(
        decision, view=tag_recently_touched_rows(decision.view, (decision.gold_row,))
    )
    vocabulary = OperatorFeatureVocabularyV1.from_inputs([decision.view])
    torch.manual_seed(0)
    encoder = OperatorFeatureEncoder(vocabulary, dim=6, arm=arm)

    plain = encoder.encode_reference_rows(decision.view)
    marked = encoder.encode_reference_rows(tagged.view)
    assert not torch.allclose(plain[decision.gold_row], marked[decision.gold_row])
    for row in range(len(decision.view.reference_rows)):
        if row != decision.gold_row:
            assert torch.allclose(plain[row], marked[row])


@pytest.mark.parametrize("arm", [FeatureArm.TYPED, FeatureArm.HASH_SCALAR])
def test_recently_touched_stays_permutation_invariant(arm) -> None:
    """The regression the field's whole admissibility rests on.

    Under a row-order/opaque-ID permutation that preserves all content, the
    tagged row's embedding must be unchanged (it travels with its row, not
    with its index) — and must stay distinct from an untagged row that is
    otherwise identical in every allowed field.
    """
    plain = make_fixture_operator_decisions(1, seed=13, n_candidates=5)[0]
    # Tag a *non-gold* row, so the tagged row and its siblings are identical
    # in every other allowed field ("openui.string", no parent, same facts) --
    # only this bit can separate them, which is exactly the situation the
    # field was added for.
    focus_row = next(row for row in range(5) if row != plain.gold_row)
    sibling_row = next(
        row for row in range(5) if row not in (plain.gold_row, focus_row)
    )
    decision = FixtureOperatorDecisionV1(
        view=tag_recently_touched_rows(plain.view, (focus_row,)),
        gold_row=plain.gold_row,
    )
    permuted = permute_fixture_decision(decision, seed=77)
    permuted_focus = next(
        row.row for row in permuted.view.reference_rows if row.recently_touched
    )
    assert permuted_focus != focus_row, "fixture did not actually move the tagged row"
    assert sum(row.recently_touched for row in permuted.view.reference_rows) == 1
    # The sanitized canonical payload is identical under the permutation.
    assert decision.view.to_dict() == permuted.view.to_dict()

    vocabulary = OperatorFeatureVocabularyV1.from_inputs([decision.view])
    torch.manual_seed(0)
    encoder = OperatorFeatureEncoder(vocabulary, dim=6, arm=arm)

    original = encoder.encode_reference_rows(decision.view)
    reshuffled = encoder.encode_reference_rows(permuted.view)
    assert torch.allclose(original[focus_row], reshuffled[permuted_focus], atol=1e-6)
    # ...and the bit is real signal: an otherwise-identical untagged sibling
    # does not collapse onto it.
    assert not torch.allclose(original[focus_row], original[sibling_row])


def test_row_identity_bucket_wraps_and_validates() -> None:
    assert row_identity_bucket(0, 4) == 0
    assert row_identity_bucket(5, 4) == 1
    with pytest.raises(ValueError):
        row_identity_bucket(0, 0)


def test_train_fixture_arm_reduces_loss() -> None:
    decisions = make_fixture_operator_decisions(12, seed=5, n_candidates=4)
    trained = train_fixture_arm(decisions, arm=FeatureArm.TYPED, dim=6, steps=40, lr=0.1, seed=0)
    assert trained["history"][-1] <= trained["history"][0]


def test_matched_arms_fixture_report_shape_and_disposition() -> None:
    report = run_matched_arms_fixture(n_train=8, n_eval=6, n_candidates=4, steps=15)
    payload = report.to_dict()
    assert payload["claim_class"] == "wiring"
    assert set(payload["arms"]) == {"hash_scalar", "typed", "typed_identity_bucket"}
    for arm_metrics in payload["arms"].values():
        assert 0.0 <= arm_metrics["top1_recall"] <= 1.0
        assert 0.0 <= arm_metrics["ndcg"] <= 1.0
        assert 0.0 <= arm_metrics["accepted_set_mass"] <= 1.0
    assert payload["permutation_robustness"]["typed"]["invariant"] is True
    assert payload["permutation_robustness"]["hash_scalar"]["invariant"] is True
    assert payload["disposition"] in {
        "wiring_underpowered",
        "supported",
        "closed_negative",
        "invalid_permutation_variance",
    }
    # Fixture scale never clears the powered bar in this repo's own convention.
    assert payload["disposition"] != "invalid_permutation_variance"
