"""AP-016 (SLM-302): AbstractPlanV1 reservation, collision-freedom, and
default-off tokenizer parity."""

from __future__ import annotations

import json

import pytest

from tests.casefiles import case_values
import torch

from slm_training.dsl.abstract_plan import (
    AbstractPlanV1,
    RoleSpanV1,
    resize_embedding_preserving_rows,
    verify_embedding_resize_preserved_old_rows,
)
from slm_training.dsl.openui_tokens import (
    ABSTRACT_PLAN_BEGIN,
    ABSTRACT_PLAN_END,
    TOKEN_ID_NAMESPACE_RANGES,
)
from slm_training.models.choice_tokenizer import ChoiceTokenizer
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, TokenKind


# --- AbstractPlanV1 contract -------------------------------------------------


def test_default_plan_is_valid_and_non_interpretable() -> None:
    plan = AbstractPlanV1()
    assert plan.slot_count == 64
    assert plan.max_slot_count == 128
    assert plan.rounds == 3
    assert plan.begin_token == ABSTRACT_PLAN_BEGIN
    assert plan.end_token == ABSTRACT_PLAN_END
    assert not plan.is_interpretable
    assert len(plan.slot_tokens) == 64
    assert len(plan.token_ids) == 66  # begin + 64 slots + end


def test_token_ids_are_unique_and_inside_the_reserved_namespace() -> None:
    plan = AbstractPlanV1()
    ids = plan.token_ids
    assert len(set(ids)) == len(ids)
    ns = TOKEN_ID_NAMESPACE_RANGES["abstract_plan"]
    assert all(tid in ns for tid in ids)
    other_ranges = [
        r for name, r in TOKEN_ID_NAMESPACE_RANGES.items() if name != "abstract_plan"
    ]
    assert all(tid not in other for tid in ids for other in other_ranges)


def test_to_dict_from_dict_round_trip() -> None:
    plan = AbstractPlanV1(slot_count=8, role_metadata=tuple([None] * 8))
    restored = AbstractPlanV1.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.compatibility_fingerprint == plan.compatibility_fingerprint


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slot_count": 0},
        {"slot_count": 200},
        {"max_slot_count": 0},
        {"max_slot_count": 999_999},
        {"rounds": 0},
        {"role_metadata": ("bad role!",) * 64},
        {"slot_count": 4, "role_metadata": (None, None)},
        {"source": "unknown_source"},
        {"schema": "abstract_plan/v2"},
        {"plan_version": "2"},
        {"codebook_version": 0},
        {"codebook_version": 2},
        {"begin_token": "<other_begin>"},
        {"end_token": "<other_end>"},
        {"provenance": ""},
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AbstractPlanV1(**kwargs)


def test_non_canonical_delimiters_and_versions_are_rejected_explicitly() -> None:
    """AP-016 review: a V1 plan cannot claim a mapping its ids don't back."""
    with pytest.raises(ValueError, match="begin_token"):
        AbstractPlanV1(begin_token="<not_canonical>")
    with pytest.raises(ValueError, match="end_token"):
        AbstractPlanV1(end_token="<not_canonical>")
    with pytest.raises(ValueError, match="codebook_version"):
        AbstractPlanV1(codebook_version=2)


def test_assert_no_collisions_passes_against_default_off_vocab() -> None:
    plan = AbstractPlanV1()
    dsl_tok = DSLNativeTokenizer.build()
    choice_tok = ChoiceTokenizer.build()
    plan.assert_no_collisions(dsl_tok.token_to_id)
    plan.assert_no_collisions(choice_tok.token_to_id)


def test_assert_no_collisions_fails_closed_on_overlap() -> None:
    plan = AbstractPlanV1(slot_count=4)
    with pytest.raises(ValueError, match="collide"):
        plan.assert_no_collisions({"<ABS_0>", "unrelated"})
    with pytest.raises(ValueError, match="collide"):
        plan.assert_no_collisions({ABSTRACT_PLAN_BEGIN})


# --- RoleSpanV1 / role_spans (AP-025 / SLM-318) -----------------------------


def test_role_span_valid_defaults() -> None:
    span = RoleSpanV1(role="intent", start=0, length=4)
    assert span.end == 4
    assert span.max_length is None


@pytest.mark.parametrize(
    "kwargs",
    case_values(__file__, "test_role_span_rejects_invalid_configurations"),
)
def test_role_span_rejects_invalid_configurations(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RoleSpanV1(**kwargs)


def test_role_span_to_dict_from_dict_round_trip() -> None:
    span = RoleSpanV1(role="topology", start=4, length=3, max_length=2)
    restored = RoleSpanV1.from_dict(span.to_dict())
    assert restored == span


def _role_factorized_plan() -> AbstractPlanV1:
    return AbstractPlanV1(
        slot_count=8,
        max_slot_count=8,
        role_spans=(
            RoleSpanV1(role="intent", start=0, length=3),
            RoleSpanV1(role="topology", start=3, length=5, max_length=2),
        ),
    )


def test_role_factorized_plan_is_interpretable_and_reports_roles() -> None:
    plan = _role_factorized_plan()
    assert plan.is_role_factorized
    assert plan.is_interpretable
    assert plan.role_names == ("intent", "topology")
    assert plan.role_for_slot(0) == "intent"
    assert plan.role_for_slot(2) == "intent"
    assert plan.role_for_slot(3) == "topology"
    assert plan.role_for_slot(7) == "topology"


def test_role_slot_indices_and_token_ids_and_mask() -> None:
    plan = _role_factorized_plan()
    assert plan.role_slot_indices("intent") == (0, 1, 2)
    assert plan.role_slot_indices("topology") == (3, 4, 5, 6, 7)
    assert plan.role_slot_token_ids("intent") == plan.slot_token_ids[:3]
    mask = plan.role_slot_mask("intent")
    assert mask == (True, True, True, False, False, False, False, False)


def test_role_span_lookup_rejects_unknown_role() -> None:
    plan = _role_factorized_plan()
    with pytest.raises(KeyError):
        plan.role_span("unknown")
    with pytest.raises(KeyError):
        plan.role_slot_indices("unknown")


def test_homogeneous_plan_has_no_roles() -> None:
    plan = AbstractPlanV1()
    assert not plan.is_role_factorized
    assert plan.role_names == ()
    assert plan.role_for_slot(0) is None


def test_role_spans_reject_overlap_and_out_of_order() -> None:
    with pytest.raises(ValueError, match="overlap"):
        AbstractPlanV1(
            slot_count=8,
            max_slot_count=8,
            role_spans=(
                RoleSpanV1(role="intent", start=0, length=4),
                RoleSpanV1(role="topology", start=3, length=4),
            ),
        )


def test_role_spans_reject_duplicate_role_name() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AbstractPlanV1(
            slot_count=8,
            max_slot_count=8,
            role_spans=(
                RoleSpanV1(role="intent", start=0, length=2),
                RoleSpanV1(role="intent", start=2, length=2),
            ),
        )


def test_role_spans_reject_range_exceeding_slot_count() -> None:
    with pytest.raises(ValueError, match="exceeds slot_count"):
        AbstractPlanV1(
            slot_count=4,
            max_slot_count=4,
            role_spans=(RoleSpanV1(role="intent", start=0, length=5),),
        )


def test_role_spans_and_role_metadata_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="alternate role"):
        AbstractPlanV1(
            slot_count=4,
            max_slot_count=4,
            role_metadata=(None, None, None, None),
            role_spans=(RoleSpanV1(role="intent", start=0, length=4),),
        )


def test_role_factorized_plan_to_dict_from_dict_round_trip() -> None:
    plan = _role_factorized_plan()
    restored = AbstractPlanV1.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.compatibility_fingerprint == plan.compatibility_fingerprint


def test_role_spans_do_not_change_token_ids_or_collide() -> None:
    """Spans only label existing slot indices -- no new reserved token text,
    so collision-freedom and the codebook_version pin are both unaffected."""
    homogeneous = AbstractPlanV1(slot_count=8, max_slot_count=8)
    factorized = _role_factorized_plan()
    assert homogeneous.token_ids == factorized.token_ids
    factorized.assert_no_collisions({"unrelated"})


# --- Embedding resize ---------------------------------------------------


def test_resize_embedding_preserves_old_rows_bit_exactly() -> None:
    torch.manual_seed(0)
    original = torch.nn.Embedding(10, 4)
    resized = resize_embedding_preserving_rows(original, 14)
    assert resized.weight.shape == (14, 4)
    assert torch.equal(resized.weight[:10], original.weight)
    verify_embedding_resize_preserved_old_rows(original, resized)


def test_resize_embedding_rejects_shrink() -> None:
    original = torch.nn.Embedding(10, 4)
    with pytest.raises(ValueError, match="shrink"):
        resize_embedding_preserving_rows(original, 5)


def test_resize_embedding_noop_when_same_size() -> None:
    original = torch.nn.Embedding(10, 4)
    same = resize_embedding_preserving_rows(original, 10)
    assert same is original


def test_verify_resize_fails_closed_on_mutated_old_rows() -> None:
    original = torch.nn.Embedding(4, 3)
    resized = resize_embedding_preserving_rows(original, 6)
    with torch.no_grad():
        resized.weight[0, 0] += 1.0
    with pytest.raises(ValueError, match="mutated"):
        verify_embedding_resize_preserved_old_rows(original, resized)


def test_resize_embedding_preserves_non_default_config_and_forward_behavior() -> None:
    torch.manual_seed(0)
    original = torch.nn.Embedding(
        5,
        3,
        padding_idx=0,
        max_norm=0.5,
        norm_type=2.0,
        scale_grad_by_freq=True,
    )
    with torch.no_grad():
        original.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    resized = resize_embedding_preserving_rows(original, 8)
    verify_embedding_resize_preserved_old_rows(original, resized)

    # Config carried over, not just values.
    assert resized.padding_idx == original.padding_idx
    assert resized.max_norm == original.max_norm
    assert resized.norm_type == original.norm_type
    assert resized.scale_grad_by_freq == original.scale_grad_by_freq

    # Forward behavior on pre-existing rows (incl. max_norm renormalization)
    # is identical, not just the stored weights.
    idx = torch.tensor([0, 1, 2, 3, 4])
    assert torch.allclose(original(idx), resized(idx))


def test_resize_embedding_preserves_requires_grad() -> None:
    original = torch.nn.Embedding(4, 3)
    original.weight.requires_grad_(False)
    resized = resize_embedding_preserving_rows(original, 6)
    assert resized.weight.requires_grad is False
    verify_embedding_resize_preserved_old_rows(original, resized)

    resized.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="requires_grad"):
        verify_embedding_resize_preserved_old_rows(original, resized)


# --- Tokenizer wiring: default-off parity + append-only enablement ---------


def test_dsl_native_tokenizer_feature_off_parity() -> None:
    baseline = DSLNativeTokenizer.build()
    explicit_off = DSLNativeTokenizer.build(abstract_plan_slots=0)
    assert baseline.token_to_id == explicit_off.token_to_id
    assert baseline.id_to_kind == explicit_off.id_to_kind
    assert baseline.vocab_size == explicit_off.vocab_size


def test_dsl_native_tokenizer_enabled_is_append_only() -> None:
    baseline = DSLNativeTokenizer.build()
    enabled = DSLNativeTokenizer.build(abstract_plan_slots=8)
    for tok, tid in baseline.token_to_id.items():
        assert enabled.token_to_id[tok] == tid
    assert enabled.vocab_size == baseline.vocab_size + 10  # begin + 8 slots + end
    assert enabled.is_abstract_id(enabled.abstract_begin_id)
    assert enabled.is_abstract_id(enabled.abstract_end_id)
    assert enabled.kind_of(enabled.abstract_slot_id(3)) == TokenKind.ABSTRACT
    assert enabled.abstract_slot_of(enabled.abstract_slot_id(3)) == 3


def test_dsl_native_tokenizer_rejects_out_of_range_slots() -> None:
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        DSLNativeTokenizer.build(abstract_plan_slots=129)
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        DSLNativeTokenizer.build(abstract_plan_slots=-1)


def test_dsl_native_tokenizer_load_rejects_out_of_range_slots(tmp_path) -> None:
    tok = DSLNativeTokenizer.build(abstract_plan_slots=5)
    path = tmp_path / "dsl_native_bad.json"
    tok.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["abstract_plan_slots"] = -1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        DSLNativeTokenizer.load(path)


def test_dsl_native_tokenizer_save_load_round_trip(tmp_path) -> None:
    tok = DSLNativeTokenizer.build(abstract_plan_slots=5)
    path = tmp_path / "dsl_native.json"
    tok.save(path)
    loaded = DSLNativeTokenizer.load(path)
    assert loaded.token_to_id == tok.token_to_id
    assert loaded.abstract_plan_slots == 5


def test_choice_tokenizer_feature_off_parity() -> None:
    baseline = ChoiceTokenizer.build()
    explicit_off = ChoiceTokenizer.build(abstract_plan_slots=0)
    assert baseline.token_to_id == explicit_off.token_to_id
    assert baseline.id_to_kind == explicit_off.id_to_kind


def test_choice_tokenizer_enabled_is_append_only() -> None:
    baseline = ChoiceTokenizer.build()
    enabled = ChoiceTokenizer.build(abstract_plan_slots=6)
    for tok, tid in baseline.token_to_id.items():
        assert enabled.token_to_id[tok] == tid
    assert enabled.vocab_size == baseline.vocab_size + 8  # begin + 6 slots + end
    assert enabled.is_abstract_id(enabled.abstract_begin_id)
    assert enabled.abstract_slot_id(2) in enabled.id_to_token


def test_choice_tokenizer_save_load_round_trip(tmp_path) -> None:
    tok = ChoiceTokenizer.build(abstract_plan_slots=3)
    path = tmp_path / "choice.json"
    tok.save(path)
    loaded = ChoiceTokenizer.load(path)
    assert loaded.token_to_id == tok.token_to_id
    assert loaded.abstract_plan_slots == 3


def test_choice_tokenizer_rejects_out_of_range_slots() -> None:
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        ChoiceTokenizer.build(abstract_plan_slots=129)
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        ChoiceTokenizer.build(abstract_plan_slots=-1)


def test_choice_tokenizer_load_rejects_out_of_range_slots(tmp_path) -> None:
    tok = ChoiceTokenizer.build(abstract_plan_slots=3)
    path = tmp_path / "choice_bad.json"
    tok.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["abstract_plan_slots"] = -1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="abstract_plan_slots"):
        ChoiceTokenizer.load(path)
