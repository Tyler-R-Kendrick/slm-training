"""Tests for slm_training.models.semantic_plan_predictor (SLM-144)."""

from __future__ import annotations

import torch

from slm_training.models.semantic_plan_predictor import (
    ArchetypeClassifierHead,
    PlanBatchCollator,
    PlanTrainingExample,
    RoleSetPredictorHead,
    SerializedInventoryHead,
    build_role_set_target,
    predict_role_set_from_logits,
    predict_serialized_inventory,
    train_fixture_predictor,
)


def _example(
    *,
    example_id: str = "ex",
    input_dim: int = 4,
    archetype_label: int = 0,
    num_roles: int = 5,
    role_ids: tuple[str, ...] = ("r0", "r2"),
    role_vocab: dict[str, int] | None = None,
    max_len: int = 4,
) -> PlanTrainingExample:
    vocab = role_vocab or {f"r{i}": i for i in range(num_roles)}
    features = torch.randn(input_dim)
    mask = build_role_set_target(role_ids, vocab, num_roles)
    serial = torch.full((max_len,), -1, dtype=torch.long)
    for idx, role in enumerate(role_ids):
        if idx < max_len and role in vocab:
            serial[idx] = vocab[role]
    return PlanTrainingExample(
        example_id=example_id,
        input_features=features,
        archetype_label=archetype_label,
        role_set_mask=mask,
        serialized_roles=serial,
    )


def test_archetype_classifier_output_shape() -> None:
    head = ArchetypeClassifierHead(input_dim=4, num_archetypes=3)
    x = torch.randn(2, 4)
    logits = head(x)
    assert logits.shape == (2, 3)


def test_role_set_predictor_output_shape() -> None:
    head = RoleSetPredictorHead(input_dim=4, num_roles=5, num_slots=3)
    x = torch.randn(2, 4)
    logits = head(x)
    assert logits.shape == (2, 3, 6)  # 5 roles + blank


def test_serialized_inventory_output_shape() -> None:
    head = SerializedInventoryHead(input_dim=4, num_roles=5, max_len=4)
    x = torch.randn(2, 4)
    logits = head(x)
    assert logits.shape == (2, 4, 5)


def test_collator_stacks_examples() -> None:
    ex1 = _example(example_id="a", role_ids=("r0", "r2"))
    ex2 = _example(example_id="b", role_ids=("r1",))
    batch = PlanBatchCollator()([ex1, ex2])
    assert batch["example_ids"] == ["a", "b"]
    assert batch["input_features"].shape == (2, 4)
    assert batch["archetype_labels"].shape == (2,)
    assert batch["role_set_masks"].shape == (2, 5)
    assert batch["serialized_roles"].shape == (2, 4)


def test_train_fixture_predictor_improves_loss() -> None:
    examples = [
        _example(example_id=f"ex{i}", archetype_label=i % 2, role_ids=(f"r{i % 3}",))
        for i in range(16)
    ]
    result = train_fixture_predictor(
        examples,
        epochs=10,
        batch_size=4,
        seed=0,
    )
    history = result["history"]
    assert len(history) == 10
    assert history[-1]["loss"] < history[0]["loss"]
    assert "archetype_head" in result
    assert "role_set_head" in result
    assert "serialized_inventory_head" in result


def test_predict_role_set_from_logits_filters_blank() -> None:
    # One slot predicts role 2, another predicts blank (index 3 with 3 roles).
    logits = torch.tensor([[[1.0, 0.0, 0.0, 5.0], [0.0, 0.0, 6.0, 0.0]]])
    preds = predict_role_set_from_logits(logits[0], blank_role=3)
    assert preds == [2]


def test_predict_serialized_inventory_stops_at_pad_and_dedupes() -> None:
    logits = torch.zeros(1, 5, 5)
    # Greedy picks index 1 for steps 0 and 1, index 3 for step 2, then pad.
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 20.0
    logits[0, 2, 3] = 10.0
    logits[0, 3, 0] = 10.0  # step 3 -> 0, which is the pad class
    preds = predict_serialized_inventory(logits[0], pad_index=0)
    # The helper deduplicates the decoded role set.
    assert preds == [1, 3]


def test_empty_role_set_target() -> None:
    mask = build_role_set_target((), {"r0": 0}, 3)
    assert mask.tolist() == [0.0, 0.0, 0.0]
