from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from slm_training.data.flow.bridge_corpus import (
    ExactLegalEditCandidateSetV1,
    load_corpus,
)
from slm_training.models.legal_edit_batch import LegalEditBatch

FIXTURE = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)


def test_ragged_batch_masks_targets_and_padding() -> None:
    rows, candidate_sets, _ = load_corpus(FIXTURE)
    batch = LegalEditBatch.pack(rows, candidate_sets)
    assert batch.row_offsets.tolist() == [0, 10, 26, 35, 54]
    assert not bool((batch.unknown_mask & batch.unsupported_mask).any())
    for index in range(len(rows)):
        start, end = int(batch.row_offsets[index]), int(batch.row_offsets[index + 1])
        assert torch.isclose(
            batch.target_distribution[start:end].sum(), torch.tensor(1.0)
        )
    padded = batch.to_padded()
    assert padded.membership_mask.sum().item() == len(batch.candidate_ids)
    assert padded.candidate_features.shape[:2] == (4, 19)


def test_request_local_pointers_are_features_not_vocab_rows() -> None:
    rows, candidate_sets, _ = load_corpus(FIXTURE)
    batch = LegalEditBatch.pack(rows[:1], candidate_sets)
    assert "node_pointer" in batch.feature_names
    pointer_index = batch.feature_names.index("node_pointer")
    assert batch.candidate_features[:, pointer_index].numel() == len(
        batch.candidate_ids
    )
    assert all(candidate_id.startswith("edit_") for candidate_id in batch.candidate_ids)


def test_singleton_candidate_is_retained() -> None:
    rows, candidate_sets, _ = load_corpus(FIXTURE)
    row = rows[0]
    original = candidate_sets[row.candidate_set_digest]
    candidate = original.candidates[0]
    singleton = ExactLegalEditCandidateSetV1(
        state_fingerprint=original.state_fingerprint,
        candidates=(candidate,),
    )
    singleton_row = replace(
        row,
        candidate_set_digest=singleton.candidate_set_digest,
        candidate_set_ref=f"candidate_sets/{singleton.candidate_set_digest}.json",
        complete_candidate_ids=(candidate.candidate_id,),
        candidate_successors={
            candidate.candidate_id: candidate.successor_fingerprint
        },
        positive_candidate_ids=(candidate.candidate_id,),
        supported_candidate_ids=(candidate.candidate_id,),
        unsupported_candidate_ids=(),
        unknown_candidate_ids=(),
        planner_selected_candidate_id=candidate.candidate_id,
    )
    batch = LegalEditBatch.pack(
        [singleton_row], {singleton.candidate_set_digest: singleton}
    )
    assert batch.row_offsets.tolist() == [0, 1]
    assert batch.to_padded().membership_mask.tolist() == [[True]]


def test_projection_cannot_change_membership() -> None:
    rows, candidate_sets, _ = load_corpus(FIXTURE)
    batch = LegalEditBatch.pack(rows[:1], candidate_sets)
    values = torch.arange(len(batch.candidate_ids), dtype=torch.float32)
    assert torch.equal(
        batch.gathered_projection(values, lambda tensor, _: tensor), values
    )
    with pytest.raises(ValueError, match="membership"):
        batch.gathered_projection(values, lambda tensor, _: tensor[:-1])
    with pytest.raises(ValueError, match=r"shape \[N\]"):
        batch.gathered_projection(values[:-1], lambda tensor, _: tensor)


def test_two_objective_consumers_receive_identical_membership() -> None:
    rows, candidate_sets, _ = load_corpus(FIXTURE)
    batch = LegalEditBatch.pack(rows[:2], candidate_sets)

    def direct_policy_consumer(value: LegalEditBatch) -> tuple[int, tuple[str, ...]]:
        return id(value), value.candidate_ids

    def flow_rate_consumer(value: LegalEditBatch) -> tuple[int, tuple[str, ...]]:
        return id(value), value.candidate_ids

    assert direct_policy_consumer(batch) == flow_rate_consumer(batch)
