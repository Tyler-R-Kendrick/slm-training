"""V6 CoRe remask / T2M selection tests."""

from __future__ import annotations

import torch

from slm_training.models.parallel_decode import (
    core_instability_scores,
    perturb_known_neighbors,
    select_remask_core_indices,
    select_remask_indices,
)


def test_core_instability_prefers_brittle_tokens() -> None:
    # Vocab size 3; committed ids = [0, 1, 2]
    probs = torch.tensor(
        [
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
            ]
        ]
    )
    # After perturbation, middle token support collapses.
    probs_pert = torch.tensor(
        [
            [
                [0.85, 0.1, 0.05],
                [0.4, 0.2, 0.4],
                [0.05, 0.1, 0.85],
            ]
        ]
    )
    ids = torch.tensor([[0, 1, 2]])
    known = torch.tensor([[True, True, True]])
    inst = core_instability_scores(probs, probs_pert, ids, known)
    assert float(inst[0, 1]) > float(inst[0, 0])
    assert float(inst[0, 1]) > float(inst[0, 2])


def test_select_remask_core_indices_ranks_instability() -> None:
    conf = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    known = torch.tensor([[True, True, True, True]])
    inst = torch.tensor([[0.01, 0.9, 0.02, 0.8]])
    idxs = select_remask_core_indices(
        conf,
        known,
        remask_ratio=0.5,
        instability=inst,
    )
    assert 0 not in idxs  # BOS protected
    assert 1 in idxs
    assert len(idxs) >= 1


def test_perturb_known_neighbors_masks_fraction() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    known = torch.tensor([[True, True, True, True, True]])
    out = perturb_known_neighbors(
        ids, known, mask_id=99, perturb_frac=0.5, protect_bos=True
    )
    assert int(out[0, 0].item()) == 1  # BOS preserved
    assert int((out == 99).sum().item()) >= 1


def test_select_remask_core_falls_back_to_confidence() -> None:
    conf = torch.tensor([[0.9, 0.1, 0.8, 0.05]])
    known = torch.tensor([[True, True, True, True]])
    idxs = select_remask_core_indices(conf, known, remask_ratio=0.5, instability=None)
    assert 0 not in idxs
    # Should behave like confidence remask when instability is absent.
    classic = select_remask_indices(conf, known, remask_ratio=0.5)
    assert set(idxs) & set(classic)
