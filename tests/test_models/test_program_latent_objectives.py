import pytest
import torch

from slm_training.models.program_latent_objectives import (
    combined_program_latent_objective,
    trace_interpolation,
)


def test_binder_and_reference_margin_and_zero_weight_parity():
    reconstruction = torch.tensor(2.0)
    positive = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    negative = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    off = combined_program_latent_objective(reconstruction, positive, negative, contrast_margin=1.0, contrast_weight=0.0)
    on = combined_program_latent_objective(reconstruction, positive, negative, contrast_margin=1.0, contrast_weight=1.0)
    assert torch.equal(off.total, reconstruction)
    assert on.contrast > 0 and on.total > reconstruction


def test_locked_promotion_metrics_are_not_losses():
    with pytest.raises(ValueError, match="locked promotion"):
        combined_program_latent_objective(torch.tensor(0.0), torch.zeros((1, 2)), torch.ones((1, 2)), contrast_margin=1.0, contrast_weight=1.0, training_metrics=("meaning_v2",))


def test_interpolation_records_every_boundary_and_fails_closed():
    with pytest.raises(ValueError, match="required"):
        trace_interpolation(torch.zeros(1), torch.ones(1), steps=3, decode_latent=None, verify_program=None, factor_fingerprint=None)
    rows = trace_interpolation(
        torch.zeros(1), torch.ones(1), steps=3,
        decode_latent=lambda z: "binder" if z.item() < 0.5 else "reference",
        verify_program=lambda source: {"valid": source in {"binder", "reference"}},
        factor_fingerprint=lambda source: {"kind": source},
    )
    assert [row["alpha"] for row in rows] == [0.0, 0.5, 1.0]
    assert rows[1]["boundary_crossing"]
    assert all("verification" in row and "factor_fingerprint" in row for row in rows)
