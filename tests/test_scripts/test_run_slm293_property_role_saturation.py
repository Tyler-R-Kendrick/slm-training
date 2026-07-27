import torch

from scripts.run_slm293_property_role_saturation import (
    _factor_controls,
    _model,
    _records,
    _semantic_connector_audit,
)


def test_factor_controls_shuffle_across_examples() -> None:
    controls = _factor_controls(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        targets=torch.tensor([0, 1]),
    )

    assert controls["learned"]["strict_factor_accuracy"] == 1.0
    assert controls["gold_substituted"]["strict_factor_accuracy"] == 1.0
    assert controls["zeroed"]["eligible_choices"] == 2
    assert controls["shuffled"]["strict_factor_accuracy"] == 0.0


def test_semantic_connector_audit_fails_closed_when_unowned() -> None:
    audit = _semantic_connector_audit(_model(_records()))

    assert audit["connector_modules_installed"] == []
    assert audit["eligible_decode_positions"] == 0
    assert audit["choice_changes"] == 0
    assert audit["structural_meaning_v2_delta"] == 0.0
    assert audit["disposition"] == "non_causal_not_on_decoder_path"
