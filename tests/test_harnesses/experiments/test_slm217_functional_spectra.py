"""Tests for exact-state functional spectral diagnostics."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from slm_training.harnesses.experiments.slm217_functional_spectra import (
    FunctionalObservationV1,
    StreamingCovariance,
    analyze_functional_spectrum,
    capture_linear_inputs,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionStateV2,
    split_for_group,
)


def _state(tag: str, *, kind: str = "component", role: str = "slot") -> DecisionStateV2:
    group_id = f"group-{kind}"
    return DecisionStateV2(
        state_id="",
        group_id=group_id,
        architecture="twotower",
        context_text=f"context {tag}",
        context_ids=None,
        canvas_ids=(1, 2, 3),
        decision_position=1,
        generation_step=0,
        legal_action_ids=(4, 5),
        decision_kind=kind,
        abstract_state_role=role,
        grammar_state_hash="grammar",
        policy_checkpoint_sha="checkpoint",
        tokenizer_sha="tokenizer",
        decode_config_hash="decode",
        verifier_bundle_hash="verifier",
        split=split_for_group(group_id),
    )


def _observations(
    values: list[tuple[float, ...]],
    *,
    kind: str = "component",
    role: str = "slot",
) -> list[FunctionalObservationV1]:
    return [
        FunctionalObservationV1.from_state(
            _state(str(index), kind=kind, role=role),
            module_path="probe",
            values=torch.tensor(row),
        )
        for index, row in enumerate(values)
    ]


def test_streaming_covariance_matches_batch_covariance() -> None:
    rows = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [4.0, 0.0, 2.0]],
        dtype=torch.float64,
    )
    accumulator = StreamingCovariance(3)
    accumulator.update(rows[:1])
    accumulator.update(rows[1:])
    assert torch.allclose(accumulator.covariance(), torch.cov(rows.T))


def test_linear_orientation_and_isotropic_reduction() -> None:
    weight = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    rows = [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ]
    snapshot = analyze_functional_spectrum(
        weight,
        _observations(rows),
        checkpoint_sha="checkpoint",
        semantic_role="probe",
        base_spectral_reference="base",
        ridge=0.0,
        bootstrap_draws=4,
    )
    assert snapshot.orientation.startswith("PyTorch nn.Linear weight")
    assert snapshot.functional_singular_values == pytest.approx(
        tuple(value * (4 / 3) ** 0.5 for value in (3.0, 2.0, 1.0))
    )
    assert snapshot.isotropic_null_esd_distance == pytest.approx(0.0)


def test_low_support_and_rank_deficiency_fail_closed() -> None:
    snapshot = analyze_functional_spectrum(
        torch.eye(3),
        _observations([(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]),
        checkpoint_sha="checkpoint",
        semantic_role="probe",
        base_spectral_reference="base",
        ridge=0.0,
        min_support=4,
        bootstrap_draws=2,
    )
    assert snapshot.eligibility in {
        "ineligible_low_support",
        "ineligible_rank_deficient",
    }
    assert snapshot.covariance_rank == 1
    assert snapshot.warnings


def test_split_or_decision_kind_cannot_mix() -> None:
    rows = _observations([(1.0, 0.0), (0.0, 1.0)])
    rows[1] = replace(rows[1], decision_kind="binding")
    with pytest.raises(ValueError, match="cannot mix"):
        analyze_functional_spectrum(
            torch.eye(2),
            rows,
            checkpoint_sha="checkpoint",
            semantic_role="probe",
            base_spectral_reference="base",
        )


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.probe(inputs)


def test_targeted_capture_preserves_outputs_mode_and_selected_rows() -> None:
    model = _Probe()
    model.train()
    states = [_state("a"), _state("b")]
    inputs = {
        states[0].state_id: torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
        states[1].state_id: torch.tensor([[4.0, 5.0, 6.0], [8.0, 8.0, 8.0]]),
    }
    expected = [model(inputs[state.state_id]) for state in states]
    observations, outputs = capture_linear_inputs(
        model,
        module_path="probe",
        states=states,
        run_state=lambda state: model(inputs[state.state_id]),
        select_input=lambda _state, tensor: tensor[:1],
    )
    assert model.training
    assert len(observations) == 2
    assert observations[0].values == (1.0, 2.0, 3.0)
    for actual, wanted in zip(outputs, expected, strict=True):
        assert torch.equal(actual, wanted)


def test_nulls_bootstrap_and_hashes_are_deterministic() -> None:
    observations = _observations(
        [(1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0)]
    )
    nulls = _observations(
        [(1.0, 1.0), (-1.0, 1.0), (2.0, 2.0), (-2.0, 2.0)],
        kind="binding",
    )
    kwargs = dict(
        checkpoint_sha="checkpoint",
        semantic_role="probe",
        base_spectral_reference="base",
        init_weight=torch.eye(2),
        null_observations=nulls,
        null_draws=8,
        bootstrap_draws=8,
        seed=7,
    )
    first = analyze_functional_spectrum(torch.diag(torch.tensor([2.0, 1.0])), observations, **kwargs)
    second = analyze_functional_spectrum(torch.diag(torch.tensor([2.0, 1.0])), observations, **kwargs)
    assert first.to_dict() == second.to_dict()
    assert first.permutation_null_interval is not None
    assert first.bootstrap_interval is not None
    assert first.init_null_esd_distance is not None
