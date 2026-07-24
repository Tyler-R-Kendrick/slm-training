"""SLM-232 latent-state decomposition, ablation, rank, and gate tests."""

from __future__ import annotations

import pytest
import torch

from slm_training.harnesses.experiments.slm232_latent_state_use import (
    LatentStateVerdict,
    RecursiveStateAblationV1,
    apply_initial_ablation,
    classify_latent_state_use,
    compose_z0,
    representation_summary,
    within_group_permutation,
)
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower


def _tower() -> SharedRecursiveDenoiserTower:
    torch.manual_seed(232)
    return SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=2,
        n_heads=2,
        max_len=8,
        recursive_steps=2,
    ).eval()


def test_initial_z_decomposition_sums_to_canonical_state_exactly() -> None:
    tower = _tower()
    noisy = torch.tensor([[1, 2, 0]])
    context = torch.randn(1, 2, 8)
    components = tower.initial_transition_components(noisy, context, 0)
    state = tower.initial_transition_state(noisy, context, 0)
    pos = torch.arange(noisy.shape[1]).unsqueeze(0)
    pooled = context.mean(dim=1)
    torch.testing.assert_close(
        components["z_latent_component"], tower.z_latent[pos], rtol=0, atol=0
    )
    torch.testing.assert_close(
        components["z_context_component"],
        tower.ctx_proj(pooled).unsqueeze(1).expand_as(state["z"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        components["z_position_component"], tower.pos(pos), rtol=0, atol=0
    )
    assert components["context_projection_applied"] is True
    torch.testing.assert_close(compose_z0(components), state["z"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mode", "removed"),
    [
        ("zero_ctx_proj", "z_context_component"),
        ("zero_z_latent", "z_latent_component"),
        ("remove_z_position", "z_position_component"),
    ],
)
def test_component_ablation_removes_only_declared_term(mode: str, removed: str) -> None:
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2]]), torch.randn(1, 2, 8), 0
    )
    baseline = compose_z0(components)
    assert baseline is not None
    ablated = apply_initial_ablation(components, RecursiveStateAblationV1(mode))
    removed_value = components[removed]
    assert isinstance(removed_value, torch.Tensor)
    torch.testing.assert_close(ablated, baseline - removed_value)
    torch.testing.assert_close(compose_z0(components), baseline, rtol=0, atol=0)


def test_unchanged_override_preserves_logits_and_source_weights_exactly() -> None:
    tower = _tower()
    before = {
        name: value.detach().clone() for name, value in tower.state_dict().items()
    }
    noisy = torch.tensor([[1, 2]])
    context = torch.randn(1, 2, 8)
    components = tower.initial_transition_components(noisy, context, 0)
    state = tower.initial_transition_state(noisy, context, 0)
    unchanged = apply_initial_ablation(
        components, RecursiveStateAblationV1("none")
    )
    assert isinstance(state["y"], torch.Tensor)
    assert isinstance(state["z"], torch.Tensor)
    assert isinstance(unchanged, torch.Tensor)
    baseline = tower.transition_step(
        state["y"], state["z"], context, state["self_pad_mask"]
    )
    override = tower.transition_step(
        state["y"], unchanged, context, state["self_pad_mask"]
    )
    torch.testing.assert_close(baseline["logits"], override["logits"], rtol=0, atol=0)
    for name, value in tower.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_shuffle_preserves_shape_and_batch_norm_distribution() -> None:
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 2, 8), 0
    )
    baseline = compose_z0(components)
    assert baseline is not None
    permutation, manifest_sha256 = within_group_permutation(
        ["same", "same"], seed=232
    )
    shuffled = apply_initial_ablation(
        components,
        RecursiveStateAblationV1("shuffle_z_across_examples"),
        permutation=permutation,
    )
    assert shuffled is not None and shuffled.shape == baseline.shape
    torch.testing.assert_close(
        shuffled.flatten(1).norm(dim=1).sort().values,
        baseline.flatten(1).norm(dim=1).sort().values,
    )
    assert len(manifest_sha256) == 64


def test_shuffle_rejects_cross_group_or_implicit_pairs() -> None:
    with pytest.raises(ValueError, match="every shuffle group"):
        within_group_permutation(["one", "two"])
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 2, 8), 0
    )
    with pytest.raises(ValueError, match="preregistered permutation"):
        apply_initial_ablation(
            components, RecursiveStateAblationV1("shuffle_z_across_examples")
        )


def test_matched_swap_requires_manifest() -> None:
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2]]), torch.randn(1, 2, 8), 0
    )
    baseline = compose_z0(components)
    assert baseline is not None
    with pytest.raises(ValueError, match="pair manifest"):
        apply_initial_ablation(
            components,
            RecursiveStateAblationV1("swap_z_matched"),
            matched_z0=baseline.clone(),
        )


def test_random_control_is_seeded_and_norm_matched() -> None:
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2]]), torch.randn(1, 2, 8), 0
    )
    ablation = RecursiveStateAblationV1("random_norm_matched", seed=7)
    first = apply_initial_ablation(components, ablation)
    second = apply_initial_ablation(components, ablation)
    baseline = compose_z0(components)
    assert first is not None and second is not None and baseline is not None
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=-1),
        torch.linalg.vector_norm(baseline, dim=-1),
    )


def test_rank_summary_separates_rank_one_and_full_rank() -> None:
    rank_one = torch.arange(1, 6, dtype=torch.float64).unsqueeze(1) @ torch.ones(
        1, 4, dtype=torch.float64
    )
    full = torch.eye(5, 4, dtype=torch.float64)
    assert representation_summary(rank_one)["matrix_rank"] == 1
    assert representation_summary(full)["matrix_rank"] == 4
    assert (
        representation_summary(full)["effective_rank"]
        > representation_summary(rank_one)["effective_rank"]
    )
    zero = representation_summary(torch.ones(3, 4, dtype=torch.float64))
    assert zero["matrix_rank"] == 0
    assert zero["effective_rank"] == 0.0
    assert zero["participation_ratio"] == 0.0


def test_unstable_prior_blocks_positive_latent_verdict() -> None:
    verdict = classify_latent_state_use(
        rank_qualified=True,
        context_only=False,
        targeted_effect_reproduced=True,
        targeted_exceeds_nuisance=True,
        matched_y_only_equivalent=False,
        powered_no_effect=False,
        actual_legal_effect=True,
        protected_outcome_effect=True,
        uncertainty_excludes_zero=True,
        unstable_dynamics=True,
        nonvacuous_outcome=True,
    )
    assert verdict is LatentStateVerdict.UNSTABLE


def test_missing_positive_evidence_fails_closed() -> None:
    verdict = classify_latent_state_use(
        rank_qualified=True,
        context_only=False,
        targeted_effect_reproduced=True,
        targeted_exceeds_nuisance=True,
        matched_y_only_equivalent=None,
        powered_no_effect=False,
        actual_legal_effect=None,
        protected_outcome_effect=None,
        uncertainty_excludes_zero=None,
        unstable_dynamics=False,
        nonvacuous_outcome=True,
    )
    assert verdict is LatentStateVerdict.INCONCLUSIVE


def test_path_ablation_is_default_off_and_changes_only_declared_route() -> None:
    tower = _tower()
    noisy = torch.tensor([[1, 2]])
    context = torch.randn(1, 2, 8)
    initial = tower.initial_transition_state(noisy, context, 0)
    y, z, mask = initial["y"], initial["z"], initial["self_pad_mask"]
    assert isinstance(y, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    baseline = tower.transition_step(y, z, context, mask)
    explicit_none = tower.transition_step(
        y, z, context, mask, state_path_ablation="none"
    )
    torch.testing.assert_close(
        baseline["logits"], explicit_none["logits"], rtol=0, atol=0
    )
    disconnected = tower.transition_step(
        y, z, context, mask, state_path_ablation="detach_z_to_y"
    )
    assert not torch.equal(baseline["logits"], disconnected["logits"])
    torch.testing.assert_close(
        baseline["z"], disconnected["z"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline["z_update"], disconnected["z_update"], rtol=0, atol=0
    )


def test_detach_y_to_z_makes_z_update_independent_of_y() -> None:
    tower = _tower()
    context = torch.randn(1, 2, 8)
    initial = tower.initial_transition_state(torch.tensor([[1, 2]]), context, 0)
    y, z, mask = initial["y"], initial["z"], initial["self_pad_mask"]
    assert isinstance(y, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    first = tower.transition_step(
        y, z, context, mask, state_path_ablation="detach_y_to_z"
    )
    second = tower.transition_step(
        y + 9.0, z, context, mask, state_path_ablation="detach_y_to_z"
    )
    torch.testing.assert_close(first["z"], second["z"], rtol=0, atol=0)
    torch.testing.assert_close(first["z_update"], second["z_update"], rtol=0, atol=0)


def test_path_ablation_rejects_y_only_state() -> None:
    tower = _tower()
    context = torch.randn(1, 2, 8)
    y = torch.randn(1, 2, 8)
    with pytest.raises(ValueError, match="requires an explicit z state"):
        tower.transition_step(
            y,
            None,
            context,
            torch.zeros(1, 2, dtype=torch.bool),
            state_path_ablation="detach_z_to_y",
        )


def test_parameter_free_context_is_not_mislabeled_as_projected() -> None:
    tower = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=2,
        n_heads=2,
        max_len=8,
        recursive_steps=2,
        z_state_mode="parameter_free",
    ).eval()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2]]), torch.randn(1, 2, 8), 0
    )
    assert components["context_projection_applied"] is False
    with pytest.raises(ValueError, match="applied learned projection"):
        apply_initial_ablation(
            components, RecursiveStateAblationV1("zero_ctx_proj")
        )


def test_gold_oracle_is_explicitly_not_applicable() -> None:
    tower = _tower()
    components = tower.initial_transition_components(
        torch.tensor([[1, 2]]), torch.randn(1, 2, 8), 0
    )
    assert (
        apply_initial_ablation(
            components, RecursiveStateAblationV1("gold_oracle_z")
        )
        is None
    )
