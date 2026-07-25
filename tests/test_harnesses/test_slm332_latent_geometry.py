import torch

from slm_training.harnesses.experiments.slm332_latent_geometry import (
    BindingContrastCorpus,
    DEFAULT_AP009_PAIRS,
    GeometryTrainingConfig,
    load_admitted_binding_contrasts,
    measure_geometry_arm,
    train_geometry_arm,
)
from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import FactorReconstructionModel
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig


def test_admitted_ap009_binding_contrasts_are_canonical_and_explicit() -> None:
    corpus = load_admitted_binding_contrasts(DEFAULT_AP009_PAIRS)

    assert corpus.positive.shape == corpus.negative.shape
    assert corpus.positive.shape[0] == 366
    assert set(corpus.transform_ids) == {"binding_swap_symbol", "binding_swap_symbol_reverse"}
    assert torch.all(torch.linalg.vector_norm(corpus.positive - corpus.negative, dim=-1) > 0)


def test_geometry_config_can_form_the_matched_reconstruction_control() -> None:
    control = GeometryTrainingConfig(contrast_weight=0.0, noise_weight=0.0)

    assert control.contrast_weight == 0.0
    assert control.noise_weight == 0.0


def test_geometry_arm_uses_codec_latents_and_never_promotes_proxy_metrics() -> None:
    corpus = BindingContrastCorpus(
        positive=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        negative=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        pair_ids=("binding-a", "binding-b"),
        transform_ids=("binding_swap_symbol", "binding_swap_symbol_reverse"),
    )
    codec = ContinuousLatentCodec(
        ContinuousLatentConfig(
            num_states=1, latent_dim=2, hidden_dim=4, feature_dim=2, mode="semantic_trace"
        )
    )
    model = FactorReconstructionModel(codec, feature_dim=2)
    config = GeometryTrainingConfig(steps=2, noise_std=0.0)

    history = train_geometry_arm(model, corpus, config)
    measurement = measure_geometry_arm(model, corpus, config)

    assert len(history) == config.steps
    assert measurement.ap030_oracle_semantic_gate.startswith("unavailable_")
    assert measurement.interpolation_traversal.startswith("unavailable_")
    assert not measurement.promotion_eligible
