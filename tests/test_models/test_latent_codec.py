"""Regression tests for the CAP2-02 latent-codec interface and families."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.binary_lfq import BinaryLFQCodec, BinaryLFQConfig
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig
from slm_training.models.latent_codec import (
    LatentCodecSpec,
    _index_from_mixed_radix,
    _symbols_from_index,
)
from slm_training.models.latent_codec_trainer import (
    LatentCodecModel,
    evaluate_latent_codec,
    train_latent_codec,
)
from slm_training.models.learned_vq import LearnedVQCodec, LearnedVQConfig
from slm_training.models.mixed_radix_fsq import (
    MixedRadixFSQCodec,
    MixedRadixFSQConfig,
    suggest_mixed_radix_levels,
)
from slm_training.models.uniform_scalar_codec import UniformScalarCodec, UniformScalarCodecConfig


def _fixture_states(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.arange(n, dtype=torch.long)
    return states, states


def test_uniform_scalar_codec_round_trip() -> None:
    cfg = UniformScalarCodecConfig(num_states=8, K=2, d=3, hidden_dim=8)
    codec = UniformScalarCodec(cfg)
    states, _ = _fixture_states(8)
    enc = codec.encode(states, hard=True)
    decoder_input = codec.decode_input(enc)
    assert decoder_input.shape == (8, 2 * 3)
    assert decoder_input.dtype == torch.float32


def test_uniform_scalar_codec_capacity_reconstruction() -> None:
    """Uniform scalar codec with K**d >= M can reconstruct all fixture states."""
    M = 41
    cfg = UniformScalarCodecConfig(num_states=M, K=2, d=6, hidden_dim=64)
    codec = UniformScalarCodec(cfg)
    model = LatentCodecModel(codec, M)
    states, targets = _fixture_states(M)
    train_latent_codec(model, states, targets, steps=1200, lr=2e-2)
    metrics = evaluate_latent_codec(model, states, targets)
    assert metrics["exact_reconstruction_rate"] == 1.0
    assert metrics["occupied_codewords"] == M


def test_uniform_scalar_codec_below_capacity_does_not_reconstruct() -> None:
    M = 41
    cfg = UniformScalarCodecConfig(num_states=M, K=2, d=5, hidden_dim=64)
    codec = UniformScalarCodec(cfg)
    model = LatentCodecModel(codec, M)
    states, targets = _fixture_states(M)
    train_latent_codec(model, states, targets, steps=600, lr=2e-2)
    metrics = evaluate_latent_codec(model, states, targets)
    assert metrics["exact_reconstruction_rate"] < 1.0


def test_mixed_radix_fsq_round_trip() -> None:
    cfg = MixedRadixFSQConfig(num_states=8, levels=(2, 3, 4), hidden_dim=8)
    codec = MixedRadixFSQCodec(cfg)
    states, _ = _fixture_states(8)
    enc = codec.encode(states, hard=True)
    assert enc.hard.shape == (8, 3)
    assert (enc.hard[:, 0] < 2).all()
    assert (enc.hard[:, 1] < 3).all()
    assert (enc.hard[:, 2] < 4).all()
    decoder_input = codec.decode_input(enc)
    assert decoder_input.shape == (8, 2 + 3 + 4)


def test_mixed_radix_fsq_capacity_reconstruction() -> None:
    M = 41
    levels = (2, 3, 3, 4, 5)
    assert math.prod(levels) >= M
    cfg = MixedRadixFSQConfig(num_states=M, levels=levels, hidden_dim=64)
    codec = MixedRadixFSQCodec(cfg)
    model = LatentCodecModel(codec, M)
    states, targets = _fixture_states(M)
    train_latent_codec(model, states, targets, steps=1600, lr=2e-2)
    metrics = evaluate_latent_codec(model, states, targets)
    assert metrics["exact_reconstruction_rate"] == 1.0
    assert metrics["occupied_codewords"] == M


def test_binary_lfq_round_trip() -> None:
    cfg = BinaryLFQConfig(num_states=8, d=4, hidden_dim=8)
    codec = BinaryLFQCodec(cfg)
    states, _ = _fixture_states(8)
    enc = codec.encode(states, hard=True)
    assert enc.hard.shape == (8, 4)
    assert set(enc.hard.flatten().tolist()).issubset({0, 1})
    decoder_input = codec.decode_input(enc)
    assert set(decoder_input.flatten().tolist()).issubset({-1.0, 1.0})


def test_binary_lfq_capacity_reconstruction() -> None:
    M = 41
    cfg = BinaryLFQConfig(num_states=M, d=6, hidden_dim=64)
    codec = BinaryLFQCodec(cfg)
    model = LatentCodecModel(codec, M)
    states, targets = _fixture_states(M)
    train_latent_codec(model, states, targets, steps=1600, lr=2e-2)
    metrics = evaluate_latent_codec(model, states, targets)
    assert metrics["exact_reconstruction_rate"] == 1.0


def test_learned_vq_round_trip() -> None:
    cfg = LearnedVQConfig(num_states=8, codebook_size=8, latent_dim=4, hidden_dim=8)
    codec = LearnedVQCodec(cfg)
    states, _ = _fixture_states(8)
    enc = codec.encode(states, hard=True)
    assert enc.code_index is not None
    assert enc.code_index.shape == (8,)
    decoder_input = codec.decode_input(enc)
    assert decoder_input.shape == (8, 4)


def test_learned_vq_capacity_reconstruction() -> None:
    M = 41
    torch.manual_seed(0)
    cfg = LearnedVQConfig(num_states=M, codebook_size=64, latent_dim=8, hidden_dim=64)
    codec = LearnedVQCodec(cfg)
    model = LatentCodecModel(codec, M)
    states, targets = _fixture_states(M)
    train_latent_codec(model, states, targets, steps=2400, lr=2e-2)
    metrics = evaluate_latent_codec(model, states, targets)
    assert metrics["exact_reconstruction_rate"] == 1.0
    assert metrics["occupied_codewords"] == M


def test_continuous_latent_records_noise_policy() -> None:
    cfg = ContinuousLatentConfig(num_states=8, latent_dim=4, noise_std=0.1, rate_penalty=0.01)
    codec = ContinuousLatentCodec(cfg)
    states, _ = _fixture_states(8)
    enc = codec.encode(states, hard=True)
    assert enc.metadata["noise_std"] == 0.1
    assert enc.metadata["rate_penalty"] == 0.01
    decoder_input = codec.decode_input(enc)
    assert decoder_input.shape == (8, 4)


def test_continuous_latent_no_discrete_capacity_claim() -> None:
    cfg = ContinuousLatentConfig(num_states=41, latent_dim=6)
    codec = ContinuousLatentCodec(cfg)
    assert not codec.spec.hard_only
    storage = codec.physical_storage((1,))
    assert storage.bytes_per_example == 6 * 4  # float32 bytes


def test_mixed_radix_allocator_emits_alternatives() -> None:
    candidates = suggest_mixed_radix_levels(6.0)
    assert len(candidates) > 0
    for levels in candidates:
        bits = sum(math.log2(level) for level in levels)
        assert bits >= 6.0
    # No duplicates.
    assert len(candidates) == len(set(candidates))


def test_mixed_radix_index_conversion() -> None:
    levels = (2, 3, 4)
    symbols = torch.tensor([[0, 0, 0], [1, 2, 3], [1, 1, 1]])
    indices = _index_from_mixed_radix(symbols, levels)
    recovered = _symbols_from_index(indices, levels)
    assert torch.equal(recovered, symbols)


def test_latent_codec_spec_nominal_bits() -> None:
    spec = LatentCodecSpec(name="test", levels=(2, 3, 4), nominal_bits=math.log2(24))
    assert spec.nominal_bits == pytest.approx(math.log2(24))
