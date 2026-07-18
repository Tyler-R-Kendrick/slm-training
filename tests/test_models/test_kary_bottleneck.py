"""Regression tests for the strict K-ary bottleneck model (CAP2-01)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.kary_bottleneck import (
    KaryBottleneck,
    KaryBottleneckConfig,
    evaluate_kary_bottleneck,
    train_kary_bottleneck,
)


def _fixture_states(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.arange(n, dtype=torch.long)
    return states, states


def test_bottleneck_no_bypass_audit() -> None:
    cfg = KaryBottleneckConfig(num_states=8, K=2, d=3, hidden_dim=8)
    model = KaryBottleneck(cfg)
    states, _ = _fixture_states(8)
    assert model.audit_no_bypass(states)


def test_bottleneck_above_capacity_learns() -> None:
    """With K**d >= M the learned bottleneck can reconstruct all fixture states."""
    M = 41
    cfg = KaryBottleneckConfig(num_states=M, K=2, d=6, hidden_dim=64, train_steps=800)
    model = KaryBottleneck(cfg)
    states, targets = _fixture_states(M)
    train_kary_bottleneck(model, states, targets, steps=cfg.train_steps)
    metrics = evaluate_kary_bottleneck(model, states, targets)
    assert metrics["exact_reconstruction_rate"] == 1.0
    assert metrics["occupied_codewords"] == M


def test_bottleneck_below_capacity_fails() -> None:
    """With K**d < M exact reconstruction is impossible (phase boundary)."""
    M = 41
    cfg = KaryBottleneckConfig(num_states=M, K=2, d=5, hidden_dim=64, train_steps=400)
    model = KaryBottleneck(cfg)
    states, targets = _fixture_states(M)
    train_kary_bottleneck(model, states, targets, steps=cfg.train_steps)
    metrics = evaluate_kary_bottleneck(model, states, targets)
    assert metrics["exact_reconstruction_rate"] < 1.0
    assert metrics["occupied_codewords"] <= 2 ** 5


def test_bottleneck_soft_training_relaxation_disabled_in_eval() -> None:
    cfg = KaryBottleneckConfig(num_states=8, K=2, d=3, hidden_dim=8)
    model = KaryBottleneck(cfg)
    states, targets = _fixture_states(8)
    model.train()
    out_train, code_train, _ = model(states, hard=False)
    # Training path is differentiable; eval uses hard argmax codes.
    assert out_train.requires_grad
    model.eval()
    out_eval, code_eval, _ = model(states, hard=True)
    assert torch.equal(code_eval, code_train)
