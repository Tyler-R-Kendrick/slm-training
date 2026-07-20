"""Tests for the semantic connector module."""

from __future__ import annotations

import torch

from slm_training.models.semantic_connector import (
    ConnectorOutput,
    CrossAttentionConnector,
    LinearConnector,
    LowRankConnector,
    SemanticConnector,
    count_connector_parameters,
    estimate_connector_flops,
)


def _synthetic_inputs(batch: int = 2, seq: int = 8, d_model: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, seq, d_model)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[:, -1] = False
    return x, mask


def test_linear_connector_forward_shape() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    conn = LinearConnector(d_model)
    out = conn(x, mask)
    assert isinstance(out, ConnectorOutput)
    assert out.context_vectors.shape == (x.shape[0], d_model)
    assert out.mask is not None and out.mask.shape == (x.shape[0], 1)
    assert out.attention_weights is None
    assert out.connector_type == "linear"


def test_low_rank_connector_forward_shape() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    conn = LowRankConnector(d_model, hidden_dim=128)
    out = conn(x, mask)
    assert out.context_vectors.shape == (x.shape[0], d_model)
    assert out.mask is not None and out.mask.shape == (x.shape[0], 1)
    assert out.attention_weights is None
    assert out.connector_type == "low_rank"


def test_cross_attention_connector_forward_shape() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    conn = CrossAttentionConnector(d_model, n_queries=4, n_heads=2, n_blocks=1)
    out = conn(x, mask)
    assert out.context_vectors.shape == (x.shape[0], 4, d_model)
    assert out.mask is not None and out.mask.shape == (x.shape[0], 4)
    assert out.attention_weights is not None
    assert out.attention_weights.shape == (x.shape[0], 4, x.shape[1])
    assert out.connector_type == "cross_attention"


def test_semantic_connector_factory_selects_variants() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    for ctype in ("none", "linear", "low_rank", "cross_attention"):
        conn = SemanticConnector(ctype, d_model=d_model, connector_n_queries=4)
        out = conn(x, mask)
        assert out.connector_type == ctype


def test_none_connector_returns_pooled_input_unchanged() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    conn = SemanticConnector("none", d_model=d_model)
    out = conn(x, mask)
    expected = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
    assert torch.allclose(out.context_vectors, expected, atol=1e-6)


def test_parameter_counts_increase_with_capacity() -> None:
    d_model = 32
    none_conn = SemanticConnector("none", d_model=d_model)
    linear_conn = LinearConnector(d_model)
    low_rank_conn = LowRankConnector(d_model, hidden_dim=20)
    cross_conn = CrossAttentionConnector(d_model, n_queries=8, n_heads=2)
    assert count_connector_parameters(none_conn) == 0
    assert count_connector_parameters(linear_conn) > 0
    assert count_connector_parameters(low_rank_conn) > count_connector_parameters(linear_conn)
    assert count_connector_parameters(cross_conn) > count_connector_parameters(low_rank_conn)


def test_estimate_flops_increases_with_capacity() -> None:
    d_model = 32
    batch, seq = 2, 4
    none_conn = SemanticConnector("none", d_model=d_model)
    linear_conn = LinearConnector(d_model)
    low_rank_conn = LowRankConnector(d_model, hidden_dim=20)
    cross_conn = CrossAttentionConnector(d_model, n_queries=8, n_heads=2)
    assert estimate_connector_flops(none_conn, batch, seq, d_model) == 0
    assert estimate_connector_flops(linear_conn, batch, seq, d_model) > 0
    assert estimate_connector_flops(low_rank_conn, batch, seq, d_model) > estimate_connector_flops(linear_conn, batch, seq, d_model)
    assert estimate_connector_flops(cross_conn, batch, seq, d_model) > estimate_connector_flops(low_rank_conn, batch, seq, d_model)


def test_state_dict_round_trip() -> None:
    x, mask = _synthetic_inputs()
    d_model = x.shape[-1]
    conn = SemanticConnector("cross_attention", d_model=d_model, connector_n_queries=4)
    state = conn.state_dict()
    conn2 = SemanticConnector("cross_attention", d_model=d_model, connector_n_queries=4)
    conn2.load_state_dict(state)
    with torch.no_grad():
        out1 = conn(x, mask)
        out2 = conn2(x, mask)
    assert torch.allclose(out1.context_vectors, out2.context_vectors, atol=1e-6)


def test_invalid_connector_type_raises() -> None:
    with torch.no_grad():
        pass
    try:
        SemanticConnector("unknown", d_model=32)
    except ValueError as exc:
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for unknown connector type")
