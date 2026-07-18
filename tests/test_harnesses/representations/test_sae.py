"""LDI4-02 SAE forward / metrics / dead-features / fail-closed artifacts."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.representations.sae import (  # noqa: E402
    SparseAutoencoder,
    dead_feature_mask,
    load_sae,
    sae_loss,
    sae_metrics,
    save_sae,
)
from slm_training.harnesses.representations.spec import SAEConfig  # noqa: E402


def _sae(**kw) -> SparseAutoencoder:
    return SparseAutoencoder(SAEConfig(d_in=8, expansion_factor=4, seed=0, **kw))


def test_forward_shapes_and_nonnegative_codes():
    sae = _sae()
    h = torch.randn(5, 8)
    h_hat, z = sae(h)
    assert h_hat.shape == (5, 8)
    assert z.shape == (5, 32)
    assert torch.all(z >= 0)  # nonnegative codes


def test_loss_finite_and_reconstruction_improves_with_training():
    torch.manual_seed(0)
    sae = _sae()
    h = torch.randn(64, 8)
    loss0, parts = sae_loss(sae, h)
    assert torch.isfinite(loss0) and parts["reconstruction_mse"] >= 0
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        loss, _ = sae_loss(sae, h)
        loss.backward()
        opt.step()
    loss1, _ = sae_loss(sae, h)
    assert float(loss1) < float(loss0)  # it actually learns to reconstruct


def test_metrics_report_expected_keys_and_ranges():
    sae = _sae()
    m = sae_metrics(sae, torch.randn(32, 8))
    for k in ("reconstruction_mse", "explained_variance", "cosine_similarity", "l0", "l1",
              "dead_feature_rate", "ultra_dense_rate"):
        assert k in m
    assert 0.0 <= m["dead_feature_rate"] <= 1.0
    assert 0.0 <= m["l0"] <= sae.config.dict_width


def test_dead_feature_mask_flags_inactive_features():
    sae = _sae()
    z = sae.encode(torch.randn(40, 8))
    mask = dead_feature_mask(z, threshold=1e-6)
    assert mask.dtype == torch.bool and mask.numel() == sae.config.dict_width


def test_jumprelu_is_sparser_than_relu_at_init():
    h = torch.randn(64, 8)
    z_relu = _sae(nonlinearity="relu").encode(h)
    z_jump = _sae(nonlinearity="jumprelu").encode(h)
    assert int((z_jump > 0).sum()) <= int((z_relu > 0).sum())


def test_save_load_roundtrip_and_fail_closed(tmp_path):
    sae = _sae()
    h = torch.randn(6, 8)
    save_sae(sae, tmp_path)
    loaded = load_sae(tmp_path, expect_d_in=8)
    assert torch.allclose(sae(h)[0], loaded(h)[0], atol=1e-6)
    # width mismatch fails closed
    with pytest.raises(ValueError):
        load_sae(tmp_path, expect_d_in=16)
    # tampered artifact kind fails closed
    import json

    manifest = json.loads((tmp_path / "sae_manifest.json").read_text())
    manifest["kind"] = "not_an_sae"
    (tmp_path / "sae_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_sae(tmp_path)
