"""LDI4-02 sparse autoencoder + quality metrics + fail-closed artifacts (SLM-136).

A small, transparent baseline SAE over captured decision-state activations:

    z     = act(encoder(h - bias_dec))          # nonnegative codes
    h_hat = decoder(z) + bias_dec
    loss  = reconstruction_mse + lambda_sparse * L1(z)

Torch is required (imported lazily so ``spec.py`` stays torch-free). A low
reconstruction loss is not evidence of interpretability or steering utility -- the
metrics here are diagnostics, and no SAE is promoted from them alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from slm_training.harnesses.representations.spec import SAEConfig

__all__ = [
    "SparseAutoencoder",
    "sae_loss",
    "sae_metrics",
    "dead_feature_mask",
    "save_sae",
    "load_sae",
    "SAE_ARTIFACT_VERSION",
]

SAE_ARTIFACT_VERSION = "ldi4-02-sae-v1"
_JUMPRELU_THETA = 1e-2


class SparseAutoencoder(nn.Module):
    """Single-hidden-layer SAE with a pre-decoder bias and a nonnegative code."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        gen = torch.Generator().manual_seed(config.seed)
        d, m = config.d_in, config.dict_width
        self.bias_dec = nn.Parameter(torch.zeros(d))
        self.encoder = nn.Linear(d, m)
        self.decoder = nn.Linear(m, d, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.randn(m, d, generator=gen) * (d ** -0.5))
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(torch.randn(d, m, generator=gen) * (m ** -0.5))
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        if self.config.decoder_norm == "unit":
            with torch.no_grad():
                norm = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
                self.decoder.weight.div_(norm)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(h - self.bias_dec)
        z = torch.relu(pre)
        if self.config.nonlinearity == "jumprelu":
            z = z * (pre > _JUMPRELU_THETA).to(z.dtype)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.bias_dec

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        return self.decode(z), z


def sae_loss(
    sae: SparseAutoencoder, h: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    h_hat, z = sae(h)
    recon = torch.mean((h_hat - h) ** 2)
    l1 = torch.mean(z.abs().sum(dim=-1))
    loss = recon + sae.config.lambda_sparse * l1
    return loss, {"reconstruction_mse": float(recon), "l1": float(l1)}


def dead_feature_mask(z: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Boolean mask over features whose activation frequency is below ``threshold``."""
    freq = (z > 0).to(torch.float32).mean(dim=0)
    return freq < threshold


@torch.no_grad()
def sae_metrics(sae: SparseAutoencoder, h: torch.Tensor) -> dict[str, Any]:
    """Reconstruction / sparsity / density diagnostics on a held or train batch."""
    h_hat, z = sae(h)
    resid = h_hat - h
    recon_mse = float(torch.mean(resid ** 2))
    total_var = float(torch.var(h, unbiased=False)) or 1e-12
    explained_variance = 1.0 - float(torch.var(resid, unbiased=False)) / total_var
    cos = float(
        torch.mean(
            torch.nn.functional.cosine_similarity(h_hat, h, dim=-1, eps=1e-8)
        )
    )
    l0 = float((z > 0).to(torch.float32).sum(dim=-1).mean())
    l1 = float(z.abs().sum(dim=-1).mean())
    freq = (z > 0).to(torch.float32).mean(dim=0)
    dead = float((freq < sae.config.dead_feature_threshold).to(torch.float32).mean())
    ultra_dense = float((freq > 0.5).to(torch.float32).mean())
    return {
        "reconstruction_mse": recon_mse,
        "explained_variance": explained_variance,
        "cosine_similarity": cos,
        "l0": l0,
        "l1": l1,
        "dead_feature_rate": dead,
        "ultra_dense_rate": ultra_dense,
        "dict_width": sae.config.dict_width,
        "n_rows": int(h.shape[0]),
    }


def save_sae(sae: SparseAutoencoder, path: Path | str) -> dict[str, Any]:
    """Persist ``sae_model.pt`` + ``sae_config.json`` + a manifest with the config
    fingerprint and artifact version (mirrors ``save_intervention``)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), out / "sae_model.pt")
    manifest = {
        "kind": "sae_diagnostic",
        "version": SAE_ARTIFACT_VERSION,
        "config": sae.config.to_dict(),
        "config_fingerprint": sae.config.fingerprint(),
        "d_in": sae.config.d_in,
        "dict_width": sae.config.dict_width,
    }
    import json

    (out / "sae_config.json").write_text(
        json.dumps(sae.config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "sae_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_sae(path: Path | str, *, expect_d_in: int | None = None) -> SparseAutoencoder:
    """Rebuild an SAE from a saved artifact. Fails closed on artifact-kind/version or
    width mismatch (a mismatched model site or activation width cannot silently load)."""
    import json

    root = Path(path)
    manifest = json.loads((root / "sae_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "sae_diagnostic":
        raise ValueError(f"not an sae_diagnostic artifact: {manifest.get('kind')!r}")
    if manifest.get("version") != SAE_ARTIFACT_VERSION:
        raise ValueError(f"sae artifact version mismatch: {manifest.get('version')!r}")
    config = SAEConfig.from_mapping(json.loads((root / "sae_config.json").read_text(encoding="utf-8")))
    if expect_d_in is not None and config.d_in != expect_d_in:
        raise ValueError(f"sae width mismatch: artifact d_in={config.d_in} expected {expect_d_in}")
    if config.fingerprint() != manifest.get("config_fingerprint"):
        raise ValueError("sae config fingerprint does not match manifest")
    sae = SparseAutoencoder(config)
    state = torch.load(root / "sae_model.pt", weights_only=True)
    sae.load_state_dict(state)
    return sae
