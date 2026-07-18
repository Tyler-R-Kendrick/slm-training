"""LDI4-01 low-rank representation interventions for TwoTower (SLM-134).

A representation-space actuator that tests *where the correction lives*: a
LoReFT-style low-rank residual intervention at one declared site/position,
alongside a non-trainable difference-in-means control and a bit-identical parent
control. It consumes the same admitted action evidence and shared local objectives
as the weight-space adapters — it does not introduce a new event schema, objective
library, or evaluation pipeline.

Intervention (LoReFT), applied at a declared site to hidden state ``h``::

    h' = h + scale * (W h + b - R h) Rᵀ

with a low-rank orthonormal ``R`` (rank r). Initialising ``W = R``, ``b = 0`` makes
the edit exactly zero, so an untrained / disabled intervention is bit-identical to
the parent. Only ``R``, ``W``, ``b`` (or the fixed DiffMean vector) receive
gradients; the base model stays frozen. This is *adapted*, not a faithful ReFT
reproduction, and it claims no representation-space superiority without matched
trainable evidence.

The versioned :class:`InterventionSpec` and the matched arm-spec builder live in the
torch-free :mod:`slm_training.models.reft_intervention_spec` (re-exported here);
this module holds the torch-dependent intervention modules and artifact lifecycle.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from slm_training.models.reft_intervention_spec import (
    InterventionMethod,
    InterventionSpec,
    matched_arm_specs,
)

__all__ = [
    "InterventionMethod",
    "InterventionSpec",
    "LowRankReft",
    "DiffMeanIntervention",
    "build_intervention",
    "diffmean_vector",
    "matched_arm_specs",
    "save_intervention",
    "load_intervention",
]


class LowRankReft(nn.Module):
    """LoReFT-style low-rank affine subspace intervention. Identity at init."""

    def __init__(self, spec: InterventionSpec) -> None:
        super().__init__()
        if spec.method not in ("reft_r1", "reft_low_rank"):
            raise ValueError("LowRankReft requires a reft method")
        self.spec = spec
        h, r = spec.hidden_size, spec.rank
        # Orthonormal rows for R (deterministic via QR of a fixed basis slice).
        eye = torch.eye(h)[:r]
        self.R = nn.Parameter(eye.clone())
        self.W = nn.Parameter(eye.clone())  # W == R at init -> zero edit -> identity
        self.b = nn.Parameter(torch.zeros(r))
        self.scale = float(spec.scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [..., hidden]
        rh = h @ self.R.t()  # [..., r]
        wh = h @ self.W.t()  # [..., r]
        edit = wh + self.b - rh  # [..., r]
        return h + self.scale * (edit @ self.R)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.R, self.W, self.b]


class DiffMeanIntervention(nn.Module):
    """Non-trainable difference-in-means steering: ``h' = h + scale * v``. ``v`` is
    computed from train groups only and serialized as part of the artifact."""

    def __init__(self, spec: InterventionSpec, vector: torch.Tensor) -> None:
        super().__init__()
        if vector.shape != (spec.hidden_size,):
            raise ValueError("diffmean vector must have shape [hidden_size]")
        self.spec = spec
        self.register_buffer("v", vector.detach().clone())
        self.scale = float(spec.scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scale * self.v

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []


class _Identity(nn.Module):
    """Bit-identical parent control (no_intervention)."""

    def __init__(self, spec: InterventionSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []


def diffmean_vector(
    positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    """Difference-in-means of train-group activations (positive minus negative).
    Callers must pass train-split activations only."""
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
        raise ValueError("activations must be [n, hidden] with matching hidden size")
    return positive.mean(dim=0) - negative.mean(dim=0)


def build_intervention(
    spec: InterventionSpec, *, diffmean: torch.Tensor | None = None
) -> nn.Module:
    """Instantiate the intervention for ``spec``. DiffMean requires a precomputed
    train-only vector; unknown sites/shapes fail closed via the spec validation."""
    if spec.method == "no_intervention":
        return _Identity(spec)
    if spec.method == "diffmean_fixed":
        if diffmean is None:
            raise ValueError("diffmean_fixed requires a precomputed train-only vector")
        return DiffMeanIntervention(spec, diffmean)
    return LowRankReft(spec)


def save_intervention(module: nn.Module, path: Any) -> dict[str, Any]:
    """Serialize an intervention artifact (config + state). Returns the manifest."""
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    spec: InterventionSpec = module.spec  # type: ignore[attr-defined]
    torch.save(module.state_dict(), path / "intervention_model.pt")
    manifest = {
        "config": asdict(spec),
        "config_fingerprint": spec.fingerprint(),
        "trainable_parameters": int(
            sum(p.numel() for p in module.trainable_parameters())  # type: ignore[attr-defined]
        ),
    }
    (path / "intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (path / "intervention_config.json").write_text(
        json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_intervention(path: Any, *, diffmean: torch.Tensor | None = None) -> nn.Module:
    """Rebuild an intervention from its artifact; fails closed on identity mismatch."""
    from pathlib import Path

    path = Path(path)
    config = json.loads((path / "intervention_config.json").read_text(encoding="utf-8"))
    spec = InterventionSpec(**config)
    if spec.method == "diffmean_fixed" and diffmean is None:
        # recover the buffer from the state dict
        state = torch.load(path / "intervention_model.pt", weights_only=True)
        diffmean = state["v"]
    module = build_intervention(spec, diffmean=diffmean)
    state = torch.load(path / "intervention_model.pt", weights_only=True)
    module.load_state_dict(state)
    return module
