"""CAP4-01: residual ternary planes for local scorer/energy heads.

Weight-space residual stack: a base ``nn.Linear`` plus ternary-quantized plane
weights.  Effective weight is ``W0 + sum_r scale_r * trits(W_r)``.  This is a
wiring implementation; no optimized packed multi-plane kernel is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.analysis.arity import (
    ResidualScaleMode,
    assert_geometric_only,
    balanced_ternary_levels,
)
from slm_training.models.quantization.cost import compute_tensor_cost
from slm_training.models.quantization.formats import residual_ternary_plane_format


def _ternarize(x: torch.Tensor) -> torch.Tensor:
    """Hard ternary {-1, 0, +1} with straight-through estimator."""
    return x + (torch.sign(x) - x).detach()


@dataclass
class PlaneOutput:
    """Diagnostics emitted by ``ResidualTritStack.forward``."""

    final_output: torch.Tensor
    cumulative_outputs: list[torch.Tensor] = field(default_factory=list)
    plane_outputs: list[torch.Tensor] = field(default_factory=list)
    symbols: list[torch.Tensor] = field(default_factory=list)
    scales: list[torch.Tensor] = field(default_factory=list)
    residual_norms: list[float] = field(default_factory=list)
    per_plane_cost_bytes: list[int] = field(default_factory=list)
    quant_errors: list[float] = field(default_factory=list)


class ResidualTritStack(nn.Module):
    """Base linear + ternary residual planes in weight space.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        R: number of residual planes.
        scale_mode: ``geometric_balanced`` | ``learned_independent`` |
            ``learned_monotone``.
        residual_normalization: ``none`` | ``rms`` | ``variance_preserving``.
        group_size: group size for byte accounting (not for quantization,
            since plane weights are ternarized element-wise here).
        bias: whether the base linear has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        R: int,
        scale_mode: Literal[
            "geometric_balanced",
            "learned_independent",
            "learned_monotone",
        ] = "geometric_balanced",
        residual_normalization: Literal["none", "rms", "variance_preserving"] = "none",
        group_size: int = 128,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if R < 0:
            raise ValueError("R must be non-negative")
        self.in_features = in_features
        self.out_features = out_features
        self.R = R
        self.scale_mode = scale_mode
        self.residual_normalization = residual_normalization
        self.group_size = group_size

        self.base_module = nn.Linear(in_features, out_features, bias=bias)
        self.planes = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(R)]
        )
        # Initialize planes near zero so the stack starts near the base module.
        for plane in self.planes:
            nn.init.normal_(plane.weight, mean=0.0, std=0.001)

        base_scale = float(self.base_module.weight.data.abs().max().item())
        base_scale = max(base_scale, 1e-6)

        if scale_mode == "geometric_balanced":
            # Fixed radix-3 schedule.  Only this mode may claim 3**R grid levels.
            scales = [base_scale * (3.0 ** (-r)) for r in range(R)]
            self.scales = nn.ParameterList(
                [nn.Parameter(torch.tensor(s), requires_grad=False) for s in scales]
            )
        elif scale_mode == "learned_independent":
            self.scales = nn.ParameterList(
                [nn.Parameter(torch.tensor(base_scale * (0.5 ** r))) for r in range(R)]
            )
        elif scale_mode == "learned_monotone":
            # Parameterize positive scales; soft regularization in training keeps
            # them non-increasing.  Initialize with a decaying schedule.
            self._scale_logits = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(math.log(base_scale * (0.5 ** r))))
                    for r in range(R)
                ]
            )
            self._monotone_penalty_weight = 1e-3
        else:
            raise ValueError(f"unknown scale_mode: {scale_mode!r}")

    def effective_scale(self, r: int) -> torch.Tensor:
        """Return the (possibly learned) scale for plane ``r``."""
        if self.scale_mode == "learned_monotone":
            return self._scale_logits[r].exp()
        return self.scales[r]

    def all_scales(self) -> list[torch.Tensor]:
        """Return a list of scale tensors, one per plane."""
        if self.scale_mode == "learned_monotone":
            return [s.exp() for s in self._scale_logits]
        return list(self.scales)

    def grid_levels(self) -> int:
        """Return 3**R; only valid under geometric_balanced mode."""
        mode = (
            ResidualScaleMode.GEOMETRIC_BALANCED
            if self.scale_mode == "geometric_balanced"
            else ResidualScaleMode.LEARNED_INDEPENDENT
        )
        assert_geometric_only(mode)
        return balanced_ternary_levels(self.R)

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residual normalization and return (normalized, rescale_factor)."""
        if self.residual_normalization == "none":
            return x, torch.tensor(1.0, dtype=x.dtype, device=x.device)
        if self.residual_normalization == "rms":
            rms = x.pow(2).mean().sqrt().clamp_min(1e-8)
            return x / rms, rms
        if self.residual_normalization == "variance_preserving":
            std = x.std(unbiased=False).clamp_min(1e-8)
            return x / std, std
        raise ValueError(f"unknown residual_normalization: {self.residual_normalization!r}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        max_planes: int | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | PlaneOutput:
        """Forward through base + residual planes.

        If ``return_diagnostics`` is True, returns a ``PlaneOutput``; otherwise
        returns only the final tensor.
        """
        max_planes = self.R if max_planes is None else min(max_planes, self.R)
        y = self.base_module(x)

        diagnostics: PlaneOutput | None = PlaneOutput(final_output=y) if return_diagnostics else None
        effective_weight = self.base_module.weight

        for r in range(max_planes):
            plane = self.planes[r]
            raw = plane.weight
            norm_raw, rescale = self._normalize(raw)
            symbols = _ternarize(norm_raw)
            scale = self.effective_scale(r)
            quantized_weight = symbols * scale * rescale

            effective_weight = effective_weight + quantized_weight
            plane_out = F.linear(x, quantized_weight)
            y = y + plane_out

            if diagnostics is not None:
                diagnostics.cumulative_outputs.append(y)
                diagnostics.plane_outputs.append(plane_out)
                diagnostics.symbols.append(symbols)
                diagnostics.scales.append(scale.detach())
                diagnostics.residual_norms.append(float(raw.detach().norm().item()))
                cost = compute_tensor_cost(
                    f"residual_plane.{r}.weight",
                    plane.weight,
                    residual_ternary_plane_format(group_size=self.group_size),
                    group_size=self.group_size,
                    metadata_overhead_bytes=32,
                )
                diagnostics.per_plane_cost_bytes.append(cost.total_bytes)
                quant_err = float((quantized_weight - raw).pow(2).mean().item())
                diagnostics.quant_errors.append(quant_err)

        out = F.linear(x, effective_weight, self.base_module.bias)
        if diagnostics is not None:
            diagnostics.final_output = out
            return diagnostics
        return out

    def effective_weight(self, max_planes: int | None = None) -> torch.Tensor:
        """Return the full-precision effective weight (base + quantized planes)."""
        max_planes = self.R if max_planes is None else min(max_planes, self.R)
        w = self.base_module.weight
        for r in range(max_planes):
            raw = self.planes[r].weight
            norm_raw, rescale = self._normalize(raw)
            symbols = _ternarize(norm_raw)
            w = w + symbols * self.effective_scale(r) * rescale
        return w

    def fit_planes_sequential(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        *,
        steps: int = 20,
        lr: float = 1e-2,
        freeze_previous: bool = True,
    ) -> dict[str, Any]:
        """Sequentially fit each plane to the residual of the previous prefix.

        ``target`` is a teacher output tensor of shape ``(batch, out_features)``.
        Returns per-plane loss histories.
        """
        histories: list[list[float]] = []
        y = self.base_module(x)

        for r in range(self.R):
            plane = self.planes[r]
            params = list(plane.parameters())
            if self.scale_mode == "learned_monotone":
                params.extend(self._scale_logits.parameters())
            elif self.scale_mode == "learned_independent":
                params.extend(self.scales.parameters())
            optimizer = torch.optim.SGD(params, lr=lr)
            history: list[float] = []

            original_requires_grad: dict[int, bool] = {}
            if freeze_previous:
                for p in self.parameters():
                    original_requires_grad[id(p)] = p.requires_grad
                    p.requires_grad = False
                for p in params:
                    p.requires_grad = True

            for _ in range(steps):
                optimizer.zero_grad()
                residual = target - y.detach()
                raw = plane.weight
                norm_raw, rescale = self._normalize(raw)
                quantized_weight = _ternarize(norm_raw) * self.effective_scale(r) * rescale
                pred = F.linear(x, quantized_weight)
                loss = F.mse_loss(pred, residual)
                loss.backward()
                optimizer.step()
                history.append(float(loss.item()))

            if freeze_previous:
                for p in self.parameters():
                    p.requires_grad = original_requires_grad.get(id(p), p.requires_grad)

            # Add the (quantized) contribution of this plane to the running prefix.
            with torch.no_grad():
                raw = plane.weight
                norm_raw, rescale = self._normalize(raw)
                quantized_weight = _ternarize(norm_raw) * self.effective_scale(r) * rescale
                y = y + F.linear(x, quantized_weight)

            histories.append(history)

        return {"loss_histories": histories}


# Re-export ResidualScaleMode so callers only need one import.
__all__ = [
    "PlaneOutput",
    "ResidualTritStack",
    "_ternarize",
]
