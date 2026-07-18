"""Per-tensor quantization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TensorDiagnostics:
    """Diagnostics comparing a tensor to its fake-quantized version."""

    name: str
    format_id: str
    level_occupancy: dict[int, int]
    zero_rate: float
    mse: float
    max_error: float
    cosine_similarity: float | None
    kurtosis: float | None
    outlier_rate: float
    symbol_entropy: float | None
    scale_min: float | None
    scale_max: float | None
    scale_mean: float | None
    excluded_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format_id": self.format_id,
            "level_occupancy": self.level_occupancy,
            "zero_rate": self.zero_rate,
            "mse": self.mse,
            "max_error": self.max_error,
            "cosine_similarity": self.cosine_similarity,
            "kurtosis": self.kurtosis,
            "outlier_rate": self.outlier_rate,
            "symbol_entropy": self.symbol_entropy,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_mean": self.scale_mean,
            "excluded_reason": self.excluded_reason,
        }


def diagnose_tensor(
    original: torch.Tensor,
    quantized: torch.Tensor,
    fmt_id: str,
    name: str = "",
    levels: tuple[float, ...] | None = None,
    scale: torch.Tensor | None = None,
    excluded_reason: str | None = None,
) -> TensorDiagnostics:
    """Compute diagnostics between ``original`` and ``quantized``."""
    flat_orig = original.detach().float().flatten()
    flat_quant = quantized.detach().float().flatten()
    diff = flat_orig - flat_quant
    mse = float(diff.pow(2).mean().item())
    max_error = float(diff.abs().max().item()) if flat_orig.numel() else 0.0

    norm_orig = flat_orig.norm(2)
    norm_quant = flat_quant.norm(2)
    cosine = (
        float((flat_orig * flat_quant).sum().item() / (norm_orig * norm_quant).item())
        if norm_orig > 0 and norm_quant > 0
        else None
    )

    mean = flat_orig.mean()
    std = flat_orig.std(unbiased=False)
    kurtosis = (
        float(((flat_orig - mean).pow(4).mean() / std.pow(4)).item() - 3.0)
        if std > 0
        else None
    )
    threshold = mean + 3.0 * std
    outlier_rate = float((flat_orig.abs() > threshold.abs()).float().mean().item()) if std > 0 else 0.0

    zero_rate = float((flat_quant == 0.0).float().mean().item()) if flat_quant.numel() else 0.0

    occupancy: dict[int, int] = {}
    symbol_entropy: float | None = None
    if levels:
        level_t = torch.tensor(levels, dtype=flat_quant.dtype)
        idx = flat_quant.sub(level_t.view(-1, 1)).abs().argmin(dim=0)
        counts = torch.bincount(idx, minlength=len(levels))
        occupancy = {i: int(c.item()) for i, c in enumerate(counts)}
        probs = counts.float() / counts.sum()
        probs = probs[probs > 0]
        if probs.numel():
            symbol_entropy = float(-(probs * torch.log2(probs)).sum().item())

    scale_min = scale_min_max_mean(scale)
    scale_max = None
    scale_mean_val = None
    if scale is not None and scale.numel():
        scale_max = float(scale.max().item())
        scale_mean_val = float(scale.mean().item())

    return TensorDiagnostics(
        name=name,
        format_id=fmt_id,
        level_occupancy=occupancy,
        zero_rate=zero_rate,
        mse=mse,
        max_error=max_error,
        cosine_similarity=cosine,
        kurtosis=kurtosis,
        outlier_rate=outlier_rate,
        symbol_entropy=symbol_entropy,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_mean=scale_mean_val,
        excluded_reason=excluded_reason,
    )


def scale_min_max_mean(scale: torch.Tensor | None) -> float | None:
    if scale is None or scale.numel() == 0:
        return None
    return float(scale.min().item())
