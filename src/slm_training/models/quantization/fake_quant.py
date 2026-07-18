"""Reference fake-quantization forwards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.autograd import Function

from slm_training.models.quantization.observers import (
    observe_asymmetric_scale,
    observe_symmetric_scale,
)

if TYPE_CHECKING:
    from slm_training.models.quantization.formats import QuantFormat


class _FakeQuantSTE(Function):
    """Straight-through estimator for fake-quantized tensors."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, quant_fn):  # noqa: ANN001, ANN205
        out, scale, zp = quant_fn(input)
        ctx.save_for_backward(scale, zp)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: ANN001, ANN205
        return grad_output, None


def _nearest_level_index(normalized: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Argmin over integer/float quantization levels."""
    normalized = normalized.unsqueeze(-1)
    levels = levels.view(1, -1)
    return normalized.sub(levels).abs().argmin(dim=-1)


def _normalize_levels(fmt: QuantFormat) -> tuple[torch.Tensor, int, int, int, int]:
    """Return (level_tensor, grid_min, grid_max, idx_min, idx_max) for a format."""
    if fmt.is_learned:
        levels = torch.tensor(fmt.learned_levels, dtype=torch.float32)
        return levels, 0, len(levels) - 1, 0, len(levels) - 1
    levels = torch.tensor(fmt.weight_levels, dtype=torch.float32)
    int_levels = levels.to(torch.int64)
    if torch.equal(levels, int_levels.to(levels.dtype)):
        grid_min = int(int_levels.min().item())
        grid_max = int(int_levels.max().item())
    else:
        grid_min = 0
        grid_max = len(levels) - 1
    return levels, grid_min, grid_max, 0, len(levels) - 1


def _group_shape(shape: tuple[int, ...], group_size: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return (grouped_shape, original_shape, padded_last) for last-dim grouping."""
    if len(shape) == 1:
        last = shape[0]
        padded = ((last + group_size - 1) // group_size) * group_size
        return ((padded // group_size, group_size), shape, padded)
    last = shape[-1]
    padded = ((last + group_size - 1) // group_size) * group_size
    batch = 1
    for s in shape[:-1]:
        batch *= s
    grouped_shape = (batch, padded // group_size, group_size)
    return grouped_shape, shape, padded


def _reshape_to_group(tensor: torch.Tensor, group_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Reshape tensor for groupwise quantization; pads last dimension if needed."""
    grouped_shape, orig_shape, padded_last = _group_shape(tuple(tensor.shape), group_size)
    if tensor.dim() == 1:
        padded = torch.nn.functional.pad(tensor, (0, padded_last - tensor.shape[0]))
        return padded.view(grouped_shape), orig_shape, padded_last
    # >= 2 dims: flatten leading, pad last, group.
    flat = tensor.flatten(0, -2)
    padded = torch.nn.functional.pad(flat, (0, padded_last - flat.shape[-1]))
    grouped = padded.view(grouped_shape)
    return grouped, orig_shape, padded_last


def _reshape_back(
    grouped: torch.Tensor,
    orig_shape: tuple[int, ...],
    padded_last: int,
) -> torch.Tensor:
    """Undo grouping and trim padding to restore original shape."""
    if len(orig_shape) == 1:
        flat = grouped.view(-1)
        return flat[: orig_shape[0]]
    batch = 1
    for s in orig_shape[:-1]:
        batch *= s
    flat = grouped.view(batch, padded_last)
    flat = flat[:, : orig_shape[-1]]
    return flat.view(orig_shape)


def fake_quantize(
    tensor: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fake-quantize an arbitrary tensor using ``fmt``.

    Returns ``(quantized_tensor, scale, zero_point)``.  ``group_size=None``
    selects per-tensor scaling; a positive value selects last-dim groupwise
    scaling.  When ``tensor`` requires gradients the forward is wrapped in a
    straight-through estimator.
    """

    def _quant_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return _fake_quantize_tensor(x, fmt, group_size=group_size, symmetric=symmetric)

    if tensor.requires_grad:
        return _FakeQuantSTE.apply(tensor, _quant_fn)
    return _quant_fn(tensor)


def fake_quantize_weight(
    weight: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fake-quantize a weight tensor and return dequantized values plus scale/zp."""
    return _fake_quantize_tensor(weight, fmt, group_size=group_size, symmetric=symmetric)


def _fake_quantize_tensor(
    tensor: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if fmt.format_id in ("fp16", "bf16"):
        return tensor, torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device), None

    levels, grid_min, grid_max, idx_min, idx_max = _normalize_levels(fmt)
    levels = levels.to(tensor.device, tensor.dtype)

    if fmt.is_learned:
        idx = _nearest_level_index(tensor, levels)
        quant = levels[idx]
        scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
        return quant, scale, None

    if fmt.format_id == "binary_plus_mask":
        scale = tensor.abs().amax()
        mask = tensor.abs().gt(scale * 0.5)
        sign = torch.where(tensor >= 0, torch.ones_like(tensor), torch.full_like(tensor, -1.0))
        quant = sign * mask.float() * scale
        return quant, scale, None

    if group_size is None or tensor.numel() < group_size:
        quant, scale, zp = _observe_and_quantize(
            tensor, levels, grid_min, grid_max, idx_min, idx_max, symmetric, group_size=None
        )
        return quant, scale, zp

    grouped, orig_shape, padded_last = _reshape_to_group(tensor, group_size)
    quant_grouped, scale, zp = _observe_and_quantize(
        grouped, levels, grid_min, grid_max, idx_min, idx_max, symmetric, group_size=group_size
    )
    quant = _reshape_back(quant_grouped, orig_shape, padded_last)
    return quant, scale, zp


def _observe_and_quantize(
    tensor: torch.Tensor,
    levels: torch.Tensor,
    grid_min: int,
    grid_max: int,
    idx_min: int,
    idx_max: int,
    symmetric: bool,
    group_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Per-tensor or already-grouped quantization helper."""
    eps = 1e-8
    if group_size is not None and tensor.shape[-1] == group_size:
        # ``tensor`` is already reshaped so the last dimension is the group.
        if symmetric:
            abs_max = tensor.abs().amax(dim=-1, keepdim=True)
            scale = abs_max / max(abs(grid_min), abs(grid_max))
            zp = torch.zeros_like(scale)
        else:
            wmin = tensor.amin(dim=-1, keepdim=True)
            wmax = tensor.amax(dim=-1, keepdim=True)
            scale = ((wmax - wmin) / max(grid_max - grid_min, eps)).clamp_min(eps)
            zp = grid_min - wmin / scale
    else:
        if symmetric:
            scale, zp = observe_symmetric_scale(tensor, grid_min, grid_max, group_size=None)
        else:
            scale, zp = observe_asymmetric_scale(tensor, grid_min, grid_max, group_size=None)

    # Broadcast scale to the grouped tensor shape.
    while scale.dim() < tensor.dim():
        scale = scale.unsqueeze(-1)
    if zp is not None:
        while zp.dim() < tensor.dim():
            zp = zp.unsqueeze(-1)

    normalized = tensor / scale
    if zp is not None:
        normalized = normalized + zp

    idx = _nearest_level_index(normalized, levels)
    idx = idx.clamp(idx_min, idx_max)
    quant_levels = levels[idx]
    if zp is not None:
        quant_levels = quant_levels - zp
    quant = quant_levels * scale
    return quant, scale.squeeze() if scale.numel() > 1 else scale, zp.squeeze() if zp is not None and zp.numel() > 1 else zp
