"""Deterministic min-max scale observers for reference quantization."""

from __future__ import annotations

import torch


def _reshape_for_group(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reshape so the last dimension is the group, padding if necessary.

    For a 2-D weight (out, in) this yields per-output-channel groups along the
    input dimension.  For 1-D tensors it yields (ceil(n/g), g).
    """
    if group_size <= 0 or tensor.numel() < group_size:
        return tensor.view(-1, 1)
    if tensor.dim() == 1:
        padded = _pad_dim(tensor, tensor.numel(), group_size)
        return padded.view(-1, group_size)
    # For >= 2 dims, group the last dimension.
    last = tensor.shape[-1]
    padded = _pad_dim(tensor.flatten(0, -2), last, group_size)
    grouped = padded.view(padded.shape[0], -1, group_size)
    return grouped


def _pad_dim(tensor: torch.Tensor, size: int, multiple: int) -> torch.Tensor:
    """Pad the last dimension of ``tensor`` so ``size`` becomes a multiple."""
    remainder = size % multiple
    if remainder == 0:
        return tensor
    pad = multiple - remainder
    return torch.nn.functional.pad(tensor, (0, pad))


def observe_symmetric_scale(
    tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    group_size: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scale, zero_point) for symmetric quantization.

    ``group_size`` selects groupwise scaling on the last dimension.  When None,
    the returned scale is a scalar per-tensor value.
    """
    grid_max = max(abs(qmin), abs(qmax))
    if group_size is None or tensor.numel() == 0:
        abs_max = tensor.abs().max() if tensor.numel() else torch.tensor(0.0, device=tensor.device)
        scale = torch.tensor(
            abs_max.item() / max(grid_max, eps),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return scale, torch.zeros_like(scale)

    grouped = _reshape_for_group(tensor, group_size)
    abs_max = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / max(grid_max, eps)).clamp_min(eps)
    return scale, torch.zeros_like(scale)


def observe_asymmetric_scale(
    tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    group_size: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scale, zero_point) for asymmetric quantization."""
    qrange = max(qmax - qmin, eps)
    if group_size is None or tensor.numel() == 0:
        if tensor.numel() == 0:
            scale = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
            zp = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
            return scale, zp
        wmin = tensor.min()
        wmax = tensor.max()
        scale = ((wmax - wmin) / qrange).clamp_min(eps)
        zp = qmin - wmin / scale
        return scale, zp

    grouped = _reshape_for_group(tensor, group_size)
    wmin = grouped.amin(dim=-1, keepdim=True)
    wmax = grouped.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / qrange).clamp_min(eps)
    zp = qmin - wmin / scale
    return scale, zp


def dtype_bits(dtype_name: str) -> int:
    """Return the bit width of a named scalar dtype used by the ledger."""
    mapping = {
        "fp32": 32,
        "fp16": 16,
        "bf16": 16,
        "int32": 32,
        "int16": 16,
        "int8": 8,
        "int4": 4,
        "uint8": 8,
        "uint4": 4,
    }
    return mapping.get(dtype_name, 16)
