"""Block-structured bottleneck attention (AP-019 / SLM-307).

Reproduces the core Abstract-CoT bottleneck (arXiv:2604.22709): a sequence is
split into four ordered segments -- ``prompt``, ``privileged_plan``,
``abstract`` (the reasoning bottleneck), and ``target`` (the answer/program) --
and abstract tokens may read the privileged plan, but target tokens may
**never** attend privileged-plan positions directly, even though a plain
causal mask would otherwise expose them (the plan precedes the target in
sequence order). This module is deliberately independent of any specific
model/harness: it only builds the additive attention mask and the label mask,
following the same "4D float mask overrides the backbone's causal mask"
convention already used by ``models/hf_denoiser.py``.

Nothing in the repository calls this module by default; it is opt-in
plumbing consumed by ``CausalLMOpenUIPlugin.forward_with_segments``.
"""

from __future__ import annotations

import enum

import torch


class SegmentKind(enum.IntEnum):
    """Per-token segment identity within a block-bottleneck sequence."""

    PROMPT = 0
    PRIVILEGED_PLAN = 1
    ABSTRACT = 2
    TARGET = 3


def build_block_bottleneck_mask(
    segment_ids: torch.Tensor,
    example_ids: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive ``[batch, 1, seq, seq]`` bottleneck attention mask.

    ``segment_ids`` is ``[batch, seq]`` with values from :class:`SegmentKind`.
    Visibility is plain causal (query position ``i`` may see key position
    ``j`` only when ``j <= i``) with exactly one additional restriction: a
    ``TARGET`` query may never see a ``PRIVILEGED_PLAN`` key, regardless of
    causal order. Every other documented rule -- abstract tokens read prompt
    + privileged plan + prior abstract; target tokens read prompt + full
    abstract span + prior targets -- falls out of causal order alone, since
    the four segments appear in that fixed sequence order.

    ``example_ids`` (``[batch, seq]``, optional) supports packed batches: a
    query may only attend keys sharing its own example id, preventing
    cross-example leakage inside one packed row.
    """
    if segment_ids.dim() != 2:
        raise ValueError("segment_ids must be a [batch, seq_len] tensor")
    batch_size, seq_len = segment_ids.shape
    device = segment_ids.device

    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    query_is_target = (segment_ids == SegmentKind.TARGET).unsqueeze(-1)
    key_is_plan = (segment_ids == SegmentKind.PRIVILEGED_PLAN).unsqueeze(-2)
    bottleneck_block = query_is_target & key_is_plan

    visible = causal.unsqueeze(0) & ~bottleneck_block

    if example_ids is not None:
        if example_ids.shape != segment_ids.shape:
            raise ValueError("example_ids must match segment_ids shape")
        same_example = example_ids.unsqueeze(-1) == example_ids.unsqueeze(-2)
        visible = visible & same_example

    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def loss_position_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Boolean ``[batch, seq]`` mask: ``True`` at ABSTRACT/TARGET positions.

    Only these positions may contribute to the masked SFT loss.
    """
    return (segment_ids == SegmentKind.ABSTRACT) | (segment_ids == SegmentKind.TARGET)


def apply_loss_mask(
    labels: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return ``labels`` with every non-(abstract|target) position set to ``ignore_index``."""
    if labels.shape != segment_ids.shape:
        raise ValueError("labels must match segment_ids shape")
    keep = loss_position_mask(segment_ids)
    ignore = torch.full_like(labels, ignore_index)
    return torch.where(keep, labels, ignore)
