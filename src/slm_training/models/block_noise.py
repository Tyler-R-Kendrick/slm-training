"""Block corruption schedules shared by grammar-diffusion train and decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockNoiseSchedule:
    """Mask/unmask entire production blocks (not independent token noise)."""

    block_size: int = 4
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not 0.0 <= self.mask_min <= self.mask_max <= 1.0:
            raise ValueError("mask rates must satisfy 0 <= mask_min <= mask_max <= 1")


def num_blocks(seq_len: int, block_size: int) -> int:
    if seq_len <= 0:
        return 0
    return (seq_len + block_size - 1) // block_size


def block_index(position: int, block_size: int) -> int:
    return position // block_size


def block_positions(block_idx: int, seq_len: int, block_size: int) -> range:
    start = block_idx * block_size
    end = min(seq_len, start + block_size)
    return range(start, end)


def corrupt_blocks_for_training(
    seq_len: int,
    *,
    schedule: BlockNoiseSchedule,
    mask_id: int,
    pad_id: int,
    frozen: torch.Tensor,
    target_ids: torch.Tensor,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mask whole blocks for training.

    Returns ``(noisy_ids, predict_mask)`` with the same block boundaries used
    at inference by :func:`select_blocks_to_unmask`.
    """
    device = target_ids.device
    bsz = target_ids.size(0)
    noisy = target_ids.clone()
    predict_mask = torch.zeros_like(target_ids, dtype=torch.bool)

    n_blk = num_blocks(seq_len, schedule.block_size)
    if n_blk == 0:
        return noisy, predict_mask

    for row in range(bsz):
        valid = ~frozen[row]
        if not bool(valid.any()):
            continue
        if rng is not None:
            rate = torch.empty(1, device=device).uniform_(schedule.mask_min, schedule.mask_max, generator=rng).item()
        else:
            rate = torch.empty(1, device=device).uniform_(schedule.mask_min, schedule.mask_max).item()
        masked_blocks: list[int] = []
        for b in range(n_blk):
            if rng is not None:
                draw = torch.rand((), generator=rng, device=device).item()
            else:
                draw = torch.rand((), device=device).item()
            if draw < rate:
                masked_blocks.append(b)
        if not masked_blocks:
            # Guarantee at least one predictable block.
            candidates = [
                b
                for b in range(n_blk)
                if any(valid[p].item() for p in block_positions(b, seq_len, schedule.block_size))
            ]
            if candidates:
                if rng is not None:
                    pick = int(
                        torch.randint(len(candidates), (1,), generator=rng, device=device).item()
                    )
                else:
                    pick = int(torch.randint(len(candidates), (1,), device=device).item())
                masked_blocks = [candidates[pick]]
        for b in masked_blocks:
            for pos in block_positions(b, seq_len, schedule.block_size):
                if not bool(valid[pos].item()):
                    continue
                noisy[row, pos] = mask_id
                predict_mask[row, pos] = True

    return noisy, predict_mask


def block_unknown_mask(unknown: torch.Tensor, block_size: int) -> torch.Tensor:
    """True for blocks that still contain at least one masked position."""
    bsz, seq = unknown.shape
    n_blk = num_blocks(seq, block_size)
    out = torch.zeros(bsz, n_blk, dtype=torch.bool, device=unknown.device)
    for b in range(n_blk):
        cols = list(block_positions(b, seq, block_size))
        if cols:
            out[:, b] = unknown[:, cols].any(dim=1)
    return out


def unmask_budget(*, remaining_blocks: int, step: int, steps: int) -> int:
    """MaskGIT-style per-step block budget (train/infer parity)."""
    if remaining_blocks <= 0:
        return 0
    steps_left = max(1, steps - step)
    return max(1, math.ceil(remaining_blocks / steps_left))


def select_blocks_to_unmask(
    block_conf: torch.Tensor,
    block_unknown: torch.Tensor,
    *,
    step: int,
    schedule: BlockNoiseSchedule,
    mode: str = "adaptive",
    min_spacing: int = 1,
) -> list[int]:
    """
    Choose block indices to reveal this decode step.

    ``block_conf`` / ``block_unknown``: ``[B, n_blocks]``.
    """
    flat_conf = block_conf.view(-1)
    flat_unk = block_unknown.view(-1)
    remaining = int(flat_unk.sum().item())
    if remaining <= 0:
        return []

    budget = unmask_budget(
        remaining_blocks=remaining,
        step=step,
        steps=max(1, schedule.gen_steps),
    )

    if mode == "topk":
        return flat_conf.topk(min(budget, remaining)).indices.tolist()

    masked_conf = flat_conf[flat_unk]
    if masked_conf.numel() == 0:
        return []
    frac = step / max(1, schedule.gen_steps - 1)
    q = 0.35 + 0.45 * frac
    try:
        tau = float(torch.quantile(masked_conf.float(), q).item())
    except Exception:  # noqa: BLE001
        tau = float(masked_conf.median().item())
    cand = (flat_conf >= tau) & flat_unk
    idxs = cand.nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        return flat_conf.topk(min(budget, remaining)).indices.tolist()

    if mode == "confidence":
        if len(idxs) > budget * 2:
            scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
            return scored[: budget * 2]
        return idxs

    # adaptive: greedy block picks with optional spacing in block index space.
    scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
    chosen: list[int] = []
    taken: set[int] = set()
    n_blocks = block_conf.size(-1)
    for flat in scored:
        b = flat % n_blocks
        if any(abs(b - t) < min_spacing for t in taken):
            continue
        chosen.append(flat)
        taken.add(b)
        if len(chosen) >= budget * 2:
            break
    if not chosen:
        return flat_conf.topk(min(budget, remaining)).indices.tolist()
    return chosen


def positions_from_blocks(
    flat_block_indices: list[int],
    seq_len: int,
    block_size: int,
) -> list[int]:
    """Expand flat ``[B * n_blocks]`` indices to absolute sequence positions."""
    n_blk = num_blocks(seq_len, block_size)
    positions: list[int] = []
    for flat in flat_block_indices:
        b = flat % n_blk
        positions.extend(block_positions(b, seq_len, block_size))
    return sorted(set(positions))


def aggregate_block_confidence(
    conf: torch.Tensor,
    unknown: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean confidence per block; unknown mask per block."""
    bsz, seq = conf.shape
    n_blk = num_blocks(seq, block_size)
    block_conf = torch.full(
        (bsz, n_blk), -1.0, device=conf.device, dtype=conf.dtype
    )
    block_unk = torch.zeros(bsz, n_blk, dtype=torch.bool, device=conf.device)
    for b in range(n_blk):
        cols = list(block_positions(b, seq, block_size))
        if not cols:
            continue
        col_t = torch.tensor(cols, device=conf.device, dtype=torch.long)
        sub_conf = conf.index_select(1, col_t)
        sub_unk = unknown.index_select(1, col_t)
        block_unk[:, b] = sub_unk.any(dim=1)
        masked = sub_conf.masked_fill(~sub_unk, float("-inf"))
        # Blocks with no unknown positions stay at -1 and are filtered downstream.
        block_conf[:, b] = masked.max(dim=1).values
        no_unk = ~block_unk[:, b]
        if bool(no_unk.any()):
            block_conf[no_unk, b] = -1.0
    return block_conf, block_unk


__all__ = [
    "BlockNoiseSchedule",
    "aggregate_block_confidence",
    "block_index",
    "block_positions",
    "block_unknown_mask",
    "corrupt_blocks_for_training",
    "num_blocks",
    "positions_from_blocks",
    "select_blocks_to_unmask",
    "unmask_budget",
]
