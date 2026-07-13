"""Training-free parallel unmasking policies for MaskGIT / discrete diffusion.

Adapted from MaskGIT schedules (Chang et al., 2022) plus confidence-threshold /
neighbor-spacing heuristics used in discrete diffusion LLM decode. Not a
faithful reimplementation of a single dLLM paper — see
``docs/design/research-lineage.md``. No auxiliary model — drops into MaskGIT.
"""

from __future__ import annotations

import math

import torch


def select_unmask_indices(
    conf: torch.Tensor,
    unknown: torch.Tensor,
    *,
    step: int,
    steps: int,
    mode: str = "adaptive",
    min_spacing: int = 2,
) -> list[int]:
    """
    Choose flat indices to unmask this step.

    conf/unknown: [B, T] (typically B=1 for MaskGIT path).
    Modes:
      - topk: classic MaskGIT ceil(remaining / remaining_steps)
      - confidence: all masked positions above adaptive tau
      - adaptive: confidence with mean-field spacing (skip neighbors)
    """
    flat_conf = conf.view(-1)
    flat_unk = unknown.view(-1)
    remaining = int(flat_unk.sum().item())
    if remaining <= 0:
        return []

    steps_left = max(1, steps - step)
    topk_n = max(1, math.ceil(remaining / steps_left))

    if mode == "topk":
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()

    masked_conf = flat_conf[flat_unk]
    if masked_conf.numel() == 0:
        return []
    # Rising threshold: early steps more aggressive, later more selective.
    frac = step / max(1, steps - 1)
    q = 0.35 + 0.45 * frac
    try:
        tau = float(torch.quantile(masked_conf.float(), q).item())
    except Exception:  # noqa: BLE001
        tau = float(masked_conf.median().item())
    cand = (flat_conf >= tau) & flat_unk
    idxs = cand.nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()

    if mode == "confidence":
        # Cap to 2x classic topk to bound quality risk.
        if len(idxs) > topk_n * 2:
            scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
            return scored[: topk_n * 2]
        return idxs

    # adaptive: greedy independent set with spacing (mean-field-lite).
    scored = sorted(idxs, key=lambda i: float(flat_conf[i]), reverse=True)
    chosen: list[int] = []
    taken_pos: set[int] = set()
    length = conf.size(-1)
    for flat in scored:
        t = flat % length
        if any(abs(t - p) < min_spacing for p in taken_pos):
            continue
        chosen.append(flat)
        taken_pos.add(t)
        if len(chosen) >= topk_n * 2:
            break
    if not chosen:
        return flat_conf.topk(min(topk_n, remaining)).indices.tolist()
    return chosen


def select_remask_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
) -> list[int]:
    """
    Remask lowest-confidence already-unmasked tokens (GIDD / ReMDM-lite).

    conf: [B, T] confidence of current committed tokens (higher = keep).
    known: [B, T] True where token is currently unmasked / committed.
    """
    if remask_ratio <= 0.0:
        return []
    flat_conf = conf.view(-1)
    flat_known = known.view(-1).clone()
    if protect_bos and flat_known.numel() > 0:
        # Position 0 is BOS on the single-sequence MaskGIT path.
        length = conf.size(-1)
        for b in range(conf.size(0)):
            flat_known[b * length] = False
    eligible = int(flat_known.sum().item())
    if eligible <= 0:
        return []
    k = max(1, int(math.ceil(eligible * float(remask_ratio))))
    k = min(k, eligible)
    # Lowest confidence among known positions.
    scores = flat_conf.clone()
    scores = scores.masked_fill(~flat_known, float("inf"))
    return scores.topk(k, largest=False).indices.tolist()
