"""Training-free parallel unmasking policies for MaskGIT / discrete diffusion.

Adapted from MaskGIT schedules (Chang et al., 2022) plus confidence-threshold /
neighbor-spacing heuristics used in discrete diffusion LLM decode. Not a
faithful reimplementation of a single dLLM paper — see
``docs/design/research-lineage.md``. No auxiliary model — drops into MaskGIT.

V4 (E33): budgeted remask can mix grammar hard-errors, trust-gate scores, and
token entropy — remask, don't replace.
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


def select_remask_policy_indices(
    conf: torch.Tensor,
    known: torch.Tensor,
    *,
    remask_ratio: float = 0.15,
    protect_bos: bool = True,
    grammar_positions: list[int] | None = None,
    gate_trust: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    gate_threshold: float = 0.5,
) -> list[int]:
    """
    E33: budgeted remask mixing grammar hard-errors, trust gate, and entropy.

    Priority order for the remask budget:
      1. Grammar hard-error positions (always included if known)
      2. Lowest trust-gate scores beneath ``gate_threshold`` (if provided)
      3. Highest entropy among remaining known tokens (if provided)
      4. Lowest confidence (classic E22 fallback)

    Returns flat indices; size capped by ``ceil(eligible * remask_ratio)``
    (at least the grammar set when larger).
    """
    if remask_ratio <= 0.0 and not grammar_positions:
        return []
    flat_conf = conf.view(-1)
    flat_known = known.view(-1).clone()
    length = conf.size(-1)
    if protect_bos and flat_known.numel() > 0:
        for b in range(conf.size(0)):
            flat_known[b * length] = False
    eligible_idx = flat_known.nonzero(as_tuple=False).flatten().tolist()
    if not eligible_idx and not grammar_positions:
        return []

    chosen: list[int] = []
    seen: set[int] = set()

    def _add(idx: int) -> None:
        i = int(idx)
        if i in seen:
            return
        if i < 0 or i >= flat_known.numel():
            return
        if protect_bos and (i % length) == 0:
            return
        # Grammar remasks are allowed even if already unknown.
        if not bool(flat_known[i].item()) and i not in set(grammar_positions or []):
            return
        seen.add(i)
        chosen.append(i)

    for g in grammar_positions or []:
        _add(g)

    eligible = max(len(eligible_idx), len(chosen))
    if remask_ratio > 0:
        budget = max(1, int(math.ceil(eligible * float(remask_ratio))))
    else:
        budget = len(chosen)
    budget = max(budget, len(chosen))

    # Trust gate: remask low-trust known tokens.
    if gate_trust is not None and len(chosen) < budget:
        flat_gate = gate_trust.view(-1)
        scored = sorted(
            eligible_idx,
            key=lambda i: float(flat_gate[i].item()) if i < flat_gate.numel() else 1.0,
        )
        for i in scored:
            if len(chosen) >= budget:
                break
            trust = float(flat_gate[i].item()) if i < flat_gate.numel() else 1.0
            if trust <= float(gate_threshold):
                _add(i)

    # Entropy: remask high-entropy known tokens.
    if entropy is not None and len(chosen) < budget:
        flat_ent = entropy.view(-1)
        scored = sorted(
            eligible_idx,
            key=lambda i: float(flat_ent[i].item()) if i < flat_ent.numel() else 0.0,
            reverse=True,
        )
        for i in scored:
            if len(chosen) >= budget:
                break
            _add(i)

    # Confidence fallback to fill remaining budget.
    if len(chosen) < budget:
        scores = flat_conf.clone()
        scores = scores.masked_fill(~flat_known, float("inf"))
        for i in chosen:
            if i < scores.numel():
                scores[i] = float("inf")
        need = budget - len(chosen)
        known_count = int(flat_known.sum().item())
        if need > 0 and known_count > 0:
            for idx in scores.topk(
                min(need, known_count), largest=False
            ).indices.tolist():
                _add(idx)

    return chosen
