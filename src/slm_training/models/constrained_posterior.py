"""Constrained posterior for grammar-native block diffusion decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from slm_training.models.block_noise import (
    BlockNoiseSchedule,
    aggregate_block_confidence,
    positions_from_blocks,
    select_blocks_to_unmask,
)


class ProductionCodecLike(Protocol):
    mask_id: int
    pad_id: int
    eos_id: int

    def decode(
        self,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
        *,
        stop_at_mask: bool = False,
    ) -> str:
        ...


@dataclass
class ExtendabilityChecker:
    """Grammar extendability probes for partial production canvases."""

    grammar_dsl: str = "openui"
    require_bridge: bool = False

    def prefix_extendable(
        self,
        codec: ProductionCodecLike,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
    ) -> bool:
        """
        True when the leftmost unmasked span can still complete in the grammar.

        Mirrors the MaskGIT ``admit_fill`` specialization: tokens after the
        first hole are ignored because holes may rewrite the suffix.
        """
        text = codec.decode(
            production_ids,
            slot_ids,
            slot_inventory,
            stop_at_mask=True,
        )
        if not text.strip():
            return True
        try:
            from slm_training.grammar_fastpath import engine_for_dsl
            from slm_training.models.grammar import active_dsl, stream_check

            engine = engine_for_dsl(self.grammar_dsl or active_dsl())
            if engine is not None:
                left = text.split("<mask>", 1)[0] if "<mask>" in text else text
                if not left:
                    return True
                if engine.set_prefix(left):
                    return True
                return bool(engine.can_complete_with_holes(text))
            status = stream_check(text if text.endswith(("\n", " ", "(", "[", ",")) else f"{text}(")
            return not status.hard_error
        except Exception:  # noqa: BLE001
            if self.require_bridge:
                return False
            return True

    def trial_extendable(
        self,
        codec: ProductionCodecLike,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
        position: int,
        production_id: int,
        slot_id: int,
    ) -> bool:
        trial_prod = list(production_ids)
        trial_slot = list(slot_ids)
        trial_prod[position] = int(production_id)
        trial_slot[position] = int(slot_id)
        return self.prefix_extendable(codec, trial_prod, trial_slot, slot_inventory)


def _topk_candidates(logits_1d: torch.Tensor, k: int) -> list[int]:
    k = min(max(1, k), logits_1d.numel())
    return torch.topk(logits_1d, k=k).indices.tolist()


def pick_constrained_production(
    production_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    *,
    position: int,
    production_ids: list[int],
    slot_ids: list[int],
    slot_inventory: list[str],
    codec: ProductionCodecLike,
    checker: ExtendabilityChecker,
    top_k: int = 8,
    slot_none_id: int = 0,
) -> tuple[int, int, float] | None:
    """Return ``(production_id, slot_id, confidence)`` or ``None`` if no legal fill."""
    if production_logits.dim() == 3:
        prod_scores = production_logits[0, position]
        slot_scores = slot_logits[0, position]
    else:
        prod_scores = production_logits[position]
        slot_scores = slot_logits[position]
    prod_probs = F.softmax(prod_scores, dim=-1)
    conf, _ = prod_probs.max(dim=-1)
    confidence = float(conf.item())

    for prod_id in _topk_candidates(prod_scores, top_k):
        if int(prod_id) in {
            codec.pad_id,
            codec.mask_id,
        }:
            continue
        slot_id = slot_none_id
        prod_name = ""
        try:
            prod_name = codec.id_to_production[int(prod_id)]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            prod_name = ""
        if prod_name == "SLOT":
            slot_id = int(slot_scores.argmax(dim=-1).item())
            if slot_id == slot_none_id and slot_inventory:
                slot_id = 1
        if checker.trial_extendable(
            codec,
            production_ids,
            slot_ids,
            slot_inventory,
            position,
            int(prod_id),
            int(slot_id),
        ):
            return int(prod_id), int(slot_id), confidence
    return None


def parallel_commit_selection(
    production_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    confidence: torch.Tensor,
    unknown: torch.Tensor,
    *,
    production_ids: list[int],
    slot_ids: list[int],
    slot_inventory: list[str],
    codec: ProductionCodecLike,
    checker: ExtendabilityChecker,
    schedule: BlockNoiseSchedule,
    step: int,
    mode: str = "adaptive",
    top_k: int = 8,
) -> list[tuple[int, int, int, float]]:
    """
    Select parallel commits for one decode step.

    Returns a list of ``(position, production_id, slot_id, confidence)``.
    """
    block_conf, block_unk = aggregate_block_confidence(
        confidence, unknown, schedule.block_size
    )
    flat_blocks = select_blocks_to_unmask(
        block_conf,
        block_unk,
        step=step,
        schedule=schedule,
        mode=mode,
    )
    seq_len = unknown.size(-1)
    positions = positions_from_blocks(flat_blocks, seq_len, schedule.block_size)
    commits: list[tuple[int, int, int, float]] = []
    prod_canvas = list(production_ids)
    slot_canvas = list(slot_ids)

    for pos in positions:
        if pos >= seq_len or not bool(unknown[0, pos].item()):
            continue
        picked = pick_constrained_production(
            production_logits,
            slot_logits,
            position=pos,
            production_ids=prod_canvas,
            slot_ids=slot_canvas,
            slot_inventory=slot_inventory,
            codec=codec,
            checker=checker,
            top_k=top_k,
        )
        if picked is None:
            continue
        prod_id, slot_id, conf = picked
        prod_canvas[pos] = prod_id
        slot_canvas[pos] = slot_id
        commits.append((pos, prod_id, slot_id, conf))
    return commits


def adaptive_should_stop(
    unknown: torch.Tensor,
    confidence: torch.Tensor,
    *,
    step: int,
    schedule: BlockNoiseSchedule,
    min_confidence: float = 0.55,
) -> bool:
    """Stop early when remaining masks are high-confidence or budget is exhausted."""
    if not bool(unknown.any()):
        return True
    if step + 1 >= schedule.gen_steps:
        return True
    masked = confidence.masked_select(unknown)
    if masked.numel() == 0:
        return True
    return float(masked.min().item()) >= min_confidence


def apply_commits(
    production_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    unknown: torch.Tensor,
    commits: list[tuple[int, int, int, float]],
) -> None:
    """In-place apply parallel commits to batch 0."""
    for pos, prod_id, slot_id, _conf in commits:
        production_ids[0, pos] = prod_id
        slot_ids[0, pos] = slot_id
        unknown[0, pos] = False


__all__ = [
    "ExtendabilityChecker",
    "ProductionCodecLike",
    "adaptive_should_stop",
    "apply_commits",
    "parallel_commit_selection",
    "pick_constrained_production",
]
