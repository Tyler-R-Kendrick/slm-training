"""MaskGIT hole-admissibility (constrained-diffusion-inspired)."""

from __future__ import annotations

from slm_training.grammar_fastpath.engine import OpenUIIncrementalEngine
from slm_training.models.tokenizer import OpenUITokenizer


def admit_fill(
    engine: OpenUIIncrementalEngine,
    tokenizer: OpenUITokenizer,
    token_ids: list[int],
    *,
    mask_id: int | None = None,
) -> bool:
    """
    Return True if the canvas (with remaining masks as holes) can still complete.

    Cheap OpenUI specialization of CFG ∩ completion emptiness
    (Mündler et al. 2025, arXiv:2508.10111 / constrained-diffusion.ai):
    require the contiguous unmasked left-span to be a valid incomplete
    InteractiveParser prefix. Tokens after the first hole are ignored
    (holes can rewrite the suffix).
    """
    mask_id = tokenizer.mask_id if mask_id is None else mask_id
    pieces: list[str] = []
    for tid in token_ids:
        if tid == mask_id:
            pieces.append("<mask>")
        elif tid in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}:
            continue
        else:
            pieces.append(tokenizer.id_to_token.get(tid, ""))
    text = "".join(pieces)
    if "<mask>" in text:
        left = text.split("<mask>", 1)[0]
        if not left:
            return True
        return bool(engine.set_prefix(left))
    # Fully filled — admit if prefix syncs or full completion probe passes.
    if engine.set_prefix(text):
        return True
    return engine.can_complete_with_holes(text)
