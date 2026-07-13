"""Auxiliary force-align loss for grammar fast-path training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from slm_training.models.tokenizer import OpenUITokenizer


def force_align_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: OpenUITokenizer,
    *,
    pad_id: int,
) -> torch.Tensor:
    """
    Cheap structural prior: CE on positions whose gold token is punctuation
    that is almost always grammar-forced (`= ( ) [ ] ,`).

    Avoids walking a full DFA over every gold prefix (too expensive for train).
    """
    forced_strings = {"=", "(", ")", "[", "]", ","}
    forced_ids = {
        tokenizer.token_to_id[s]
        for s in forced_strings
        if s in tokenizer.token_to_id
    }
    if not forced_ids:
        return logits.sum() * 0.0
    tid = torch.tensor(sorted(forced_ids), device=target_ids.device, dtype=target_ids.dtype)
    mask = torch.isin(target_ids, tid) & target_ids.ne(pad_id)
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target_ids[mask])
