"""Grammar-constrained decode helpers using official OpenUI streaming parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from slm_training.models.tokenizer import OpenUITokenizer

# Soft structural prior for MaskGIT logits (OpenUI subset).
STRUCTURAL_TOKENS = frozenset(
    {
        "root",
        "Stack",
        "Card",
        "Text",
        "Button",
        "vertical",
        "horizontal",
        "=",
        "(",
        ")",
        "[",
        "]",
        ",",
        "\n",
        " ",
        '"',
    }
)

PREFERRED_COMPONENT_NAMES = frozenset({"Stack", "Card", "Text", "Button"})


@dataclass(frozen=True)
class StreamStatus:
    ok: bool
    incomplete: bool
    has_root: bool
    error_codes: tuple[str, ...]
    unresolved: tuple[str, ...]
    serialized: str | None = None

    @property
    def hard_error(self) -> bool:
        hard = {
            "unknown-component",
            "invalid-type",
            "unexpected-token",
            "placeholder_required",
        }
        return any(code in hard for code in self.error_codes)

    @property
    def complete_ok(self) -> bool:
        return (
            self.ok
            and self.has_root
            and not self.incomplete
            and not self.error_codes
            and not self.unresolved
        )


def stream_check(source: str) -> StreamStatus:
    """Run official createStreamingParser over a (possibly partial) program."""
    from slm_training.dsl import lang_core

    result = lang_core.stream_check(source)
    errors = result.get("errors") or []
    codes = tuple(
        str(e.get("code") or e.get("message") or "error")
        for e in errors
        if isinstance(e, dict)
    )
    return StreamStatus(
        ok=bool(result.get("ok")),
        incomplete=bool(result.get("incomplete")),
        has_root=bool(result.get("has_root")),
        error_codes=codes,
        unresolved=tuple(result.get("unresolved") or []),
        serialized=result.get("serialized"),
    )


def structural_token_ids(tokenizer: OpenUITokenizer) -> set[int]:
    ids: set[int] = set()
    for tok in STRUCTURAL_TOKENS:
        if tok in tokenizer.token_to_id:
            ids.add(tokenizer.token_to_id[tok])
    for tok, tid in tokenizer.token_to_id.items():
        if tok.startswith(":") or (tok.startswith('"') and tok.endswith('"')):
            ids.add(tid)
    ids.update(
        {
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.mask_id,
        }
    )
    return ids


def apply_structural_bias(
    logits,  # torch.Tensor [B, T, V]
    tokenizer: OpenUITokenizer,
    *,
    bias: float = 1.5,
):
    """Boost known OpenUI structural tokens (returns new tensor)."""
    import torch

    allowed = structural_token_ids(tokenizer)
    if not allowed:
        return logits
    boost = torch.zeros(logits.size(-1), device=logits.device, dtype=logits.dtype)
    idx = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
    boost.index_fill_(0, idx, bias)
    return logits + boost.view(1, 1, -1)


def pick_constrained_token(
    logits_1d,
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    top_k: int = 16,
) -> int:
    """
    Choose a next token from top-k that does not introduce a hard streaming error.

    Probes `prefix + token + "("` so bare identifiers are checked as component names.
    Prefers known OpenUI components / structural tokens.
    """
    import torch

    k = min(top_k, int(logits_1d.numel()))
    _values, indices = torch.topk(logits_1d, k=k)
    fallback = int(indices[0].item())
    try:
        from slm_training.dsl import bridge_available
    except Exception:  # noqa: BLE001
        return fallback
    if not bridge_available():
        return fallback

    preferred: list[int] = []
    acceptable: list[int] = []

    for idx in indices.tolist():
        token_id = int(idx)
        token = tokenizer.id_to_token.get(token_id, "")
        trial_ids = prefix_ids + [token_id]
        text = tokenizer.decode(trial_ids)
        probe = text if text.endswith(("(", "[", ",", "=", " ", "\n")) else f"{text}("
        try:
            status = stream_check(probe)
        except Exception:  # noqa: BLE001
            acceptable.append(token_id)
            continue
        if status.hard_error:
            continue
        if (
            token in PREFERRED_COMPONENT_NAMES
            or token in STRUCTURAL_TOKENS
            or status.has_root
            or status.incomplete
            or status.complete_ok
        ):
            preferred.append(token_id)
        else:
            acceptable.append(token_id)

    if preferred:
        return preferred[0]
    if acceptable:
        return acceptable[0]
    return fallback


def filter_ids_by_stream(
    tokenizer: OpenUITokenizer,
    token_ids: list[int],
    newly_filled: Iterable[int],
) -> list[int]:
    """
    If the current decode has hard errors, return positions among newly_filled
    that should be remasked.
    """
    text = tokenizer.decode(token_ids)
    try:
        status = stream_check(text)
    except Exception:  # noqa: BLE001
        return []
    if not status.hard_error:
        return []
    return list(newly_filled)
