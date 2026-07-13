"""Grammar-constrained decode helpers using official OpenUI streaming parser.

Stream checks go through the OpenUI Node bridge (not the Cactus kernel).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from slm_training.models.tokenizer import OpenUITokenizer

# Soft structural prior for MaskGIT logits (official openuiLibrary).
STRUCTURAL_TOKENS = frozenset(
    {
        "root",
        "Stack",
        "Card",
        "CardHeader",
        "TextContent",
        "Button",
        "Buttons",
        "Input",
        "Form",
        "FormControl",
        "Label",
        "TextArea",
        "Select",
        "SelectItem",
        "CheckBoxGroup",
        "CheckBoxItem",
        "RadioGroup",
        "RadioItem",
        "SwitchGroup",
        "SwitchItem",
        "Slider",
        "DatePicker",
        "Image",
        "ImageBlock",
        "ImageGallery",
        "Modal",
        "Tabs",
        "TabItem",
        "Callout",
        "TextCallout",
        "Separator",
        "Table",
        "Col",
        "column",
        "row",
        "none",
        "xs",
        "s",
        "m",
        "l",
        "xl",
        "2xl",
        "primary",
        "secondary",
        "tertiary",
        "small",
        "default",
        "large",
        "small-heavy",
        "large-heavy",
        "=",
        "(",
        ")",
        "[",
        "]",
        ",",
        "\n",
        " ",
        '"',
        "null",
        "true",
        "false",
    }
)

PREFERRED_COMPONENT_NAMES = frozenset(
    {
        "Stack",
        "Card",
        "TextContent",
        "Button",
        "Input",
        "Form",
        "ImageBlock",
        "Modal",
        "Tabs",
        "Slider",
        "CheckBoxItem",
        "RadioItem",
        "SwitchItem",
        "DatePicker",
    }
)

_STREAM_CACHE: dict[str, "StreamStatus"] = {}
_STREAM_CACHE_MAX = 2048
_STRUCT_ID_CACHE: dict[int, set[int]] = {}
_BIAS_CACHE: dict[tuple[int, float, str, str], object] = {}


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
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()
    hit = _STREAM_CACHE.get(key)
    if hit is not None:
        return hit

    from slm_training.dsl import lang_core

    result = lang_core.stream_check(source)
    errors = result.get("errors") or []
    codes = tuple(
        str(e.get("code") or e.get("message") or "error")
        for e in errors
        if isinstance(e, dict)
    )
    status = StreamStatus(
        ok=bool(result.get("ok")),
        incomplete=bool(result.get("incomplete")),
        has_root=bool(result.get("has_root")),
        error_codes=codes,
        unresolved=tuple(result.get("unresolved") or []),
        serialized=result.get("serialized"),
    )
    if len(_STREAM_CACHE) >= _STREAM_CACHE_MAX:
        _STREAM_CACHE.pop(next(iter(_STREAM_CACHE)))
    _STREAM_CACHE[key] = status
    return status


def structural_token_ids(tokenizer: OpenUITokenizer) -> set[int]:
    cache_key = id(tokenizer.token_to_id)
    cached = _STRUCT_ID_CACHE.get(cache_key)
    if cached is not None and len(cached) > 0:
        return cached

    ids: set[int] = set()
    for tok in STRUCTURAL_TOKENS:
        if tok in tokenizer.token_to_id:
            ids.add(tokenizer.token_to_id[tok])
    for tok, tid in tokenizer.token_to_id.items():
        if tok.startswith(":") or (tok.startswith('"') and tok.endswith('"')):
            ids.add(tid)
        if tok[:1].isupper() and tok.isidentifier():
            ids.add(tid)
    ids.update(
        {
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.mask_id,
        }
    )
    _STRUCT_ID_CACHE[cache_key] = ids
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
    cache_key = (
        id(tokenizer.token_to_id),
        float(bias),
        str(logits.device),
        str(logits.dtype),
    )
    boost = _BIAS_CACHE.get(cache_key)
    if boost is None or getattr(boost, "numel", lambda: 0)() != logits.size(-1):
        boost = torch.zeros(logits.size(-1), device=logits.device, dtype=logits.dtype)
        idx = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        boost.index_fill_(0, idx, bias)
        _BIAS_CACHE[cache_key] = boost
    return logits + boost.view(1, 1, -1)  # type: ignore[union-attr]


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
