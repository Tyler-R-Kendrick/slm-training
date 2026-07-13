"""Grammar-constrained decode helpers via pluggable GrammarBackend.

Default backend is OpenUI hybrid (official lang-core when available, else Lark).
Stream checks and structural priors come from the active DSL backend so other
grammars can drive the same MaskGIT / LTR decode path.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

from slm_training.dsl.openui_tokens import (
    PREFERRED_COMPONENT_NAMES,
    STRUCTURAL_TOKENS,
)
from slm_training.dsl.stream_types import StreamStatus
from slm_training.models.tokenizer import OpenUITokenizer

_STREAM_CACHE: dict[str, StreamStatus] = {}
_STREAM_CACHE_MAX = 2048
_STRUCT_ID_CACHE: dict[int, set[int]] = {}
_BIAS_CACHE: dict[tuple[int, float, str, str], object] = {}
_ACTIVE_DSL: str | None = None


def set_active_dsl(dsl: str | None) -> None:
    """Select grammar backend id used by stream_check / structural priors."""
    global _ACTIVE_DSL, _STREAM_CACHE, _STRUCT_ID_CACHE, _BIAS_CACHE
    _ACTIVE_DSL = dsl
    _STREAM_CACHE.clear()
    _STRUCT_ID_CACHE.clear()
    _BIAS_CACHE.clear()
    if dsl:
        from slm_training.grammar_backends import set_default_backend

        set_default_backend(dsl)


def active_dsl() -> str:
    return _ACTIVE_DSL or os.getenv("SLM_GRAMMAR_DSL") or "openui"


def _backend():
    from slm_training.grammar_backends import get_backend

    return get_backend(active_dsl())


def stream_check(source: str) -> StreamStatus:
    """Run the active DSL backend's streaming / incremental check."""
    key = hashlib.sha256(f"{active_dsl()}|{source}".encode("utf-8")).hexdigest()
    hit = _STREAM_CACHE.get(key)
    if hit is not None:
        return hit

    status = _backend().stream_check(source)
    if len(_STREAM_CACHE) >= _STREAM_CACHE_MAX:
        _STREAM_CACHE.pop(next(iter(_STREAM_CACHE)))
    _STREAM_CACHE[key] = status
    return status


def structural_tokens() -> frozenset[str]:
    try:
        return _backend().structural_tokens()
    except Exception:  # noqa: BLE001
        return STRUCTURAL_TOKENS


def preferred_components() -> frozenset[str]:
    try:
        comps = _backend().component_names()
        if comps:
            return frozenset(c for c in comps if c in PREFERRED_COMPONENT_NAMES) or comps
    except Exception:  # noqa: BLE001
        pass
    return PREFERRED_COMPONENT_NAMES


def structural_token_ids(tokenizer: OpenUITokenizer) -> set[int]:
    cache_key = id(tokenizer.token_to_id)
    cached = _STRUCT_ID_CACHE.get(cache_key)
    if cached is not None and len(cached) > 0:
        return cached

    ids: set[int] = set()
    for tok in structural_tokens():
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
    """Boost known structural tokens (returns new tensor)."""
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


def force_emit_token_id(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    grammar_dsl: str | None = None,
) -> int | None:
    """Return a forced next token id when the grammar DFA has a singleton structural emit."""
    dsl = grammar_dsl or active_dsl()
    try:
        from slm_training.grammar_fastpath import engine_for_dsl, force_next_token_id
    except Exception:  # noqa: BLE001
        return None
    engine = engine_for_dsl(dsl)
    if engine is None:
        return None
    prefix_text = tokenizer.decode(prefix_ids)
    return force_next_token_id(engine, tokenizer, prefix_text)


def _incomplete_quoted_string(prefix_text: str) -> bool:
    """True when the prefix ends inside an unclosed double-quoted string."""
    return prefix_text.count('"') % 2 == 1


def contract_allowed_token_ids(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    slot_contract: list[str] | None,
) -> set[int] | None:
    """
    When building a quoted placeholder, return allowed next token ids from the
    slot contract inventory. None means no contract filter applies.
    """
    if not slot_contract:
        return None
    prefix_text = tokenizer.decode(prefix_ids)
    if not _incomplete_quoted_string(prefix_text):
        return None

    from slm_training.models.tokenizer import tokenize_text

    last_open = prefix_text.rfind('"')
    built = prefix_text[last_open + 1 :]
    # Ordinary string literals ("column", "row") must not use placeholder contract.
    if not built.startswith(":"):
        return None

    built_seq = tokenize_text(f'"{built}')
    allowed: set[int] = set()
    for ph in slot_contract:
        target = ph if ph.startswith(":") else f":{ph}"
        target_seq = tokenize_text(f'"{target}"')
        if len(built_seq) > len(target_seq):
            continue
        if target_seq[: len(built_seq)] != built_seq:
            continue
        if len(built_seq) < len(target_seq):
            tok = target_seq[len(built_seq)]
            tid = tokenizer.token_to_id.get(tok)
            if tid is not None:
                allowed.add(tid)
        else:
            # Complete placeholder — allow closing quote if present in target
            if len(target_seq) > len(built_seq):
                tok = target_seq[len(built_seq)]
                tid = tokenizer.token_to_id.get(tok)
                if tid is not None:
                    allowed.add(tid)
    return allowed or None


def pick_constrained_token(
    logits_1d,
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    top_k: int = 16,
    forced_token_id: int | None = None,
    slot_contract: list[str] | None = None,
) -> int:
    """
    Choose a next token from top-k that does not introduce a hard streaming error.

    Probes `prefix + token + "("` so bare identifiers are checked as component names.
    Prefers known components / structural tokens from the active backend.

    When ``forced_token_id`` is set (grammar DFA force-emit), that id is returned
    if it does not introduce a hard streaming error.
    """
    import torch

    contract_allowed = contract_allowed_token_ids(
        tokenizer, prefix_ids, slot_contract
    )

    if forced_token_id is not None:
        if contract_allowed is None or int(forced_token_id) in contract_allowed:
            trial_ids = prefix_ids + [int(forced_token_id)]
            text = tokenizer.decode(trial_ids)
            probe = text if text.endswith(("(", "[", ",", "=", " ", "\n")) else f"{text}("
            try:
                status = stream_check(probe)
                if not status.hard_error:
                    return int(forced_token_id)
            except Exception:  # noqa: BLE001
                return int(forced_token_id)

    k = min(top_k, int(logits_1d.numel()))
    _values, indices = torch.topk(logits_1d, k=k)
    fallback = int(indices[0].item())

    backend = _backend()
    if not backend.available():
        return fallback

    preferred_names = preferred_components()
    struct = structural_tokens()
    preferred: list[int] = []
    acceptable: list[int] = []

    for idx in indices.tolist():
        token_id = int(idx)
        if contract_allowed is not None and token_id not in contract_allowed:
            continue
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
            token in preferred_names
            or token in struct
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
    if contract_allowed:
        return next(iter(contract_allowed))
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


__all__ = [
    "PREFERRED_COMPONENT_NAMES",
    "STRUCTURAL_TOKENS",
    "StreamStatus",
    "active_dsl",
    "apply_structural_bias",
    "contract_allowed_token_ids",
    "filter_ids_by_stream",
    "force_emit_token_id",
    "pick_constrained_token",
    "preferred_components",
    "set_active_dsl",
    "stream_check",
    "structural_token_ids",
    "structural_tokens",
]
