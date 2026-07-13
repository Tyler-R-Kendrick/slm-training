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


def _dfa_engine(grammar_dsl: str | None = None):
    try:
        from slm_training.grammar_fastpath import engine_for_dsl
    except Exception:  # noqa: BLE001
        return None
    return engine_for_dsl(grammar_dsl or active_dsl())


def dfa_admits_token(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    token_id: int,
    *,
    grammar_dsl: str | None = None,
    engine=None,
) -> bool:
    """True iff Lark incremental parse accepts ``prefix + token`` as a legal prefix."""
    eng = engine if engine is not None else _dfa_engine(grammar_dsl)
    if eng is None:
        return True  # no DFA available — defer to stream_check
    text = tokenizer.decode([*prefix_ids, int(token_id)])
    try:
        return bool(eng.set_prefix(text))
    except Exception:  # noqa: BLE001
        return False


def _stream_probe_ok(tokenizer: OpenUITokenizer, prefix_ids: list[int], token_id: int) -> bool:
    """Reject unknown-component / typed hard errors via streaming semantic check."""
    trial_ids = [*prefix_ids, int(token_id)]
    text = tokenizer.decode(trial_ids)
    probe = text if text.endswith(("(", "[", ",", "=", " ", "\n")) else f"{text}("
    try:
        status = stream_check(probe)
    except Exception:  # noqa: BLE001
        return False
    return not status.hard_error


def pick_constrained_token(
    logits_1d,
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    top_k: int = 16,
    forced_token_id: int | None = None,
) -> int | None:
    """
    Speculative constrained pick: only tokens admitted by the grammar DFA
    (and not rejected by stream hard-errors) may be selected.

    This is *pseudo* speculative decoding (verify against the OpenUI acceptor),
    not draft-model speculative decoding — see docs/design/research-lineage.md.

    When ``forced_token_id`` is set (singleton DFA structural emit), that id is
    returned only if the DFA still admits it.

    Returns ``None`` when no legal candidate exists (never returns a DFA-illegal
    or hard-error token).
    """
    import torch

    engine = _dfa_engine()
    allowed: set[int] | None = None
    if engine is not None:
        prefix_text = tokenizer.decode(prefix_ids)
        try:
            synced = bool(engine.set_prefix(prefix_text))
        except Exception:  # noqa: BLE001
            synced = False
        if not synced and prefix_text.strip():
            # Prefix already illegal — no legal continuation.
            return None
        try:
            from slm_training.grammar_fastpath.token_map import allowed_id_set

            allowed = allowed_id_set(tokenizer, engine.next_terminals())
        except Exception:  # noqa: BLE001
            allowed = None

    def _legal(token_id: int) -> bool:
        tid = int(token_id)
        if tid in {
            tokenizer.pad_id,
            tokenizer.mask_id,
            tokenizer.bos_id,
        }:
            return False
        if allowed is not None and tid not in allowed:
            return False
        if engine is not None and not dfa_admits_token(
            tokenizer, prefix_ids, tid, engine=engine
        ):
            return False
        return _stream_probe_ok(tokenizer, prefix_ids, tid)

    if forced_token_id is not None:
        if _legal(int(forced_token_id)):
            return int(forced_token_id)
        forced_token_id = None

    backend = _backend()
    vocab = int(logits_1d.numel())
    search_k = min(max(top_k, 1), vocab)

    # When DFA gave a concrete allowed set, score only those ids (true mask).
    if allowed is not None and allowed:
        scored: list[tuple[float, int]] = []
        for tid in allowed:
            if tid < 0 or tid >= vocab:
                continue
            if not _legal(tid):
                continue
            scored.append((float(logits_1d[tid].item()), tid))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            preferred_names = preferred_components()
            struct = structural_tokens()
            for _score, tid in scored:
                token = tokenizer.id_to_token.get(tid, "")
                if token in preferred_names or token in struct:
                    return tid
            return scored[0][1]

    # Escalate top-k search if no allowed-set hit (or allowed was broad/None).
    for k in (search_k, min(max(search_k * 4, 64), vocab), vocab):
        _values, indices = torch.topk(logits_1d, k=k)
        if not backend.available() and engine is None:
            # Cannot certify legality — refuse rather than emit unconstrained top-1.
            return None

        preferred_names = preferred_components()
        struct = structural_tokens()
        preferred: list[int] = []
        acceptable: list[int] = []

        for idx in indices.tolist():
            token_id = int(idx)
            if not _legal(token_id):
                continue
            token = tokenizer.id_to_token.get(token_id, "")
            text = tokenizer.decode([*prefix_ids, token_id])
            probe = text if text.endswith(("(", "[", ",", "=", " ", "\n")) else f"{text}("
            try:
                status = stream_check(probe)
            except Exception:  # noqa: BLE001
                status = None
            if (
                token in preferred_names
                or token in struct
                or (status is not None and (status.has_root or status.incomplete or status.complete_ok))
            ):
                preferred.append(token_id)
            else:
                acceptable.append(token_id)

        if preferred:
            return preferred[0]
        if acceptable:
            return acceptable[0]
        if k >= vocab:
            break
    return None


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
        # Also remask when the Lark DFA rejects the full prefix.
        engine = _dfa_engine()
        if engine is not None:
            try:
                if not engine.set_prefix(text):
                    return list(newly_filled)
            except Exception:  # noqa: BLE001
                pass
        return []
    return list(newly_filled)


__all__ = [
    "PREFERRED_COMPONENT_NAMES",
    "STRUCTURAL_TOKENS",
    "StreamStatus",
    "active_dsl",
    "apply_structural_bias",
    "dfa_admits_token",
    "filter_ids_by_stream",
    "force_emit_token_id",
    "pick_constrained_token",
    "preferred_components",
    "set_active_dsl",
    "stream_check",
    "structural_token_ids",
    "structural_tokens",
]
