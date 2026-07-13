"""Deterministic force-emit when the grammar allows a singleton continuation."""

from __future__ import annotations

from slm_training.grammar_fastpath.engine import OpenUIIncrementalEngine
from slm_training.grammar_fastpath.token_map import string_to_token_ids
from slm_training.models.tokenizer import OpenUITokenizer


def force_next_token_id(
    engine: OpenUIIncrementalEngine,
    tokenizer: OpenUITokenizer,
    prefix_text: str,
) -> int | None:
    """If DFA says the next lexeme is fully determined, return its token id."""
    if not engine.set_prefix(prefix_text):
        return None
    forced = engine.is_deterministic_next()
    if forced is None:
        return None
    ids = string_to_token_ids(tokenizer, forced)
    if len(ids) == 1:
        return ids[0]
    return None


def draft_forced_ids(
    engine: OpenUIIncrementalEngine,
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    max_tokens: int = 8,
) -> list[int]:
    """Speculatively draft up to max_tokens forced continuations."""
    text = tokenizer.decode(prefix_ids)
    drafted: list[int] = []
    for _ in range(max_tokens):
        tid = force_next_token_id(engine, tokenizer, text)
        if tid is None:
            break
        drafted.append(tid)
        text = tokenizer.decode(prefix_ids + drafted)
    return drafted
