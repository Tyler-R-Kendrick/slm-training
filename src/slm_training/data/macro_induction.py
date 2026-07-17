"""C3 (SLM-27): Stitch/LILO-style macro-token induction over OpenUI corpora.

Mines recurring fixed-vocabulary token spans from canonicalized programs and
mints ``<MACRO_i>`` expansions for the DSL-native tokenizer: like gist tokens
(Mu et al., NeurIPS 2023) but deterministic and lossless — every macro expands
back to its exact token span at decode.

Design constraints (see ``dsl/canonicalize.py`` header for the α-equivalence
caveat this sidesteps):

* Candidate spans contain only fixed-vocabulary tokens
  (``MACRO_EXPANDABLE_KINDS``): never per-example ``<SYM_i>``/``<BIND_*>``/
  ``<STATE_k>`` rows, never ``NL`` (spans stay within one statement), never
  other macros. Expansions are therefore context-free by construction.
* Induction is greedy MDL-style: each round re-applies accepted macros and
  picks the candidate with the highest net token saving
  ``freq * (len - 1) - len`` (occurrences collapse to one token; the table
  stores ``len`` tokens once). Fully deterministic: ties break on longer
  span, then lexicographic token order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from slm_training.dsl.canonicalize import canonicalize
from slm_training.models.dsl_tokenizer import (
    MACRO_EXPANDABLE_KINDS,
    NL,
    DSLNativeTokenizer,
    SymbolTable,
)


@dataclass(frozen=True)
class MacroInductionConfig:
    min_frequency: int = 3
    min_gain_tokens: int = 4
    max_macros: int = 16
    max_span_len: int = 8


@dataclass(frozen=True)
class MacroInductionResult:
    expansions: tuple[tuple[str, ...], ...]
    stats: dict[str, Any]


def _encode_corpus(
    sources: Iterable[str], tokenizer: DSLNativeTokenizer
) -> list[list[int]]:
    encoded: list[list[int]] = []
    for source in sources:
        canonical = canonicalize(source)
        encoded.append(
            tokenizer.encode(canonical, add_special=False, table=SymbolTable())
        )
    return encoded


def _description_length_bits(streams: list[list[int]]) -> float:
    from slm_training.evals.semantic_bits import _description_length

    tokens = [str(tid) for stream in streams for tid in stream]
    if not tokens:
        return 0.0
    return float(_description_length(tokens)["total_bits"])


def _candidate_counts(
    streams: list[list[int]],
    tokenizer: DSLNativeTokenizer,
    config: MacroInductionConfig,
) -> Counter[tuple[int, ...]]:
    nl_id = tokenizer.token_to_id[NL]
    expandable = {
        tid
        for tid, kind in tokenizer.id_to_kind.items()
        if kind in MACRO_EXPANDABLE_KINDS and tid != nl_id
    }
    counts: Counter[tuple[int, ...]] = Counter()
    for stream in streams:
        for start in range(len(stream)):
            if stream[start] not in expandable:
                continue
            for span in range(2, config.max_span_len + 1):
                end = start + span
                if end > len(stream):
                    break
                if stream[end - 1] not in expandable:
                    break
                counts[tuple(stream[start:end])] += 1
    return counts


def _net_gain(span_len: int, frequency: int) -> int:
    return frequency * (span_len - 1) - span_len


def induce_macros(
    sources: Iterable[str],
    tokenizer: DSLNativeTokenizer,
    config: MacroInductionConfig | None = None,
) -> MacroInductionResult:
    """Mine a deterministic macro table from corpus sources.

    ``tokenizer`` must not already carry macro expansions — induction defines
    the table that will then be installed via ``set_macro_expansions``.
    """
    cfg = config or MacroInductionConfig()
    if tokenizer.macro_expansions:
        raise ValueError("tokenizer already carries a macro table")
    streams = _encode_corpus(sources, tokenizer)
    tokens_before = sum(len(stream) for stream in streams)
    bits_before = _description_length_bits(streams)

    accepted: list[tuple[str, ...]] = []
    per_macro: list[dict[str, Any]] = []
    working = [list(stream) for stream in streams]
    max_macros = min(cfg.max_macros, tokenizer.macro_slots)
    for _ in range(max_macros):
        counts = _candidate_counts(working, tokenizer, cfg)
        best: tuple[int, int, tuple[str, ...], tuple[int, ...]] | None = None
        for span_ids, frequency in counts.items():
            if frequency < cfg.min_frequency:
                continue
            gain = _net_gain(len(span_ids), frequency)
            if gain < cfg.min_gain_tokens:
                continue
            span_tokens = tuple(
                tokenizer.id_to_token[tid] for tid in span_ids
            )
            candidate = (gain, len(span_ids), span_tokens, span_ids)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None:
            break
        gain, span_len, span_tokens, span_ids = best
        frequency = counts[span_ids]
        accepted.append(span_tokens)
        per_macro.append(
            {
                "tokens": list(span_tokens),
                "frequency": int(frequency),
                "net_gain_tokens": int(gain),
            }
        )
        # Collapse occurrences before the next round so later picks reflect
        # the already-compressed corpus (iterative greedy, Stitch-lite).
        placeholder = -1 - len(accepted)
        compressed: list[list[int]] = []
        for stream in working:
            out: list[int] = []
            index = 0
            while index < len(stream):
                if tuple(stream[index : index + span_len]) == span_ids:
                    out.append(placeholder)
                    index += span_len
                else:
                    out.append(stream[index])
                    index += 1
            compressed.append(out)
        working = compressed

    tokens_after = sum(len(stream) for stream in working)
    table_tokens = sum(len(exp) for exp in accepted)
    bits_after = _description_length_bits(working)
    stats = {
        "n_sources": len(streams),
        "n_macros": len(accepted),
        "tokens_before": int(tokens_before),
        "tokens_after_with_table": int(tokens_after + table_tokens),
        "table_tokens": int(table_tokens),
        "description_length_bits_before": round(bits_before, 2),
        "description_length_bits_after": round(bits_after, 2),
        "macros": per_macro,
        "config": {
            "min_frequency": cfg.min_frequency,
            "min_gain_tokens": cfg.min_gain_tokens,
            "max_macros": cfg.max_macros,
            "max_span_len": cfg.max_span_len,
        },
    }
    return MacroInductionResult(expansions=tuple(accepted), stats=stats)


__all__ = [
    "MacroInductionConfig",
    "MacroInductionResult",
    "induce_macros",
]
