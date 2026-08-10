"""Request-independent packed semantic kind summary (EXP-SR-7 / RSP-003).

Built solely from :func:`tokenizer_authority` — token ids, kinds, and special
ids. Macro bodies, runtime symbols, binder visibility, and placeholder spellings
are excluded by construction (mirrors PCT-006's static/request-boundary guard).

The summary is a cold-path preload layer on top of the certified completion
artifact; it never becomes production authority and never widens legality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    REQUEST_DEPENDENT_AUTHORITY_EXCLUDED,
    _canonical_json,
    _sha256_bytes,
    tokenizer_authority,
)

SUMMARY_SCHEMA = "openui_packed_semantic_summary/v1"

# Kinds that ``semantic_state.advance`` treats as no-ops (request-independent).
_NO_EFFECT_KINDS = frozenset({"special", "comment", "ws"})


@dataclass(frozen=True)
class PackedSemanticSummaryV1:
    """Frozen tokenizer-id → semantic kind projection."""

    schema: str
    authority_sha256: str
    kind_by_id: tuple[str, ...]
    special_ids: frozenset[int]
    no_effect_kinds: frozenset[str]

    def kind_of(self, token_id: int) -> str:
        tid = int(token_id)
        if not 0 <= tid < len(self.kind_by_id):
            raise IndexError(f"token_id {tid} out of range")
        return str(self.kind_by_id[tid])

    def has_semantic_effect(self, token_id: int) -> bool:
        tid = int(token_id)
        if tid in self.special_ids:
            return False
        kind = self.kind_of(tid)
        if kind in self.no_effect_kinds:
            return False
        raw = kind  # kind-only gate; literal bodies stay request-local
        if raw in {"", "COMMENT", "WS_INLINE"}:
            return False
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_sha256": self.authority_sha256,
            "vocab_size": len(self.kind_by_id),
            "special_id_count": len(self.special_ids),
            "request_dependent_authority_excluded": list(
                REQUEST_DEPENDENT_AUTHORITY_EXCLUDED
            ),
        }


def build_packed_semantic_summary(tokenizer: Any) -> PackedSemanticSummaryV1:
    """Compile the summary from live tokenizer authority only."""

    auth = tokenizer_authority(tokenizer)
    vocab = int(getattr(tokenizer, "vocab_size", 0))
    if vocab <= 0:
        raise ValueError("tokenizer vocab_size must be positive")
    kind_by_id = [""] * vocab
    for token_id, kind in auth["id_to_kind"]:
        tid = int(token_id)
        if 0 <= tid < vocab:
            kind_by_id[tid] = str(kind)
    special = auth.get("special_ids") or {}
    special_ids = frozenset(int(v) for v in special.values())
    return PackedSemanticSummaryV1(
        schema=SUMMARY_SCHEMA,
        authority_sha256=_sha256_bytes(_canonical_json(auth)),
        kind_by_id=tuple(kind_by_id),
        special_ids=special_ids,
        no_effect_kinds=_NO_EFFECT_KINDS,
    )


def load_checked_packed_semantic_summary(tokenizer: Any) -> PackedSemanticSummaryV1:
    """Fail closed when the packed summary diverges from live tokenizer authority."""

    summary = build_packed_semantic_summary(tokenizer)
    from slm_training.dsl.grammar.fastpath import semantic_state as _ss

    for tid in range(len(summary.kind_by_id)):
        live_kind = _ss._kind_of(tokenizer, tid)  # noqa: SLF001 — checker seam
        if live_kind != summary.kind_of(tid):
            raise ValueError(
                f"packed semantic summary kind mismatch at id {tid}: "
                f"{summary.kind_of(tid)!r} vs live {live_kind!r}"
            )
        if (tid in summary.special_ids) != (tid in _ss._special_ids(tokenizer)):  # noqa: SLF001
            raise ValueError(f"packed semantic summary special-id mismatch at {tid}")
    return summary


def try_load_packed_semantic_summary(
    tokenizer: Any,
) -> tuple[PackedSemanticSummaryV1 | None, str | None]:
    """Never raise: checker failures return ``(None, reason)`` for safe fallback."""

    try:
        return load_checked_packed_semantic_summary(tokenizer), None
    except (OSError, ValueError, TypeError, IndexError) as exc:
        return None, str(exc)


def assert_request_independent(summary: PackedSemanticSummaryV1) -> None:
    """Guard that the summary carries no request-local authority keys."""

    banned = {
        "runtime_symbols",
        "scope",
        "binders",
        "placeholders",
        "macro_bodies",
        "literal_body",
        "slot_contract",
    }
    payload = summary.describe()
    overlap = banned & set(payload)
    if overlap:
        raise ValueError(f"request-local keys leaked into summary: {sorted(overlap)}")


__all__ = [
    "PackedSemanticSummaryV1",
    "SUMMARY_SCHEMA",
    "assert_request_independent",
    "build_packed_semantic_summary",
    "load_checked_packed_semantic_summary",
    "try_load_packed_semantic_summary",
]
