"""Deterministic speculative ranking over forward-calculated symbol tables.

Decode invariant I3 (`docs/design/decode-invariants.md`): symbol tables are
computed *before* the model. At a non-singleton branch point, the choice among
already-legal candidates may be made by a **deterministic** scorer instead of a
neural forward, and a run of such choices may be committed as one speculative
span — lookahead-then-verify (arXiv:2602.00612), with the grammar oracle as the
verifier (intersection-witness completions, arXiv:2508.10111).

Two rules hold without exception:

* **Ranking is a lever; legality is not.** Everything here only *orders* or
  *selects among* candidates that the completion forest already proved legal.
  No function in this module can add a candidate.
* **Speculation verifies before it commits.** Every token in a span is drawn
  from a forest whose coverage is ``complete``; a partial proof stops the span
  rather than guessing.

The scorer is a back-off n-gram over decoder token ids, built from the training
corpus and persisted as a content-addressed artifact so a run can say exactly
which table produced its decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)

PathKey = tuple[int, ...]

NGRAM_TABLE_SCHEMA = "speculative_ngram_table/v1"
SPECULATIVE_RANK_IMPL_VERSION = "speculative_rank/v1"

# Stupid-backoff discount (Brants et al. 2007). Deterministic and cheap; the
# absolute value never matters, only the induced order and the margin.
_BACKOFF = 0.4
_MIN_LOG_PROB = -30.0


def _fingerprint(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NgramTableV1:
    """Back-off n-gram counts over decoder token ids.

    ``counts`` maps a context (0 to ``order - 1`` trailing token ids) to the
    observed next-token counts. Contexts are stored as tuples; the empty tuple
    is the unigram level and is always present in a non-empty table.
    """

    order: int
    counts: Mapping[PathKey, Mapping[int, int]]
    corpus_fingerprint: str
    sequences: int = 0
    tokens: int = 0
    schema: str = NGRAM_TABLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NGRAM_TABLE_SCHEMA:
            raise ValueError(f"unsupported n-gram table schema {self.schema!r}")
        if self.order < 1:
            raise ValueError("n-gram order must be >= 1")

    @property
    def is_empty(self) -> bool:
        return not self.counts

    def log_prob(self, context: Sequence[int], token_id: int) -> float:
        """Return a stupid-backoff log score for ``token_id`` after ``context``.

        Unseen everywhere returns ``_MIN_LOG_PROB`` rather than ``-inf`` so a
        margin stays finite and comparisons stay total.
        """
        trimmed = tuple(int(item) for item in context)[-(self.order - 1) :]
        penalty = 0.0
        while True:
            row = self.counts.get(trimmed)
            if row:
                hit = row.get(int(token_id))
                if hit:
                    total = sum(row.values())
                    return math.log(hit / total) + penalty
            if not trimmed:
                return _MIN_LOG_PROB + penalty
            trimmed = trimmed[1:]
            penalty += math.log(_BACKOFF)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "impl_version": SPECULATIVE_RANK_IMPL_VERSION,
            "order": self.order,
            "sequences": self.sequences,
            "tokens": self.tokens,
            "corpus_fingerprint": self.corpus_fingerprint,
            "counts": [
                {
                    "context": list(context),
                    "next": {str(token): int(count) for token, count in sorted(row.items())},
                }
                for context, row in sorted(self.counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NgramTableV1":
        schema = str(payload.get("schema") or "")
        if schema != NGRAM_TABLE_SCHEMA:
            raise ValueError(f"unsupported n-gram table schema {schema!r}")
        counts: dict[PathKey, dict[int, int]] = {}
        for row in payload.get("counts") or ():
            context = tuple(int(item) for item in row.get("context") or ())
            counts[context] = {
                int(token): int(count) for token, count in (row.get("next") or {}).items()
            }
        return cls(
            order=int(payload["order"]),
            counts=counts,
            corpus_fingerprint=str(payload.get("corpus_fingerprint") or ""),
            sequences=int(payload.get("sequences") or 0),
            tokens=int(payload.get("tokens") or 0),
        )

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "NgramTableV1":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_ngram_table(
    sequences: Iterable[Sequence[int]], *, order: int = 3
) -> NgramTableV1:
    """Count every context of length ``0..order-1`` over ``sequences``."""
    if order < 1:
        raise ValueError("n-gram order must be >= 1")
    counts: dict[PathKey, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_sequences = 0
    n_tokens = 0
    digest = hashlib.sha256()
    for sequence in sequences:
        ids = [int(token) for token in sequence]
        if not ids:
            continue
        n_sequences += 1
        n_tokens += len(ids)
        digest.update(json.dumps(ids, separators=(",", ":")).encode("utf-8"))
        for index, token in enumerate(ids):
            for back in range(order):
                if back > index:
                    break
                context = tuple(ids[index - back : index])
                counts[context][token] += 1
    return NgramTableV1(
        order=order,
        counts={context: dict(row) for context, row in counts.items()},
        corpus_fingerprint=digest.hexdigest(),
        sequences=n_sequences,
        tokens=n_tokens,
    )


@dataclass(frozen=True)
class SpeculativeChoiceV1:
    """One deterministic ranking decision over an already-legal domain."""

    scores: tuple[float, ...]
    order: tuple[int, ...]
    margin: float
    confident: bool

    @property
    def best_index(self) -> int:
        return self.order[0]


@dataclass(frozen=True)
class SpeculativeSpanV1:
    """A verified run of tokens committed without a neural forward."""

    token_ids: tuple[int, ...]
    forced_tokens: int
    ranked_tokens: int
    stop_reason: str

    @property
    def is_empty(self) -> bool:
        return not self.token_ids


@dataclass(frozen=True)
class SpeculativeRankerV1:
    """Deterministic scorer over forward-calculated completion domains.

    ``margin`` is the minimum log-probability gap between the best and the
    runner-up candidate required to commit *without* consulting the model. A
    margin of ``0.0`` (the default) means the ranker only ever supplies an
    ordering — the model still decides — which is the safe way to introduce a
    new table before a campaign certifies it.
    """

    table: NgramTableV1
    margin: float = 0.0
    impl_version: str = SPECULATIVE_RANK_IMPL_VERSION

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "impl_version": self.impl_version,
            "schema": self.table.schema,
            "order": self.table.order,
            "margin": self.margin,
            "corpus_fingerprint": self.table.corpus_fingerprint,
        }

    def score_paths(
        self, prefix: Sequence[int], paths: Sequence[CompletionPath]
    ) -> dict[PathKey, float]:
        """Score each legal path. Membership is untouched — this only orders."""
        scored: dict[PathKey, float] = {}
        for path in paths:
            key = tuple(int(token) for token in path.token_ids)
            scored[key] = self._score_span(prefix, key)
        return scored

    def _score_span(self, prefix: Sequence[int], span: Sequence[int]) -> float:
        context = list(int(token) for token in prefix)
        total = 0.0
        for token in span:
            total += self.table.log_prob(context, int(token))
            context.append(int(token))
        # Length-normalize so a long forced suffix does not out-vote a short
        # action purely by accumulating more (negative) mass.
        return total / max(1, len(tuple(span)))

    def choose(
        self, prefix: Sequence[int], paths: Sequence[CompletionPath]
    ) -> SpeculativeChoiceV1 | None:
        """Rank ``paths`` deterministically; ``None`` when there is nothing to rank."""
        if not paths:
            return None
        keys = [tuple(int(token) for token in path.token_ids) for path in paths]
        scores = [self._score_span(prefix, key) for key in keys]
        # Total order: score desc, then the path key itself, so the decision is
        # reproducible for a given (prefix, domain) regardless of input order.
        order = tuple(
            sorted(range(len(paths)), key=lambda index: (-scores[index], keys[index]))
        )
        if len(order) == 1:
            margin = math.inf
        else:
            margin = scores[order[0]] - scores[order[1]]
        confident = self.margin > 0.0 and margin >= self.margin
        return SpeculativeChoiceV1(
            scores=tuple(scores), order=order, margin=margin, confident=confident
        )


def speculative_span(
    prefix: Sequence[int],
    build_forest: Callable[[list[int]], CompletionForest],
    *,
    ranker: SpeculativeRankerV1 | None = None,
    max_tokens: int = 8,
) -> SpeculativeSpanV1:
    """Draft a verified continuation from the symbol table alone.

    At each step the completion domain is re-derived for the extended prefix
    (the *verify* half of lookahead-then-verify). A singleton domain is always
    committed; a wider domain is committed only when ``ranker`` is confident by
    its configured margin. Anything less — a partial proof, an empty domain, a
    close call — stops the span, and the caller falls back to the model.
    """
    drafted: list[int] = []
    forced_tokens = 0
    ranked_tokens = 0
    working = [int(token) for token in prefix]
    budget = max(0, int(max_tokens))
    while len(drafted) < budget:
        forest = build_forest(working)
        if forest.coverage != "complete":
            return SpeculativeSpanV1(
                tuple(drafted), forced_tokens, ranked_tokens, "incomplete_coverage"
            )
        paths = tuple(path for path in forest.paths if path.token_ids)
        if not paths:
            return SpeculativeSpanV1(
                tuple(drafted), forced_tokens, ranked_tokens, "empty_domain"
            )
        if len(paths) == 1:
            chosen = paths[0]
            source_forced = True
        else:
            if ranker is None:
                return SpeculativeSpanV1(
                    tuple(drafted), forced_tokens, ranked_tokens, "no_ranker"
                )
            choice = ranker.choose(working, paths)
            if choice is None or not choice.confident:
                return SpeculativeSpanV1(
                    tuple(drafted), forced_tokens, ranked_tokens, "low_margin"
                )
            chosen = paths[choice.best_index]
            source_forced = False
        span = [int(token) for token in chosen.token_ids]
        remaining = budget - len(drafted)
        if len(span) > remaining:
            return SpeculativeSpanV1(
                tuple(drafted), forced_tokens, ranked_tokens, "budget"
            )
        drafted.extend(span)
        working.extend(span)
        if source_forced:
            forced_tokens += len(span)
        else:
            ranked_tokens += len(span)
    return SpeculativeSpanV1(tuple(drafted), forced_tokens, ranked_tokens, "budget")


def load_ranker(
    table_path: Path | str | None, *, margin: float = 0.0
) -> SpeculativeRankerV1 | None:
    """Load a committed table, or ``None`` when the lever is unconfigured.

    A configured-but-unreadable table is an error: silently decoding without
    the ranker a run claims to use would make the result uncomparable.
    """
    if not table_path:
        return None
    table = NgramTableV1.load(table_path)
    if table.is_empty:
        raise ValueError(f"speculative rank table is empty: {table_path}")
    return SpeculativeRankerV1(table=table, margin=float(margin))


__all__ = [
    "NGRAM_TABLE_SCHEMA",
    "NgramTableV1",
    "PathKey",
    "SPECULATIVE_RANK_IMPL_VERSION",
    "SpeculativeChoiceV1",
    "SpeculativeRankerV1",
    "SpeculativeSpanV1",
    "build_ngram_table",
    "load_ranker",
    "speculative_span",
]
