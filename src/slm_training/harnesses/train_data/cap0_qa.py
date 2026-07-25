"""CAP0 QA views: identity, canonicalization, and accepted-set suffix completion.

SLM-360 (DSH1-08). QA rows are generated from verified answer artifacts
**through the closed Harness DSL** (DSH1-03
:class:`~slm_training.dsl.harness_dsl.HarnessTaskV1` serialization/validation)
in three kinds:

* ``IDENTITY`` — copy the payload verbatim. Capped at the declared
  ``identity_cap`` fraction of the dataset and marked ``diagnostic``: identity
  rows measure the copy shortcut, they never dominate training. Every atomic
  production used by an answer must also appear in at least one nonidentity
  context (coverage guard, tested).
* ``CANONICALIZE`` — expanded surface form → canonical form pairs from the
  DSH1-05 proven equivalence transforms
  (:mod:`~slm_training.harnesses.train_data.equivalence_transforms`), with the
  ``CANONICALLY_PREFERRED_TO`` relation recorded per pair.
* ``COMPLETE_SUFFIX`` — statement-boundary prefix → **accepted set**. For
  ambiguous prefixes every verified accepted completion is persisted (each
  replayed to a verified terminal through the DSH1-07 classifier), plus a
  separate canonical preference. No arbitrary single-gold label is ever
  attached where several completions are accepted
  (:func:`audit_no_single_gold_where_multi` rejects such a dataset).

Anti-identity hard negatives accompany the views in both required shapes:

* ``same_symbols_different_structure`` — an identical symbol multiset with a
  different verified structure (different canonical fingerprint);
* ``matched_structure_permuted_markers`` — the identical production structure
  with marker surfaces permuted under the marker-table authority discipline
  (deterministic derangement, parser-verified, different canonical
  fingerprint).

Scoring guidance is uniform: rows carry ``scoring = accepted_set_membership``
— membership/mass over the accepted set, never a penalty against a valid
alternative derivation.

Stop rule: :func:`identity_baseline_matches` flags a suite that an
identity-only baseline would fully match — CAP0 must not train on such a
suite.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    production_id,
)
from slm_training.dsl.harness_dsl import (
    HarnessOperation,
    HarnessPayloadKind,
    HarnessTaskV1,
    runtime_symbols_for_payload,
    serialize_harness_task,
)
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.harnesses.train_data.equivalence_transforms import (
    CANONICALLY_PREFERRED_TO,
    canonical_cost,
    materialize_equivalence_class,
)
from slm_training.harnesses.train_data.partial_states import (
    PartialStateClassV1,
    classify_partial_state,
)

SCHEMA_VERSION = "cap0_qa/v1"
GENERATOR_VERSION = "v1"
SCORING_ACCEPTED_SET = "accepted_set_membership"

RowKind = Literal["identity", "canonicalize", "complete_suffix"]
NegativeKind = Literal[
    "same_symbols_different_structure", "matched_structure_permuted_markers"
]

_TOKEN_RE = re.compile(r"[A-Za-z_$:@][A-Za-z0-9_.$:@]*|\S")
_LIST_RE = re.compile(r"\[([^\[\]]*,[^\[\]]*)\]")


class Cap0QAAuditError(ValueError):
    """A CAP0 QA dataset violated a labeling or coverage guard."""


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cap0AcceptedCompletionV1:
    """One verified accepted completion of a prefix (replays to a terminal)."""

    source: str  # full verified terminal document
    suffix: str  # completion text after the prefix cut
    canonical_fingerprint: str
    terminal_state_class: str  # classify_partial_state verdict on the source
    terminal_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "suffix": self.suffix,
            "canonical_fingerprint": self.canonical_fingerprint,
            "terminal_state_class": self.terminal_state_class,
            "terminal_verified": self.terminal_verified,
        }


@dataclass(frozen=True)
class Cap0QARowV1:
    """One Harness-DSL QA row."""

    row_id: str
    kind: RowKind
    task: HarnessTaskV1
    prompt: str  # canonical Harness serialization (validated)
    completion: str | None = None  # single gold: identity/canonicalize only
    accepted_completions: tuple[Cap0AcceptedCompletionV1, ...] = ()
    canonical_preference: str | None = None  # separate from the accepted set
    relations: tuple[str, ...] = ()
    diagnostic: bool = False
    scoring: str = SCORING_ACCEPTED_SET
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "completion": self.completion,
            "accepted_completions": [
                item.to_dict() for item in self.accepted_completions
            ],
            "canonical_preference": self.canonical_preference,
            "relations": list(self.relations),
            "diagnostic": self.diagnostic,
            "scoring": self.scoring,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class Cap0HardNegativeV1:
    """Anti-identity hard negative: near-miss form that must not score."""

    negative_id: str
    negative_kind: NegativeKind
    source_answer: str
    negative_form: str
    source_fingerprint: str
    negative_fingerprint: str
    verified: bool  # parses + fingerprint differs (+ structure/symbol proof)

    def to_dict(self) -> dict[str, Any]:
        return {
            "negative_id": self.negative_id,
            "negative_kind": self.negative_kind,
            "source_answer": self.source_answer,
            "negative_form": self.negative_form,
            "source_fingerprint": self.source_fingerprint,
            "negative_fingerprint": self.negative_fingerprint,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class Cap0QADatasetV1:
    """The three CAP0 QA views plus hard negatives and coverage evidence."""

    rows: tuple[Cap0QARowV1, ...]
    hard_negatives: tuple[Cap0HardNegativeV1, ...]
    identity_cap: float
    seed: int
    uncovered_productions: tuple[str, ...]
    stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "identity_cap": self.identity_cap,
            "seed": self.seed,
            "uncovered_productions": list(self.uncovered_productions),
            "stats": dict(self.stats),
            "rows": [row.to_dict() for row in self.rows],
            "hard_negatives": [item.to_dict() for item in self.hard_negatives],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _row_id(kind: str, *parts: str) -> str:
    raw = ":".join((kind, *parts))
    return f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}_cap0"


def _task(
    operation: HarnessOperation, payload: str
) -> tuple[HarnessTaskV1, str]:
    task = HarnessTaskV1(
        operation=operation,
        pack_id="openui",
        payload_kind=HarnessPayloadKind.DOCUMENT,
        grammar_category=None,
        payload=payload,
        runtime_symbols=tuple(
            sorted(
                runtime_symbols_for_payload(payload),
                key=lambda symbol: (symbol.role, symbol.surface),
            )
        ),
    )
    return task, serialize_harness_task(task)


def _verified_terminal(source: str, *, adapter: GrammarCapabilityAdapterV1) -> bool:
    """DSH1-07 replay: the source classifies COMPLETE (verified terminal)."""
    report = classify_partial_state(source, adapter=adapter)
    return report.state_class is PartialStateClassV1.COMPLETE


def _accepted_completions(
    prefix: str,
    *,
    candidates: Iterable[str],
    adapter: GrammarCapabilityAdapterV1,
    dsl: str | None,
) -> tuple[Cap0AcceptedCompletionV1, ...]:
    """Every candidate extending the prefix that replays to a verified terminal."""
    accepted: list[Cap0AcceptedCompletionV1] = []
    for candidate in sorted({item.strip("\n") for item in candidates}):
        if not candidate.startswith(prefix) or candidate == prefix:
            continue
        report = classify_partial_state(candidate, adapter=adapter)
        if report.state_class is not PartialStateClassV1.COMPLETE:
            continue
        accepted.append(
            Cap0AcceptedCompletionV1(
                source=candidate,
                suffix=candidate[len(prefix) :],
                canonical_fingerprint=canonical_fingerprint(candidate, dsl=dsl),
                terminal_state_class=report.state_class.value,
                terminal_verified=True,
            )
        )
    return tuple(accepted)


def _canonical_preference(
    accepted: tuple[Cap0AcceptedCompletionV1, ...], *, dsl: str | None
) -> str | None:
    """Soft canonical preference over the accepted set (never a single gold)."""
    if not accepted:
        return None
    ranked = sorted(
        accepted,
        key=lambda item: (canonical_cost(item.source), item.source),
    )
    return ranked[0].source


def _statement_prefixes(source: str) -> tuple[str, ...]:
    """Proper statement-boundary prefixes (never a mid-token cut)."""
    lines = [line for line in source.splitlines() if line.strip()]
    return tuple(
        "\n".join(lines[: count]) for count in range(1, len(lines))
    )


def _production_ids(
    adapter: GrammarCapabilityAdapterV1, source: str
) -> frozenset[str]:
    """Atomic production ids used by one parseable source."""
    starts = adapter.start_symbols
    primary = str(starts[0]) if starts else "start"
    trace = adapter.authority.production_trace
    if trace is None:
        return frozenset()
    try:
        occurrences = trace(primary, source if source.endswith("\n") else source + "\n")
    except Exception:  # noqa: BLE001 - unparseable source: no productions
        return frozenset()
    return frozenset(production_id(item.production) for item in occurrences)


# --------------------------------------------------------------------------- #
# Hard negatives
# --------------------------------------------------------------------------- #


def _same_symbols_negative(
    answer: str, *, adapter: GrammarCapabilityAdapterV1, dsl: str | None
) -> str | None:
    """Swap two container children: identical symbols, different structure."""
    match = _LIST_RE.search(answer)
    if match is None:
        return None
    items = [item.strip() for item in match.group(1).split(",")]
    if len(items) < 2:
        return None
    swapped = ", ".join([items[1], items[0], *items[2:]])
    negative = answer[: match.start()] + f"[{swapped}]" + answer[match.end() :]
    if negative == answer:
        return None
    if sorted(_TOKEN_RE.findall(negative)) != sorted(_TOKEN_RE.findall(answer)):
        return None
    try:
        adapter.static_validate(negative)
    except Exception:  # noqa: BLE001 - unverified negatives are dropped
        return None
    if canonical_fingerprint(negative, dsl=dsl) == canonical_fingerprint(
        answer, dsl=dsl
    ):
        return None
    return negative


def _permuted_markers_negative(
    answer: str,
    *,
    adapter: GrammarCapabilityAdapterV1,
    dsl: str | None,
    seed: int,
) -> str | None:
    """Derange marker surfaces (marker-table authority discipline), structure fixed."""
    markers = sorted(set(extract_placeholders(answer)))
    if len(markers) < 2:
        return None
    rng = random.Random(seed)
    while True:
        permuted = list(markers)
        rng.shuffle(permuted)
        if all(a != b for a, b in zip(markers, permuted)):
            break
    mapping = dict(zip(markers, permuted))
    pattern = re.compile("|".join(re.escape(item) for item in markers))
    negative = pattern.sub(lambda match: mapping[match.group(0)], answer)
    if negative == answer:
        return None
    try:
        adapter.static_validate(negative)
    except Exception:  # noqa: BLE001 - unverified negatives are dropped
        return None
    if canonical_fingerprint(negative, dsl=dsl) == canonical_fingerprint(
        answer, dsl=dsl
    ):
        return None
    # Matched structure: identical production multiset under the permutation.
    if _production_ids(adapter, negative) != _production_ids(adapter, answer):
        return None
    return negative


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def generate_cap0_qa(
    answers: Iterable[str],
    *,
    adapter: GrammarCapabilityAdapterV1,
    identity_cap: float = 0.1,
    seed: int = 0,
    dsl: str | None = None,
) -> Cap0QADatasetV1:
    """Generate the three CAP0 QA views plus anti-identity hard negatives.

    Identity rows are capped at ``identity_cap`` of the total row count and
    marked diagnostic; every answer's atomic productions must also appear in
    at least one nonidentity context (recorded as ``uncovered_productions``).
    """
    if not 0 < identity_cap < 1:
        raise ValueError(f"identity_cap must be in (0, 1), got {identity_cap!r}")
    staged = sorted({str(answer).strip("\n") for answer in answers if str(answer).strip()})
    if not staged:
        raise ValueError("generate_cap0_qa requires at least one answer artifact")

    witnesses = [
        str(item.source).strip("\n")
        for item in (adapter.authority.witness_candidates or (lambda: ()))()
    ]
    completion_domain = (*staged, *witnesses)

    nonidentity: list[Cap0QARowV1] = []
    identity_candidates: list[Cap0QARowV1] = []
    hard_negatives: list[Cap0HardNegativeV1] = []
    context_sources: list[str] = []

    for answer in staged:
        report = classify_partial_state(answer, adapter=adapter)
        if report.state_class is not PartialStateClassV1.COMPLETE:
            raise Cap0QAAuditError(
                f"answer artifact is not a verified terminal: {answer!r} "
                f"({report.state_class.value}:{report.reason})"
            )
        equivalence = materialize_equivalence_class(answer, dsl=dsl)
        canonical = equivalence.canonical_form

        identity_task, identity_prompt = _task(HarnessOperation.IDENTITY, canonical)
        identity_candidates.append(
            Cap0QARowV1(
                row_id=_row_id("identity", canonical),
                kind="identity",
                task=identity_task,
                prompt=identity_prompt,
                completion=canonical,
                diagnostic=True,
                provenance={"answer": answer, "cap": identity_cap},
            )
        )

        for expansion in equivalence.expansions:
            task, prompt = _task(HarnessOperation.CANONICALIZE, expansion.form)
            nonidentity.append(
                Cap0QARowV1(
                    row_id=_row_id("canonicalize", expansion.form, canonical),
                    kind="canonicalize",
                    task=task,
                    prompt=prompt,
                    completion=canonical,
                    relations=(CANONICALLY_PREFERRED_TO,),
                    provenance={
                        "answer": answer,
                        "transform_id": expansion.transform_id,
                        "transform_version": expansion.transform_version,
                        "margin": expansion.margin,
                        "equivalence_class": equivalence.canonical_fingerprint,
                    },
                )
            )
            context_sources.extend([expansion.form, canonical])

        for prefix in _statement_prefixes(answer):
            try:
                task, prompt = _task(HarnessOperation.COMPLETE_SUFFIX, prefix)
            except Exception:  # noqa: BLE001 - not a Harness-serializable prefix
                continue
            accepted = _accepted_completions(
                prefix, candidates=completion_domain, adapter=adapter, dsl=dsl
            )
            if not accepted:
                continue
            prefix_report = classify_partial_state(prefix, adapter=adapter)
            nonidentity.append(
                Cap0QARowV1(
                    row_id=_row_id(
                        "complete_suffix", prefix, *(item.source for item in accepted)
                    ),
                    kind="complete_suffix",
                    task=task,
                    prompt=prompt,
                    completion=None,  # accepted set, never an arbitrary single gold
                    accepted_completions=accepted,
                    canonical_preference=_canonical_preference(accepted, dsl=dsl),
                    provenance={
                        "answer": answer,
                        "prefix_state_class": prefix_report.state_class.value,
                        "prefix_certificate": (
                            prefix_report.certificate.to_dict()
                            if prefix_report.certificate is not None
                            else None
                        ),
                        "ambiguous": len(accepted) > 1,
                    },
                )
            )
            context_sources.append(prefix)
            context_sources.extend(item.source for item in accepted)

        same_symbols = _same_symbols_negative(answer, adapter=adapter, dsl=dsl)
        if same_symbols is not None:
            hard_negatives.append(
                Cap0HardNegativeV1(
                    negative_id=_row_id("same_symbols", answer, same_symbols),
                    negative_kind="same_symbols_different_structure",
                    source_answer=answer,
                    negative_form=same_symbols,
                    source_fingerprint=canonical_fingerprint(answer, dsl=dsl),
                    negative_fingerprint=canonical_fingerprint(same_symbols, dsl=dsl),
                    verified=True,
                )
            )
        permuted = _permuted_markers_negative(
            answer, adapter=adapter, dsl=dsl, seed=seed
        )
        if permuted is not None:
            hard_negatives.append(
                Cap0HardNegativeV1(
                    negative_id=_row_id("permuted_markers", answer, permuted),
                    negative_kind="matched_structure_permuted_markers",
                    source_answer=answer,
                    negative_form=permuted,
                    source_fingerprint=canonical_fingerprint(answer, dsl=dsl),
                    negative_fingerprint=canonical_fingerprint(permuted, dsl=dsl),
                    verified=True,
                )
            )

    # Identity cap: keep at most floor(cap / (1 - cap) * n_nonidentity) rows so
    # identity_rows / total_rows <= cap exactly.
    max_identity = int(identity_cap / (1 - identity_cap) * len(nonidentity))
    rng = random.Random(seed)
    selected = rng.sample(identity_candidates, k=min(max_identity, len(identity_candidates)))
    identity_rows = sorted(selected, key=lambda row: row.row_id)

    rows = tuple(sorted((*nonidentity, *identity_rows), key=lambda row: row.row_id))

    # Atomic coverage: every production used by a staged answer must appear in
    # at least one nonidentity context source.
    answer_productions: set[str] = set()
    for answer in staged:
        answer_productions.update(_production_ids(adapter, answer))
    context_productions: set[str] = set()
    for source in sorted(set(context_sources)):
        context_productions.update(_production_ids(adapter, source))
    uncovered = tuple(sorted(answer_productions - context_productions))

    identity_count = len(identity_rows)
    total = len(rows)
    stats = {
        "answers": len(staged),
        "rows_total": total,
        "rows_by_kind": {
            kind: sum(1 for row in rows if row.kind == kind)
            for kind in ("identity", "canonicalize", "complete_suffix")
        },
        "identity_fraction": (identity_count / total) if total else 0.0,
        "identity_cap": identity_cap,
        "ambiguous_prefixes": sum(
            1 for row in rows if len(row.accepted_completions) > 1
        ),
        "hard_negatives": len(hard_negatives),
        "hard_negatives_by_kind": {
            kind: sum(1 for item in hard_negatives if item.negative_kind == kind)
            for kind in (
                "same_symbols_different_structure",
                "matched_structure_permuted_markers",
            )
        },
        "scoring": SCORING_ACCEPTED_SET,
    }
    return Cap0QADatasetV1(
        rows=rows,
        hard_negatives=tuple(hard_negatives),
        identity_cap=identity_cap,
        seed=seed,
        uncovered_productions=uncovered,
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Guards and stop rule
# --------------------------------------------------------------------------- #


def identity_rows_within_cap(dataset: Cap0QADatasetV1) -> bool:
    """Guard: identity rows obey the declared cap and stay diagnostic."""
    if not dataset.rows:
        return False
    identity = [row for row in dataset.rows if row.kind == "identity"]
    if any(not row.diagnostic for row in identity):
        return False
    return len(identity) / len(dataset.rows) <= dataset.identity_cap


def require_atomic_coverage(dataset: Cap0QADatasetV1) -> None:
    """Guard: fail closed when an atomic production lacks nonidentity context."""
    if dataset.uncovered_productions:
        raise Cap0QAAuditError(
            "atomic productions without nonidentity context: "
            + ", ".join(dataset.uncovered_productions)
        )


def audit_no_single_gold_where_multi(dataset: Cap0QADatasetV1) -> None:
    """Reject any multi-completion prefix carrying an arbitrary single-gold label."""
    findings = [
        row.row_id
        for row in dataset.rows
        if len(row.accepted_completions) > 1 and row.completion is not None
    ]
    if findings:
        raise Cap0QAAuditError(
            "single-gold label on ambiguous accepted-set rows: "
            + ", ".join(sorted(findings))
        )


def require_verified_terminals(dataset: Cap0QADatasetV1) -> None:
    """Guard: every accepted completion must replay to a verified terminal."""
    findings = [
        (row.row_id, item.source)
        for row in dataset.rows
        for item in row.accepted_completions
        if not item.terminal_verified
        or item.terminal_state_class != PartialStateClassV1.COMPLETE.value
    ]
    if findings:
        raise Cap0QAAuditError(
            "accepted completions that do not replay to verified terminals: "
            + ", ".join(f"{row}:{source!r}" for row, source in findings)
        )


def identity_baseline_matches(dataset: Cap0QADatasetV1) -> dict[str, Any]:
    """Stop rule: would an identity-only baseline match the staged suite?

    Scores the copy shortcut on every non-diagnostic row: the payload verbatim
    must be the row's gold or a member of its accepted set. When the baseline
    matches the whole suite the QA views carry no signal beyond copying and
    CAP0 must not train.
    """
    scored = [row for row in dataset.rows if not row.diagnostic]

    def copied(row: Cap0QARowV1) -> bool:
        payload = row.task.payload
        if row.completion is not None and row.completion == payload:
            return True
        return any(item.source == payload for item in row.accepted_completions)

    hits = sum(1 for row in scored if copied(row))
    score = (hits / len(scored)) if scored else 0.0
    matches = bool(scored) and hits == len(scored)
    return {
        "identity_baseline_score": score,
        "non_diagnostic_rows": len(scored),
        "identity_solved_rows": hits,
        "matches": matches,
        "do_not_train": matches,
    }


__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "SCORING_ACCEPTED_SET",
    "Cap0AcceptedCompletionV1",
    "Cap0HardNegativeV1",
    "Cap0QAAuditError",
    "Cap0QADatasetV1",
    "Cap0QARowV1",
    "audit_no_single_gold_where_multi",
    "generate_cap0_qa",
    "identity_baseline_matches",
    "identity_rows_within_cap",
    "require_atomic_coverage",
    "require_verified_terminals",
]
