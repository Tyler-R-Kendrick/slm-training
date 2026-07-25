"""DSH1-08 (SLM-360): Harness-DSL QA views with capped identity and verified
suffix-completion accepted sets.

Builds on the existing, already-canonical machinery instead of inventing new
parsing or scoring paths:

* IDENTITY / CANONICALIZE rows are the existing :mod:`scope_corpus` families
  (:func:`~slm_training.harnesses.train_data.scope_corpus.build_scope_corpus`);
  this module only adds a hard, declared identity cap plus explicit
  ``diagnostic`` marking (:func:`cap_identity_records`).
* COMPLETE_SUFFIX is net new: :func:`build_accepted_completion_set` splits a
  canonical multi-statement document at a legal statement boundary (both
  halves independently re-validate through
  :func:`slm_training.dsl.parser.validate` — no tokenizer or decode-time
  ``CompletionForest`` is needed for this document-level split) and verifies
  every accepted completion, reusing the byte-for-byte
  :func:`~slm_training.harnesses.train_data.scope_corpus.decanonicalize_variants`
  round-trip check already trusted by the canonical-pair family for surface
  equivalents, plus any caller-supplied ``peer_suffixes`` observed elsewhere
  in the corpus for the same prefix. The full accepted set is carried in
  :class:`~slm_training.dsl.schema.ExampleRecord.accepted_outputs` — the
  schema field the existing scorer (``score_output_targets``,
  ``evals/task_scoreboard.py``) already scores by membership — so no new
  mass/weighting mechanism is introduced. Anti-identity hard negatives
  (:func:`same_symbols_different_structure_negative`,
  :func:`matched_structure_permuted_markers_negative`) reuse the existing
  verified corruption oracle
  (:func:`slm_training.data.corrupt.oracle.build_scoped_corruptions`) and the
  existing runtime-marker extractor
  (:func:`slm_training.dsl.harness_dsl.runtime_symbols_for_payload`).

No decode-path integration, training run, eval, benchmark, or checkpoint
claim is made by this module.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from slm_training.data.corrupt.oracle import build_scoped_corruptions
from slm_training.dsl.harness_dsl import (
    HarnessOperation,
    HarnessPayloadKind,
    HarnessTaskV1,
    runtime_symbols_for_payload,
    serialize_harness_task,
)
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, OutputKind, OutputTarget
from slm_training.harnesses.train_data.scope_corpus import decanonicalize_variants

_SCOPED_CORRUPTION_KINDS = frozenset({"statement", "expression", "lexical"})


def _record_id(*parts: str) -> str:
    raw = ":".join(parts)
    return f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}_harness_qa"


# --------------------------------------------------------------------------- #
# COMPLETE_SUFFIX accepted-completion sets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AcceptedCompletionSetV1:
    """All verified-accepted suffix completions for one legal document prefix.

    ``canonical`` is the single designated canonical completion (always a
    member of ``accepted``); every other member of ``accepted`` is a
    verified-but-non-canonical alternative — a genuinely different admissible
    completion (e.g. a caller-supplied peer suffix) or a verified surface
    variant of the canonical suffix. Membership, not a single arbitrary gold
    label, is what scoring reads: see :func:`accepted_completion_outputs`.
    """

    prefix: str
    canonical: str
    accepted: tuple[str, ...]
    full_canonical: str

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("prefix must be non-empty")
        if not self.canonical:
            raise ValueError("canonical completion must be non-empty")
        if self.canonical not in self.accepted:
            raise ValueError("canonical completion must be a member of accepted")
        if len(set(self.accepted)) != len(self.accepted):
            raise ValueError("accepted completions must be unique")
        if f"{self.prefix}\n{self.canonical}" != self.full_canonical:
            raise ValueError(
                "prefix + canonical completion must equal full_canonical"
            )


def statement_prefix_boundaries(canonical: str) -> tuple[int, ...]:
    """Legal interior statement-line cut points for a canonical document.

    Each canonical OpenUI document serializes one top-level statement per
    line (the same convention :func:`~scope_corpus._rotate_statements`
    relies on); every ``1 <= cut < n`` line count is a candidate boundary,
    verified independently by :func:`build_accepted_completion_set`.
    """
    lines = canonical.splitlines()
    return tuple(range(1, len(lines)))


def build_accepted_completion_set(
    canonical: str,
    *,
    cut: int,
    peer_suffixes: Sequence[str] = (),
) -> AcceptedCompletionSetV1:
    """Split ``canonical`` at statement-line ``cut`` and verify the accepted set.

    The canonical suffix is always accepted. Deterministic non-canonical
    surface rewrites of that suffix
    (:func:`~scope_corpus.decanonicalize_variants`, reused unmodified from the
    existing canonical-pair machinery) are added only when they independently
    parse. ``peer_suffixes`` — completions observed for this exact prefix
    elsewhere in the corpus — are added only when ``prefix + "\\n" +
    peer_suffix`` independently validates; a peer suffix that does not
    legally continue this prefix is silently dropped, never accepted
    unverified.
    """
    lines = canonical.splitlines()
    if not (0 < cut < len(lines)):
        raise ValueError(f"cut={cut} is not a legal interior statement boundary")
    prefix = "\n".join(lines[:cut])
    canonical_suffix = "\n".join(lines[cut:])
    try:
        validate(prefix)
        validate(canonical_suffix)
    except ParseError as exc:
        raise ValueError(
            f"statement cut {cut} is not independently valid on both sides: {exc}"
        ) from exc

    accepted: list[str] = [canonical_suffix]
    for _name, variant in decanonicalize_variants(canonical_suffix):
        if variant in accepted:
            continue
        try:
            validate(variant)
        except ParseError:
            continue
        accepted.append(variant)
    for peer in peer_suffixes:
        if not peer or peer in accepted:
            continue
        try:
            validate(f"{prefix}\n{peer}")
        except ParseError:
            continue
        accepted.append(peer)

    return AcceptedCompletionSetV1(
        prefix=prefix,
        canonical=canonical_suffix,
        accepted=tuple(accepted),
        full_canonical=canonical,
    )


def accepted_completion_outputs(
    completion_set: AcceptedCompletionSetV1,
    *,
    kind: OutputKind = "document",
    category: str | None = None,
) -> tuple[OutputTarget, ...]:
    """The non-canonical accepted members as ``OutputTarget``\\ s.

    The canonical member belongs in ``ExampleRecord.openui``; this is exactly
    the rest of :attr:`AcceptedCompletionSetV1.accepted`, so a record built
    with ``openui=completion_set.canonical`` and
    ``accepted_outputs=accepted_completion_outputs(completion_set, ...)``
    carries the *full* accepted set through the existing
    ``ExampleRecord.output_targets`` / ``score_output_targets`` scoring path
    with no new mechanism.
    """
    return tuple(
        OutputTarget(text=member, kind=kind, category=category)
        for member in completion_set.accepted
        if member != completion_set.canonical
    )


def build_complete_suffix_task(
    completion_set: AcceptedCompletionSetV1,
    *,
    pack_id: str = "openui",
    payload_kind: HarnessPayloadKind = HarnessPayloadKind.DOCUMENT,
    grammar_category: str | None = None,
) -> HarnessTaskV1:
    """The Harness-DSL ``COMPLETE_SUFFIX`` task: payload is the legal prefix."""
    return HarnessTaskV1(
        operation=HarnessOperation.COMPLETE_SUFFIX,
        pack_id=pack_id,
        payload_kind=payload_kind,
        grammar_category=grammar_category,
        payload=completion_set.prefix,
        runtime_symbols=runtime_symbols_for_payload(completion_set.prefix),
    )


def complete_suffix_record(
    completion_set: AcceptedCompletionSetV1,
    *,
    root_id: str,
    split: str = "train",
    pack_id: str = "openui",
    payload_kind: HarnessPayloadKind = HarnessPayloadKind.DOCUMENT,
    grammar_category: str | None = None,
) -> ExampleRecord:
    """One COMPLETE_SUFFIX training example for the given accepted set.

    ``openui`` is the canonical completion; every other accepted member is
    recorded in ``accepted_outputs`` (never dropped, never treated as a
    penalized wrong answer).
    """
    task = build_complete_suffix_task(
        completion_set,
        pack_id=pack_id,
        payload_kind=payload_kind,
        grammar_category=grammar_category,
    )
    prompt = serialize_harness_task(task)
    output_kind: OutputKind = "document" if payload_kind == HarnessPayloadKind.DOCUMENT else payload_kind.value  # type: ignore[assignment]
    return ExampleRecord(
        id=_record_id(root_id, "complete_suffix", completion_set.prefix),
        prompt=prompt,
        openui=completion_set.canonical,
        split=split,
        source="harness_complete_suffix",
        target_kind=output_kind,
        target_category=grammar_category,
        accepted_outputs=list(
            accepted_completion_outputs(
                completion_set, kind=output_kind, category=grammar_category
            )
        ),
        meta={
            "task": "completion",
            "source_kind": "deterministic",
            "determinacy": "deterministic",
            "parent_id": root_id,
            "harness_dsl": {
                "schema": task.schema,
                "operation": task.operation.value,
                "pack_id": task.pack_id,
                "payload_kind": task.payload_kind.value,
                "grammar_category": task.grammar_category,
            },
            "complete_suffix": {
                "accepted_count": len(completion_set.accepted),
                "full_canonical_sha256": hashlib.sha256(
                    completion_set.full_canonical.encode("utf-8")
                ).hexdigest(),
            },
        },
    )


def build_complete_suffix_records(
    canonical: str,
    *,
    root_id: str,
    split: str = "train",
    max_cuts: int | None = None,
    peer_suffixes_by_cut: dict[int, Sequence[str]] | None = None,
) -> list[ExampleRecord]:
    """All (or the first ``max_cuts``) COMPLETE_SUFFIX records for one root."""
    cuts = statement_prefix_boundaries(canonical)
    if max_cuts is not None:
        cuts = cuts[:max_cuts]
    peer_by_cut = peer_suffixes_by_cut or {}
    records: list[ExampleRecord] = []
    for cut in cuts:
        completion_set = build_accepted_completion_set(
            canonical, cut=cut, peer_suffixes=peer_by_cut.get(cut, ())
        )
        records.append(
            complete_suffix_record(completion_set, root_id=root_id, split=split)
        )
    return records


# --------------------------------------------------------------------------- #
# Identity cap + diagnostic marking
# --------------------------------------------------------------------------- #


def _harness_meta(record: ExampleRecord) -> dict[str, Any] | None:
    meta = record.meta.get("harness_dsl")
    return meta if isinstance(meta, dict) else None


def cap_identity_records(
    records: Sequence[ExampleRecord], *, cap: int
) -> tuple[ExampleRecord, ...]:
    """Enforce a hard identity-row cap; mark every kept identity row diagnostic.

    Non-Harness and non-IDENTITY rows pass through unchanged. IDENTITY rows
    are grouped by ``(payload_kind, grammar_category)`` family, in input
    order; rows beyond ``cap`` within a family are dropped (never silently
    retained past the declared cap), and every kept IDENTITY row gets
    ``meta["diagnostic"] = True`` so downstream training/scoring code can
    treat it as a capped diagnostic anchor rather than ordinary generative
    supervision.
    """
    if cap < 0:
        raise ValueError("identity cap must be non-negative")
    kept: list[ExampleRecord] = []
    seen_per_family: dict[tuple[str, str | None], int] = {}
    for record in records:
        harness_meta = _harness_meta(record)
        if harness_meta is None or harness_meta.get("operation") != HarnessOperation.IDENTITY.value:
            kept.append(record)
            continue
        family = (harness_meta["payload_kind"], harness_meta.get("grammar_category"))
        count = seen_per_family.get(family, 0)
        if count >= cap:
            continue
        seen_per_family[family] = count + 1
        meta = dict(record.meta)
        meta["diagnostic"] = True
        kept.append(replace(record, meta=meta))
    return tuple(kept)


def identity_row_counts(
    records: Sequence[ExampleRecord],
) -> dict[tuple[str, str | None], int]:
    """IDENTITY row counts per ``(payload_kind, grammar_category)`` family."""
    counts: dict[tuple[str, str | None], int] = {}
    for record in records:
        harness_meta = _harness_meta(record)
        if harness_meta is None or harness_meta.get("operation") != HarnessOperation.IDENTITY.value:
            continue
        family = (harness_meta["payload_kind"], harness_meta.get("grammar_category"))
        counts[family] = counts.get(family, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Atomic-form reuse coverage
# --------------------------------------------------------------------------- #


def atomic_form_coverage_violations(records: Sequence[ExampleRecord]) -> tuple[str, ...]:
    """Lexical atomic forms used ONLY as an identity anchor, never reused.

    Identity anchors are every record with ``meta["task"] == "identity"``
    (the general marker every identity producer in this repo sets, harness-
    wrapped or not — see ``scope_corpus._record``). For each such record at
    lexical granularity (``meta["scope_kind"] == "lexical"``), this checks
    whether its exact surface text is reused anywhere — as a substring of a
    prompt or target — in a non-identity record from the same ``records``.
    An empty tuple means every observed lexical atomic form is exercised in
    at least one non-identity task too: the DSH1-08 acceptance criterion
    "reuse every atomic production in a nonidentity context".
    """
    identity_lexical_forms = {
        record.openui
        for record in records
        if record.meta.get("task") == "identity"
        and record.meta.get("scope_kind") == "lexical"
    }
    if not identity_lexical_forms:
        return ()
    haystack = "\n".join(
        f"{record.prompt}\n{record.openui}"
        for record in records
        if record.meta.get("task") != "identity"
    )
    return tuple(
        sorted(form for form in identity_lexical_forms if form not in haystack)
    )


# --------------------------------------------------------------------------- #
# Anti-identity hard negatives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AntiIdentityNegative:
    """One preference decoy contrasted against an exact IDENTITY echo."""

    prompt: str
    chosen: str
    rejected: str
    variant: str

    def __post_init__(self) -> None:
        if self.chosen == self.rejected:
            raise ValueError("an anti-identity negative must differ from chosen")


def _require_identity(record: ExampleRecord) -> dict[str, Any]:
    harness_meta = _harness_meta(record)
    if harness_meta is None or harness_meta.get("operation") != HarnessOperation.IDENTITY.value:
        raise ValueError("expected an IDENTITY harness record")
    return harness_meta


def same_symbols_different_structure_negative(
    identity_record: ExampleRecord,
) -> AntiIdentityNegative | None:
    """A same-vocabulary, differently-structured decoy for one IDENTITY row.

    Reuses the existing verified corruption oracle
    (:func:`build_scoped_corruptions`) instead of inventing new mutation
    logic; returns ``None`` when the row's scope has no corruption operators
    (``document``/``typed_node``) or none apply.
    """
    harness_meta = _require_identity(identity_record)
    kind = harness_meta["payload_kind"]
    if kind not in _SCOPED_CORRUPTION_KINDS:
        return None
    cases = build_scoped_corruptions(
        identity_record.openui,
        kind,  # type: ignore[arg-type]
        category=harness_meta.get("grammar_category"),
        limit=1,
    )
    if not cases:
        return None
    return AntiIdentityNegative(
        prompt=identity_record.prompt,
        chosen=identity_record.openui,
        rejected=cases[0].broken_text,
        variant="same_symbols_different_structure",
    )


def _permute_marker_swap(text: str) -> dict[str, str]:
    by_role: dict[str, list[str]] = {}
    for symbol in runtime_symbols_for_payload(text):
        by_role.setdefault(symbol.role, []).append(symbol.surface)
    swap: dict[str, str] = {}
    for surfaces in by_role.values():
        names = sorted(set(surfaces))
        if len(names) < 2:
            continue
        rotated = names[1:] + names[:1]
        swap.update(dict(zip(names, rotated)))
    return swap


def matched_structure_permuted_markers_negative(
    identity_record: ExampleRecord,
) -> AntiIdentityNegative | None:
    """A same-shape decoy with its runtime markers cyclically permuted.

    Structure is byte-identical apart from marker identity: every occurrence
    of each declared runtime marker (:func:`runtime_symbols_for_payload`) is
    whole-word-swapped for another declared marker of the same role, in a
    single simultaneous pass so no marker chains into another's target.
    Returns ``None`` when fewer than two same-role markers exist to permute.
    """
    _require_identity(identity_record)
    text = identity_record.openui
    swap = _permute_marker_swap(text)
    if not swap:
        return None
    pattern = re.compile(
        "|".join(re.escape(name) for name in sorted(swap, key=len, reverse=True))
    )
    decoy = pattern.sub(lambda match: swap[match.group()], text)
    if decoy == text:
        return None
    return AntiIdentityNegative(
        prompt=identity_record.prompt,
        chosen=text,
        rejected=decoy,
        variant="matched_structure_permuted_markers",
    )


__all__ = [
    "AcceptedCompletionSetV1",
    "AntiIdentityNegative",
    "accepted_completion_outputs",
    "atomic_form_coverage_violations",
    "build_accepted_completion_set",
    "build_complete_suffix_records",
    "build_complete_suffix_task",
    "cap_identity_records",
    "complete_suffix_record",
    "identity_row_counts",
    "matched_structure_permuted_markers_negative",
    "same_symbols_different_structure_negative",
    "statement_prefix_boundaries",
]
