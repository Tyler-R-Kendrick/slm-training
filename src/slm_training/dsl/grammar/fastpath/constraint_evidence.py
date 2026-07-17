"""Reason-coded constraint evidence for the compiler completion forest (VSS0-02).

This module instruments :func:`build_completion_forest` so a later certificate
builder can tell **which hard-constraint stage admitted or rejected each
considered semantic action**. See
[verified-scope-solver.md](../../../../../docs/design/verified-scope-solver.md)
("Constraint evidence") and
[lattice-recursive-search.md](../../../../../docs/design/lattice-recursive-search.md).

Honesty boundary (do not drift): evidence records **prefix legality**, i.e. which
stage of the deterministic bounded enumerator excluded a candidate as
not-admissible-*now*. It is **not** a support-participation verdict. Only a forest
whose ``coverage == "complete"`` licenses treating its considered set as
exhaustive, and even then legality is not support — asserting ``SUPPORTED`` still
requires a verifier-accepted witness (see the support contract). Evidence for a
``partial``/``none`` forest never authorizes candidate removal and must not be
serialized as an exhaustive proof.

The recorder is only instantiated when ``build_completion_forest`` is called with
``explain=True``; the default path allocates nothing here and keeps byte-for-byte
candidate-set parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ConstraintStage(str, Enum):
    """The hard-constraint stage that admitted or rejected a considered action.

    Stages mirror the deterministic narrowing order of
    :func:`build_completion_forest`. ``str`` mixin keeps values JSON-stable.
    """

    GRAMMAR = "grammar"
    SCHEMA = "schema"
    BINDING = "binding"
    SLOT_CONTRACT = "slot_contract"
    DATAFLOW = "dataflow"
    LITERAL_FRAME = "literal_frame"
    MIN_CONTENT = "min_content"
    TERMINAL = "terminal"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class ConstraintEvidence:
    """One immutable admit/reject decision for a considered candidate action.

    ``candidate_id`` is the first token id of the considered action (``None`` only
    for a whole-forest verdict such as an unparseable prefix or the terminal
    coverage record). ``path_token_ids`` is the admitted action's forced suffix
    (empty for rejections). ``details`` carries bounded, safe metadata only —
    never decoded prompt/user literals.
    """

    candidate_id: int | None
    path_token_ids: tuple[int, ...]
    stage: ConstraintStage
    admitted: bool
    reason_code: str
    details: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Deterministic, JSON-safe projection (round-trips via ``from_dict``)."""
        return {
            "candidate_id": self.candidate_id,
            "path_token_ids": list(self.path_token_ids),
            "stage": self.stage.value,
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "details": [list(pair) for pair in self.details],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ConstraintEvidence:
        candidate = data["candidate_id"]
        return cls(
            candidate_id=None if candidate is None else int(candidate),  # type: ignore[arg-type]
            path_token_ids=tuple(int(token) for token in data["path_token_ids"]),  # type: ignore[union-attr]
            stage=ConstraintStage(str(data["stage"])),
            admitted=bool(data["admitted"]),
            reason_code=str(data["reason_code"]),
            details=tuple(
                (str(pair[0]), str(pair[1])) for pair in data.get("details", ())  # type: ignore[index,union-attr]
            ),
        )


def stage_counts(
    evidence: Iterable[ConstraintEvidence],
) -> tuple[tuple[str, int, int], ...]:
    """Aggregate ``(stage, admitted_count, rejected_count)`` in stage order.

    Deterministic and independent of model logits: rows follow the declaration
    order of :class:`ConstraintStage`, and only stages with at least one record
    appear.
    """
    admitted: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for record in evidence:
        bucket = admitted if record.admitted else rejected
        bucket[record.stage.value] = bucket.get(record.stage.value, 0) + 1
    rows: list[tuple[str, int, int]] = []
    for stage in ConstraintStage:
        key = stage.value
        passed = admitted.get(key, 0)
        failed = rejected.get(key, 0)
        if passed or failed:
            rows.append((key, passed, failed))
    return tuple(rows)


class ConstraintEvidenceRecorder:
    """Collects per-stage admit/reject evidence during forest enumeration.

    The recorder tracks a live set of *considered* candidate ids and attributes
    each removal to the stage during which it dropped out. It never scans the
    full vocabulary: it is seeded from the grammar-admitted candidate set and
    only ever narrows. Added candidates (e.g. a forced next binder) are absorbed
    into the live set without a fabricated rejection.
    """

    __slots__ = ("_live", "_records", "_frozen")

    def __init__(self) -> None:
        self._live: set[int] = set()
        self._records: list[ConstraintEvidence] = []
        self._frozen = False

    # -- seeding / narrowing ------------------------------------------------
    def seed(self, candidate_ids: Iterable[int]) -> None:
        """Record the grammar-admitted considered set (no per-candidate rows)."""
        self._live = {int(token) for token in candidate_ids}

    def narrow(
        self,
        stage: ConstraintStage,
        reason_code: str,
        surviving: Iterable[int],
        *,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Attribute every candidate removed by this stage to ``reason_code``.

        Candidates newly present in ``surviving`` are absorbed silently. A no-op
        once :meth:`freeze` has been called (used when a schema enum replaces the
        considered set and later ``candidates`` mutations no longer affect the
        emitted paths).
        """
        if self._frozen:
            return
        survivors = {int(token) for token in surviving}
        for candidate in sorted(self._live - survivors):
            self._records.append(
                ConstraintEvidence(candidate, (), stage, False, reason_code, details)
            )
        self._live = survivors

    def freeze(self) -> None:
        """Stop attributing further set narrowing (schema-enum takeover)."""
        self._frozen = True

    def exclude_specials(self, special_ids: Iterable[int]) -> None:
        """Drop non-semantic special tokens from the live set without a row."""
        self._live -= {int(token) for token in special_ids}

    def note_unparseable_prefix(
        self, reason_code: str = "prefix_not_parseable"
    ) -> None:
        """Record a whole-forest grammar rejection (engine could not parse prefix)."""
        self._records.append(
            ConstraintEvidence(None, (), ConstraintStage.GRAMMAR, False, reason_code)
        )

    # -- terminal / EOS decisions ------------------------------------------
    def note_eos(
        self,
        eos_id: int,
        admitted: bool,
        stage: ConstraintStage,
        reason_code: str,
        *,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Record whether EOS was admitted, keeping the reason distinguishable.

        A withheld EOS records its own reject row (min-content withholding is a
        distinct :class:`ConstraintStage.MIN_CONTENT` reason, never conflated with
        grammar rejection) and leaves the live set without EOS. Admission is left
        to the path loop, which emits the ``eos`` path.
        """
        eos = int(eos_id)
        if admitted:
            self._live.add(eos)
        else:
            self._records.append(
                ConstraintEvidence(eos, (), stage, False, reason_code, details)
            )
            self._live.discard(eos)

    def admit_path(
        self, candidate_id: int, path_token_ids: Iterable[int], kind: str
    ) -> None:
        """Record a candidate that reached a compiler-valid completion path."""
        candidate = int(candidate_id)
        self._records.append(
            ConstraintEvidence(
                candidate,
                tuple(int(token) for token in path_token_ids),
                ConstraintStage.TERMINAL,
                True,
                "admitted",
                (("kind", str(kind)),),
            )
        )
        self._live.discard(candidate)

    def reject_unreachable(
        self, candidate_id: int, reason_code: str = "not_grammar_reachable"
    ) -> None:
        """Record a candidate that passed set filters but failed CFG reachability."""
        candidate = int(candidate_id)
        self._records.append(
            ConstraintEvidence(
                candidate, (), ConstraintStage.TERMINAL, False, reason_code
            )
        )
        self._live.discard(candidate)

    # -- finalization -------------------------------------------------------
    def finalize(self, coverage: str) -> tuple[ConstraintEvidence, ...]:
        """Append the coverage verdict and return the immutable evidence tuple.

        The coverage record makes exhaustiveness explicit: a consumer must see
        ``coverage == "complete"`` (admitted) before treating the rejections as an
        exhaustive account of the considered set. ``partial``/``none`` forests are
        recorded as not-admitted and must not be read as a support proof.
        """
        self._records.append(
            ConstraintEvidence(
                None,
                (),
                ConstraintStage.COVERAGE,
                coverage == "complete",
                f"coverage_{coverage}",
                (("coverage", str(coverage)),),
            )
        )
        return tuple(self._records)

    @property
    def records(self) -> tuple[ConstraintEvidence, ...]:
        return tuple(self._records)


__all__ = [
    "ConstraintStage",
    "ConstraintEvidence",
    "ConstraintEvidenceRecorder",
    "stage_counts",
]
