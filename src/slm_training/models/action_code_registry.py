"""Versioned action-code registry for state-local action heads (CAP2-03).

Each registry entry describes the discrete code assigned to the legal actions of a
state family.  The registry is keyed by a stable state-family/action-schema hash
rather than runtime object identity, so codes can be shared across states with the
same aligned action schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ActionSchema:
    """Stable identity for a set of legal semantic actions.

    Attributes:
        state_family_id: human-readable label, e.g. "openui:card:arg0".
        action_identities: ordered tuple of stable action identifiers.  The order
            determines the code assignment and must be deterministic.
        version: schema version string.
    """

    state_family_id: str
    action_identities: tuple[str, ...]
    version: str = "v1"

    def schema_hash(self) -> str:
        payload = json.dumps(
            {
                "state_family_id": self.state_family_id,
                "action_identities": self.action_identities,
                "version": self.version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class CodeAssignment:
    """Codeword assignment for one action in a registry entry."""

    action_identity: str
    codeword: tuple[int, ...]
    alphabet_size: int


@dataclass(frozen=True)
class ActionCodeEntry:
    """A versioned code entry for one state-family/action-schema.

    Attributes:
        schema: the stable action schema.
        code_family: "flat", "base3", "ternary_ecoc_d2", etc.
        alphabet_radices: per-coordinate alphabet sizes.  For a uniform ternary
            code this is ``(3,) * m``.
        assignments: ordered code assignments, one per legal action.
        minimum_hamming_distance: verified minimum distance, or None for flat.
        unused_codewords: reserved but unassigned codewords.
        invalid_code_policy: fallback behavior for out-of-code words.
        cost_matrix_source: provenance of the semantic-cost matrix, if any.
        entry_hash: deterministic hash of the entry contents.
    """

    schema: ActionSchema
    code_family: str
    alphabet_radices: tuple[int, ...]
    assignments: tuple[CodeAssignment, ...]
    minimum_hamming_distance: int | None
    unused_codewords: tuple[tuple[int, ...], ...]
    invalid_code_policy: Literal["abstain", "refine", "detected_error", "unused_valid"] = "abstain"
    cost_matrix_source: str = ""
    entry_hash: str = ""

    def __post_init__(self) -> None:
        # entry_hash is computed on construction when empty.
        if not self.entry_hash:
            object.__setattr__(
                self,
                "entry_hash",
                self._compute_hash(),
            )

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "schema_hash": self.schema.schema_hash(),
                "code_family": self.code_family,
                "alphabet_radices": self.alphabet_radices,
                "assignments": [
                    {"action": a.action_identity, "codeword": a.codeword}
                    for a in self.assignments
                ],
                "minimum_hamming_distance": self.minimum_hamming_distance,
                "unused_codewords": self.unused_codewords,
                "invalid_code_policy": self.invalid_code_policy,
                "cost_matrix_source": self.cost_matrix_source,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

    @property
    def nominal_bits(self) -> float:
        import math

        return sum(math.log2(r) for r in self.alphabet_radices)

    @property
    def action_count(self) -> int:
        return len(self.assignments)

    def codeword_for(self, action_identity: str) -> tuple[int, ...] | None:
        for assignment in self.assignments:
            if assignment.action_identity == action_identity:
                return assignment.codeword
        return None

    def action_for_codeword(self, codeword: tuple[int, ...]) -> str | None:
        for assignment in self.assignments:
            if assignment.codeword == codeword:
                return assignment.action_identity
        return None


class ActionCodeRegistry:
    """In-memory registry of action-code entries keyed by schema hash."""

    def __init__(self) -> None:
        self._entries: dict[str, ActionCodeEntry] = {}

    def register(self, entry: ActionCodeEntry) -> None:
        key = entry.schema.schema_hash()
        if key in self._entries:
            raise ValueError(f"schema already registered: {entry.schema.state_family_id}")
        self._entries[key] = entry

    def get(self, schema: ActionSchema) -> ActionCodeEntry | None:
        return self._entries.get(schema.schema_hash())

    def get_or_create(
        self,
        schema: ActionSchema,
        *,
        code_family: str,
        builder: Any,
    ) -> ActionCodeEntry:
        """Return an existing entry or build and register one using ``builder``.

        The builder callable must accept an ActionSchema and return an
        ActionCodeEntry.
        """
        entry = self.get(schema)
        if entry is not None:
            return entry
        entry = builder(schema)
        self.register(entry)
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[ActionCodeEntry, ...]:
        return tuple(self._entries.values())
