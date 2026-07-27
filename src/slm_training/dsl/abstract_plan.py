"""AbstractPlanV1: a collision-free, default-off discrete latent-plan codebook (AP-016).

Reserves ``<beginabstract>``/``<endabstract>`` delimiters plus a fixed pool of
abstract codebook tokens inside the pre-allocated ``abstract_plan`` logical
token-id namespace (:mod:`slm_training.dsl.openui_tokens`). This module is
serialization/config only: no tokenizer emits these ids until a caller
explicitly builds with ``abstract_plan_slots > 0``
(:mod:`slm_training.models.dsl_tokenizer`, :mod:`slm_training.models.choice_tokenizer`),
and slot identity is intentionally non-interpretable in the base variant —
``role_metadata`` stays empty unless a later, separately versioned experiment
assigns roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from slm_training.dsl.openui_tokens import (
    ABSTRACT_PLAN_BEGIN,
    ABSTRACT_PLAN_BEGIN_LOCAL_ID,
    ABSTRACT_PLAN_CODEBOOK_SIZE,
    ABSTRACT_PLAN_CODEBOOK_VERSION,
    ABSTRACT_PLAN_END,
    ABSTRACT_PLAN_END_LOCAL_ID,
    ABSTRACT_PLAN_SLOT_LOCAL_OFFSET,
    DEFAULT_ABSTRACT_PLAN_ROUNDS,
    MAX_ABSTRACT_PLAN_SLOTS,
    TOKEN_ID_NAMESPACE_RANGES,
    abstract_plan_logical_id,
    abstract_plan_slot_token,
)
from slm_training.dsl.operators.contracts import _fingerprint

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch

_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Closed set of allowed provenance sources; extend only with a new versioned
# schema (never repurpose an existing source's meaning).
ABSTRACT_PLAN_SOURCES = frozenset({"reserved_codebook"})


@dataclass(frozen=True)
class RoleSpanV1:
    """One named, contiguous, non-overlapping codebook range (AP-025 / SLM-318).

    Partitions a slice of an :class:`AbstractPlanV1`'s ``slot_count`` codebook
    into a role family (e.g. ``intent``, ``inventory``, ``cardinality``,
    ``topology``, ``bindings``, ``style`` -- the families ``semantic_plan_factors``
    already uses for the richer ``SemanticPlanV1``). This only labels which
    codebook indices belong to which role; it reserves no new vocabulary token
    and changes no token id, so it never touches ``codebook_version``.
    """

    role: str
    start: int
    length: int
    # Optional per-role budget on how many *occupied* rounds of this role may
    # remain non-truncated; must be <= length when set. None means "no cap
    # beyond length itself".
    max_length: int | None = None

    def __post_init__(self) -> None:
        if not _ROLE_RE.fullmatch(self.role):
            raise ValueError(f"invalid RoleSpanV1 role: {self.role!r}")
        if self.start < 0:
            raise ValueError("RoleSpanV1.start must be >= 0")
        if self.length < 1:
            raise ValueError("RoleSpanV1.length must be >= 1")
        if self.max_length is not None and not 1 <= self.max_length <= self.length:
            raise ValueError(
                "RoleSpanV1.max_length must be within [1, length] when set, "
                f"got {self.max_length} for length {self.length}"
            )

    @property
    def end(self) -> int:
        """Exclusive end of this role's codebook range."""
        return self.start + self.length

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "start": self.start,
            "length": self.length,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoleSpanV1":
        max_length = value.get("max_length")
        return cls(
            role=str(value["role"]),
            start=int(value["start"]),
            length=int(value["length"]),
            max_length=int(max_length) if max_length is not None else None,
        )


@dataclass(frozen=True)
class AbstractPlanV1:
    """Versioned, collision-free discrete latent-plan codebook contract."""

    schema: str = "abstract_plan/v1"
    plan_version: str = "1"
    codebook_version: int = ABSTRACT_PLAN_CODEBOOK_VERSION
    slot_count: int = ABSTRACT_PLAN_CODEBOOK_SIZE  # M
    max_slot_count: int = MAX_ABSTRACT_PLAN_SLOTS  # m_max
    rounds: int = DEFAULT_ABSTRACT_PLAN_ROUNDS  # T
    begin_token: str = ABSTRACT_PLAN_BEGIN
    end_token: str = ABSTRACT_PLAN_END
    # Per-slot optional role label. Empty by default: the base variant is
    # intentionally non-interpretable. Must be empty or exactly slot_count long.
    role_metadata: tuple[str | None, ...] = ()
    # Ordered, non-overlapping named codebook ranges (AP-025 / SLM-318): an
    # alternate, span-based role representation. Empty by default -- the base
    # variant is homogeneous. Mutually exclusive with role_metadata (pick one
    # role representation per plan) so a slot never carries two conflicting
    # role identities.
    role_spans: tuple[RoleSpanV1, ...] = ()
    source: str = "reserved_codebook"
    provenance: str = "ap2_causal_pilot_warm_up"

    def __post_init__(self) -> None:
        if self.schema != "abstract_plan/v1":
            raise ValueError("unsupported AbstractPlanV1 schema")
        if self.plan_version != "1":
            raise ValueError("unsupported AbstractPlanV1 plan_version")
        if self.codebook_version != ABSTRACT_PLAN_CODEBOOK_VERSION:
            # token_ids always uses the V1 local-id mapping below; accepting
            # any other codebook_version would let a serialized plan claim a
            # future codebook while its ids/fingerprint stay V1.
            raise ValueError(
                "codebook_version must equal the supported V1 codebook "
                f"version {ABSTRACT_PLAN_CODEBOOK_VERSION}, got "
                f"{self.codebook_version}"
            )
        if self.begin_token != ABSTRACT_PLAN_BEGIN:
            raise ValueError(
                f"begin_token must be the canonical {ABSTRACT_PLAN_BEGIN!r} "
                "delimiter (this V1 contract cannot represent another mapping)"
            )
        if self.end_token != ABSTRACT_PLAN_END:
            raise ValueError(
                f"end_token must be the canonical {ABSTRACT_PLAN_END!r} "
                "delimiter (this V1 contract cannot represent another mapping)"
            )
        if self.max_slot_count < 1:
            raise ValueError("max_slot_count must be >= 1")
        if not 1 <= self.slot_count <= self.max_slot_count:
            raise ValueError(
                f"slot_count {self.slot_count} must be within "
                f"[1, {self.max_slot_count}]"
            )
        capacity = (
            len(TOKEN_ID_NAMESPACE_RANGES["abstract_plan"])
            - ABSTRACT_PLAN_SLOT_LOCAL_OFFSET
        )
        if self.max_slot_count > capacity:
            raise ValueError(
                f"max_slot_count {self.max_slot_count} exceeds the reserved "
                f"abstract_plan namespace capacity ({capacity})"
            )
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.role_metadata and len(self.role_metadata) != self.slot_count:
            raise ValueError(
                "role_metadata must be empty or match slot_count exactly"
            )
        for role in self.role_metadata:
            if role is not None and not _ROLE_RE.fullmatch(role):
                raise ValueError(f"invalid role_metadata entry: {role!r}")
        if self.role_spans:
            if self.role_metadata:
                raise ValueError(
                    "role_spans and role_metadata are alternate role "
                    "representations; set at most one"
                )
            seen_roles: set[str] = set()
            prev_end = 0
            for span in self.role_spans:
                if span.start < prev_end:
                    raise ValueError(
                        "role_spans must be ordered by ascending start with no "
                        f"overlap: role {span.role!r} starts at {span.start}, "
                        f"before the previous span's end {prev_end}"
                    )
                if span.role in seen_roles:
                    raise ValueError(f"duplicate RoleSpanV1 role: {span.role!r}")
                if span.end > self.slot_count:
                    raise ValueError(
                        f"role_spans role {span.role!r} range [{span.start}, "
                        f"{span.end}) exceeds slot_count {self.slot_count}"
                    )
                seen_roles.add(span.role)
                prev_end = span.end
        if self.source not in ABSTRACT_PLAN_SOURCES:
            raise ValueError(f"unknown AbstractPlanV1 source: {self.source!r}")
        if not self.provenance:
            raise ValueError("provenance must be non-empty")

    @property
    def is_interpretable(self) -> bool:
        """False in the base variant: no slot carries an assigned role."""
        return any(role is not None for role in self.role_metadata) or bool(
            self.role_spans
        )

    @property
    def is_role_factorized(self) -> bool:
        """True iff this plan uses the span-based (not per-slot) role representation."""
        return bool(self.role_spans)

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(span.role for span in self.role_spans)

    def role_span(self, role: str) -> RoleSpanV1:
        for span in self.role_spans:
            if span.role == role:
                return span
        raise KeyError(f"unknown role: {role!r}")

    def role_for_slot(self, slot_index: int) -> str | None:
        """The role owning ``slot_index``, or None if unassigned/homogeneous."""
        for span in self.role_spans:
            if span.start <= slot_index < span.end:
                return span.role
        return None

    def role_slot_indices(self, role: str) -> tuple[int, ...]:
        span = self.role_span(role)
        return tuple(range(span.start, span.end))

    def role_slot_token_ids(self, role: str) -> tuple[int, ...]:
        """Absolute reserved vocabulary ids of ``role``'s codebook range."""
        all_ids = self.slot_token_ids
        return tuple(all_ids[i] for i in self.role_slot_indices(role))

    def role_slot_mask(self, role: str) -> tuple[bool, ...]:
        """Boolean mask over ``range(slot_count)``, True where the slot is ``role``'s."""
        span = self.role_span(role)
        return tuple(span.start <= i < span.end for i in range(self.slot_count))

    @property
    def slot_tokens(self) -> tuple[str, ...]:
        return tuple(abstract_plan_slot_token(i) for i in range(self.slot_count))

    @property
    def delimiter_token_ids(self) -> tuple[int, int]:
        return (
            abstract_plan_logical_id(ABSTRACT_PLAN_BEGIN_LOCAL_ID),
            abstract_plan_logical_id(ABSTRACT_PLAN_END_LOCAL_ID),
        )

    @property
    def slot_token_ids(self) -> tuple[int, ...]:
        return tuple(
            abstract_plan_logical_id(ABSTRACT_PLAN_SLOT_LOCAL_OFFSET + i)
            for i in range(self.slot_count)
        )

    @property
    def token_ids(self) -> tuple[int, ...]:
        begin_id, end_id = self.delimiter_token_ids
        return (begin_id, *self.slot_token_ids, end_id)

    @property
    def compatibility_fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_version": self.plan_version,
            "codebook_version": self.codebook_version,
            "slot_count": self.slot_count,
            "max_slot_count": self.max_slot_count,
            "rounds": self.rounds,
            "begin_token": self.begin_token,
            "end_token": self.end_token,
            "role_metadata": list(self.role_metadata),
            "role_spans": [span.to_dict() for span in self.role_spans],
            "source": self.source,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AbstractPlanV1":
        return cls(
            schema=str(value.get("schema", "abstract_plan/v1")),
            plan_version=str(value.get("plan_version", "1")),
            codebook_version=int(
                value.get("codebook_version", ABSTRACT_PLAN_CODEBOOK_VERSION)
            ),
            slot_count=int(value.get("slot_count", ABSTRACT_PLAN_CODEBOOK_SIZE)),
            max_slot_count=int(
                value.get("max_slot_count", MAX_ABSTRACT_PLAN_SLOTS)
            ),
            rounds=int(value.get("rounds", DEFAULT_ABSTRACT_PLAN_ROUNDS)),
            begin_token=str(value.get("begin_token", ABSTRACT_PLAN_BEGIN)),
            end_token=str(value.get("end_token", ABSTRACT_PLAN_END)),
            role_metadata=tuple(value.get("role_metadata") or ()),
            role_spans=tuple(
                RoleSpanV1.from_dict(span) for span in value.get("role_spans") or ()
            ),
            source=str(value.get("source", "reserved_codebook")),
            provenance=str(
                value.get("provenance", "ap2_causal_pilot_warm_up")
            ),
        )

    def assert_no_collisions(self, existing_tokens: Iterable[str]) -> None:
        """Fail closed if any reserved token text already exists in a vocabulary."""
        existing = set(existing_tokens)
        reserved = {self.begin_token, self.end_token, *self.slot_tokens}
        collisions = reserved & existing
        if collisions:
            raise ValueError(
                "AbstractPlanV1 reserved tokens collide with existing "
                f"vocabulary: {sorted(collisions)}"
            )


def resize_embedding_preserving_rows(
    embedding: "torch.nn.Embedding",
    new_num_embeddings: int,
    *,
    generator: "torch.Generator | None" = None,
) -> "torch.nn.Embedding":
    """Grow an embedding table, bit-exactly preserving every existing row.

    New rows are initialized independently (matched to the existing table's
    mean/std) so growth never perturbs previously trained rows. ``padding_idx``,
    ``max_norm``, ``norm_type``, ``scale_grad_by_freq``, ``sparse``, and
    ``requires_grad`` are carried over from ``embedding`` so forward/gradient
    behavior on pre-existing rows is unchanged, not just their stored values —
    required before any checkpoint migration that adds AbstractPlanV1 rows.
    """
    import torch
    from torch import nn

    old_num_embeddings, dim = embedding.weight.shape
    if new_num_embeddings < old_num_embeddings:
        raise ValueError(
            f"cannot shrink embedding from {old_num_embeddings} to "
            f"{new_num_embeddings} rows"
        )
    if new_num_embeddings == old_num_embeddings:
        return embedding
    resized = nn.Embedding(
        new_num_embeddings,
        dim,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    with torch.no_grad():
        old_weight = embedding.weight.detach()
        std = float(old_weight.std().item()) if old_num_embeddings > 1 else 0.0
        mean = float(old_weight.mean().item()) if old_num_embeddings > 0 else 0.0
        resized.weight[:old_num_embeddings].copy_(old_weight)
        resized.weight[old_num_embeddings:].normal_(
            mean=mean, std=std or 0.02, generator=generator
        )
    resized.weight.requires_grad_(embedding.weight.requires_grad)
    return resized


def verify_embedding_resize_preserved_old_rows(
    original: "torch.nn.Embedding", resized: "torch.nn.Embedding"
) -> None:
    """Fail closed unless every pre-existing row and config is unchanged."""
    import torch

    old_n = original.weight.shape[0]
    if resized.weight.shape[0] < old_n:
        raise ValueError("resized embedding has fewer rows than the original")
    if resized.weight.shape[1] != original.weight.shape[1]:
        raise ValueError("resized embedding changed embedding dimension")
    if not torch.equal(resized.weight[:old_n], original.weight):
        raise ValueError("embedding resize mutated pre-existing rows")
    for attr in (
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ):
        if getattr(resized, attr) != getattr(original, attr):
            raise ValueError(f"embedding resize changed {attr}")
    if resized.weight.requires_grad != original.weight.requires_grad:
        raise ValueError("embedding resize changed requires_grad")


__all__ = [
    "ABSTRACT_PLAN_SOURCES",
    "AbstractPlanV1",
    "RoleSpanV1",
    "resize_embedding_preserving_rows",
    "verify_embedding_resize_preserved_old_rows",
]
