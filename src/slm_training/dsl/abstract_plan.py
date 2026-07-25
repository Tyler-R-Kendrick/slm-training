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
    source: str = "reserved_codebook"
    provenance: str = "ap2_causal_pilot_warm_up"

    def __post_init__(self) -> None:
        if self.schema != "abstract_plan/v1":
            raise ValueError("unsupported AbstractPlanV1 schema")
        if self.plan_version != "1":
            raise ValueError("unsupported AbstractPlanV1 plan_version")
        if self.codebook_version < 1:
            raise ValueError("codebook_version must be >= 1")
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
        if self.source not in ABSTRACT_PLAN_SOURCES:
            raise ValueError(f"unknown AbstractPlanV1 source: {self.source!r}")
        if not self.provenance:
            raise ValueError("provenance must be non-empty")

    @property
    def is_interpretable(self) -> bool:
        """False in the base variant: no slot carries an assigned role."""
        return any(role is not None for role in self.role_metadata)

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
    mean/std) so growth never perturbs previously trained rows — required
    before any checkpoint migration that adds AbstractPlanV1 rows.
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
    return resized


def verify_embedding_resize_preserved_old_rows(
    original: "torch.nn.Embedding", resized: "torch.nn.Embedding"
) -> None:
    """Fail closed unless every pre-existing row is bit-for-bit unchanged."""
    import torch

    old_n = original.weight.shape[0]
    if resized.weight.shape[0] < old_n:
        raise ValueError("resized embedding has fewer rows than the original")
    if resized.weight.shape[1] != original.weight.shape[1]:
        raise ValueError("resized embedding changed embedding dimension")
    if not torch.equal(resized.weight[:old_n], original.weight):
        raise ValueError("embedding resize mutated pre-existing rows")


__all__ = [
    "ABSTRACT_PLAN_SOURCES",
    "AbstractPlanV1",
    "resize_embedding_preserving_rows",
    "verify_embedding_resize_preserved_old_rows",
]
