"""Deterministic root-family split assignment and inheritance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

SCHEMA_VERSION = "root_family_split/v1"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class RootFamilySplitPolicyV1:
    modulus: int = 100
    validation_buckets: tuple[int, ...] = tuple(range(80, 90))
    test_buckets: tuple[int, ...] = tuple(range(90, 100))
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported split policy {self.schema_version!r}")
        if self.modulus < 3:
            raise ValueError("split modulus must be at least 3")
        reserved = (*self.validation_buckets, *self.test_buckets)
        if len(reserved) != len(set(reserved)):
            raise ValueError("validation and test buckets must be disjoint")
        if any(bucket < 0 or bucket >= self.modulus for bucket in reserved):
            raise ValueError("split bucket is outside the modulus")

    def assign(self, root_family_id: str) -> str:
        if not root_family_id:
            raise ValueError("root_family_id is required before expansion")
        bucket = (
            int.from_bytes(
                hashlib.sha256(root_family_id.encode("utf-8")).digest()[:8], "big"
            )
            % self.modulus
        )
        if bucket in self.validation_buckets:
            return "validation"
        if bucket in self.test_buckets:
            return "test"
        return "train"

    def require_inherited(
        self,
        *,
        root_family_id: str,
        split_group_id: str,
        split: str,
        parent_splits: tuple[str, ...] = (),
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if split_group_id != root_family_id:
            raise ValueError("descendant split_group_id must equal its root family")
        expected = self.assign(root_family_id)
        if split != expected:
            raise ValueError(
                f"root family {root_family_id!r} belongs to {expected}, not {split}"
            )
        if any(parent != split for parent in parent_splits):
            raise ValueError(
                f"composition has incompatible source splits: {parent_splits}"
            )
