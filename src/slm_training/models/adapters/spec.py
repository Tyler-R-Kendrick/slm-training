"""Versioned spec for the removable TwoTower low-rank adapter (LDI2-01).

Torch-free so it round-trips through CLI / config / checkpoint metadata without
importing torch. The spec names *what* to adapt and the base identity it is bound to;
resolving the spec against a concrete model (and failing closed on a mismatch) is the
model's job, not the spec's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ADAPTER_SCHEMA_VERSION = 1

__all__ = ["ADAPTER_SCHEMA_VERSION", "TwoTowerAdapterSpec"]


@dataclass(frozen=True)
class TwoTowerAdapterSpec:
    """A removable low-rank delta adapter configuration bound to a base checkpoint.

    ``target_modules`` names the linear-projection categories to adapt (matched
    deterministically against the denoiser blocks); ``target_layer_indices`` optionally
    restricts to selected layers (``None`` means every layer). ``include_output_head``
    opts the denoiser output head in. The context tower is never adapted by this spec.
    """

    method: Literal["low_rank"]
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    base_compatibility_fingerprint: str
    base_checkpoint_sha: str
    tokenizer_sha: str
    target_layer_indices: tuple[int, ...] | None = None
    include_output_head: bool = False
    train_bias: Literal["none"] = "none"
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(
            self,
            "target_layer_indices",
            None
            if self.target_layer_indices is None
            else tuple(int(i) for i in self.target_layer_indices),
        )
        if self.method != "low_rank":
            raise ValueError("the first adapter backend supports only method='low_rank'")
        if self.rank <= 0:
            raise ValueError("adapter rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("adapter alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("adapter dropout must be in [0, 1)")
        if self.train_bias != "none":
            raise ValueError("the first adapter backend supports only train_bias='none'")
        if not self.target_modules:
            raise ValueError("adapter target_modules must not be empty")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("adapter target_modules must be unique")
        if self.target_layer_indices is not None and any(
            i < 0 for i in self.target_layer_indices
        ):
            raise ValueError("adapter target_layer_indices must be non-negative")
        for name in (
            "base_compatibility_fingerprint",
            "base_checkpoint_sha",
            "tokenizer_sha",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"adapter spec field {name!r} must be non-empty")

    @property
    def scaling(self) -> float:
        """The low-rank delta multiplier ``alpha / rank`` applied to ``B(A(x))``."""
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-JSON dict (lists, not tuples) round-trippable via ``from_dict``."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "target_layer_indices": (
                None
                if self.target_layer_indices is None
                else list(self.target_layer_indices)
            ),
            "include_output_head": self.include_output_head,
            "train_bias": self.train_bias,
            "base_compatibility_fingerprint": self.base_compatibility_fingerprint,
            "base_checkpoint_sha": self.base_checkpoint_sha,
            "tokenizer_sha": self.tokenizer_sha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TwoTowerAdapterSpec:
        """Rebuild a spec from ``to_dict`` output, failing closed on unknown fields.

        An unrecognized key raises ``ValueError`` rather than being silently dropped, so a
        newer or corrupted config never round-trips into a partially-applied adapter.
        """
        known = {
            "schema_version",
            "method",
            "rank",
            "alpha",
            "dropout",
            "target_modules",
            "target_layer_indices",
            "include_output_head",
            "train_bias",
            "base_compatibility_fingerprint",
            "base_checkpoint_sha",
            "tokenizer_sha",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown adapter spec fields: {sorted(unknown)}")
        layers = data.get("target_layer_indices")
        return cls(
            method=data["method"],
            rank=data["rank"],
            alpha=data["alpha"],
            dropout=data["dropout"],
            target_modules=tuple(data["target_modules"]),
            base_compatibility_fingerprint=data["base_compatibility_fingerprint"],
            base_checkpoint_sha=data["base_checkpoint_sha"],
            tokenizer_sha=data["tokenizer_sha"],
            target_layer_indices=None if layers is None else tuple(layers),
            include_output_head=bool(data.get("include_output_head", False)),
            train_bias=data.get("train_bias", "none"),
            schema_version=int(data.get("schema_version", ADAPTER_SCHEMA_VERSION)),
        )
