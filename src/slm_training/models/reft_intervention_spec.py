"""Versioned spec + arm-spec builder for LDI4-01 representation interventions (SLM-134).

Torch-free so it round-trips through CLI / config / checkpoint metadata and lets a
campaign harness enumerate the matched R0-R4 arm set without importing torch. The
spec names *what* to intervene on (method / site / position / rank) and the base
identity it is bound to; instantiating a concrete intervention module against a
model (and failing closed on a shape mismatch) is the torch module's job, not the
spec's. Import :class:`LowRankReft` / :func:`build_intervention` from
:mod:`slm_training.models.reft_intervention` where torch is available.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

InterventionMethod = Literal["no_intervention", "reft_r1", "reft_low_rank", "diffmean_fixed"]
_METHODS: frozenset[str] = frozenset(
    {"no_intervention", "reft_r1", "reft_low_rank", "diffmean_fixed"}
)

__all__ = ["InterventionMethod", "InterventionSpec", "matched_arm_specs"]


@dataclass(frozen=True)
class InterventionSpec:
    """Versioned, fail-closed intervention config. Its fingerprint is part of the
    artifact identity together with the base/module-shape fingerprints."""

    method: InterventionMethod
    site: str  # named hook / layer, e.g. "denoiser.block.3.residual"
    hidden_size: int
    rank: int = 1
    scale: float = 1.0
    position: Literal["exact_decision", "context_pooled"] = "exact_decision"
    base_checkpoint_sha: str = ""
    module_shape_sha: str = ""
    objective_fingerprint: str = ""
    corpus_fingerprint: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"unknown intervention method {self.method!r}")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.method == "reft_r1" and self.rank != 1:
            raise ValueError("reft_r1 requires rank == 1")
        if self.method in ("reft_r1", "reft_low_rank") and not 1 <= self.rank <= self.hidden_size:
            raise ValueError("rank must be in [1, hidden_size]")
        if self.scale < 0:
            raise ValueError("scale must be non-negative")

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


def matched_arm_specs(
    *,
    site: str,
    hidden_size: int,
    base_checkpoint_sha: str,
    ranks: tuple[int, ...] = (1, 4, 8),
    scale: float = 1.0,
) -> dict[str, InterventionSpec]:
    """The matched R0-R5 arm set (parent / weight-adapter placeholder / DiffMean /
    ReFT rank sweep). Only the declared actuator/rank differs per arm; the matched
    trainable comparison itself is a downstream GPU run."""
    common = dict(
        site=site, hidden_size=hidden_size, base_checkpoint_sha=base_checkpoint_sha, scale=scale
    )
    arms = {
        "R0_parent": InterventionSpec(method="no_intervention", **common),
        "R2_diffmean": InterventionSpec(method="diffmean_fixed", **common),
        "R3_reft_r1": InterventionSpec(method="reft_r1", rank=1, **common),
    }
    for r in ranks:
        if r > 1:
            arms[f"R4_reft_r{r}"] = InterventionSpec(method="reft_low_rank", rank=r, **common)
    return arms
