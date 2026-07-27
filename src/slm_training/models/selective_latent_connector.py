"""SelectiveLatentConnector: a default-off, gated-additive continuous-factor
bias that conditions MaskGIT/tree denoising on *soft* program factors while
leaving precision-critical factors fully discrete (AP-033 / SLM-333,
milestone "Selective Hybrid Objective").

This is the continuous-latent sibling of
:mod:`slm_training.models.abstract_plan_connector` (AP-024 / SLM-316): same
idiom (a single learnable scalar gate initialized at ``0.0`` times a linear
projection, added to logits/targets as a post-hoc bias, never a change to the
transformer stack itself), same discipline (every named arm resolves through
one function so control arms cannot silently diverge), same default-off
guarantee ("off" is "the connector is not constructed at all", not merely "a
disabled flag").

It differs from :class:`AbstractPlanConnector` in exactly one place: the
input is already a continuous vector (a *soft-factor* latent -- style/layout/
global-intent signal), not a sequence of codebook-local slot ids that needs
its own embedding table. There is nothing to pool, so
:meth:`SelectiveLatentConnector.bias_for_vocab` projects its input directly,
with the identical signature (``vector``, optional ``candidate_ids``) as
:meth:`AbstractPlanConnector.bias_for_vocab` so the two are structurally
interchangeable at any call site that only relies on that method plus the
``gate`` attribute -- in particular
:meth:`slm_training.models.blocks.DenoiserTower._apply_plan_connector_bias`,
which never checks ``isinstance`` and therefore accepts either connector type
via :meth:`DenoiserTower.set_plan_connector` unmodified. See
``tests/test_models/test_selective_latent_connector.py`` for a direct proof
of that drop-in composability, and
``docs/design/selective-hybrid-latent-objective.md``'s "Scope decision" for
why this repo does not also duplicate AP-024's much larger production wiring
inside ``twotower.py``/``blocks.py`` itself.

Factor routing (the issue's "Define factor routing between continuous soft
factors and discrete precision-critical factors" requirement): the six
:data:`~slm_training.models.semantic_plan_factors.PROGRAM_FACTOR_FAMILIES`
are partitioned into :data:`PRECISION_CRITICAL_FACTOR_FAMILIES` (topology,
cardinality, binder_reference, and inventory -- exact structural/executable
state the issue explicitly says "remain discrete") and
:data:`SOFT_FACTOR_FAMILIES` (style_layout, property_role_value -- the
closest existing proxy for "style/layout/global intent"; see the module
docstring caveat below for the honesty note on ``property_role_value``).
Family order in ``PROGRAM_FACTOR_FAMILIES`` already places the four
precision-critical families first and the two soft families last, so the
split is a single contiguous slice boundary, not a gather -- see
:func:`split_soft_precision_features`.

Honest caveat on the routing split: ``property_role_value`` (per-role
required flags plus mean factor confidence) is not literally "global intent"
-- ``SemanticPlanV1`` has no first-class intent field (the same caveat
``semantic_plan_factors.py`` already documents for ``style_layout``'s
archetype-string bridge). It is grouped as "soft" here because, unlike
topology/cardinality/binder_reference, corrupting it does not change what
components exist or how they are bound -- it changes confidence/requiredness
shading, the closest available proxy for a soft, non-executable-semantics
signal in the existing schema. This is a judgment call, documented rather
than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from slm_training.models.semantic_plan_factors import PROGRAM_FACTOR_FAMILIES

__all__ = [
    "PRECISION_CRITICAL_FACTOR_FAMILIES",
    "SOFT_FACTOR_FAMILIES",
    "SelectiveLatentArm",
    "SelectiveLatentConnector",
    "SelectiveLatentTrace",
    "resolve_selective_latent_vector",
    "split_soft_precision_features",
]

# Exact topology, cardinality, and binder/reference state -- the issue's own
# "remain discrete" list -- plus inventory (which component families exist),
# since that is equally executable-semantics-determining, not stylistic.
PRECISION_CRITICAL_FACTOR_FAMILIES: tuple[str, ...] = (
    "inventory",
    "cardinality",
    "topology",
    "binder_reference",
)

# Style/layout/global-intent proxy factors -- see the module docstring's
# honest caveat on why property_role_value is grouped here.
SOFT_FACTOR_FAMILIES: tuple[str, ...] = (
    "property_role_value",
    "style_layout",
)

if set(PRECISION_CRITICAL_FACTOR_FAMILIES) | set(SOFT_FACTOR_FAMILIES) != set(PROGRAM_FACTOR_FAMILIES):
    raise RuntimeError(
        "PRECISION_CRITICAL_FACTOR_FAMILIES + SOFT_FACTOR_FAMILIES must exactly "
        "partition PROGRAM_FACTOR_FAMILIES"
    )
if tuple(PRECISION_CRITICAL_FACTOR_FAMILIES) != PROGRAM_FACTOR_FAMILIES[: len(PRECISION_CRITICAL_FACTOR_FAMILIES)]:
    raise RuntimeError(
        "PRECISION_CRITICAL_FACTOR_FAMILIES must be the leading contiguous slice "
        "of PROGRAM_FACTOR_FAMILIES so the soft/precision split is a single slice "
        "boundary, not a gather"
    )


def split_soft_precision_features(
    features: torch.Tensor,
    dims: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a concatenated ``PROGRAM_FACTOR_FAMILIES`` feature tensor into
    ``(precision_critical, soft)`` slices along the last dimension.

    ``dims`` is the per-family width mapping from
    :func:`slm_training.models.semantic_plan_factors.factor_dims`. Since
    ``PRECISION_CRITICAL_FACTOR_FAMILIES`` is exactly the leading contiguous
    slice of ``PROGRAM_FACTOR_FAMILIES`` (asserted at import time above),
    this is a single slice, never a gather -- so no family's tensor content
    is reordered relative to how
    :func:`slm_training.models.semantic_plan_factors.concat_factor_tensors`
    produced it.
    """
    boundary = sum(dims[name] for name in PRECISION_CRITICAL_FACTOR_FAMILIES)
    return features[..., :boundary], features[..., boundary:]


class SelectiveLatentArm(str, Enum):
    """Named arms sharing one connector code path (AP-033's own arm set)."""

    DISCRETE_ONLY = "discrete_only"
    CONTINUOUS_ONLY = "continuous_only"
    SELECTIVE_HYBRID = "selective_hybrid"
    ALL_CONTINUOUS = "all_continuous"
    RANDOM_CONTINUOUS = "random_continuous"
    ORACLE_CONTINUOUS = "oracle_continuous"


# Arms whose connector input is the trained soft-latent vector, undisturbed.
# CONTINUOUS_ONLY / SELECTIVE_HYBRID / ALL_CONTINUOUS differ from each other
# in how the *harness* composes the connector's output with the rest of the
# reconstruction (see cap2_selective_hybrid_latent.py's SelectiveHybridModel),
# not in what content reaches the connector -- exactly the same separation of
# concerns AbstractPlanConnector's LEARNED/ORACLE arms use (both passthrough;
# the distinction is upstream of the connector).
_PASSTHROUGH = (
    SelectiveLatentArm.CONTINUOUS_ONLY,
    SelectiveLatentArm.SELECTIVE_HYBRID,
    SelectiveLatentArm.ALL_CONTINUOUS,
)


def resolve_selective_latent_vector(
    vector: torch.Tensor,
    *,
    arm: SelectiveLatentArm,
    oracle_vector: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Resolve the connector's input vector for one arm from the same call site.

    ``vector`` supplies the shape/device/dtype template for every arm
    (including the content-null ones), mirroring
    :func:`slm_training.models.abstract_plan_connector.resolve_plan_vector`.
    """
    if arm is SelectiveLatentArm.DISCRETE_ONLY:
        raise ValueError(
            "resolve_selective_latent_vector called with arm=discrete_only; "
            "the connector must not be constructed or invoked at all in that case"
        )
    if generator is not None and torch.device(generator.device) != vector.device:
        raise ValueError(
            f"generator device {generator.device!r} does not match latent "
            f"vector device {vector.device!r}; construct the generator on "
            "the same device"
        )
    if arm in _PASSTHROUGH:
        return vector
    if arm is SelectiveLatentArm.ORACLE_CONTINUOUS:
        if oracle_vector is None:
            raise ValueError("oracle_continuous requires oracle_vector")
        if oracle_vector.shape != vector.shape:
            raise ValueError(
                f"oracle_vector shape {tuple(oracle_vector.shape)} != connector "
                f"input shape {tuple(vector.shape)}"
            )
        return oracle_vector
    if arm is SelectiveLatentArm.RANDOM_CONTINUOUS:
        return torch.randn(
            vector.shape, generator=generator, device=vector.device, dtype=vector.dtype
        )
    raise ValueError(f"unknown arm {arm!r}")  # pragma: no cover - exhaustive above


@dataclass(frozen=True)
class SelectiveLatentTrace:
    """One forward call's worth of connector evidence (mirrors ``PlanConnectorTrace``)."""

    arm: str
    gate_value: float
    bias_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "gate_value": self.gate_value,
            "bias_norm": self.bias_norm,
        }


class SelectiveLatentConnector(nn.Module):
    """Gated-additive bias from a continuous soft-factor vector.

    Unlike :class:`AbstractPlanConnector` (which owns an embedding table for
    discrete codebook-local slot ids), this connector's input is already a
    continuous vector, so it owns only a single linear projection -- no
    embedding lookup, no pooling. The gate is a single learnable scalar
    initialized at ``0.0`` (``tanh(0) == 0``), so an enabled-but-freshly-
    initialized connector starts as a numerical no-op, matching
    ``AbstractPlanConnector``'s convention.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def bias_for_vocab(
        self,
        vector: torch.Tensor,
        *,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Additive bias, gathered to ``candidate_ids`` when given.

        Same signature as :meth:`AbstractPlanConnector.bias_for_vocab` so
        this connector is a structural drop-in wherever that method's
        ``(vector, gate)`` duck-typed interface is used (see the module
        docstring).
        """
        full = torch.tanh(self.gate) * self.proj(vector)
        if candidate_ids is None:
            return full
        return full.index_select(-1, candidate_ids)
