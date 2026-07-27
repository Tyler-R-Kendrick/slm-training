"""Deterministic tensorization of SemanticPlanV1 program factors (CAP2-05 / AP-029).

Bridges the CAP2-02 common ``LatentCodec`` interface (uniform scalar,
mixed-radix FSQ, binary LFQ, learned VQ, continuous) onto real OpenUI
``SemanticPlanV1`` program factors instead of oracle integer state ids, so the
existing no-bypass bottleneck harness can be run over program-scale semantic
structure rather than a synthetic state space.

Six factor families are tensorized into fixed-width vectors:

* ``inventory`` -- component-family counts over ``role_slots``.
* ``cardinality`` -- per-role min/max cardinality bounds.
* ``topology`` -- depth bounds and parent/sibling edge counts.
* ``binder_reference`` -- symbol/binding counts, fallback rate, and declared
  opaque binding-edge ordinals.
* ``property_role_value`` -- per-role required flag plus mean factor confidence.
* ``style_layout`` -- derived from ``archetype.id`` (see caveat below).

Every family is a fixed-size vector regardless of plan size: role slots,
symbols, and bindings beyond the configured maxima are truncated
deterministically by canonical (component_family, role_id) / declaration
order, never dropped silently in a way that changes vector width.

Honest caveat: ``SemanticPlanV1`` has no first-class style/layout field today.
The reference extractor (``OpenUISemanticPlanExtractor``) folds layout signal
(e.g. a ``Stack``'s ``direction``) into the ``archetype.id`` string instead
(see ``docs/design/iter-cap2-05-semantic-plan-bottleneck-20260725.md``). This
module derives the ``style_layout`` factor by matching a fixed vocabulary
against that string rather than inventing a new schema field. It is a
faithful bridge to the *existing* schema, not a claim that style/layout is
already a first-class structured factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.canonicalize import canonicalize_plan

__all__ = [
    "PROGRAM_FACTOR_FAMILIES",
    "ProgramFactorTensorizerConfig",
    "factor_dims",
    "total_feature_dim",
    "tensorize_semantic_plan",
    "concat_factor_tensors",
]

PROGRAM_FACTOR_FAMILIES: tuple[str, ...] = (
    "inventory",
    "cardinality",
    "topology",
    "binder_reference",
    "property_role_value",
    "style_layout",
)

_DEFAULT_COMPONENT_VOCAB: tuple[str, ...] = (
    "Stack",
    "Card",
    "List",
    "Grid",
    "TextContent",
    "Button",
    "Input",
    "CardHeader",
)
_DEFAULT_STYLE_VOCAB: tuple[str, ...] = ("column", "row")


@dataclass(frozen=True)
class ProgramFactorTensorizerConfig:
    """Fixed vocabulary/bounds for deterministic ``SemanticPlanV1`` tensorization.

    Every bound is a hard cap used for truncation, not a soft target: plans
    with more role slots/symbols/bindings than the configured maxima are
    truncated deterministically by canonical order so the emitted tensor
    width never changes.
    """

    component_vocab: tuple[str, ...] = _DEFAULT_COMPONENT_VOCAB
    style_vocab: tuple[str, ...] = _DEFAULT_STYLE_VOCAB
    max_role_slots: int = 8
    max_symbols: int = 8
    max_bindings: int = 8
    cardinality_cap: float = 8.0
    max_depth: float = 16.0
    max_edges: float = 32.0


def factor_dims(config: ProgramFactorTensorizerConfig) -> dict[str, int]:
    """Fixed per-family tensor widths for the given tensorizer config."""
    return {
        "inventory": len(config.component_vocab) + 1,  # +1 unknown/other bucket
        "cardinality": config.max_role_slots * 2,  # (min, max) per slot, padded
        "topology": 4,  # depth_low, depth_high, parent_edges, sibling_groups
        # Counts plus (role ordinal, target-symbol ordinal) for each retained
        # declared binding edge. Ordinals are request-local opaque identities,
        # never raw binder names or surface text.
        "binder_reference": 4 + config.max_bindings * 2,
        "property_role_value": config.max_role_slots + 1,  # required flags + mean confidence
        "style_layout": len(config.style_vocab) + 1,  # +1 unrecognized-style bucket
    }


def total_feature_dim(config: ProgramFactorTensorizerConfig) -> int:
    return sum(factor_dims(config).values())


def _inventory_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    dim = len(config.component_vocab) + 1
    vec = torch.zeros(dim, dtype=torch.float32)
    vocab_index = {name: i for i, name in enumerate(config.component_vocab)}
    total = max(1, len(plan.role_slots))
    for slot in plan.role_slots:
        idx = vocab_index.get(slot.component_family or "", dim - 1)
        vec[idx] += 1.0
    return vec / total


def _ordered_role_slots(plan: SemanticPlanV1, limit: int) -> list:
    ordered = sorted(plan.role_slots, key=lambda slot: slot.role_id)
    return ordered[:limit]


def _cardinality_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    dim = config.max_role_slots * 2
    vec = torch.zeros(dim, dtype=torch.float32)
    for i, slot in enumerate(_ordered_role_slots(plan, config.max_role_slots)):
        min_card = float(slot.min_cardinality) if slot.min_cardinality is not None else 0.0
        if slot.max_cardinality is not None:
            max_card = float(slot.max_cardinality)
        else:
            max_card = min_card
        vec[2 * i] = min(min_card, config.cardinality_cap) / config.cardinality_cap
        vec[2 * i + 1] = min(max_card, config.cardinality_cap) / config.cardinality_cap
    return vec


def _topology_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    vec = torch.zeros(4, dtype=torch.float32)
    topo = plan.topology
    if topo.depth_bounds is not None:
        low, high = topo.depth_bounds
        vec[0] = min(float(low), config.max_depth) / config.max_depth
        vec[1] = min(float(high), config.max_depth) / config.max_depth
    if topo.parent_relation_candidates is not None:
        vec[2] = min(float(len(topo.parent_relation_candidates)), config.max_edges) / config.max_edges
    if topo.sibling_order_groups is not None:
        vec[3] = min(float(len(topo.sibling_order_groups)), config.max_edges) / config.max_edges
    return vec


def _binder_reference_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    vec = torch.zeros(4 + config.max_bindings * 2, dtype=torch.float32)
    symbol_cap = max(1.0, float(config.max_symbols))
    binding_cap = max(1.0, float(config.max_bindings))
    vec[0] = min(float(len(plan.symbols)), float(config.max_symbols)) / symbol_cap
    vec[1] = min(float(len(plan.bindings)), float(config.max_bindings)) / binding_cap
    if plan.bindings:
        avg_candidates = sum(len(b.candidate_symbols or ()) for b in plan.bindings) / len(plan.bindings)
        vec[2] = min(avg_candidates, float(config.max_symbols)) / symbol_cap
        fallback = sum(1 for b in plan.bindings if b.placeholder_fallback) / len(plan.bindings)
        vec[3] = fallback
    role_ordinals = {
        slot.role_id: index + 1
        for index, slot in enumerate(sorted(plan.role_slots, key=lambda slot: slot.role_id))
    }
    symbol_ordinals = {
        symbol.symbol_id: index + 1
        for index, symbol in enumerate(sorted(plan.symbols, key=lambda symbol: symbol.symbol_id))
    }
    for index, binding in enumerate(
        sorted(plan.bindings, key=lambda binding: binding.role_slot_id)[: config.max_bindings]
    ):
        offset = 4 + index * 2
        vec[offset] = min(float(role_ordinals.get(binding.role_slot_id, 0)), symbol_cap) / symbol_cap
        # Candidate symbols are declared authority. Their canonical ordinal is
        # stable under alpha-renaming but still retains a binding swap.
        candidate = min(binding.candidate_symbols or (), key=symbol_ordinals.get, default=None)
        vec[offset + 1] = min(float(symbol_ordinals.get(candidate, 0)), symbol_cap) / symbol_cap
    return vec


def _property_role_value_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    dim = config.max_role_slots + 1
    vec = torch.zeros(dim, dtype=torch.float32)
    for i, slot in enumerate(_ordered_role_slots(plan, config.max_role_slots)):
        vec[i] = 1.0 if slot.required else 0.0
    confidences = plan.confidence_calibration.per_factor_confidence or {}
    if confidences:
        vec[-1] = sum(confidences.values()) / len(confidences)
    return vec


def _style_layout_tensor(plan: SemanticPlanV1, config: ProgramFactorTensorizerConfig) -> torch.Tensor:
    """Derive a style/layout tensor from ``archetype.id``.

    See the module docstring's honest caveat: this is a bridge onto an
    existing string field, not a claim that style/layout is first-class in
    ``SemanticPlanV1`` today.
    """
    dim = len(config.style_vocab) + 1
    vec = torch.zeros(dim, dtype=torch.float32)
    archetype_id = plan.archetype.id or ""
    matched = False
    for i, token in enumerate(config.style_vocab):
        if token in archetype_id:
            vec[i] = 1.0
            matched = True
    if not matched and archetype_id:
        vec[-1] = 1.0
    return vec


_FACTOR_BUILDERS = {
    "inventory": _inventory_tensor,
    "cardinality": _cardinality_tensor,
    "topology": _topology_tensor,
    "binder_reference": _binder_reference_tensor,
    "property_role_value": _property_role_value_tensor,
    "style_layout": _style_layout_tensor,
}


def tensorize_semantic_plan(
    plan: SemanticPlanV1,
    config: ProgramFactorTensorizerConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministically tensorize one ``SemanticPlanV1`` into factor vectors.

    The plan is canonicalized first so that two plans with identical
    structure but different raw role/symbol ids produce bit-identical
    tensors. (Gold-provenance plans compile to their own baseline under
    ``canonicalize_plan`` and are returned unchanged; the reference
    extractor already assigns role/symbol ids in stable structural-walk
    order, so this is a safe no-op for the fixture corpus.)
    """
    cfg = config or ProgramFactorTensorizerConfig()
    canonical = canonicalize_plan(plan)
    return {name: _FACTOR_BUILDERS[name](canonical, cfg) for name in PROGRAM_FACTOR_FAMILIES}


def concat_factor_tensors(factors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-family tensors in the canonical family order."""
    return torch.cat([factors[name] for name in PROGRAM_FACTOR_FAMILIES], dim=-1)
