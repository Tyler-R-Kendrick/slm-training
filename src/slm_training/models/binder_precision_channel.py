"""BinderPrecisionChannel: a default-off, gated connector conditioning
binder/reference-eligible decisions on a :class:`BinderGraphV1` (AP-026 /
SLM-320).

Mirrors ``models/abstract_plan_connector.py`` (AP-024 / SLM-316)'s
single-code-path, matched-arm discipline, but the content is discrete graph
structure rather than a continuous plan vector:

* :class:`BinderChannelArm` names the modes the issue asks for (``oracle``,
  ``predicted_soft``, ``absent``, ``shuffled``, ``corrupted``) plus
  ``disabled``; every arm resolves through :func:`resolve_binder_graph` so
  control arms cannot silently diverge from ``predicted_soft`` in structure.
* :func:`confirm_predicted_edges` is the non-negotiable gate: a predicted
  reference edge is *never* treated as legal on its own authority, no matter
  how confident the arm/model is. It is only "confirmed" when the same
  ``(source, target, role)`` edge also appears in the deterministic verifier
  graph (:meth:`~slm_training.data.progspec.binder_graph.BinderGraphV1.check_g3`'s
  input) -- this is decode-invariant I3 in one function: "ranking is a
  lever; legality is not" (``docs/design/decode-invariants.md``).
* :class:`BinderChannelGate` is the smallest possible neural hook: a single
  learnable scalar (init ``0.0``, matching ``AbstractPlanConnector.gate``)
  that scales caller-supplied per-edge features, masked to confirmed edges
  only. It does not own an embedding table or wire into
  ``DenoiserTower``/``TwoTowerModel`` -- that integration (a concrete
  per-edge feature representation reused from the encoder/decoder ops
  vocabulary) is deferred, honestly, to a follow-up issue; wiring it in
  before there is a trained producer of predicted binder graphs would be
  untested plumbing pretending to be a shipped feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from slm_training.data.progspec.binder_graph import BinderGraphV1
from slm_training.data.progspec.capsules import DependencyKind, ScopeEdge

__all__ = [
    "BinderChannelArm",
    "BinderChannelGate",
    "BinderChannelTrace",
    "confirm_predicted_edges",
    "resolve_binder_graph",
]


class BinderChannelArm(str, Enum):
    """Named conditioning arms sharing one resolution path."""

    DISABLED = "disabled"
    ORACLE = "oracle"
    PREDICTED_SOFT = "predicted_soft"
    ABSENT = "absent"
    SHUFFLED = "shuffled"
    CORRUPTED = "corrupted"


# Arms whose graph is the caller-supplied graph's own content, undisturbed.
_PASSTHROUGH = (BinderChannelArm.ORACLE, BinderChannelArm.PREDICTED_SOFT)


def resolve_binder_graph(
    graph: BinderGraphV1,
    *,
    arm: BinderChannelArm,
    generator: torch.Generator | None = None,
) -> BinderGraphV1:
    """Resolve the channel's input graph for one arm from the same call site.

    ``graph`` supplies the node/edge-count template for every arm (including
    the content-null ones), so every arm's downstream cost is comparable and
    only the channel's *content* differs -- the same discipline
    ``resolve_plan_vector`` uses for :class:`PlanConnectorArm`.
    """
    if arm is BinderChannelArm.DISABLED:
        raise ValueError(
            "resolve_binder_graph called with arm=disabled; the channel "
            "must not be constructed or invoked at all in that case"
        )
    if arm in _PASSTHROUGH:
        return graph
    if arm is BinderChannelArm.ABSENT:
        return BinderGraphV1(
            root_id=graph.root_id,
            nodes=(),
            edges=(),
            unresolved=(),
            spec_id=graph.spec_id,
            version=graph.version,
        )
    if arm is BinderChannelArm.SHUFFLED:
        return _shuffled(graph, generator=generator)
    # CORRUPTED: redirect each reference edge's target to a different,
    # deterministically-chosen wrong node -- plausible-shaped but wrong,
    # distinct from SHUFFLED's full batch-level permutation.
    return _corrupted(graph, generator=generator)


def _shuffled(
    graph: BinderGraphV1, *, generator: torch.Generator | None
) -> BinderGraphV1:
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) < 2:
        return graph
    perm = torch.randperm(len(node_ids), generator=generator).tolist()
    remap = {original: node_ids[new_index] for original, new_index in zip(node_ids, perm)}
    shuffled_edges = tuple(
        ScopeEdge(
            source=remap.get(edge.source, edge.source),
            target=remap.get(edge.target, edge.target),
            kind=edge.kind,
            role=edge.role,
        )
        for edge in graph.edges
    )
    return BinderGraphV1(
        root_id=remap.get(graph.root_id, graph.root_id),
        nodes=graph.nodes,
        edges=shuffled_edges,
        unresolved=graph.unresolved,
        spec_id=graph.spec_id,
        version=graph.version,
    )


def _corrupted(
    graph: BinderGraphV1, *, generator: torch.Generator | None
) -> BinderGraphV1:
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) < 2:
        return graph
    corrupted_edges = []
    for offset, edge in enumerate(graph.edges):
        if edge.kind != DependencyKind.REFERENCE:
            corrupted_edges.append(edge)
            continue
        candidates = [n for n in node_ids if n != edge.target]
        if not candidates:
            corrupted_edges.append(edge)
            continue
        wrong_target = candidates[
            torch.randint(
                0, len(candidates), (1,), generator=generator
            ).item()
            if generator is not None
            else offset % len(candidates)
        ]
        corrupted_edges.append(
            ScopeEdge(
                source=edge.source, target=wrong_target, kind=edge.kind, role=edge.role
            )
        )
    return BinderGraphV1(
        root_id=graph.root_id,
        nodes=graph.nodes,
        edges=tuple(corrupted_edges),
        unresolved=graph.unresolved,
        spec_id=graph.spec_id,
        version=graph.version,
    )


def confirm_predicted_edges(
    predicted: BinderGraphV1, *, verifier: BinderGraphV1
) -> tuple[ScopeEdge, ...]:
    """Return only the predicted reference edges the verifier also contains.

    This is the gate the issue's acceptance criteria require: "never
    hard-enforce learned edges without verifier confirmation." A predicted
    edge earns no legal authority from being predicted -- it is confirmed
    if and only if the exact ``(source, target, role)`` triple is also a
    reference edge in ``verifier`` (the deterministic
    :func:`~slm_training.data.progspec.binder_graph.derive_binder_graph`
    output for the same program). Edges present only in ``predicted`` are
    silently dropped, never promoted; edges present only in ``verifier`` are
    not added (this is a *filter*, not a merge).
    """
    verified = {
        (edge.source, edge.target, edge.role) for edge in verifier.reference_edges()
    }
    return tuple(
        edge for edge in predicted.reference_edges() if (edge.source, edge.target, edge.role) in verified
    )


@dataclass(frozen=True)
class BinderChannelTrace:
    """One channel call's worth of verifier-gating evidence.

    ``rejected_edge_count`` is the number of predicted reference edges that
    did *not* survive :func:`confirm_predicted_edges` -- a nonzero count is
    expected and healthy (it is the gate doing its job), not an error.
    """

    arm: str
    predicted_edge_count: int
    confirmed_edge_count: int
    rejected_edge_count: int
    gate_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "predicted_edge_count": self.predicted_edge_count,
            "confirmed_edge_count": self.confirmed_edge_count,
            "rejected_edge_count": self.rejected_edge_count,
            "gate_value": self.gate_value,
        }


class BinderChannelGate(nn.Module):
    """Gated scaling of caller-supplied per-edge features, masked to
    verifier-confirmed edges only.

    Owns a single learnable scalar initialized at ``0.0`` (``tanh(0) == 0``),
    matching ``AbstractPlanConnector.gate`` -- an enabled-but-freshly
    -initialized gate starts as a numerical no-op. Does not own an embedding
    table: callers supply their own ``edge_features`` (e.g. a pooled
    encoder/decoder ops-vocabulary representation of each edge's source and
    target), keeping this module agnostic to how that representation is
    produced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def bias_for_confirmed_edges(
        self,
        edge_features: torch.Tensor,
        *,
        confirmed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gate-scaled features, zeroed for every unconfirmed edge.

        ``edge_features``: ``[num_edges, ...]``. ``confirmed_mask``: a
        boolean or 0/1 tensor of shape ``[num_edges]`` (broadcastable over
        the trailing dims of ``edge_features``) -- callers build this from
        :func:`confirm_predicted_edges`'s output. An edge with
        ``confirmed_mask == 0`` contributes exactly zero regardless of
        ``gate``, so an unconfirmed predicted edge can never leak into the
        bias even if the gate is fully open.
        """
        if edge_features.shape[0] != confirmed_mask.shape[0]:
            raise ValueError(
                "edge_features and confirmed_mask must share their leading "
                f"(num_edges) dimension; got {edge_features.shape[0]} and "
                f"{confirmed_mask.shape[0]}"
            )
        mask = confirmed_mask.to(dtype=edge_features.dtype)
        extra_dims = edge_features.dim() - mask.dim()
        if extra_dims > 0:
            mask = mask.view(mask.shape + (1,) * extra_dims)
        return torch.tanh(self.gate) * edge_features * mask
