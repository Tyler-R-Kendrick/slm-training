"""Advisory semantic-factor evidence plane (default-off residual experiment).

Exact legality stays with compiler/solver/verifier. Factors and ports are a
**read-only projection** of exact and external evidence used only to rank
already-legal candidates. Nothing here may widen or shrink legal support.

See ``docs/design/semantic-factor-frontier.md`` and the anti-E237 campaign.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    "AdvisoryEvidenceFactorV1",
    "EvidencePolarity",
    "EvidenceSourceKind",
    "FactorPortDirection",
    "FactorPortV1",
    "FactorRepresentation",
    "SemanticEvidenceSnapshotV1",
    "build_factor_representation",
    "canonical_json",
    "digest_snapshot",
    "lossy_pairwise_edges",
    "project_program_factors",
    "reconstruct_membership",
    "shuffle_factor_membership",
    "shuffle_port_roles",
]


class FactorPortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    UNORDERED = "unordered"


class EvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceSourceKind(str, Enum):
    EXACT_STATE = "exact_state"
    SEMANTIC_PLAN = "semantic_plan"
    BINDER_GRAPH = "binder_graph"
    SCHEMA = "schema"
    RETRIEVED_EXAMPLE = "retrieved_example"
    VERIFIER_FEEDBACK = "verifier_feedback"
    PREDICTED = "predicted"


class FactorRepresentation(str, Enum):
    NONE = "none"
    DIRECT_FACTORS = "direct_factors"
    LOSSY_PAIRWISE = "lossy_pairwise"
    FACTOR_NODE = "factor_node"
    ROLE_PORTED_FACTOR = "role_ported_factor"
    FLAT_HYPEREDGE = "flat_hyperedge"
    ROLE_SHUFFLED = "role_shuffled"
    FACTOR_SHUFFLED = "factor_shuffled"
    EXACT_TYPED_ZERO_PARAMETER = "exact_typed_zero_parameter"
    ORACLE_DIAGNOSTIC = "oracle_diagnostic"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class FactorPortV1:
    """One role-ported attachment of a node to an evidence factor."""

    port_id: str
    role: str
    node_id: str
    node_kind: str
    direction: FactorPortDirection
    ordinal: int

    def __post_init__(self) -> None:
        if not self.port_id or not self.role or not self.node_id or not self.node_kind:
            raise ValueError("FactorPortV1 requires nonempty ids/role/kind")
        if int(self.ordinal) < 0:
            raise ValueError("ordinal must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "role": self.role,
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "direction": self.direction.value,
            "ordinal": int(self.ordinal),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorPortV1:
        return cls(
            port_id=str(data["port_id"]),
            role=str(data["role"]),
            node_id=str(data["node_id"]),
            node_kind=str(data["node_kind"]),
            direction=FactorPortDirection(str(data["direction"])),
            ordinal=int(data["ordinal"]),
        )


@dataclass(frozen=True)
class AdvisoryEvidenceFactorV1:
    """One advisory higher-order evidence factor (never legal authority)."""

    factor_version: str
    factor_id: str
    factor_type: str
    ports: tuple[FactorPortV1, ...]
    polarity: EvidencePolarity
    source_kind: EvidenceSourceKind
    source_ref: str
    confidence: float = 1.0
    advisory: bool = True

    def __post_init__(self) -> None:
        if not self.factor_id or not self.factor_type:
            raise ValueError("factor_id and factor_type required")
        if not self.advisory:
            raise ValueError("factors must be advisory=True")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be in [0,1]")
        # Reject raw binder surface names as node identity (alpha-stable ids only).
        for port in self.ports:
            if ":" in port.node_id and port.node_id.startswith(":"):
                raise ValueError("slot surface paths are not canonical node ids")

    def membership(self) -> frozenset[str]:
        return frozenset(p.node_id for p in self.ports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_version": self.factor_version,
            "factor_id": self.factor_id,
            "factor_type": self.factor_type,
            "ports": [p.to_dict() for p in self.ports],
            "polarity": self.polarity.value,
            "source_kind": self.source_kind.value,
            "source_ref": self.source_ref,
            "confidence": float(self.confidence),
            "advisory": True,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdvisoryEvidenceFactorV1:
        ports = tuple(FactorPortV1.from_dict(p) for p in data.get("ports") or ())
        return cls(
            factor_version=str(data["factor_version"]),
            factor_id=str(data["factor_id"]),
            factor_type=str(data["factor_type"]),
            ports=ports,
            polarity=EvidencePolarity(str(data["polarity"])),
            source_kind=EvidenceSourceKind(str(data["source_kind"])),
            source_ref=str(data["source_ref"]),
            confidence=float(data.get("confidence", 1.0)),
            advisory=bool(data.get("advisory", True)),
        )


@dataclass(frozen=True)
class SemanticEvidenceSnapshotV1:
    """Immutable request-local evidence view over exact/external sources."""

    schema: str
    source_digest: str
    legal_candidate_ids: tuple[str, ...]
    coverage: str
    factors: tuple[AdvisoryEvidenceFactorV1, ...]
    extractor_version: str
    claim_class: str = "fixture_or_scratch"
    advisory: bool = True

    def __post_init__(self) -> None:
        if self.schema != "semantic_evidence_snapshot/v1":
            raise ValueError("unsupported semantic evidence snapshot schema")
        if not self.advisory:
            raise ValueError("snapshot must be advisory")
        if len(self.legal_candidate_ids) != len(set(self.legal_candidate_ids)):
            raise ValueError("legal_candidate_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_digest": self.source_digest,
            "legal_candidate_ids": list(self.legal_candidate_ids),
            "coverage": self.coverage,
            "factors": [f.to_dict() for f in self.factors],
            "extractor_version": self.extractor_version,
            "claim_class": self.claim_class,
            "advisory": True,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticEvidenceSnapshotV1:
        factors = tuple(
            AdvisoryEvidenceFactorV1.from_dict(f) for f in data.get("factors") or ()
        )
        return cls(
            schema=str(data["schema"]),
            source_digest=str(data["source_digest"]),
            legal_candidate_ids=tuple(str(x) for x in data.get("legal_candidate_ids") or ()),
            coverage=str(data["coverage"]),
            factors=factors,
            extractor_version=str(data["extractor_version"]),
            claim_class=str(data.get("claim_class", "fixture_or_scratch")),
            advisory=bool(data.get("advisory", True)),
        )


def digest_snapshot(snapshot: SemanticEvidenceSnapshotV1) -> str:
    """Content digest excluding claim_class timestamps (identity-stable)."""

    payload = snapshot.to_dict()
    payload.pop("claim_class", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _load_projection_payload() -> dict[str, Any]:
    """Optional external projection.v1.json; fall back to built-in aliases."""

    try:
        from slm_training.harnesses.experiments.semantic_factor_config import (
            load_sff_projection,
        )

        return dict(load_sff_projection().payload)
    except Exception:
        return {
            "extractor_version": "semantic_evidence_projector/v1",
            "default_claim_class": "fixture_or_scratch",
            "component_role": "COMPONENT",
            "component_confidence": 1.0,
            "binder_confidence": 1.0,
            "source_role_aliases": {
                "DEF": "DEFINITION",
                "DEFINITION": "DEFINITION",
                "default": "SOURCE",
            },
            "target_role_aliases": {
                "USE": "USE",
                "REF": "USE",
                "REFERENCE": "USE",
                "default": "TARGET",
            },
        }


def _alias_role(role: str, aliases: Mapping[str, Any]) -> str:
    key = str(role).upper()
    if key in aliases:
        return str(aliases[key])
    return str(aliases.get("default") or role)


def project_program_factors(
    *,
    program_id: str,
    components: Sequence[tuple[str, str]],
    binder_edges: Sequence[tuple[str, str, str]] = (),
    legal_candidate_ids: Sequence[str],
    coverage: str = "complete",
    extractor_version: str | None = None,
) -> SemanticEvidenceSnapshotV1:
    """Pure projection from structural program facts into advisory factors.

    ``components`` is ``(node_id, component_kind)``.
    ``binder_edges`` is ``(source_id, target_id, role)``.
    Role aliases / confidences come from ``projection.v1.json`` when available.
    """

    proj = _load_projection_payload()
    extractor = str(
        extractor_version
        or proj.get("extractor_version")
        or "semantic_evidence_projector/v1"
    )
    component_role = str(proj.get("component_role") or "COMPONENT")
    component_conf = float(proj.get("component_confidence", 1.0))
    binder_conf = float(proj.get("binder_confidence", 1.0))
    source_aliases = dict(proj.get("source_role_aliases") or {})
    target_aliases = dict(proj.get("target_role_aliases") or {})
    claim_class = str(proj.get("default_claim_class") or "fixture_or_scratch")

    factors: list[AdvisoryEvidenceFactorV1] = []
    for index, (node_id, kind) in enumerate(components):
        ports = (
            FactorPortV1(
                port_id=f"{program_id}:comp:{index}:self",
                role=component_role,
                node_id=str(node_id),
                node_kind=str(kind),
                direction=FactorPortDirection.UNORDERED,
                ordinal=0,
            ),
        )
        factors.append(
            AdvisoryEvidenceFactorV1(
                factor_version="AdvisoryEvidenceFactorV1",
                factor_id=f"{program_id}:component:{index}",
                factor_type="component_presence",
                ports=ports,
                polarity=EvidencePolarity.SUPPORTS,
                source_kind=EvidenceSourceKind.EXACT_STATE,
                source_ref=f"component:{node_id}",
                confidence=component_conf,
            )
        )
    for index, (src, tgt, role) in enumerate(binder_edges):
        # Source port uses source_aliases (DEF→DEFINITION, else SOURCE).
        # Target port uses target_aliases (USE/REF→USE, else TARGET).
        src_role = _alias_role(role, source_aliases)
        tgt_role = _alias_role(role, target_aliases)
        ports = (
            FactorPortV1(
                port_id=f"{program_id}:bind:{index}:src",
                role=src_role,
                node_id=str(src),
                node_kind="binder",
                direction=FactorPortDirection.OUTPUT,
                ordinal=0,
            ),
            FactorPortV1(
                port_id=f"{program_id}:bind:{index}:tgt",
                role=tgt_role,
                node_id=str(tgt),
                node_kind="binder",
                direction=FactorPortDirection.INPUT,
                ordinal=1,
            ),
        )
        factors.append(
            AdvisoryEvidenceFactorV1(
                factor_version="AdvisoryEvidenceFactorV1",
                factor_id=f"{program_id}:binder:{index}",
                factor_type="binder_reference",
                ports=ports,
                polarity=EvidencePolarity.SUPPORTS,
                source_kind=EvidenceSourceKind.BINDER_GRAPH,
                source_ref=f"edge:{src}->{tgt}:{role}",
                confidence=binder_conf,
            )
        )
    source = {
        "program_id": program_id,
        "components": list(components),
        "binder_edges": list(binder_edges),
        "legal_candidate_ids": list(legal_candidate_ids),
        "coverage": coverage,
        "extractor_version": extractor,
    }
    snapshot = SemanticEvidenceSnapshotV1(
        schema="semantic_evidence_snapshot/v1",
        source_digest=hashlib.sha256(canonical_json(source)).hexdigest(),
        legal_candidate_ids=tuple(str(x) for x in legal_candidate_ids),
        coverage=str(coverage),
        factors=tuple(factors),
        extractor_version=extractor,
        claim_class=claim_class,
    )
    return snapshot



@dataclass(frozen=True)
class FactorGraphView:
    """Lossless bipartite factor-node encoding of advisory factors."""

    vertex_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    incidence: tuple[tuple[int, int], ...]  # (vertex_index, factor_index)
    port_roles: tuple[str, ...]  # parallel to incidence edges
    port_directions: tuple[str, ...]
    representation: FactorRepresentation
    polarity: tuple[str, ...]  # per factor


def reconstruct_membership(view: FactorGraphView) -> dict[str, frozenset[str]]:
    """Round-trip factor membership from a factor-node incidence view."""

    out: dict[str, set[str]] = {fid: set() for fid in view.factor_ids}
    for v_idx, f_idx in view.incidence:
        out[view.factor_ids[f_idx]].add(view.vertex_ids[v_idx])
    return {fid: frozenset(nodes) for fid, nodes in out.items()}


def lossy_pairwise_edges(
    factors: Sequence[AdvisoryEvidenceFactorV1],
) -> tuple[tuple[str, str], ...]:
    """Lose factor identity: undirected pairs among each factor's nodes."""

    pairs: set[tuple[str, str]] = set()
    for factor in factors:
        nodes = sorted(factor.membership())
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                pairs.add((a, b) if a <= b else (b, a))
    return tuple(sorted(pairs))


def shuffle_port_roles(
    factors: Sequence[AdvisoryEvidenceFactorV1], *, seed: int
) -> tuple[AdvisoryEvidenceFactorV1, ...]:
    """Deterministic role permutation within each factor (destructive control)."""

    out: list[AdvisoryEvidenceFactorV1] = []
    for offset, factor in enumerate(factors):
        roles = [p.role for p in factor.ports]
        if len(roles) > 1:
            # Rotate by seed+offset — preserves multiset of roles, changes assignment.
            k = (int(seed) + offset) % len(roles)
            roles = roles[k:] + roles[:k]
        ports = tuple(
            FactorPortV1(
                port_id=p.port_id,
                role=roles[i],
                node_id=p.node_id,
                node_kind=p.node_kind,
                direction=p.direction,
                ordinal=p.ordinal,
            )
            for i, p in enumerate(factor.ports)
        )
        out.append(
            AdvisoryEvidenceFactorV1(
                factor_version=factor.factor_version,
                factor_id=factor.factor_id,
                factor_type=factor.factor_type,
                ports=ports,
                polarity=factor.polarity,
                source_kind=factor.source_kind,
                source_ref=factor.source_ref,
                confidence=factor.confidence,
            )
        )
    return tuple(out)


def shuffle_factor_membership(
    factors: Sequence[AdvisoryEvidenceFactorV1],
    *,
    seed: int,
) -> tuple[AdvisoryEvidenceFactorV1, ...]:
    """Reassign node membership deterministically while keeping counts."""

    all_nodes: list[tuple[str, str]] = []
    for factor in factors:
        for port in factor.ports:
            all_nodes.append((port.node_id, port.node_kind))
    if not all_nodes:
        return tuple(factors)
    # Deterministic rotation of the node multiset.
    k = int(seed) % len(all_nodes)
    rotated = all_nodes[k:] + all_nodes[:k]
    cursor = 0
    out: list[AdvisoryEvidenceFactorV1] = []
    for factor in factors:
        ports: list[FactorPortV1] = []
        for port in factor.ports:
            node_id, node_kind = rotated[cursor % len(rotated)]
            cursor += 1
            ports.append(
                FactorPortV1(
                    port_id=port.port_id,
                    role=port.role,
                    node_id=node_id,
                    node_kind=node_kind,
                    direction=port.direction,
                    ordinal=port.ordinal,
                )
            )
        out.append(
            AdvisoryEvidenceFactorV1(
                factor_version=factor.factor_version,
                factor_id=factor.factor_id,
                factor_type=factor.factor_type,
                ports=tuple(ports),
                polarity=factor.polarity,
                source_kind=factor.source_kind,
                source_ref=factor.source_ref,
                confidence=factor.confidence,
            )
        )
    return tuple(out)


def build_factor_representation(
    snapshot: SemanticEvidenceSnapshotV1,
    representation: FactorRepresentation,
    *,
    seed: int = 0,
) -> FactorGraphView | None:
    """Build a shared-interface graph view for the named representation arm."""

    if representation in {
        FactorRepresentation.NONE,
        FactorRepresentation.EXACT_TYPED_ZERO_PARAMETER,
        FactorRepresentation.ORACLE_DIAGNOSTIC,
        FactorRepresentation.DIRECT_FACTORS,
        FactorRepresentation.LOSSY_PAIRWISE,
    }:
        return None

    factors = snapshot.factors
    if representation is FactorRepresentation.ROLE_SHUFFLED:
        factors = shuffle_port_roles(factors, seed=seed)
    elif representation is FactorRepresentation.FACTOR_SHUFFLED:
        factors = shuffle_factor_membership(factors, seed=seed)

    strip_roles = representation in {
        FactorRepresentation.FACTOR_NODE,
        FactorRepresentation.FLAT_HYPEREDGE,
    }
    vertex_ids = sorted({p.node_id for f in factors for p in f.ports})
    factor_ids = tuple(f.factor_id for f in factors)
    v_index = {vid: i for i, vid in enumerate(vertex_ids)}
    incidence: list[tuple[int, int]] = []
    port_roles: list[str] = []
    port_directions: list[str] = []
    polarity = tuple(f.polarity.value for f in factors)
    for f_idx, factor in enumerate(factors):
        for port in sorted(factor.ports, key=lambda p: (p.ordinal, p.port_id)):
            incidence.append((v_index[port.node_id], f_idx))
            if strip_roles:
                port_roles.append("MEMBER")
                port_directions.append(FactorPortDirection.UNORDERED.value)
            else:
                port_roles.append(port.role)
                port_directions.append(port.direction.value)
    return FactorGraphView(
        vertex_ids=tuple(vertex_ids),
        factor_ids=factor_ids,
        incidence=tuple(incidence),
        port_roles=tuple(port_roles),
        port_directions=tuple(port_directions),
        representation=representation,
        polarity=polarity,
    )
