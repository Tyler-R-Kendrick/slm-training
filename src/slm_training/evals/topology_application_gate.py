"""SLM-384 (DSH3-16) topology-application activation gate.

DSH3-16 asks to compare recompute-only, hierarchical-head-with-ordinary-
lowering, and topology-apply arms for applying verified operator edits
directly to typed topology-diffusion nodes. The terminal CAP2 disposition
(``docs/design/dsh3-17-cap2-disposition-20260723``,
``slm_training.evals.cap2_disposition``) already records why this stays
closed: ``TOPOLOGY_APPLICATION`` is ``UNRUN_CONDITIONAL`` because "no
accepted hierarchical-head arm exists" -- the hierarchical-head arm is the
operator-selection mechanism DSH3-16's own comparison would lower/apply, and
SLM-383's causal-attribution experiment rejected it (ties the token
baseline; no causal improvement).

That reason was previously prose only. This module makes the gate
mechanically checkable against the disposition's own typed capability rows
instead of duplicating or re-deriving the verdict, and is fail-closed: any
``HIERARCHICAL_HEAD`` verdict other than an implemented-benefit ``SUPPORTED``
keeps ``TOPOLOGY_APPLICATION`` not authorized. Per DSH3-16's Decision text
this closes in plan-only mode -- no ``ActionEffectV1``-to-``TopologyNode``
mutation bridge, no node-pass/model-forward instrumentation, and no
production model code -- mirroring the SLM-250/LOT1-01 precedent for a
gated issue whose activation condition is unmet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from slm_training.evals.cap2_disposition import Cap2Capability, Cap2CapabilityVerdict

TOPOLOGY_APPLICATION_GATE_SCHEMA = "topology_application_activation_gate/v1"


class TopologyApplicationGateVerdict(str, Enum):
    AUTHORIZED = "authorized"
    NOT_AUTHORIZED = "not_authorized"


@dataclass(frozen=True)
class TopologyApplicationActivationGateV1:
    """SLM-384's activation decision, derived from the CAP2 disposition ledger."""

    verdict: TopologyApplicationGateVerdict
    reason: str
    hierarchical_head_verdict: Cap2CapabilityVerdict
    hierarchical_head_evidence_ids: tuple[str, ...]
    schema: str = TOPOLOGY_APPLICATION_GATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "hierarchical_head_verdict": self.hierarchical_head_verdict.value,
            "hierarchical_head_evidence_ids": list(self.hierarchical_head_evidence_ids),
        }


def evaluate_topology_application_gate(
    capabilities: Sequence[Mapping[str, Any]],
) -> TopologyApplicationActivationGateV1:
    """Evaluate SLM-384's activation gate from a CAP2 disposition's capability rows.

    ``capabilities`` is the ``disposition["capabilities"]`` list as produced by
    ``Cap2CapabilityDispositionV1.to_dict()`` (or the equivalent published
    ``docs/design/dsh3-17-cap2-disposition-20260723/report.json``). Authorized
    only when ``HIERARCHICAL_HEAD`` is ``SUPPORTED`` with an implemented
    benefit -- a decodable-but-unused, rejected, contract-only, unavailable,
    or still-unrun verdict all keep this gate closed.
    """
    hierarchical_head = next(
        (
            item
            for item in capabilities
            if item.get("capability") == Cap2Capability.HIERARCHICAL_HEAD.value
        ),
        None,
    )
    if hierarchical_head is None:
        raise ValueError(
            "disposition capabilities are missing the hierarchical_head row"
        )
    verdict = Cap2CapabilityVerdict(hierarchical_head["verdict"])
    evidence_ids = tuple(hierarchical_head.get("evidence_ids", ()))

    if verdict is Cap2CapabilityVerdict.SUPPORTED and hierarchical_head.get(
        "implemented_benefit"
    ):
        return TopologyApplicationActivationGateV1(
            verdict=TopologyApplicationGateVerdict.AUTHORIZED,
            reason=(
                "hierarchical_head is SUPPORTED with an implemented benefit; "
                "DSH3-16 may build the ActionEffectV1-to-TopologyNode mutation "
                "bridge and run the matched-quality recompute-only/"
                "hierarchical-head/topology-apply comparison."
            ),
            hierarchical_head_verdict=verdict,
            hierarchical_head_evidence_ids=evidence_ids,
        )
    return TopologyApplicationActivationGateV1(
        verdict=TopologyApplicationGateVerdict.NOT_AUTHORIZED,
        reason=(
            f"hierarchical_head verdict is {verdict.value!r}, not an accepted "
            "SUPPORTED operator-selection mechanism. DSH3-16's topology-apply "
            "comparison has no accepted arm to lower or apply, and no "
            "ActionEffectV1-to-TopologyNode mutation bridge exists between "
            "the operators package's AST-dict representation and the "
            "topology-diffusion model's typed node tree. Close in plan-only "
            "mode without production model code (SLM-250/LOT1-01 precedent)."
        ),
        hierarchical_head_verdict=verdict,
        hierarchical_head_evidence_ids=evidence_ids,
    )


__all__ = [
    "TOPOLOGY_APPLICATION_GATE_SCHEMA",
    "TopologyApplicationActivationGateV1",
    "TopologyApplicationGateVerdict",
    "evaluate_topology_application_gate",
]
