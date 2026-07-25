"""Generic LOT-project downstream launch-gate evaluator (plan-only closeout).

Every LOT2/LOT3/LOT4 issue (SLM-252 … SLM-258) declares the same shape of
activation gate: the LOT1 chain must first authorize it — concretely, the
SLM-251 (LOT1-02) launch gate must be met, which itself requires the
SLM-248 transfer authorization and the SLM-249 oracle ceiling (evaluated
by :mod:`slm_training.harnesses.experiments.lot1_02_activation_gate`).
When the upstream chain is unmet, the downstream issue closes
``not_authorized`` in plan-only/bounded-diagnostic mode with no training,
factorial, intervention, or systems code.

This module evaluates that shared upstream gate once and stamps a
per-issue ``LotDownstreamGateV1`` disposition. It contains no model or
campaign code — evaluating the gate honestly *is* the deliverable while
the prerequisites are unmet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    AUTHORIZED_WIRING_ONLY,
    NOT_AUTHORIZED,
)
from slm_training.harnesses.experiments.lot1_02_activation_gate import (
    evaluate_lot1_02_launch_gate,
)

__all__ = [
    "DOWNSTREAM_ISSUES",
    "GATE_SCHEMA_VERSION",
    "PENDING_CAMPAIGN_GATE",
    "DownstreamIssueSpec",
    "LotDownstreamGateV1",
    "evaluate_downstream_gate",
    "render_markdown",
]

GATE_SCHEMA_VERSION = "lot_downstream_gate/v1"

#: Verdict when the upstream contract chain is met: the issue's own
#: activation still requires the LOT1-02 campaign's published gate
#: (``FaithfulLatentTransferGateV1``), which does not exist until the
#: campaign actually runs. This is never a quality or launch claim.
PENDING_CAMPAIGN_GATE = "upstream_authorized_pending_campaign_gate"


@dataclass(frozen=True)
class DownstreamIssueSpec:
    """Identity of one LOT2+ issue whose activation rides on the LOT1 chain."""

    linear_issue: str
    alias: str
    title: str
    gate_contract_id: str
    slug: str
    activation_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


DOWNSTREAM_ISSUES: dict[str, DownstreamIssueSpec] = {
    "SLM-252": DownstreamIssueSpec(
        linear_issue="SLM-252",
        alias="LOT2-01",
        title="Run the causal supervision routing x timing x grounding factorial",
        gate_contract_id="causal-supervision-routing-gate-v1",
        slug="lot2-01-supervision-routing",
        activation_requirement=(
            "SLM-251 (LOT1-02) authorizes LOT2 and provides the selected "
            "parent, faithful treatment checkpoint/recipe, continued explicit "
            "control, curriculum, and fairness manifests."
        ),
    ),
}


@dataclass(frozen=True)
class LotDownstreamGateV1:
    """Per-issue downstream disposition derived from the LOT1 launch gate."""

    schema_version: str = GATE_SCHEMA_VERSION
    contract_id: str = ""
    issue: DownstreamIssueSpec | None = None
    lot1_02_disposition: str = NOT_AUTHORIZED
    lot1_01_disposition: str = NOT_AUTHORIZED
    verdict: str = NOT_AUTHORIZED
    verdict_rationale: str = ""
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["issue"] = self.issue.to_dict() if self.issue is not None else None
        data["contract_hash"] = self.contract_hash()
        return data

    def contract_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "issue": self.issue.to_dict() if self.issue is not None else None,
            "lot1_02_disposition": self.lot1_02_disposition,
            "verdict": self.verdict,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def evaluate_downstream_gate(
    spec: DownstreamIssueSpec,
    fidelity_contract: dict[str, Any],
    trace_gate_contract: dict[str, Any],
) -> LotDownstreamGateV1:
    """Evaluate one LOT2+ issue's activation against real upstream contracts.

    The verdict is ``not_authorized`` unless the LOT1-02 launch gate is
    met; even then it is only ``upstream_authorized_pending_campaign_gate``
    — the issue's own campaign gate must still be published by a real run
    before any factorial/intervention/systems work is authorized.
    """
    lot1_02 = evaluate_lot1_02_launch_gate(fidelity_contract, trace_gate_contract)

    if lot1_02.verdict == AUTHORIZED_WIRING_ONLY:
        verdict = PENDING_CAMPAIGN_GATE
        rationale = (
            "The LOT1 contract-level prerequisites are met, but "
            f"{spec.alias} ({spec.linear_issue}) still requires the LOT1-02 "
            "campaign's published FaithfulLatentTransferGateV1 with a "
            "positive authorization before any objective/architecture work "
            "is authorized. No campaign has run; this is not a launch claim."
        )
    else:
        verdict = NOT_AUTHORIZED
        rationale = (
            f"{spec.alias} ({spec.linear_issue}) activation requires: "
            f"{spec.activation_requirement} The LOT1-02 launch gate reports "
            f"'{lot1_02.verdict}' (LOT1-01 disposition "
            f"'{lot1_02.lot1_01_disposition}'): no faithful K x c model "
            "path, curriculum, Stage 0 parent, or continued-explicit control "
            "exists, and SLM-249's oracle ceiling is not positive. Closing "
            "not_authorized in plan-only/bounded-diagnostic mode; no "
            "training, factorial, intervention, or systems code is added by "
            "this disposition."
        )

    return LotDownstreamGateV1(
        contract_id=spec.gate_contract_id,
        issue=spec,
        lot1_02_disposition=lot1_02.verdict,
        lot1_01_disposition=lot1_02.lot1_01_disposition,
        verdict=verdict,
        verdict_rationale=rationale,
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(contract: LotDownstreamGateV1) -> str:
    spec = contract.issue
    lines = [
        f"# {spec.linear_issue} {spec.alias} — Downstream-gate disposition ({contract.contract_id})",
        "",
        f"Verdict: **{contract.verdict}**",
        "",
        contract.verdict_rationale,
        "",
        "## Upstream chain",
        "",
        f"- LOT1-02 launch gate (SLM-251): `{contract.lot1_02_disposition}`",
        f"- LOT1-01 activation gate (SLM-250): `{contract.lot1_01_disposition}`",
        f"- Activation requirement: {spec.activation_requirement}",
        "",
        "## Non-goals honored",
        "",
        "No training campaign, factorial, readout/decoder, intervention, "
        "systems measurement, or production default change. This disposition "
        "is itself the deliverable while the upstream activation gate is unmet.",
        "",
    ]
    return "\n".join(lines)
