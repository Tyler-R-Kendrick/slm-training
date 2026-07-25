"""Generic DCA-chain activation-gate evaluator (fail-closed, no-spend).

Every DCA0/DCA1/DCA2 issue (SLM-270 … SLM-276) rides on the same three
prerequisite evidence legs:

* SLM-261 — LossLedgerV1 / trainer loss accounting
  (``docs/design/training-loss-ledger.md``);
* SLM-268 — selected verified-data regime or explicit large-scale negative
  result (``docs/design/iter-slm268-verified-data-scaling-activation-20260725.json``);
* SLM-269 — selected target policy or no-effect disposition
  (``docs/design/iter-slm269-dca0-01-activation-20260725.json``).

The artifacts are audited, not trusted because the Linear issues are Done.
When any leg is missing/blocked/inconclusive, the downstream DCA issue
closes ``blocked``/``inconclusive`` with successor conditions; no training,
surgery, curriculum, or tournament is launched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "DCA_ISSUES",
    "GATE_SCHEMA_VERSION",
    "PENDING_CAMPAIGN",
    "DcaActivationGateV1",
    "DcaIssueSpec",
    "evaluate_dca_gate",
    "render_markdown",
]

GATE_SCHEMA_VERSION = "dca_activation_gate/v1"

#: Verdict when every prerequisite leg is satisfied: the issue's own
#: preregistered campaign/matrix must still run before any result claim.
PENDING_CAMPAIGN = "prerequisites_met_pending_campaign"

LOSS_LEDGER_PATH = "docs/design/training-loss-ledger.md"
VSD2_ACTIVATION_PATH = (
    "docs/design/iter-slm268-verified-data-scaling-activation-20260725.json"
)
DCA0_ACTIVATION_PATH = "docs/design/iter-slm269-dca0-01-activation-20260725.json"


@dataclass(frozen=True)
class DcaIssueSpec:
    linear_issue: str
    alias: str
    title: str
    gate_contract_id: str
    slug: str
    activation_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


DCA_ISSUES: dict[str, DcaIssueSpec] = {
    "SLM-270": DcaIssueSpec(
        linear_issue="SLM-270",
        alias="DCA0-02",
        title="Run principal-only, every single auxiliary, full-stack, and full-minus-one TwoTower objective surgery",
        gate_contract_id="objective-surgery-activation-gate-v1",
        slug="dca0-02-objective-surgery",
        activation_requirement=(
            "SLM-261 complete loss ledger and trainer validity; SLM-268 "
            "selected verified data regime/control; SLM-269 selected target "
            "policy or no-effect disposition."
        ),
    ),
}


@dataclass(frozen=True)
class DcaActivationGateV1:
    schema_version: str = GATE_SCHEMA_VERSION
    contract_id: str = ""
    issue: DcaIssueSpec | None = None
    prerequisite_verdicts: dict[str, str] = field(default_factory=dict)
    verdict: str = "blocked"
    disposition: str = "inconclusive"
    verdict_rationale: str = ""
    successor_conditions: list[str] = field(default_factory=list)
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
            "prerequisite_verdicts": self.prerequisite_verdicts,
            "verdict": self.verdict,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def _audit_prerequisites(root: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}

    ledger = root / LOSS_LEDGER_PATH
    verdicts["SLM-261_loss_ledger"] = "qualified" if ledger.is_file() else "missing_or_unqualified"

    vsd2 = root / VSD2_ACTIVATION_PATH
    if not vsd2.is_file():
        verdicts["SLM-268_data_regime"] = "missing"
    else:
        status = json.loads(vsd2.read_text(encoding="utf-8")).get("activation_status", "")
        verdicts["SLM-268_data_regime"] = (
            "selected_or_negative_result" if status != "blocked" else "blocked"
        )

    dca0 = root / DCA0_ACTIVATION_PATH
    if not dca0.is_file():
        verdicts["SLM-269_target_policy"] = "missing"
    else:
        report = json.loads(dca0.read_text(encoding="utf-8"))
        status = report.get("activation_status", "")
        disposition = report.get("disposition", "")
        verdicts["SLM-269_target_policy"] = (
            "selected_or_no_effect"
            if status != "blocked" and disposition != "inconclusive"
            else "blocked_or_inconclusive"
        )
    return verdicts


_QUALIFIED = {
    "SLM-261_loss_ledger": "qualified",
    "SLM-268_data_regime": "selected_or_negative_result",
    "SLM-269_target_policy": "selected_or_no_effect",
}


def evaluate_dca_gate(spec: DcaIssueSpec, root: Path) -> DcaActivationGateV1:
    """Evaluate one DCA issue's activation against the real prerequisite artifacts."""
    verdicts = _audit_prerequisites(root)
    unmet = [leg for leg, verdict in verdicts.items() if verdict != _QUALIFIED[leg]]

    if unmet:
        verdict = "blocked"
        rationale = (
            f"{spec.alias} ({spec.linear_issue}) activation requires: "
            f"{spec.activation_requirement} Audited prerequisite artifacts "
            "are unqualified: "
            + ", ".join(f"{leg}={verdicts[leg]}" for leg in unmet)
            + ". No training, surgery, curriculum, or tournament is launched; "
            "closing blocked/inconclusive."
        )
    else:
        verdict = PENDING_CAMPAIGN
        rationale = (
            "All prerequisite legs are qualified. The issue's own "
            "preregistered campaign must still run before any result claim; "
            "this gate authorizes no quality or default-change claim."
        )

    return DcaActivationGateV1(
        contract_id=spec.gate_contract_id,
        issue=spec,
        prerequisite_verdicts=verdicts,
        verdict=verdict,
        disposition="inconclusive" if unmet else "pending",
        verdict_rationale=rationale,
        successor_conditions=[
            "qualified LossLedgerV1 artifact (SLM-261 successor)",
            "SLM-268 successor with activation_status != blocked (selected regime or negative result)",
            "SLM-269 successor with a selected target policy or no-effect disposition",
            "then re-run scripts/audit_dca_activation --issue "
            f"{spec.linear_issue} (reads artifacts live) and re-file the campaign",
        ]
        if unmet
        else [],
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(contract: DcaActivationGateV1) -> str:
    spec = contract.issue
    lines = [
        f"# {spec.linear_issue} {spec.alias} — DCA activation gate ({contract.contract_id})",
        "",
        f"Verdict: **{contract.verdict}** (disposition: `{contract.disposition}`)",
        "",
        contract.verdict_rationale,
        "",
        "## Prerequisite legs",
        "",
    ]
    lines += [f"- `{leg}`: `{verdict}`" for leg, verdict in contract.prerequisite_verdicts.items()]
    if contract.successor_conditions:
        lines += ["", "## Successor conditions", ""]
        lines += [f"- {c}" for c in contract.successor_conditions]
    lines += [
        "",
        "## Non-goals honored",
        "",
        "No training, objective surgery, curriculum, tournament, checkpoint, "
        "or production default change. This disposition is itself the "
        "deliverable while the prerequisites are unqualified.",
        "",
    ]
    return "\n".join(lines)
