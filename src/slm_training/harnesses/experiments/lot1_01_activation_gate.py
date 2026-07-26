"""SLM-250 LOT1-01 hard-activation-gate evaluator.

LOT1-01 ("Implement the faithful causal K x c looped-latent model path")
declares two hard activation gates against its own upstream contracts:

1. SLM-248 (LOT0-01) ``LotusOpenUIFidelityContractV1.authorization.verdict``
   must be ``authorize_bounded_implementation``.
2. SLM-249 (LOT0-02) ``CompilerReasoningTraceGateV1.gate.verdict`` must be
   ``oracle_ceiling_positive`` (or another explicit authorization that
   supplies K/c/stage targets).

"Otherwise close ``not_authorized`` in plan-only mode without production
model code." This module evaluates those two gates against the real,
committed upstream contract artifacts (``docs/design/lotus-openui-fidelity-contract-v1.json``
and ``docs/design/compiler-reasoning-trace-v1.json``) and emits the required
``LotusOpenUIModelContractV1`` disposition. It contains no model, training,
or K x c workspace code -- evaluating this gate honestly *is* the LOT1-01
deliverable when the gate is unmet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp

__all__ = [
    "AUTHORIZED_WIRING_ONLY",
    "CONTRACT_ID",
    "MODEL_CONTRACT_SCHEMA_VERSION",
    "NOT_AUTHORIZED",
    "REQUIRED_FIDELITY_VERDICT",
    "REQUIRED_TRACE_VERDICT",
    "GateEvaluation",
    "LotusOpenUIModelContractV1",
    "evaluate_activation_gates",
    "load_upstream_contract",
    "render_markdown",
]

MODEL_CONTRACT_SCHEMA_VERSION = "lotus_openui_model_contract/v1"
CONTRACT_ID = "lotus-openui-model-contract-v1"

# The exact verdict strings LOT1-01's own hard activation gates require.
REQUIRED_FIDELITY_VERDICT = "authorize_bounded_implementation"
REQUIRED_TRACE_VERDICT = "oracle_ceiling_positive"

NOT_AUTHORIZED = "not_authorized"
AUTHORIZED_WIRING_ONLY = "authorized_wiring_only"


@dataclass(frozen=True)
class GateEvaluation:
    """One hard-activation-gate check against a single upstream contract."""

    gate_name: str
    source_contract_id: str
    source_contract_hash: str
    source_linear_issue: str
    required_verdict: str
    actual_verdict: str
    met: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class LotusOpenUIModelContractV1:
    schema_version: str = MODEL_CONTRACT_SCHEMA_VERSION
    contract_id: str = CONTRACT_ID
    linear_issue: str = "SLM-250"
    gate_1_fidelity_authorization: GateEvaluation | None = None
    gate_2_trace_oracle_ceiling: GateEvaluation | None = None
    verdict: str = NOT_AUTHORIZED
    verdict_rationale: str = ""
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["gate_1_fidelity_authorization"] = (
            self.gate_1_fidelity_authorization.to_dict()
            if self.gate_1_fidelity_authorization is not None
            else None
        )
        data["gate_2_trace_oracle_ceiling"] = (
            self.gate_2_trace_oracle_ceiling.to_dict()
            if self.gate_2_trace_oracle_ceiling is not None
            else None
        )
        data["contract_hash"] = self.contract_hash()
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def contract_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "linear_issue": self.linear_issue,
            "gate_1_fidelity_authorization": (
                self.gate_1_fidelity_authorization.to_dict()
                if self.gate_1_fidelity_authorization is not None
                else None
            ),
            "gate_2_trace_oracle_ceiling": (
                self.gate_2_trace_oracle_ceiling.to_dict()
                if self.gate_2_trace_oracle_ceiling is not None
                else None
            ),
            "verdict": self.verdict,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def load_upstream_contract(path: Path) -> dict[str, Any]:
    """Load an upstream contract JSON artifact from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_fidelity_gate(fidelity_contract: dict[str, Any]) -> GateEvaluation:
    authorization = fidelity_contract.get("authorization") or {}
    actual = authorization.get("verdict", "")
    met = actual == REQUIRED_FIDELITY_VERDICT
    return GateEvaluation(
        gate_name="fidelity_contract_authorization",
        source_contract_id=fidelity_contract.get("contract_id", ""),
        source_contract_hash=fidelity_contract.get("contract_hash", ""),
        source_linear_issue=authorization.get("linear_issue", "SLM-248"),
        required_verdict=REQUIRED_FIDELITY_VERDICT,
        actual_verdict=actual,
        met=met,
        rationale=authorization.get("verdict_rationale", ""),
    )


def _evaluate_trace_gate(trace_gate_contract: dict[str, Any]) -> GateEvaluation:
    gate = trace_gate_contract.get("gate") or {}
    actual = gate.get("verdict", "")
    met = actual == REQUIRED_TRACE_VERDICT
    return GateEvaluation(
        gate_name="trace_oracle_ceiling",
        source_contract_id=trace_gate_contract.get("contract_id", ""),
        source_contract_hash=trace_gate_contract.get("contract_hash", ""),
        source_linear_issue=gate.get("linear_issue", "SLM-249"),
        required_verdict=REQUIRED_TRACE_VERDICT,
        actual_verdict=actual,
        met=met,
        rationale=gate.get("verdict_rationale", ""),
    )


def evaluate_activation_gates(
    fidelity_contract: dict[str, Any],
    trace_gate_contract: dict[str, Any],
) -> LotusOpenUIModelContractV1:
    """Evaluate LOT1-01's two hard activation gates against real upstream contracts.

    Returns a :class:`LotusOpenUIModelContractV1` whose ``verdict`` is
    ``not_authorized`` unless both upstream gates report their exact required
    verdict strings. No K x c model/training code path is implied or run by
    a positive result; per the issue text, even an authorized gate delivers
    "wiring/contract evidence only" (``authorized_wiring_only``), never a
    semantic-quality claim.
    """
    gate_1 = _evaluate_fidelity_gate(fidelity_contract)
    gate_2 = _evaluate_trace_gate(trace_gate_contract)

    if gate_1.met and gate_2.met:
        verdict = AUTHORIZED_WIRING_ONLY
        rationale = (
            "Both hard activation gates report their required verdicts. "
            "LOT1-01 may proceed in wiring/contract-evidence-only mode; "
            "this contract authorizes no training or semantic-quality claim."
        )
    else:
        verdict = NOT_AUTHORIZED
        unmet = [g.gate_name for g in (gate_1, gate_2) if not g.met]
        rationale = (
            "Hard activation gate(s) unmet: "
            + ", ".join(unmet)
            + f". Gate 1 (SLM-248 fidelity authorization) reports verdict "
            f"'{gate_1.actual_verdict}' (requires '{REQUIRED_FIDELITY_VERDICT}'). "
            f"Gate 2 (SLM-249 trace oracle ceiling) reports verdict "
            f"'{gate_2.actual_verdict}' (requires '{REQUIRED_TRACE_VERDICT}'). "
            "Closing not_authorized in plan-only mode; no K x c model or "
            "training code is added by this disposition."
        )

    return LotusOpenUIModelContractV1(
        gate_1_fidelity_authorization=gate_1,
        gate_2_trace_oracle_ceiling=gate_2,
        verdict=verdict,
        verdict_rationale=rationale,
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(contract: LotusOpenUIModelContractV1) -> str:
    g1 = contract.gate_1_fidelity_authorization
    g2 = contract.gate_2_trace_oracle_ceiling
    lines = [
        f"# SLM-250 LOT1-01 — Activation-gate disposition ({contract.contract_id})",
        "",
        f"Verdict: **{contract.verdict}**",
        "",
        contract.verdict_rationale,
        "",
        "## Gate 1 — fidelity contract authorization (SLM-248)",
        "",
        f"- Source contract: `{g1.source_contract_id}` (`{g1.source_contract_hash}`)",
        f"- Required verdict: `{g1.required_verdict}`",
        f"- Actual verdict: `{g1.actual_verdict}`",
        f"- Met: **{g1.met}**",
        "",
        "## Gate 2 — trace oracle-ceiling (SLM-249)",
        "",
        f"- Source contract: `{g2.source_contract_id}` (`{g2.source_contract_hash}`)",
        f"- Required verdict: `{g2.required_verdict}`",
        f"- Actual verdict: `{g2.actual_verdict}`",
        f"- Met: **{g2.met}**",
        "",
        "## Non-goals honored",
        "",
        "No K x c latent-workspace model code, no learned plan predictor, no "
        "GPU training, no production default change. This disposition is "
        "itself the LOT1-01 deliverable when the activation gate is unmet.",
        "",
    ]
    return "\n".join(lines)
