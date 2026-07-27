"""SLM-251 LOT1-02 launch-gate evaluator (plan-only closeout).

LOT1-02 ("Run the explicit-to-latent curriculum against an update-matched
continued explicit-trace control") declares three upstream dependencies:

1. SLM-248 (LOT0-01) transfer authorization -- preregistered margins and a
   ``authorize_bounded_implementation`` verdict.
2. SLM-249 (LOT0-02) ``CompilerReasoningTraceGateV1`` -- K/c/stage support
   and a positive oracle ceiling.
3. SLM-250 (LOT1-01) -- the faithful K x c looped-latent model path and
   curriculum hooks.

The campaign cannot launch honestly unless every dependency is authorized:
there is no treatment arm without the LOT1-01 model path, and no causal
attribution without the SLM-249 oracle ceiling. This module derives the
LOT1-02 launch disposition from the same real, committed upstream contract
artifacts as the LOT1-01 gate (reusing
:mod:`slm_training.harnesses.experiments.lot1_01_activation_gate`) plus the
trace contract's own ``allowed_lot1_implementation`` field. It contains no
training, curriculum, or model code -- evaluating this gate honestly *is*
the LOT1-02 deliverable while the prerequisites are unmet.
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
    GateEvaluation,
    evaluate_activation_gates,
)

__all__ = [
    "GATE_SCHEMA_VERSION",
    "LAUNCH_GATE_CONTRACT_ID",
    "Lot102LaunchGateV1",
    "evaluate_lot1_02_launch_gate",
    "render_markdown",
]

GATE_SCHEMA_VERSION = "lot1_02_launch_gate/v1"
LAUNCH_GATE_CONTRACT_ID = "lot1-02-launch-gate-v1"


@dataclass(frozen=True)
class Lot102LaunchGateV1:
    """LOT1-02 launch disposition derived from real upstream contracts."""

    schema_version: str = GATE_SCHEMA_VERSION
    contract_id: str = LAUNCH_GATE_CONTRACT_ID
    linear_issue: str = "SLM-251"
    lot1_01_disposition: str = NOT_AUTHORIZED
    gate_1_transfer_authorization: GateEvaluation | None = None
    gate_2_trace_oracle_ceiling: GateEvaluation | None = None
    gate_3_explicit_kc_stage_authorization: GateEvaluation | None = None
    verdict: str = NOT_AUTHORIZED
    verdict_rationale: str = ""
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        for key in (
            "gate_1_transfer_authorization",
            "gate_2_trace_oracle_ceiling",
            "gate_3_explicit_kc_stage_authorization",
        ):
            gate = getattr(self, key)
            data[key] = gate.to_dict() if gate is not None else None
        data["contract_hash"] = self.contract_hash()
        return data

    def contract_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "linear_issue": self.linear_issue,
            "lot1_01_disposition": self.lot1_01_disposition,
            "verdict": self.verdict,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def _evaluate_kc_stage_authorization(trace_gate_contract: dict[str, Any]) -> GateEvaluation:
    """Check SLM-249's own explicit LOT1 implementation allowance.

    ``allowed_lot1_implementation`` is a list whose only honest negative
    entry starts with ``none``. Any other entry is an explicit authorization
    carrying K/c/stage targets.
    """
    gate = trace_gate_contract.get("gate") or {}
    allowed = gate.get("allowed_lot1_implementation") or []
    explicit = [entry for entry in allowed if not str(entry).strip().lower().startswith("none")]
    met = bool(explicit)
    return GateEvaluation(
        gate_name="explicit_kc_stage_authorization",
        source_contract_id=trace_gate_contract.get("contract_id", ""),
        source_contract_hash=trace_gate_contract.get("contract_hash", ""),
        source_linear_issue=gate.get("linear_issue", "SLM-249"),
        required_verdict="explicit allowed_lot1_implementation entry with K/c/stage targets",
        actual_verdict="; ".join(str(e) for e in allowed) or "<missing>",
        met=met,
        rationale=(
            "SLM-249 explicitly authorizes LOT1 implementation: " + "; ".join(explicit)
            if met
            else "SLM-249's own allowed_lot1_implementation field authorizes no LOT1 "
            "implementation; the oracle ceiling is inconclusive and K/c is a bounded "
            "probe recommendation only, not a data-derived production budget."
        ),
    )


def evaluate_lot1_02_launch_gate(
    fidelity_contract: dict[str, Any],
    trace_gate_contract: dict[str, Any],
) -> Lot102LaunchGateV1:
    """Evaluate LOT1-02's launch prerequisites against real upstream contracts.

    Reuses the LOT1-01 evaluator for the shared transfer-authorization and
    oracle-ceiling gates, then adds LOT1-02's own requirement that SLM-249
    explicitly authorize LOT1 implementation. The verdict is
    ``not_authorized`` unless every prerequisite is met; even a met gate
    authorizes only wiring/fixture evidence, never a trained-quality claim.
    """
    lot1_01 = evaluate_activation_gates(fidelity_contract, trace_gate_contract)
    gate_3 = _evaluate_kc_stage_authorization(trace_gate_contract)

    all_met = (
        lot1_01.gate_1_fidelity_authorization.met
        and lot1_01.gate_2_trace_oracle_ceiling.met
        and gate_3.met
    )
    if all_met:
        verdict = AUTHORIZED_WIRING_ONLY
        rationale = (
            "All LOT1-02 launch prerequisites are met. The curriculum "
            "campaign may be wired in fixture/plan-only mode; this gate "
            "authorizes no GPU run or semantic-quality claim by itself."
        )
    else:
        verdict = NOT_AUTHORIZED
        unmet = []
        if not lot1_01.gate_1_fidelity_authorization.met:
            unmet.append("transfer_authorization (SLM-248)")
        if not lot1_01.gate_2_trace_oracle_ceiling.met:
            unmet.append("trace_oracle_ceiling (SLM-249)")
        if not gate_3.met:
            unmet.append("explicit_kc_stage_authorization (SLM-249 allowed_lot1_implementation)")
        rationale = (
            "LOT1-02 launch prerequisite(s) unmet: " + ", ".join(unmet) + ". "
            f"LOT1-01's disposition is '{lot1_01.verdict}', so no faithful "
            "K x c model path or curriculum hooks exist to train; SLM-249's "
            "oracle ceiling is not positive and authorizes no LOT1 "
            "implementation. There is no treatment arm to run and no matched "
            "continued-explicit control to attribute against. Closing "
            "not_authorized in plan-only mode; no training, curriculum, or "
            "model code is added by this disposition."
        )

    return Lot102LaunchGateV1(
        lot1_01_disposition=lot1_01.verdict,
        gate_1_transfer_authorization=lot1_01.gate_1_fidelity_authorization,
        gate_2_trace_oracle_ceiling=lot1_01.gate_2_trace_oracle_ceiling,
        gate_3_explicit_kc_stage_authorization=gate_3,
        verdict=verdict,
        verdict_rationale=rationale,
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(contract: Lot102LaunchGateV1) -> str:
    lines = [
        f"# SLM-251 LOT1-02 — Launch-gate disposition ({contract.contract_id})",
        "",
        f"Verdict: **{contract.verdict}**",
        "",
        contract.verdict_rationale,
        "",
        f"LOT1-01 disposition: `{contract.lot1_01_disposition}`",
        "",
    ]
    for title, gate in (
        ("Gate 1 — transfer authorization (SLM-248)", contract.gate_1_transfer_authorization),
        ("Gate 2 — trace oracle ceiling (SLM-249)", contract.gate_2_trace_oracle_ceiling),
        (
            "Gate 3 — explicit LOT1 implementation allowance (SLM-249)",
            contract.gate_3_explicit_kc_stage_authorization,
        ),
    ):
        if gate is None:
            continue
        lines += [
            f"## {title}",
            "",
            f"- Source contract: `{gate.source_contract_id}` (`{gate.source_contract_hash}`)",
            f"- Required: {gate.required_verdict}",
            f"- Actual verdict: `{gate.actual_verdict}`",
            f"- Met: **{gate.met}**",
            "",
        ]
    lines += [
        "## Non-goals honored",
        "",
        "No training campaign, no curriculum code, no Stage 0 checkpoint, no "
        "GPU dispatch, and no production default change. This disposition is "
        "itself the LOT1-02 deliverable while the launch prerequisites are unmet.",
        "",
    ]
    return "\n".join(lines)
