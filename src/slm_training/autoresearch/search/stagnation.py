"""Typed stagnation diagnoses dispatch work through the existing activity owner."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .evidence import Contract, Digest, contract_digest


class SearchSignal(Contract):
    family_identity: Digest
    evidence_identity: Digest
    cause: Literal[
        "missing_measurement",
        "invalid_data",
        "unknown_fidelity",
        "representation_ceiling",
        "powered_null",
        "complete_comparison",
        "receipt",
        "retry",
    ]
    comparison_id: Digest | None = None
    # Closure is a reference to an EXISTING authorized verdict, not a new rule.
    closure_receipt: Digest | None = None


class SearchAction(Contract):
    family_identity: Digest
    evidence_identity: Digest
    action: Literal[
        "repair_harness",
        "rebuild_data",
        "calibrate_continuation",
        "implement_mechanism",
        "apply_authorized_retirement",
    ]
    unmet_predicate: str = Field(min_length=1)
    authority_receipt: Digest | None = None


_ACTIONS = {
    "missing_measurement": ("repair_harness", "locked_measurement_complete"),
    "invalid_data": ("rebuild_data", "target_specific_data_ready"),
    "unknown_fidelity": ("calibrate_continuation", "fidelity_adequacy_measured"),
    "representation_ceiling": ("implement_mechanism", "registered_mechanism_executed"),
    "powered_null": ("apply_authorized_retirement", "existing_closure_receipt_applied"),
}


def stagnation_actions(signals) -> list[SearchAction]:
    """Logs/retries never clear a cause; only a new compatible comparison does.

    All inputs are produced by the trusted measurement/statistics controller.
    This reducer deliberately cannot manufacture an adequately powered null.
    """
    outstanding = {}
    comparisons = set()
    for raw in signals:
        signal = SearchSignal.model_validate(
            raw.model_dump() if isinstance(raw, SearchSignal) else raw
        )
        key = signal.family_identity, signal.evidence_identity
        if signal.cause == "complete_comparison":
            unit = (*key, signal.comparison_id)
            if signal.comparison_id and unit not in comparisons:
                outstanding.pop(key, None)
                comparisons.add(unit)
        elif signal.cause in _ACTIONS:
            if signal.cause == "powered_null" and signal.closure_receipt is None:
                continue
            action, predicate = _ACTIONS[signal.cause]
            outstanding[key] = SearchAction(
                family_identity=signal.family_identity,
                evidence_identity=signal.evidence_identity,
                action=action,
                unmet_predicate=predicate,
                authority_receipt=signal.closure_receipt,
            )
    return list(outstanding.values())


def dispatch_stagnation(runtime, signals, *, factories) -> dict:
    """Register real work from controller-approved factories; never invent grants.

    The action digest is bound as the immutable activity input. A missing
    executor remains a precise capability wait; unrelated jobs remain runnable.
    """
    result = {"registered": [], "waiting_capability": []}
    for action in stagnation_actions(signals):
        factory = factories.get(action.action)
        if factory is None:
            result["waiting_capability"].append(
                {
                    "action": action.action,
                    "predicate": action.unmet_predicate,
                    "identity": contract_digest(action),
                }
            )
            continue
        spec = factory(action)
        spec = type(spec).model_validate(spec.model_dump())
        if (
            spec.input_digest != contract_digest(action)
            or spec.family != action.family_identity
        ):
            raise ValueError("stagnation factory changed cause identity")
        runtime.register(spec)
        result["registered"].append(spec.activity_id)
    return result
