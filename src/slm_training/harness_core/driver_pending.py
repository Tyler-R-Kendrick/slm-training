"""Canonical typed driver yield; never a scientific completion receipt."""

from slm_training.harness_core.activity_contract import ActivityOutcome, WakeCondition


def validate_pending(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != "driver_pending/v1":
        raise ValueError("unknown driver pending contract")
    outcome = ActivityOutcome(payload["outcome"])
    if outcome not in {ActivityOutcome.CAPABILITY, ActivityOutcome.DEPENDENCY, ActivityOutcome.YIELDED}:
        raise ValueError("pending driver output cannot authorize a terminal outcome")
    if not payload.get("reason") or payload.get("measurement_complete") is not False:
        raise ValueError("driver pending requires an unmet predicate, not a measurement")
    return outcome, WakeCondition.model_validate(payload["wake"])

