"""OFREP-shaped request/response helpers (OpenFeature Remote Evaluation Protocol)."""

from __future__ import annotations

from typing import Any, Mapping

from slm_training.flags.api import EvaluationContext, FlagClient
from slm_training.flags.apply import evaluate_levers
from slm_training.flags.levers import LEVER_BY_KEY


def context_from_ofrep(payload: Mapping[str, Any] | None) -> EvaluationContext:
    raw = dict(payload or {})
    targeting = raw.pop("targetingKey", None) or raw.pop("targeting_key", None)
    # Remaining keys are evaluation attributes.
    return EvaluationContext(targeting_key=targeting, attributes=raw)


def evaluate_ofrep(
    client: FlagClient,
    *,
    context_payload: Mapping[str, Any] | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    """Return an OFREP bulk-evaluation style payload.

    Shape (subset of OFREP):
    ``{"flags": {"flagKey": {"value", "reason", "variant?", "metadata?", ...}}}``
    """
    context = context_from_ofrep(context_payload)
    if flags:
        unknown = [k for k in flags if k not in LEVER_BY_KEY]
    else:
        unknown = []
        flags = list(LEVER_BY_KEY)
    details = evaluate_levers(client, context=context, keys=flags)
    out: dict[str, Any] = {
        "flags": {d.flag_key: d.to_ofrep() for d in details},
    }
    if unknown:
        out["errors"] = [
            {"flagKey": key, "errorCode": "FLAG_NOT_FOUND"} for key in unknown
        ]
    return out
