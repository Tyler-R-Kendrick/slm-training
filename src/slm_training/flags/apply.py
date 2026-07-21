"""Apply OpenFeature-resolved experiment levers onto ModelBuildConfig."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from slm_training.flags.api import (
    EvaluationContext,
    EvaluationDetails,
    FlagClient,
    FlagValueType,
    Reason,
)
from slm_training.flags.levers import LEVER_BY_KEY, LEVER_FLAGS


def experiment_context(
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
    matrix: str | None = None,
    model_name: str | None = None,
    context_backend: str | None = None,
    **extra: Any,
) -> EvaluationContext:
    attrs: dict[str, Any] = dict(extra)
    if experiment_id is not None:
        attrs["experiment_id"] = experiment_id
    if matrix is not None:
        attrs["matrix"] = matrix
    if model_name is not None:
        attrs["model_name"] = model_name
    if context_backend is not None:
        attrs["context_backend"] = context_backend
    return EvaluationContext(targeting_key=run_id, attributes=attrs)


def evaluate_levers(
    client: FlagClient,
    *,
    context: EvaluationContext | None = None,
    keys: list[str] | None = None,
) -> list[EvaluationDetails]:
    """Evaluate registered levers (or an explicit key subset)."""
    specs = LEVER_FLAGS if keys is None else [
        LEVER_BY_KEY[k] for k in keys if k in LEVER_BY_KEY
    ]
    details: list[EvaluationDetails] = []
    for spec in specs:
        if spec.value_type is FlagValueType.BOOLEAN:
            details.append(
                client.get_boolean_details(spec.key, bool(spec.default), context)
            )
        elif spec.value_type is FlagValueType.STRING:
            details.append(
                client.get_string_details(spec.key, str(spec.default), context)
            )
        elif spec.value_type is FlagValueType.NUMBER:
            details.append(
                client.get_number_details(spec.key, float(spec.default), context)
            )
        else:
            details.append(
                client.get_object_details(spec.key, dict(spec.default or {}), context)
            )
    return details


def apply_experiment_flags(
    config: Any,
    *,
    client: FlagClient | None = None,
    context: EvaluationContext | None = None,
    overrides: Mapping[str, Any] | None = None,
    only_reasons: frozenset[Reason] | None = None,
) -> tuple[Any, list[EvaluationDetails]]:
    """Overlay OpenFeature lever values onto ``config``.

    Precedence: ``overrides`` > provider evaluation > existing config field.

    By default only non-DEFAULT evaluations mutate the config (so a missing
    ruleset leaves ``ModelBuildConfig`` defaults untouched / byte-identical).
    Pass ``only_reasons=None`` to apply every evaluated value including defaults.
    """
    if client is None:
        return config, []
    if not is_dataclass(config):
        raise TypeError("apply_experiment_flags expects a dataclass config")

    apply_reasons = (
        frozenset({Reason.STATIC, Reason.TARGETING_MATCH, Reason.SPLIT})
        if only_reasons is None
        else only_reasons
    )
    field_names = {f.name for f in fields(config)}
    details = evaluate_levers(client, context=context)
    applied: list[EvaluationDetails] = []
    override_keys = set(overrides or {})

    for detail in details:
        key = detail.flag_key
        if key not in field_names:
            continue
        if key in override_keys:
            setattr(config, key, overrides[key])  # type: ignore[index]
            applied.append(
                EvaluationDetails(
                    flag_key=key,
                    value=overrides[key],  # type: ignore[index]
                    variant=_variant(overrides[key]),  # type: ignore[index]
                    reason=Reason.STATIC,
                    flag_metadata={
                        **dict(detail.flag_metadata),
                        "source": "override",
                    },
                )
            )
            continue
        if detail.reason not in apply_reasons:
            continue
        if detail.error_code is not None and detail.reason is Reason.ERROR:
            continue
        setattr(config, key, detail.value)
        applied.append(detail)
    return config, applied


def assignments_payload(details: list[EvaluationDetails]) -> list[dict[str, Any]]:
    """JSON-serializable assignment records for train/eval summaries."""
    return [
        {
            "flag_key": d.flag_key,
            "value": d.value,
            "variant": d.variant,
            "reason": d.reason.value,
            "metadata": dict(d.flag_metadata),
            "error_code": d.error_code.value if d.error_code else None,
        }
        for d in details
    ]


def _variant(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)
