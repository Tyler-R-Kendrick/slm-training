"""OpenFeature representation of typed autoresearch experiments.

Two industry-standard surfaces over the existing ``ExperimentSpec`` /
``ExperimentKnobs`` system (never a parallel experiment store):

1. ``export_flagd_flags`` — serialize a set of experiments as a flagd flag
   definition document (https://flagd.dev/schema/v0/flags.json). One flag per
   allowlisted knob; one variant per experiment that sets it; targeting on the
   ``experiment_id`` evaluation-context attribute. Consumable by flagd and any
   OpenFeature provider in any language, with no extra dependency here.
2. ``ExperimentFlagProvider`` — an OpenFeature ``AbstractProvider`` backed by
   the same specs, so in-process consumers evaluate knobs through the standard
   OpenFeature client API. Requires the optional ``openfeature-sdk`` dependency
   (``pip install slm-training[openfeature]``).

Both surfaces are fail-closed: only ``ExperimentKnobs`` fields are flags, an
unset knob falls back to the caller's code default (matching the ``None``
semantics of ``ExperimentKnobs``), and unknown flags or experiments are errors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slm_training.autoresearch.schemas import ExperimentKnobs, ExperimentSpec

FLAGD_SCHEMA = "https://flagd.dev/schema/v0/flags.json"
EXPERIMENT_CONTEXT_ATTRIBUTE = "experiment_id"
KNOB_FLAG_KEYS = frozenset(ExperimentKnobs.model_fields)


def _unique_experiments(
    experiments: Iterable[ExperimentSpec],
) -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {}
    for spec in experiments:
        if spec.experiment_id in specs:
            raise ValueError(f"duplicate experiment_id: {spec.experiment_id}")
        specs[spec.experiment_id] = spec
    if not specs:
        raise ValueError("at least one experiment is required")
    return specs


def export_flagd_flags(
    experiments: Iterable[ExperimentSpec],
    *,
    flag_set_id: str,
) -> dict[str, Any]:
    """Render experiments as a flagd flag definition document.

    Flag key = knob field name; variant name = experiment id; targeting
    resolves the ``experiment_id`` context attribute to its variant and falls
    back to the code default (``defaultVariant: null``) when the targeted
    experiment does not set the knob — exactly the ``None`` knob semantics.
    """
    specs = _unique_experiments(experiments)
    flags: dict[str, Any] = {}
    for experiment_id, spec in sorted(specs.items()):
        for knob, value in sorted(
            spec.knobs.model_dump(exclude_none=True, mode="json").items()
        ):
            flag = flags.setdefault(
                knob,
                {"state": "ENABLED", "variants": {}, "defaultVariant": None},
            )
            flag["variants"][experiment_id] = value
    for flag in flags.values():
        members = sorted(flag["variants"])
        flag["targeting"] = {
            "if": [
                {"in": [{"var": EXPERIMENT_CONTEXT_ATTRIBUTE}, members]},
                {"var": EXPERIMENT_CONTEXT_ATTRIBUTE},
                None,
            ]
        }
    return {
        "$schema": FLAGD_SCHEMA,
        "flags": dict(sorted(flags.items())),
        "metadata": {"flagSetId": flag_set_id},
    }


try:  # pragma: no cover - exercised via both installed/uninstalled test paths
    from openfeature.evaluation_context import EvaluationContext
    from openfeature.exception import (
        FlagNotFoundError,
        InvalidContextError,
        TargetingKeyMissingError,
        TypeMismatchError,
    )
    from openfeature.flag_evaluation import FlagResolutionDetails, Reason
    from openfeature.provider import AbstractProvider, Metadata

    _OPENFEATURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENFEATURE_AVAILABLE = False


if _OPENFEATURE_AVAILABLE:

    class ExperimentFlagProvider(AbstractProvider):
        """OpenFeature provider resolving knob flags against experiment specs.

        The evaluation context selects the experiment: the
        ``experiment_id`` attribute (or the ``targeting_key``) must name one
        of the provider's experiments. Resolution is fail-closed and mirrors
        the typed-knob contract: unknown flag keys raise ``FLAG_NOT_FOUND``,
        wrong requested types raise ``TYPE_MISMATCH``, and unset knobs return
        the caller's code default with reason ``DEFAULT``.
        """

        def __init__(self, experiments: Iterable[ExperimentSpec]) -> None:
            self._experiments = _unique_experiments(experiments)

        @classmethod
        def from_matrix(cls, matrix: Any) -> "ExperimentFlagProvider":
            """Build a provider from a ``HypothesisMatrix``."""
            return cls(item.experiment for item in matrix.hypotheses)

        def get_metadata(self) -> Metadata:
            return Metadata(name="openui-experiment-provider")

        def _experiment(
            self, evaluation_context: EvaluationContext | None
        ) -> ExperimentSpec:
            context = evaluation_context or EvaluationContext()
            experiment_id = context.attributes.get(
                EXPERIMENT_CONTEXT_ATTRIBUTE
            ) or context.targeting_key
            if not experiment_id:
                raise TargetingKeyMissingError(
                    "evaluation context must set experiment_id or targeting_key"
                )
            spec = self._experiments.get(str(experiment_id))
            if spec is None:
                raise InvalidContextError(f"unknown experiment: {experiment_id}")
            return spec

        def _resolve(
            self,
            flag_key: str,
            default_value: Any,
            evaluation_context: EvaluationContext | None,
            accepts: tuple[type, ...],
            rejects: tuple[type, ...] = (),
        ) -> FlagResolutionDetails[Any]:
            if flag_key not in KNOB_FLAG_KEYS:
                raise FlagNotFoundError(f"not an allowlisted knob: {flag_key}")
            spec = self._experiment(evaluation_context)
            value = getattr(spec.knobs, flag_key)
            if value is None:
                return FlagResolutionDetails(
                    value=default_value, reason=Reason.DEFAULT
                )
            if isinstance(value, rejects) or not isinstance(value, accepts):
                raise TypeMismatchError(
                    f"{flag_key} is {type(value).__name__}, not the requested type"
                )
            return FlagResolutionDetails(
                value=value,
                variant=spec.experiment_id,
                reason=Reason.TARGETING_MATCH,
            )

        def resolve_boolean_details(
            self,
            flag_key: str,
            default_value: bool,
            evaluation_context: EvaluationContext | None = None,
        ) -> FlagResolutionDetails[bool]:
            return self._resolve(
                flag_key, default_value, evaluation_context, accepts=(bool,)
            )

        def resolve_string_details(
            self,
            flag_key: str,
            default_value: str,
            evaluation_context: EvaluationContext | None = None,
        ) -> FlagResolutionDetails[str]:
            return self._resolve(
                flag_key, default_value, evaluation_context, accepts=(str,)
            )

        def resolve_integer_details(
            self,
            flag_key: str,
            default_value: int,
            evaluation_context: EvaluationContext | None = None,
        ) -> FlagResolutionDetails[int]:
            return self._resolve(
                flag_key,
                default_value,
                evaluation_context,
                accepts=(int,),
                rejects=(bool,),
            )

        def resolve_float_details(
            self,
            flag_key: str,
            default_value: float,
            evaluation_context: EvaluationContext | None = None,
        ) -> FlagResolutionDetails[float]:
            details = self._resolve(
                flag_key,
                default_value,
                evaluation_context,
                accepts=(int, float),
                rejects=(bool,),
            )
            details.value = float(details.value)
            return details

        def resolve_object_details(
            self,
            flag_key: str,
            default_value: dict | list,
            evaluation_context: EvaluationContext | None = None,
        ) -> FlagResolutionDetails[dict | list]:
            return self._resolve(
                flag_key, default_value, evaluation_context, accepts=(dict, list)
            )

else:  # pragma: no cover - only reachable without the optional dependency

    class ExperimentFlagProvider:  # type: ignore[no-redef]
        """Placeholder raising until ``slm-training[openfeature]`` is installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "openfeature-sdk is not installed; "
                "install slm-training[openfeature] to use ExperimentFlagProvider"
            )

        @classmethod
        def from_matrix(cls, matrix: Any) -> "ExperimentFlagProvider":
            return cls(matrix)
