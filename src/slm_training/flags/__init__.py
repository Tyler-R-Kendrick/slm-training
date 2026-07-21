"""OpenFeature-compatible evaluation for experiment levers.

See ``docs/design/openfeature-experiments.md``.
"""

from __future__ import annotations

from slm_training.flags.api import (
    ErrorCode,
    EvaluationContext,
    EvaluationDetails,
    FlagClient,
    FlagProvider,
    FlagValueType,
    Reason,
)
from slm_training.flags.apply import (
    apply_experiment_flags,
    assignments_payload,
    experiment_context,
)
from slm_training.flags.bootstrap import client_from_environ
from slm_training.flags.in_memory import (
    FlagDefinition,
    InMemoryProvider,
    ruleset_from_defaults,
    ruleset_from_mapping,
)
from slm_training.flags.levers import LEVER_FLAGS, LeverSpec
from slm_training.flags.ofrep import evaluate_ofrep

__all__ = [
    "ErrorCode",
    "EvaluationContext",
    "EvaluationDetails",
    "FlagClient",
    "FlagDefinition",
    "FlagProvider",
    "FlagValueType",
    "InMemoryProvider",
    "LEVER_FLAGS",
    "LeverSpec",
    "Reason",
    "apply_experiment_flags",
    "assignments_payload",
    "client_from_environ",
    "evaluate_ofrep",
    "experiment_context",
    "ruleset_from_defaults",
    "ruleset_from_mapping",
]
