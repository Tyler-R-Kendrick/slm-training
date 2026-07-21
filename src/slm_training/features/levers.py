"""Product experiment levers — bridge ML matrix levers to OpenFeature flags.

ML quality-matrix rows (E*, X*, P*) are batch training ablations. Product levers
here are the *runtime rollout* surface: the same hypothesis may eventually ship
behind an OpenFeature flag for gradual exposure (LaunchDarkly, PostHog, or
in-memory). Ship gates and matrix definitions are never replaced by flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from slm_training.features.keys import (
    DASHBOARD_DEFAULT_RENDERER,
    PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
    VSS_DECODE_ENABLED,
)

LeverKind = Literal["ui", "decode", "data", "training_rollout"]
ProviderAffinity = Literal["any", "launchdarkly", "posthog", "in_memory"]


@dataclass(frozen=True)
class ProductExperimentLever:
    """Maps a research lever id to its product-flag rollout contract."""

    lever_id: str
    flag_key: str
    kind: LeverKind
    description: str
    matrix_ref: str | None = None
    provider_affinity: ProviderAffinity = "any"
    attributes: dict[str, Any] | None = None


# ponytail: static registry; extend when a matrix lever gains a runtime flag.
PRODUCT_EXPERIMENT_LEVERS: tuple[ProductExperimentLever, ...] = (
    ProductExperimentLever(
        lever_id="dashboard-renderer",
        flag_key=DASHBOARD_DEFAULT_RENDERER,
        kind="ui",
        description="Default compiled vs interpreted dashboard renderer for new sessions.",
        provider_affinity="any",
    ),
    ProductExperimentLever(
        lever_id="vss-decode",
        flag_key=VSS_DECODE_ENABLED,
        kind="decode",
        description="Enable verified-scope-solver participation in decode (VSS roadmap).",
        matrix_ref="docs/design/verified-scope-solver.md",
        provider_affinity="launchdarkly",
    ),
    ProductExperimentLever(
        lever_id="playground-grammar-default",
        flag_key=PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
        kind="ui",
        description="Default grammar-constrained generation in the playground.",
        provider_affinity="posthog",
    ),
    ProductExperimentLever(
        lever_id="E1-constrained-decode",
        flag_key="decode.grammar-ltr-repair",
        kind="decode",
        description="Roll out grammar LTR repair decode lever (quality matrix E1).",
        matrix_ref="docs/design/quality-experiment-matrix.md#e1",
        provider_affinity="launchdarkly",
    ),
    ProductExperimentLever(
        lever_id="E4-schema-conditioning",
        flag_key="decode.schema-in-context",
        kind="decode",
        description="Roll out schema-in-context decode lever (quality matrix E4).",
        matrix_ref="docs/design/quality-experiment-matrix.md#e4",
        provider_affinity="launchdarkly",
    ),
)


def lever_registry_payload() -> dict[str, Any]:
    return {
        "levers": [
            {
                "lever_id": lever.lever_id,
                "flag_key": lever.flag_key,
                "kind": lever.kind,
                "description": lever.description,
                "matrix_ref": lever.matrix_ref,
                "provider_affinity": lever.provider_affinity,
            }
            for lever in PRODUCT_EXPERIMENT_LEVERS
        ]
    }


def lever_by_flag(flag_key: str) -> ProductExperimentLever | None:
    for lever in PRODUCT_EXPERIMENT_LEVERS:
        if lever.flag_key == flag_key:
            return lever
    return None
