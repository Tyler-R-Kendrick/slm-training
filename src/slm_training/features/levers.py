"""Product experiment flags and their rollout metadata.

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
class ProductFeatureFlag:
    """Metadata for one canonical OpenFeature flag."""

    key: str
    kind: LeverKind
    description: str
    matrix_ref: str | None = None
    provider_affinity: ProviderAffinity = "any"
    attributes: dict[str, Any] | None = None


# ponytail: static registry; extend when a matrix lever gains a runtime flag.
PRODUCT_FEATURE_FLAGS: tuple[ProductFeatureFlag, ...] = (
    ProductFeatureFlag(
        key=DASHBOARD_DEFAULT_RENDERER,
        kind="ui",
        description="Default compiled vs interpreted dashboard renderer for new sessions.",
        provider_affinity="any",
    ),
    ProductFeatureFlag(
        key=VSS_DECODE_ENABLED,
        kind="decode",
        description="Enable verified-scope-solver participation in decode (VSS roadmap).",
        matrix_ref="docs/design/verified-scope-solver.md",
        provider_affinity="launchdarkly",
    ),
    ProductFeatureFlag(
        key=PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
        kind="ui",
        description="Default grammar-constrained generation in the playground.",
        provider_affinity="posthog",
    ),
    ProductFeatureFlag(
        key="decode.grammar-ltr-repair",
        kind="decode",
        description="Roll out grammar LTR repair decode lever (quality matrix E1).",
        matrix_ref="docs/design/quality-experiment-matrix.md#e1",
        provider_affinity="launchdarkly",
    ),
    ProductFeatureFlag(
        key="decode.schema-in-context",
        kind="decode",
        description="Roll out schema-in-context decode lever (quality matrix E4).",
        matrix_ref="docs/design/quality-experiment-matrix.md#e4",
        provider_affinity="launchdarkly",
    ),
)


def feature_flag_registry_payload() -> dict[str, Any]:
    return {
        "flags": [
            {
                "key": flag.key,
                "kind": flag.kind,
                "description": flag.description,
                "matrix_ref": flag.matrix_ref,
                "provider_affinity": flag.provider_affinity,
            }
            for flag in PRODUCT_FEATURE_FLAGS
        ]
    }


def feature_flag_by_key(key: str) -> ProductFeatureFlag | None:
    for flag in PRODUCT_FEATURE_FLAGS:
        if flag.key == key:
            return flag
    return None
