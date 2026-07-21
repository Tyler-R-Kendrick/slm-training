"""Fail-closed defaults for product feature flags."""

from __future__ import annotations

from typing import Any

from slm_training.features.keys import (
    DASHBOARD_DEFAULT_RENDERER,
    PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
    VSS_DECODE_ENABLED,
)

PRODUCT_FLAG_DEFAULTS: dict[str, Any] = {
    DASHBOARD_DEFAULT_RENDERER: "interpreted",
    VSS_DECODE_ENABLED: False,
    PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT: True,
}
