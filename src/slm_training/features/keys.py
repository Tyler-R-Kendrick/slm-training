"""Canonical OpenFeature flag keys for product experiments."""

from __future__ import annotations

from typing import Final

DASHBOARD_DEFAULT_RENDERER: Final = "dashboard.default-renderer"
VSS_DECODE_ENABLED: Final = "vss.decode-enabled"
PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT: Final = "playground.grammar-constrained-default"

PRODUCT_FLAG_KEYS: Final[tuple[str, ...]] = (
    DASHBOARD_DEFAULT_RENDERER,
    VSS_DECODE_ENABLED,
    PLAYGROUND_GRAMMAR_CONSTRAINED_DEFAULT,
)
