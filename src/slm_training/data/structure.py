"""Structure-only OpenUI helpers — strip style tokens from scaffold gold/eval.

Scaffold language = components + hierarchy + direction + placeholders.
Style (gaps, typography sizes, color-role variants) is out of scope for eval.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Quoted literals that encode visual style, not layout structure.
# Keep direction (column/row), field kinds (email/text), and Callout variants (info).
STYLE_STRING_TOKENS = frozenset(
    {
        # Stack gap scale
        "none",
        "xs",
        "s",
        "m",
        "l",
        "xl",
        "2xl",
        # Typography / control sizes
        "small",
        "large",
        "small-heavy",
        "large-heavy",
        # Color-role / emphasis variants on Button / TextContent
        "primary",
        "secondary",
        "tertiary",
    }
)

_STYLE_ARG_RE = re.compile(
    r',\s*"(?:'
    + "|".join(re.escape(t) for t in sorted(STYLE_STRING_TOKENS, key=len, reverse=True))
    + r')"'
)
_STYLE_ONLY_ARG_RE = re.compile(
    r'\(\s*"(?:'
    + "|".join(re.escape(t) for t in sorted(STYLE_STRING_TOKENS, key=len, reverse=True))
    + r')"\s*,'
)


@lru_cache(maxsize=1024)
def strip_style_literals(openui: str) -> str:
    """
    Remove style-only quoted args from OpenUI source while preserving structure.

    Pure str -> str, so the result is cached: eval metrics and leakage
    fingerprints strip the same source several times per record.

    Examples:
      Stack([a], "column", "m") → Stack([a], "column")
      TextContent(":t", "large-heavy") → TextContent(":t")
      Button(":x", "primary") → Button(":x")
    """
    if not openui:
        return openui
    text = _STYLE_ARG_RE.sub("", openui)
    # Rare leading-only style arg: Foo("primary", :x) — not used in fixtures.
    text = _STYLE_ONLY_ARG_RE.sub("(", text)
    # Collapse leftover empty arg lists artifacts like ", )" → ")"
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r",\s*,", ",", text)
    return text


def is_style_token(value: str) -> bool:
    return value.strip().strip('"') in STYLE_STRING_TOKENS
