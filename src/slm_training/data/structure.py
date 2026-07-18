"""Structure-only OpenUI helpers — strip style tokens from scaffold gold/eval.

Scaffold language = components + hierarchy + direction + placeholders.
Style (gaps, typography sizes, color-role variants) is out of scope for eval.
"""

from __future__ import annotations

import re

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

_STYLE_TOKEN_RE = (
    r'"(?:'
    + "|".join(re.escape(t) for t in sorted(STYLE_STRING_TOKENS, key=len, reverse=True))
    + r')"'
)
_STYLE_TEXT_ARG_RE = re.compile(
    rf"(\b(?:TextContent|Button)\([^()\n]*),\s*{_STYLE_TOKEN_RE}(?=\s*\))"
)
_STYLE_STACK_ARG_RE = re.compile(
    rf'(\bStack\(\[[^\]\n]*\]\s*,\s*"(?:column|row)")\s*,\s*{_STYLE_TOKEN_RE}'
)


def strip_style_literals(openui: str) -> str:
    """
    Remove style-only quoted args from OpenUI source while preserving structure.

    Examples:
      Stack([a], "column", "m") → Stack([a], "column")
      TextContent(":t", "large-heavy") → TextContent(":t")
      Button(":x", "primary") → Button(":x")
    """
    if not openui:
        return openui
    text = _STYLE_TEXT_ARG_RE.sub(r"\1", openui)
    text = _STYLE_STACK_ARG_RE.sub(r"\1", text)
    # Collapse leftover empty arg lists artifacts like ", )" → ")"
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r",\s*,", ",", text)
    return text


def is_style_token(value: str) -> bool:
    return value.strip().strip('"') in STYLE_STRING_TOKENS
