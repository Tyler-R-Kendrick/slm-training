"""Single source of truth for hard-blocker classification.

The continuous driver's hard-crash markers previously lived inline
(``scripts/run_autotrain_continuous.py`` ``_delivery_is_thrash_timeout_residual``)
as bare substring matches, which conflated two very different conditions:

- **environment-incomplete** — a JS bridge / AgentV dependency install is
  missing (the exact bounded local fix is documented in
  ``.agents/skills/autotrain/references/continuous.md``): playbook-eligible.
- **code crash** — a repo-internal import error or genuine harness bug:
  owner-skill territory, stays hard forever (never playbook-eligible).

Precision rule (skeptic O5): ``"no module named"`` matching alone must never
route to the environment playbook — ``no module named slm_training.x`` after a
bad merge is a code crash that ``npm ci`` cannot fix and would turn the
playbook into an infinite install-verify-fail loop.
"""

from __future__ import annotations

import re

from slm_training.autoresearch.heal.schemas import BlockerClass

__all__ = [
    "HARD_HARNESS_MARKERS",
    "classify_blocker",
    "missing_js_module",
]

#: True-crash markers that keep a ``repair_harness`` action hard at emission.
#: Imported by the continuous driver — the classification and the emission
#: share one marker list so they can never drift.
HARD_HARNESS_MARKERS: tuple[str, ...] = (
    "agentv sdk is unavailable",
    "npm ci",
    "module not found",
    "import error",
    "no module named",
    "err_module_not_found",
)

_JS_ENV_MARKERS: tuple[str, ...] = (
    "agentv sdk is unavailable",
    "npm ci",
    "err_module_not_found",
)

#: Python-module markers that indicate repo-internal code, not environment.
_REPO_INTERNAL_TOKENS: tuple[str, ...] = (
    "slm_training",
    "scripts.",
)

_FORMAL_INFRA_MARKERS: tuple[str, ...] = (
    "checker is unavailable",
    "cached_formal_preflight_invalid",
    "leverproof checker is unavailable",
    "lake build",
    "toolchain",
)

_FORMAL_CONTRADICTION_MARKERS: tuple[str, ...] = (
    "theorem_backed_band_miss",
    "theorem-backed",
    "theorem contradiction",
    "band_breach",
)

#: Markers naming decisions only a human may make (paid compute, HF writes,
#: explicit user authority). These route as class ``authority`` → owner
#: ``human`` regardless of blocker kind.
_AUTHORITY_MARKERS: tuple[str, ...] = (
    "paid gpu",
    "paid compute",
    "remote compute authority",
    "hf write",
    "requires user authority",
    "human approval",
    "billing budget",
)

_MISSING_JS_MODULE_RE = re.compile(
    r"(?:cannot find (?:module|package)|err_module_not_found)"
    r"[^'\"]*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def missing_js_module(reason: str) -> str | None:
    """The missing JS module named in a crash reason, when parsable."""
    match = _MISSING_JS_MODULE_RE.search(str(reason))
    if match:
        return match.group(1)
    return None


def classify_blocker(kind: str, reason: str) -> BlockerClass:
    """Map one hard-pending blocker to its heal class.

    Fail toward ``code`` (stays hard) whenever a marker is ambiguous: the
    cost of a wrong ``code`` verdict is a waiting agent, the cost of a wrong
    ``environment`` verdict is an install-verify-fail thrash loop.
    """
    kind_s = str(kind).strip()
    text = str(reason).lower()

    # Human-authority decisions dominate every other class: no playbook or
    # agent session may substitute for a paid-compute / approval grant.
    if any(m in text for m in _AUTHORITY_MARKERS):
        return "authority"
    if kind_s == "stop_campaign":
        return "formal_contradiction"
    if kind_s == "repair_formal":
        if any(m in text for m in _FORMAL_CONTRADICTION_MARKERS):
            return "formal_contradiction"
        if any(m in text for m in _FORMAL_INFRA_MARKERS):
            return "formal_infra"
        # Ambiguous formal reasons stay evidence, never infra.
        return "formal_contradiction"
    if kind_s == "deliver_stack":
        return "delivery"
    if kind_s == "rebuild_data":
        return "data"
    if kind_s == "foreign_dirty_tree":
        return "dirty_tree"
    if kind_s == "repair_harness":
        if any(tok in text for tok in _REPO_INTERNAL_TOKENS):
            return "code"
        if any(m in text for m in _JS_ENV_MARKERS):
            return "environment"
        if missing_js_module(text) is not None:
            return "environment"
        if "module not found" in text or "no module named" in text:
            # Unattributed module errors: python-style phrasing is repo code;
            # JS-style phrasing without a parsable module stays code too.
            return "code"
        return "code"
    return "unknown"
