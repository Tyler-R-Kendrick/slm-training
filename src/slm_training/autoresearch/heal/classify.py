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
    "DATA_PREREQUISITE_MARKERS",
    "HARD_HARNESS_MARKERS",
    "HARNESS_CRASH_REASON_RE",
    "TIMEOUT_RESIDUAL_MARKERS",
    "classify_blocker",
    "crash_arm_exits",
    "is_harness_crash",
    "missing_js_module",
    "timeout_arm_exits",
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

#: The delivery reason an arm leaves when its harness process died instead of
#: producing a scoreboard: ``harness_failure:<arm_id>:experiment_failed``.
#: With a non-124 exit this is a *crash* (class ``code``), never a thrash
#: wall/decode residual — the misroute that let cycles c536..c543 drop their
#: ``repair_harness`` blocker and continue.
HARNESS_CRASH_REASON_RE = re.compile(
    r"harness_failure:[^:\s]*:experiment_failed", re.IGNORECASE
)

#: Explicit timeout evidence. Only these (or an arm exit of 124) make an
#: incomplete measurement a soft thrash residual; ``missing_scoreboard`` /
#: ``primary_metric_unavailable`` alone never do.
TIMEOUT_RESIDUAL_MARKERS: tuple[str, ...] = (
    "decode_timeout",
    "decode timeout",
    "wall_timeout",
    "wall-timeout",
    "timed out",
    "timeout residual",
    "internal decode timeout",
    "exit=124",
    "exit 124",
)

#: Reasons naming a ``rebuild_data`` prerequisite (the artifacts / counts the
#: local rebuild seam produces) route to class ``data`` so the data playbook
#: can retry the rebuild with a measured postcondition.
DATA_PREREQUISITE_MARKERS: tuple[str, ...] = (
    "rebuild_data",
    "records_before",
    "records_after",
    "quality_report.json",
    "synthesis_feedback.json",
    "data_manifest.json",
    "screening suite",
    "smoke fixture",
    "sample_adequacy",
    "fixture volume",
    "train records",
)

#: Wall/timeout exit from the bounded-process convention in the driver.
TIMEOUT_EXIT_CODE = 124

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


def _exit_codes(arm_exits: object) -> list[int]:
    if not isinstance(arm_exits, dict):
        return []
    codes: list[int] = []
    for value in arm_exits.values():
        try:
            codes.append(int(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return codes


def timeout_arm_exits(arm_exits: object) -> bool:
    """True when at least one arm exit is the wall/timeout code (124)."""
    return any(c == TIMEOUT_EXIT_CODE for c in _exit_codes(arm_exits))


def crash_arm_exits(arm_exits: object) -> bool:
    """True when at least one arm exited non-zero and not by timeout."""
    return any(c not in (0, TIMEOUT_EXIT_CODE) for c in _exit_codes(arm_exits))


def is_harness_crash(reason: str, arm_exits: object = None) -> bool:
    """``harness_failure:*:experiment_failed`` without timeout evidence.

    A crash is a crash whenever a non-124 arm exit accompanies the marker; when
    no exits are known, the reason text must carry an explicit timeout marker
    to be read as a residual instead.
    """
    text = str(reason).lower()
    if not HARNESS_CRASH_REASON_RE.search(text):
        return False
    if crash_arm_exits(arm_exits):
        return True
    if timeout_arm_exits(arm_exits):
        return False
    return not any(m in text for m in TIMEOUT_RESIDUAL_MARKERS)


def classify_blocker(
    kind: str, reason: str, *, arm_exits: object = None
) -> BlockerClass:
    """Map one hard-pending blocker to its heal class.

    Fail toward ``code`` (stays hard) whenever a marker is ambiguous: the
    cost of a wrong ``code`` verdict is a waiting agent, the cost of a wrong
    ``environment`` verdict is an install-verify-fail thrash loop.
    ``arm_exits`` (the delivery's ``{arm_id: exit_code}`` map) sharpens the
    crash-vs-residual split when the caller has it.
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
    if kind_s in {"heal_postcondition_failed", "retry_measurement"} and any(
        m in text for m in DATA_PREREQUISITE_MARKERS
    ):
        # A failed data-rebuild postcondition / a retry blocked on rebuild
        # artifacts is data territory: the rebuild seam owns the next attempt.
        return "data"
    if kind_s in {"loop_stalled_no_campaign", "heal_postcondition_failed", "vacuous_pass"}:
        return "unknown"
    if kind_s == "repair_harness":
        if any(tok in text for tok in _REPO_INTERNAL_TOKENS):
            return "code"
        if any(m in text for m in _JS_ENV_MARKERS):
            return "environment"
        if missing_js_module(text) is not None:
            return "environment"
        if is_harness_crash(text, arm_exits):
            # An arm process died (exit != 124) without a scoreboard: a
            # harness crash the harness_crash playbook must triage, never a
            # soft wall residual.
            return "code"
        if any(m in text for m in DATA_PREREQUISITE_MARKERS):
            return "data"
        if "module not found" in text or "no module named" in text:
            # Unattributed module errors: python-style phrasing is repo code;
            # JS-style phrasing without a parsable module stays code too.
            return "code"
        return "code"
    return "unknown"
