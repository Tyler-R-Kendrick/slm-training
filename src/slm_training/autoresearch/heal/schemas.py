"""Typed contracts for bounded heal plans, receipts, and escalation records.

Design constraints (skeptic-panel dispositions, see
``docs/design/self-healing-escalation-layer.md``):

- Every plan **must** carry a verify probe; ``healed`` is decided by that
  probe's exit status alone (``exit_zero`` is the only success mode).
- Steps are exec-form argv only (no shell), each under its own wall budget
  bounded by the repo run cap, and each declares a ``writes_allowed`` path
  allowlist. Repo source, gates, metrics, manifests, ``versions.json`` and
  climb policy are permanently forbidden write targets
  (:func:`reject_forbidden_heal` fails closed).
- Escalation records key on a **cross-campaign** fingerprint
  (``kind`` + normalized reason); the campaign is payload, never key — a
  durable blocker keeps one identity across cycles so attempt budgets can
  actually accumulate.
- The escalation ledger is advisory-only: nothing in these schemas can mark a
  hard action acknowledged, soft, or stale.
"""

from __future__ import annotations

import hashlib
import posixpath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from slm_training.lineage.records import canonical_json

__all__ = [
    "FORBIDDEN_HEAL_WRITE_PREFIXES",
    "BlockerClass",
    "EscalationRecordV1",
    "HealAttemptReceiptV1",
    "HealPlanV1",
    "HealStepResultV1",
    "HealStepV1",
    "HealVerifyV1",
    "HealScopeError",
    "plan_sha256",
    "reject_forbidden_heal",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BlockerClass = Literal[
    "environment",  # npm/node/bridge deps — playbook-eligible
    "code",  # real harness bug — owner-skill only, stays hard
    "formal_infra",  # checker/toolchain/build infra — routed, not auto-healed yet
    "formal_contradiction",  # theorem-backed miss — evidence, never healed
    "delivery",  # deliver_stack ceremony — agent-owned, never healed
    "dirty_tree",  # foreign dirt — quarantine playbook (guarded, opt-in)
    "data",  # rebuild_data — driver-owned local heal already exists
    "authority",  # paid compute / human decision — human forever
    "unknown",
]

#: Paths no heal step may ever write, regardless of a playbook's allowlist.
#: The analogue of ``REVMATH_FROZEN_REPAIR_FIELDS``: repairing the environment
#: must never mutate the science surface.
FORBIDDEN_HEAL_WRITE_PREFIXES: tuple[str, ...] = (
    "src/slm_training/",
    "scripts/",
    "tests/",
    "docs/design/",
    "docs/MODEL_CARD.md",
    ".agents/",
    ".claude/",
    ".github/",
    "AGENTS.md",
    "CLAUDE.md",
)


class HealScopeError(ValueError):
    """A heal plan named a forbidden write target (fail closed)."""


def _normalize_path_prefix(prefix: str) -> str:
    """Separator-normalize and collapse traversal so prefixes compare honestly."""
    text = str(prefix).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    trailing = text.endswith("/")
    collapsed = posixpath.normpath(text) if text else text
    if collapsed in {".", ""}:
        return ""
    return collapsed + "/" if trailing else collapsed


def reject_forbidden_heal(writes_allowed: tuple[str, ...] | list[str]) -> None:
    """Fail closed when an allowlist would authorize writes to frozen surfaces."""
    for prefix in writes_allowed:
        normalized = _normalize_path_prefix(prefix)
        # Absolute or worktree-escaping prefixes can alias frozen surfaces
        # under another spelling — reject them outright.
        if normalized.startswith("/") or normalized.split("/", 1)[0] == "..":
            raise HealScopeError(
                f"heal allowlist prefix {prefix!r} is absolute or escapes the "
                "worktree; allowlists must be repo-relative"
            )
        for forbidden in FORBIDDEN_HEAL_WRITE_PREFIXES:
            if normalized.startswith(forbidden) or forbidden.startswith(normalized):
                raise HealScopeError(
                    f"heal allowlist prefix {prefix!r} overlaps frozen surface "
                    f"{forbidden!r}; environment repairs may never touch the "
                    "science surface"
                )


class HealStepV1(StrictModel):
    """One bounded exec-form command inside a heal plan."""

    step_id: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = ""
    env_clears: tuple[str, ...] = ()
    env_sets: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(ge=1)
    writes_allowed: tuple[str, ...] = ()


class HealVerifyV1(StrictModel):
    """The probe that decides ``healed`` — ideally the original failing check."""

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = ""
    env_clears: tuple[str, ...] = ()
    timeout_seconds: int = Field(ge=1)
    success: Literal["exit_zero"] = "exit_zero"


class HealPlanV1(StrictModel):
    """A deterministic, identity-preserving repair plan for one blocker."""

    schema_version: Literal["heal_plan/v1"] = "heal_plan/v1"
    playbook_id: str = Field(min_length=1)
    blocker_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocker_class: BlockerClass
    steps: tuple[HealStepV1, ...] = Field(min_length=1)
    verify: HealVerifyV1


def plan_sha256(plan: HealPlanV1) -> str:
    """Content signature of a plan — the cycle-detection identity."""
    return hashlib.sha256(
        canonical_json(plan.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


class HealStepResultV1(StrictModel):
    """Captured result of one executed step (or the verify probe)."""

    step_id: str
    returncode: int
    outcome: str = "completed"  # ProcessOutcome value
    duration_seconds: float = 0.0
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tail: str = Field(default="", max_length=2000)


HealOutcome = Literal[
    "healed",
    "verify_failed",
    "step_failed",
    "step_timeout",
    "budget_exhausted",
    "cycle_detected",
    "refused_scope",
]


class HealAttemptReceiptV1(StrictModel):
    """Append-only evidence that one bounded heal attempt ran.

    ``healed`` here is a claim about the *environment probe*, never about the
    pending action: only the continuous driver's emission-time rewrite may
    consume it, and only for environment-class ``repair_harness`` blockers.
    """

    schema_version: Literal["heal_attempt_receipt/v1"] = "heal_attempt_receipt/v1"
    loop_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    playbook_id: str = Field(min_length=1)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocker_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts_prior: int = Field(ge=0)
    step_results: tuple[HealStepResultV1, ...] = ()
    verify_result: HealStepResultV1 | None = None
    outcome: HealOutcome
    note: str = ""
    recorded_at: str = ""

    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()


class EscalationRecordV1(StrictModel):
    """Machine-readable state of one durable blocker (advisory-only).

    The fingerprint is cross-campaign — ``sha256(kind + normalized reason)``
    — so a blocker that survives cycles accumulates attempts instead of
    minting a fresh identity each time (the campaign-bound
    ``blocker_fingerprint`` in ``state.json`` never dedups). The ledger may
    schedule backoff and name the owner skill; it has no authority to
    acknowledge, soften, or expire a hard action.
    """

    schema_version: Literal["escalation_record/v1"] = "escalation_record/v1"
    loop_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(min_length=1)
    blocker_class: BlockerClass
    reason: str = ""
    campaign_ids: tuple[str, ...] = ()
    owner_skill: str = ""
    needed_authority: Literal[
        "none", "agent_session", "paid_compute", "human_decision"
    ] = "agent_session"
    first_seen_at: str = ""
    last_seen_at: str = ""
    seen_count: int = Field(default=1, ge=1)
    attempts: int = Field(default=0, ge=0)
    playbooks_tried: tuple[str, ...] = ()
    status: Literal["open", "healing", "escalated", "resolved"] = "open"
    note: str = ""
    next_backoff_seconds: int = Field(default=0, ge=0)


def _normalize_reason(reason: str) -> str:
    """Strip volatile tokens (hashes, campaign ids, counts) for fingerprinting."""
    import re

    text = str(reason).lower().strip()
    text = re.sub(r"\b[0-9a-f]{12,64}\b", "<sha>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}t?[\d:.z+-]*\b", "<ts>", text)
    # Word boundaries keep digits inside identifiers (openui_bridge2, agentv3)
    # so distinct blockers never collapse into one fingerprint/budget.
    text = re.sub(r"\bc?\d+\b", "<n>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:400]


def escalation_fingerprint(kind: str, reason: str) -> str:
    payload: dict[str, Any] = {
        "kind": str(kind),
        "reason": _normalize_reason(reason),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
