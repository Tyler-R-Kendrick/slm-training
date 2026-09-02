"""Bounded heal-playbook runner + escalation seam for the continuous loop.

The continuous driver is honest about *what* blocked it (typed
``AutotrainActionV1`` kinds, blocker fingerprints, content-bound receipts) but
has no executor for the blocked class: the supervisor answered every
``hard_pending`` blocker with a blind fixed sleep, and the documented
deterministic fixes (e.g. the ``npm ci`` bridge installs in
``.agents/skills/autotrain/references/continuous.md``) were never attempted
without a human-opened agent session. This package is that executor, with the
same discovery idiom as :mod:`slm_training.autoresearch.preflight`: every
module in ``heal/playbooks/`` exposing a module-level ``PLAYBOOK`` object is
discovered via ``pkgutil.iter_modules``; absence is never an error; a crashing
playbook yields a ``step_failed`` receipt, never an exception and never a
claimed heal.

Non-negotiable laws (mirrors the revmath self-healing contract):

- **Repair with proof.** A playbook performs real bounded commands and then a
  *verify probe*; ``healed`` is decided by the probe's exit status alone —
  never by the repair steps succeeding, never by heuristics.
- **Frozen identity.** Steps may only write under each plan's
  ``writes_allowed`` path prefixes. Repo source, gates, metrics, manifests,
  ``versions.json`` and climb policy are permanently outside every allowlist
  (:func:`reject_forbidden_heal`); a step that dirties anything else yields
  ``refused_scope``.
- **No acks, no waivers.** This package never writes an
  ``AutotrainActionReceiptV1``, never edits a ``cycle_handoff.json``, and
  never writes synthesis-feedback waivers. The only legal consumer of a
  verified heal receipt is the continuous driver's own emission-time rewrite
  (``SELF_HEAL_ENV_REPAIR`` — the ``SELF_HEAL_THRASH_TIMEOUT_REPAIR``
  precedent), which reclassifies a misdiagnosed environment-incomplete
  blocker with the receipt as evidence. ``stop_campaign``, theorem-backed
  ``repair_formal``, and ``deliver_stack`` are never playbook-eligible.
- **Kill is not evidence.** A step stopped by the wall budget yields
  ``step_timeout``: neither a heal nor a counted failure. Timeouts escalate
  immediately (zero retries) instead of thrashing under the run cap.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import traceback
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

from slm_training.autoresearch.heal.classify import classify_blocker
from slm_training.autoresearch.heal.escalation import (
    EscalationLedger,
    blocker_fingerprint,
)
from slm_training.autoresearch.heal.schemas import (
    HealAttemptReceiptV1,
    HealPlanV1,
    HealScopeError,
    HealStepResultV1,
    plan_sha256,
    reject_forbidden_heal,
)
from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import KILL_GRACE_SECONDS, MAX_RUN_SECONDS

__all__ = [
    "HealPlaybook",
    "discovered_playbooks",
    "run_playbooks",
    "write_heal_receipt",
    "load_heal_receipts",
]

HEAL_RECEIPTS_FILENAME = "heal_receipts.jsonl"


@runtime_checkable
class HealPlaybook(Protocol):
    """A pluggable heal playbook: module-level ``PLAYBOOK`` objects satisfy this."""

    playbook_id: str
    handles: frozenset[str]

    def matches(self, blocker: dict) -> bool: ...

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None: ...


def _crash_summary(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    location = ""
    if frames:
        frame = frames[-1]
        location = f" at {Path(frame.filename).name}:{frame.lineno}"
    return f"{type(exc).__name__}: {exc}{location}"


def discovered_playbooks() -> tuple[HealPlaybook, ...]:
    """Discover module-level ``PLAYBOOK`` objects under ``heal/playbooks/``."""
    from slm_training.autoresearch.heal import playbooks as _pkg

    found: list[HealPlaybook] = []
    for info in sorted(pkgutil.iter_modules(_pkg.__path__), key=lambda m: m.name):
        try:
            module = importlib.import_module(f"{_pkg.__name__}.{info.name}")
        except Exception:  # noqa: BLE001 — one broken playbook never blocks others
            continue
        candidate = getattr(module, "PLAYBOOK", None)
        if candidate is not None and isinstance(candidate, HealPlaybook):
            found.append(candidate)
    return tuple(found)


def heal_receipts_path(root: Path, loop_id: str) -> Path:
    return Path(root) / "loops" / loop_id / HEAL_RECEIPTS_FILENAME


def write_heal_receipt(root: Path, receipt: HealAttemptReceiptV1) -> Path:
    """Append one receipt to the loop's append-only heal ledger."""
    path = heal_receipts_path(root, receipt.loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(receipt.model_dump_json() + "\n")
    return path


def load_heal_receipts(root: Path, loop_id: str) -> tuple[HealAttemptReceiptV1, ...]:
    path = heal_receipts_path(root, loop_id)
    if not path.is_file():
        return ()
    receipts: list[HealAttemptReceiptV1] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            receipts.append(HealAttemptReceiptV1.model_validate_json(line))
        except Exception:  # noqa: BLE001 — a malformed row never blocks the fold
            continue
    return tuple(receipts)


def _dirty_paths(cwd: Path) -> tuple[str, ...]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) > 3:
            paths.append(line[3:].split(" -> ")[-1].strip().strip('"'))
    return tuple(paths)


def _scope_violation(
    before: Iterable[str], after: Iterable[str], allowed: Sequence[str]
) -> tuple[str, ...]:
    """Newly dirtied paths outside every ``writes_allowed`` prefix."""
    from slm_training.autoresearch.heal.schemas import _normalize_path_prefix

    fresh = set(after) - set(before)
    normalized = tuple(_normalize_path_prefix(p) for p in allowed)
    return tuple(
        sorted(
            p
            for p in fresh
            if not any(
                _normalize_path_prefix(p).startswith(prefix) for prefix in normalized
            )
        )
    )


def _run_step(
    argv: Sequence[str],
    *,
    cwd: Path,
    env_clears: Sequence[str],
    env_sets: dict[str, str],
    timeout_seconds: int,
    step_id: str,
) -> HealStepResultV1:
    import hashlib
    import os

    env = os.environ.copy()
    for name in env_clears:
        env.pop(name, None)
    env.update(env_sets)
    result = run_bounded_process(
        argv,
        interrupt_after_seconds=float(timeout_seconds),
        kill_grace_seconds=KILL_GRACE_SECONDS,
        cwd=str(cwd),
        env=env,
    )
    tail = (result.stdout[-1000:] + "\n" + result.stderr[-1000:]).strip()[:2000]
    return HealStepResultV1(
        step_id=step_id,
        returncode=-1 if result.returncode is None else int(result.returncode),
        outcome=result.outcome.value,
        duration_seconds=float(result.duration_seconds),
        stdout_sha256=hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        tail=tail,
    )


def run_playbooks(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str,
    blockers: Sequence[dict],
    cwd: Path,
    max_attempts_per_fingerprint: int = 2,
    playbooks: Sequence[HealPlaybook] | None = None,
) -> tuple[HealAttemptReceiptV1, ...]:
    """Attempt bounded heals for hard blockers; every attempt leaves a receipt.

    Laws enforced here (not in individual playbooks): escalation-ledger attempt
    budget, plan-signature cycle detection, per-step wall budgets with
    kill-is-not-evidence, ``writes_allowed`` scope enforcement, and
    verify-decides-healed.
    """
    root = Path(root)
    cwd = Path(cwd)
    ledger = EscalationLedger.load(root, loop_id)
    prior_receipts = load_heal_receipts(root, loop_id)
    # A plan that already healed is not a cycle: the environment can regress
    # (container rebuild drops node_modules) and the deterministic repair must
    # stay available. Only unresolved/failed plans count toward cycles.
    seen_plan_shas = {r.plan_sha256 for r in prior_receipts if r.outcome != "healed"}
    available = tuple(playbooks) if playbooks is not None else discovered_playbooks()

    def _persist(receipt: HealAttemptReceiptV1) -> None:
        # Incremental persistence: a later blocker's crash must never discard
        # earlier receipts or ledger attempt counts (the budget law).
        write_heal_receipt(root, receipt)
        ledger.save()

    receipts: list[HealAttemptReceiptV1] = []
    for blocker in blockers:
        kind = str(blocker.get("kind") or "")
        reason = str(blocker.get("reason") or "")
        blocker_class = str(
            blocker.get("blocker_class") or classify_blocker(kind, reason)
        )
        fingerprint = blocker_fingerprint(kind, reason)
        record = ledger.observe(
            kind=kind,
            reason=reason,
            blocker_class=blocker_class,
            campaign_id=campaign_id,
            owner_skill=str(blocker.get("owner") or ""),
        )
        matching = [
            p
            for p in available
            if blocker_class in p.handles and _safe_matches(p, blocker)
        ]
        if not matching:
            # An escalation without a receipt is invisible to the driver's
            # pass classifier (a pass that only escalated would score as
            # vacuous). Leave an ``unhandled`` receipt so the attempt is on
            # the ledger and the escalation is visible.
            ledger.escalate(fingerprint, note="no_matching_playbook")
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    "none",
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="unhandled",
                    note=f"no_matching_playbook class={blocker_class} kind={kind}",
                )
            )
            _persist(receipts[-1])
            continue
        playbook = matching[0]
        if record.attempts >= max_attempts_per_fingerprint:
            ledger.escalate(fingerprint, note="attempt_budget_exhausted")
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="budget_exhausted",
                )
            )
            _persist(receipts[-1])
            continue
        execute = getattr(playbook, "execute", None)
        if callable(execute):
            # In-process attempt: the playbook decides its own outcome (e.g.
            # ``attempted`` for a crash triage that repairs nothing, or a
            # count-postcondition ``healed``) instead of the subprocess verify
            # probe. Same attempt budget and receipt persistence as ``plan``.
            ledger.record_attempt(fingerprint, playbook.playbook_id)
            try:
                receipt = execute(
                    {
                        **blocker,
                        "campaign_id": blocker.get("campaign_id") or campaign_id,
                    },
                    cwd=cwd,
                    root=root,
                    loop_id=loop_id,
                    campaign_id=str(blocker.get("campaign_id") or campaign_id),
                    write_receipt=False,
                )
            except Exception as exc:  # noqa: BLE001 — execute crash is a failed attempt
                receipt = _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="step_failed",
                    note=_crash_summary(exc),
                )
            receipts.append(receipt)
            if receipt.outcome == "healed":
                ledger.resolve(fingerprint, note=f"healed_by:{playbook.playbook_id}")
            elif receipt.outcome == "step_timeout":
                ledger.escalate(fingerprint, note="step_timeout_under_run_cap")
            _persist(receipt)
            continue
        try:
            plan = playbook.plan(dict(blocker), cwd=cwd)
        except Exception as exc:  # noqa: BLE001 — plan crash is a failed attempt
            ledger.record_attempt(fingerprint, playbook.playbook_id)
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="step_failed",
                    note=_crash_summary(exc),
                )
            )
            _persist(receipts[-1])
            continue
        if plan is None:
            ledger.escalate(fingerprint, note="playbook_declined")
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="unhandled",
                    note=f"playbook_declined class={blocker_class} kind={kind}",
                )
            )
            _persist(receipts[-1])
            continue
        try:
            for step in plan.steps:
                reject_forbidden_heal(step.writes_allowed)
        except HealScopeError as exc:
            # A playbook naming a frozen surface is a buggy/hostile playbook:
            # record the refusal and escalate — never take down the runner.
            ledger.escalate(fingerprint, note="refused_scope_allowlist")
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=None,
                    attempts_prior=record.attempts,
                    outcome="refused_scope",
                    note=str(exc)[:400],
                )
            )
            _persist(receipts[-1])
            continue
        signature = plan_sha256(plan)
        if signature in seen_plan_shas:
            ledger.escalate(fingerprint, note="cycle_detected")
            receipts.append(
                _receipt(
                    loop_id,
                    campaign_id,
                    playbook.playbook_id,
                    fingerprint,
                    plan=plan,
                    attempts_prior=record.attempts,
                    outcome="cycle_detected",
                )
            )
            _persist(receipts[-1])
            continue
        seen_plan_shas.add(signature)
        ledger.record_attempt(fingerprint, playbook.playbook_id)

        try:
            receipt = _execute_plan(
                plan,
                loop_id=loop_id,
                campaign_id=campaign_id,
                fingerprint=fingerprint,
                attempts_prior=record.attempts,
                cwd=cwd,
            )
        except Exception as exc:  # noqa: BLE001 — execution crash is a failed attempt
            receipt = _receipt(
                loop_id,
                campaign_id,
                playbook.playbook_id,
                fingerprint,
                plan=plan,
                attempts_prior=record.attempts,
                outcome="step_failed",
                note=_crash_summary(exc),
            )
        receipts.append(receipt)
        if receipt.outcome == "healed":
            ledger.resolve(fingerprint, note=f"healed_by:{plan.playbook_id}")
        elif receipt.outcome == "step_timeout":
            # Kill is not evidence: escalate immediately, zero retries.
            ledger.escalate(fingerprint, note="step_timeout_under_run_cap")
        _persist(receipt)

    ledger.save()
    return tuple(receipts)


def _safe_matches(playbook: HealPlaybook, blocker: dict) -> bool:
    try:
        return bool(playbook.matches(dict(blocker)))
    except Exception:  # noqa: BLE001
        return False


def _execute_plan(
    plan: HealPlanV1,
    *,
    loop_id: str,
    campaign_id: str,
    fingerprint: str,
    attempts_prior: int,
    cwd: Path,
) -> HealAttemptReceiptV1:
    step_results: list[HealStepResultV1] = []
    before = _dirty_paths(cwd)
    outcome = "healed"
    verify_result: HealStepResultV1 | None = None
    for step in plan.steps:
        result = _run_step(
            step.argv,
            cwd=cwd / step.cwd if step.cwd else cwd,
            env_clears=step.env_clears,
            env_sets=dict(step.env_sets),
            timeout_seconds=min(int(step.timeout_seconds), int(MAX_RUN_SECONDS)),
            step_id=step.step_id,
        )
        step_results.append(result)
        if result.outcome in (
            ProcessOutcome.TIMED_OUT.value,
            ProcessOutcome.INTERRUPTED.value,
            ProcessOutcome.KILLED.value,
        ):
            outcome = "step_timeout"
            break
        if result.returncode != 0:
            outcome = "step_failed"
            break
        after = _dirty_paths(cwd)
        violation = _scope_violation(before, after, step.writes_allowed)
        if violation:
            outcome = "refused_scope"
            step_results[-1] = result.model_copy(
                update={
                    "tail": (result.tail + f"\nscope_violation={violation[:8]}")[:2000]
                }
            )
            break
        # Re-baseline: dirt a step legally produced must not be re-judged
        # under the next step's (disjoint) allowlist.
        before = after
    if outcome == "healed":
        verify_result = _run_step(
            plan.verify.argv,
            cwd=cwd / plan.verify.cwd if plan.verify.cwd else cwd,
            env_clears=plan.verify.env_clears,
            env_sets={},
            timeout_seconds=min(int(plan.verify.timeout_seconds), int(MAX_RUN_SECONDS)),
            step_id="verify",
        )
        if verify_result.outcome != ProcessOutcome.COMPLETED.value:
            outcome = "step_timeout"
        elif verify_result.returncode != 0:
            outcome = "verify_failed"
    return _receipt(
        loop_id,
        campaign_id,
        plan.playbook_id,
        fingerprint,
        plan=plan,
        attempts_prior=attempts_prior,
        outcome=outcome,
        step_results=tuple(step_results),
        verify_result=verify_result,
    )


def _receipt(
    loop_id: str,
    campaign_id: str,
    playbook_id: str,
    fingerprint: str,
    *,
    plan: HealPlanV1 | None,
    attempts_prior: int,
    outcome: str,
    step_results: tuple[HealStepResultV1, ...] = (),
    verify_result: HealStepResultV1 | None = None,
    note: str = "",
) -> HealAttemptReceiptV1:
    from slm_training.autoresearch.schemas import utc_now

    return HealAttemptReceiptV1(
        loop_id=loop_id,
        campaign_id=campaign_id,
        playbook_id=playbook_id,
        plan_sha256=plan_sha256(plan) if plan is not None else "0" * 64,
        blocker_fingerprint=fingerprint,
        attempts_prior=attempts_prior,
        step_results=step_results,
        verify_result=verify_result,
        outcome=outcome,  # type: ignore[arg-type]
        note=note,
        # Recency binding: the driver's rewrite rejects receipts older than
        # the blocker's handoff, so a historical heal can never authorize a
        # later regression of the same fingerprint.
        recorded_at=utc_now(),
    )
