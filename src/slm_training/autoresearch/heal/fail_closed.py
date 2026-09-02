"""Small durable primitives for fail-closed continuous-loop progress."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import fmean, stdev

from slm_training.autoresearch.heal.schemas import HealAttemptReceiptV1, HealVerifyV1

__all__ = [
    "BlockerRule",
    "BLOCKER_RULES",
    "HEAL_POSTCONDITION_FAILED",
    "append_power_deltas",
    "allocate_screening_suite_id",
    "count_records",
    "record_count_probe",
    "verify_driver_heal",
    "lease_covers",
    "read_power_evidence",
    "power_evidence_summary",
]

HEAL_POSTCONDITION_FAILED = "heal_postcondition_failed"


class BlockerRule:
    __slots__ = ("owner", "postcondition", "max_attempts", "terminal")

    def __init__(self, owner: str, postcondition: str, max_attempts: int, terminal: str):
        self.owner = owner
        self.postcondition = postcondition
        self.max_attempts = max_attempts
        self.terminal = terminal


BLOCKER_RULES = {
    "loop_stalled_no_campaign": BlockerRule("autotrain", "campaign_initialized", 3, "escalate"),
    "heal_postcondition_failed": BlockerRule("autotrain", "declared_artifact_changed", 3, "escalate"),
    "vacuous_pass": BlockerRule("autotrain", "campaign_initialized_or_typed_action", 3, "escalate"),
}


def allocate_screening_suite_id(root: Path, n: int, *, prefix: str = "e938_role_safe_all_targets_smoke") -> str:
    """Return the next unused ``<prefix><n>_v<k>`` id; frozen snapshots are never reused.

    Every published version of the ``n`` family counts (``_v1`` .. ``_vK``),
    so the allocator can never saturate at ``_v2`` and hand back an id whose
    directory already exists.
    """
    root = Path(root)
    pattern = re.compile(rf"^{re.escape(prefix)}{int(n)}_v(\d+)$")
    used: set[int] = set()
    for path in root.glob(f"{prefix}{int(n)}_v*"):
        match = pattern.match(path.name)
        if match and path.is_dir():
            used.add(int(match.group(1)))
    candidate = 1
    while candidate in used:
        candidate += 1
    return f"{prefix}{int(n)}_v{candidate}"


def count_records(path: Path) -> int:
    """Non-blank line count of a jsonl artifact (0 when the file is absent)."""
    path = Path(path)
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


_COUNT_PROBE_SOURCE = (
    "import pathlib, sys; p = pathlib.Path(sys.argv[1]); "
    "n = sum(1 for l in p.read_text(encoding='utf-8').splitlines() if l.strip()) "
    "if p.is_file() else 0; "
    "print(f'records={n} must_exceed={sys.argv[2]}'); "
    "sys.exit(0 if n > int(sys.argv[2]) else 1)"
)


def record_count_probe(path: Path, *, must_exceed: int, timeout_seconds: int = 60) -> HealVerifyV1:
    """A ``HealVerifyV1`` that re-reads ``path`` and exits zero iff its record count grew.

    The probe is a separate process reading the artifact from disk, so the
    verdict is never the heal body's own claim about what it wrote.
    """
    return HealVerifyV1(
        argv=(sys.executable, "-c", _COUNT_PROBE_SOURCE, str(path), str(int(must_exceed))),
        timeout_seconds=int(timeout_seconds),
    )


def verify_driver_heal(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
    heal_id: str,
    verify: HealVerifyV1,
    cwd: Path,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    extra_conditions: dict[str, bool] | None = None,
    note: str = "",
) -> HealAttemptReceiptV1:
    """Run a driver heal's postcondition probe and leave an append-only receipt.

    ``healed`` is decided by the probe's exit status plus every
    ``extra_conditions`` flag (e.g. ``must_generate == False``); the heal body
    never gets a vote. A failed postcondition writes a ``verify_failed``
    receipt **and** records a ``heal_postcondition_failed`` blocker in the
    escalation ledger (escalated once the class's attempt budget is spent), so
    a heal that changed nothing is counted and visible instead of vacuous.
    """
    from slm_training.autoresearch.heal import _run_step, write_heal_receipt
    from slm_training.autoresearch.heal.escalation import EscalationLedger, blocker_fingerprint
    from slm_training.autoresearch.schemas import utc_now
    from slm_training.levers import MAX_RUN_SECONDS

    conditions = dict(extra_conditions or {})
    verify_result = _run_step(
        verify.argv,
        cwd=Path(cwd) / verify.cwd if verify.cwd else Path(cwd),
        env_clears=verify.env_clears,
        env_sets={},
        timeout_seconds=min(int(verify.timeout_seconds), int(MAX_RUN_SECONDS)),
        step_id="verify",
    )
    probe_ok = verify_result.outcome == "completed" and verify_result.returncode == 0
    failed_conditions = sorted(name for name, ok in conditions.items() if not ok)
    healed = probe_ok and not failed_conditions
    detail = (
        f"{heal_id}: counts_before={json.dumps(counts_before, sort_keys=True)} "
        f"counts_after={json.dumps(counts_after, sort_keys=True)} "
        f"probe_returncode={verify_result.returncode} "
        f"failed_conditions={failed_conditions}"
    )
    if note:
        detail = f"{detail} {note}"
    reason = f"{heal_id}: declared artifact did not change ({', '.join(failed_conditions) or 'count'})"
    fingerprint = blocker_fingerprint(HEAL_POSTCONDITION_FAILED, reason)
    receipt = HealAttemptReceiptV1(
        loop_id=loop_id,
        campaign_id=campaign_id or "unknown",
        playbook_id=f"driver:{heal_id}",
        plan_sha256="0" * 64,
        blocker_fingerprint=fingerprint,
        attempts_prior=0,
        verify_result=verify_result,
        outcome="healed" if healed else "verify_failed",
        note=(("" if healed else f"{HEAL_POSTCONDITION_FAILED} ") + detail)[:2000],
        recorded_at=utc_now(),
    )
    ledger = EscalationLedger.load(root, loop_id)
    if healed:
        if fingerprint in ledger.records:
            ledger.resolve(fingerprint, note=f"healed_by:driver:{heal_id}")
    else:
        record = ledger.observe(
            kind=HEAL_POSTCONDITION_FAILED,
            reason=reason,
            blocker_class="unknown",
            campaign_id=campaign_id or "",
            owner_skill="autotrain",
        )
        rule = BLOCKER_RULES[HEAL_POSTCONDITION_FAILED]
        if record.seen_count >= rule.max_attempts:
            ledger.escalate(fingerprint, note=f"{rule.terminal}: {detail}"[:400])
        else:
            ledger.record_attempt(fingerprint, f"driver:{heal_id}")
    ledger.save()
    write_heal_receipt(root, receipt)
    return receipt


def append_power_deltas(path: Path, *, cycle: str, metric: str, deltas: list[float], costs: list[float] | None = None) -> int:
    """Append paired observations and return the new observation count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = costs or [0.0] * len(deltas)
    if len(costs) != len(deltas):
        raise ValueError("costs and deltas must have equal length")
    with path.open("a", encoding="utf-8") as fh:
        for delta, cost in zip(deltas, costs):
            fh.write(json.dumps({"cycle": cycle, "metric": metric, "delta": float(delta), "wall_seconds": float(cost)}, sort_keys=True) + "\n")
    return sum(1 for _ in path.open(encoding="utf-8"))


def read_power_evidence(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("delta"), (int, float)):
            rows.append(row)
    return rows


def power_evidence_summary(path: Path, *, min_pairs: int = 100, min_cycles: int = 10) -> dict:
    rows = read_power_evidence(path)
    cycles = {str(row.get("cycle")) for row in rows}
    ready = len(rows) >= min_pairs and len(cycles) >= min_cycles
    values = [float(row["delta"]) for row in rows]
    return {
        "paired_deltas": len(rows),
        "cycles": len(cycles),
        "ready": ready,
        "sd": stdev(values) if len(values) > 1 else None,
        "mean": fmean(values) if values else None,
        "mean_abs_cost_seconds": fmean(float(row.get("wall_seconds", 0.0)) for row in rows) if rows else None,
    }


def lease_covers(path: Path, dirty_path: str, *, now: float | None = None) -> bool:
    """Honor only an unexpired lease that explicitly covers the dirty path."""
    try:
        lease = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    now = time.time() if now is None else now
    if float(lease.get("expires_at", 0)) <= now:
        return False
    prefixes = lease.get("path_prefixes") or lease.get("paths") or []
    return any(str(dirty_path).replace("\\", "/").startswith(str(prefix).rstrip("/") + "/") or str(dirty_path) == str(prefix).rstrip("/") for prefix in prefixes)
