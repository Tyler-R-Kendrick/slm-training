"""Replayable solver-transition trace events + validator (VSS1-04 / SLM-64).

Torch-free. Turns the certified-solver artifacts (exact-closure / search results
and their certificates) into typed, replayable events recorded inside the
existing ``DecodeTraceRecorder.events`` stream, and validates that every
destructive transition is consistent and — in ``full`` certificate mode —
certificate-checked.

Honesty invariants (owned here; see ``docs/design/verified-scope-solver.md``):

* ``unknown`` support never removes a candidate;
* a ``certified_deduction`` removes only currently-live values and cites a
  certificate; in ``full`` mode that certificate must be present and its
  recomputed digest must equal its id (tamper detection);
* a ``nogood`` is never a certified deduction — a "deduction" with no certificate
  is a nogood masquerading as proof and is a violation;
* ``certified_unsat`` is impossible once any ``unknown`` / budget / truncation
  appears on the path;
* a ``solved`` terminal must carry a final verifier report;
* a truncated (bounded) snapshot is reported as **non-replayable**, never
  accepted as an exhaustive proof.

The event stream is a subset of the schema per producer: exact-closure decode
emits ``solver_state`` / ``support_result`` / ``certified_deduction`` /
``solver_terminal``; the reversible search controller additionally emits
``decision`` / ``backtrack`` / ``nogood``. The validator handles the full schema.

This module writes no Torch, runs no model, and makes no quality/ship claim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SOLVER_TRACE_SCHEMA_VERSION = 1

SOLVER_EVENT_KINDS = frozenset(
    {
        "solver_state",
        "support_result",
        "certified_deduction",
        "decision",
        "backtrack",
        "nogood",
        "solver_terminal",
    }
)

CERTIFICATE_MODES = ("none", "summary", "full")

# Bounded live-value snapshot per ``solver_state`` (privacy + boundedness).
_MAX_DOMAIN_SNAPSHOT = 512

# Verifier-report keys whose string values are provenance labels (never user
# text); every other string is dropped so no raw region text can leak.
_REPORT_STR_ALLOW = frozenset({"name", "profile", "status", "verifier", "verdict"})


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _value_key(value_dict: dict) -> str:
    return _canonical(value_dict)


def _hole_key(hole_dict: dict) -> str:
    return _canonical(hole_dict)


# --------------------------------------------------------------------------- #
# Event builders
# --------------------------------------------------------------------------- #


def solver_state_event(state, *, max_snapshot: int = _MAX_DOMAIN_SNAPSHOT) -> dict:
    """A ``solver_state`` event with a bounded live-domain snapshot.

    ``domain`` maps each hole to its live value keys so the validator can verify
    deductions/decisions remove/select only live values. If the snapshot exceeds
    ``max_snapshot`` it is truncated and ``trace_truncated`` is set — the validator
    then refuses to treat the trace as a replayable exhaustive proof.
    """
    domain: dict[str, list[str]] = {}
    total = 0
    truncated = False
    for hole in state.holes:
        hole_key = _hole_key(hole.hole_id.to_dict())
        values: list[str] = []
        for value in hole.values:
            if total >= max_snapshot:
                truncated = True
                break
            values.append(_value_key(value.to_dict()))
            total += 1
        domain[hole_key] = values
        if truncated:
            break
    return {
        "kind": "solver_state",
        "state_fingerprint": state.fingerprint,
        "problem_id": state.problem_id,
        "pack_id": state.pack_id,
        "constraint_version": state.constraint_version,
        "bounds": state.bounds.to_dict(),
        "decision_level": state.decision_level,
        "domain_summary": state.summary(),
        "domain": domain,
        "trace_truncated": truncated,
    }


def support_result_event(
    *,
    state_fingerprint: str,
    hole_id_dict: dict,
    candidate_dict: dict,
    verdict: str,
    certificate_id: str | None = None,
    witness_digest: str | None = None,
    stop_reason: str | None = None,
    coverage: tuple[str, ...] = (),
    counters: dict | None = None,
) -> dict:
    return {
        "kind": "support_result",
        "state_fingerprint": state_fingerprint,
        "hole_id": hole_id_dict,
        "candidate": candidate_dict,
        "verdict": verdict,
        "certificate_id": certificate_id,
        "witness_digest": witness_digest,
        "stop_reason": stop_reason,
        "coverage": list(coverage),
        "counters": counters or {},
    }


def certified_deduction_event(deduction) -> dict:
    return {"kind": "certified_deduction", **deduction.to_dict()}


def decision_event(decision) -> dict:
    return {"kind": "decision", **decision.to_dict()}


def backtrack_event(
    *,
    from_fingerprint: str,
    to_fingerprint: str,
    from_level: int,
    to_level: int,
    decision_id: str,
    conflict_kind: str,
) -> dict:
    return {
        "kind": "backtrack",
        "from_fingerprint": from_fingerprint,
        "to_fingerprint": to_fingerprint,
        "from_level": from_level,
        "to_level": to_level,
        "decision_id": decision_id,
        "conflict_kind": conflict_kind,
    }


def nogood_event(nogood) -> dict:
    return {"kind": "nogood", **nogood.to_dict()}


def _summarize_report(report: Any) -> Any:
    if report is None:
        return None
    if isinstance(report, bool):
        return report
    if isinstance(report, (int, float)):
        return report
    if isinstance(report, str):
        return report[:64]
    if isinstance(report, dict):
        summary: dict[str, Any] = {}
        for key, value in report.items():
            if isinstance(value, bool) or isinstance(value, (int, float)):
                summary[key] = value
            elif isinstance(value, str) and key in _REPORT_STR_ALLOW:
                summary[key] = value[:64]
        return summary
    return None


def solver_terminal_event(
    *,
    status: str,
    source_digest: str | None = None,
    verifier_report: Any = None,
    certificate_mode: str = "full",
    trace_truncated: bool = False,
) -> dict:
    return {
        "kind": "solver_terminal",
        "status": str(status),
        "source_digest": source_digest,
        "verifier_report": _summarize_report(verifier_report),
        "certificate_mode": certificate_mode,
        "trace_truncated": bool(trace_truncated),
    }


# --------------------------------------------------------------------------- #
# Certificate serialization by mode
# --------------------------------------------------------------------------- #


def serialize_certificates(certificate_store: dict, mode: str) -> dict:
    """Bounded, mode-gated certificate artifacts keyed by certificate id.

    ``none`` → ``{}`` (aggregate counters/status only); ``summary`` → compact,
    non-replayable descriptors; ``full`` → the replay material (each cert's
    ``to_dict()``, whose recomputed digest must equal its id).
    """
    if mode not in CERTIFICATE_MODES:
        raise ValueError(f"unsupported solver_certificate_mode: {mode!r}")
    if mode == "none":
        return {}
    out: dict[str, dict] = {}
    for cid, cert in certificate_store.items():
        payload = cert.to_dict()
        if mode == "full":
            out[cid] = payload
        else:  # summary
            out[cid] = {
                "schema_version": payload.get("schema_version"),
                "verdict": payload.get("verdict"),
                "exhausted": payload.get("exhausted"),
                "coverage_observations": payload.get("coverage_observations"),
                "witness_digest": payload.get("witness_digest"),
            }
    return out


# --------------------------------------------------------------------------- #
# Producers: solver result -> ordered event stream
# --------------------------------------------------------------------------- #


def solver_events_from_closure(
    result,
    root_state,
    *,
    certificate_mode: str = "full",
) -> list[dict]:
    """Event stream for one exact-closure decode prune.

    Ordered: root ``solver_state``, ``support_result`` (supported witnesses then
    unknown queries), ``certified_deduction`` (closure application order — passes
    chain by fingerprint), and a ``solver_terminal``.
    """
    events: list[dict] = [solver_state_event(root_state)]
    for witness in result.witnesses:
        events.append(
            support_result_event(
                state_fingerprint=root_state.fingerprint,
                hole_id_dict=witness.hole_id.to_dict(),
                candidate_dict=witness.value.to_dict(),
                verdict="supported",
                certificate_id=witness.certificate_id,
                witness_digest=witness.witness_digest,
                coverage=("complete",),
            )
        )
    for query in result.unknown_queries:
        events.append(
            support_result_event(
                state_fingerprint=query.state_fingerprint,
                hole_id_dict=query.hole_id.to_dict(),
                candidate_dict=query.candidate.to_dict(),
                verdict="unknown",
                stop_reason=result.stop_reason,
            )
        )
    for deduction in result.deductions:
        events.append(certified_deduction_event(deduction))
    truncated = any(e.get("trace_truncated") for e in events)
    events.append(
        solver_terminal_event(
            status=closure_status(result),
            certificate_mode=certificate_mode,
            trace_truncated=truncated,
        )
    )
    return events


def closure_status(result) -> str:
    """Honest terminal status for a closure *prune*.

    Closure never claims ``solved``: it prunes to a live subset but does not
    itself materialize a verifier-accepted terminal (that is the controller's job,
    and the decode's own final validate). It reports ``certified_unsat`` only on a
    certified bottom, ``budget_exhausted`` on a budget stop, else ``unknown``.
    """
    if result.state.is_bottom:
        return "certified_unsat"
    if result.stop_reason and result.stop_reason.startswith("budget"):
        return "budget_exhausted"
    return "unknown"


def solver_events_from_search(
    result,
    root_state,
    *,
    certificate_mode: str = "full",
    verifier_report: Any = None,
) -> list[dict]:
    """Event stream for a reversible search-controller run (decisions/nogoods)."""
    events: list[dict] = [solver_state_event(root_state)]
    for deduction in result.deductions:
        events.append(certified_deduction_event(deduction))
    for decision in result.decisions:
        events.append(decision_event(decision))
    for nogood in result.nogoods:
        events.append(nogood_event(nogood))
    events.append(
        solver_terminal_event(
            status=getattr(result.status, "value", str(result.status)),
            verifier_report=verifier_report
            if verifier_report is not None
            else result.verifier_report,
            certificate_mode=certificate_mode,
        )
    )
    return events


# --------------------------------------------------------------------------- #
# Aggregate counters
# --------------------------------------------------------------------------- #


def solver_trace_counters(events: list[dict]) -> dict[str, int]:
    """Per-kind aggregate counts derived from an event stream (for invariant 9)."""
    counts = {
        "solver_states": 0,
        "support_supported": 0,
        "support_unsupported": 0,
        "support_unknown": 0,
        "certified_deductions": 0,
        "certified_removed": 0,
        "decisions": 0,
        "backtracks": 0,
        "nogoods": 0,
    }
    for event in events:
        kind = event.get("kind")
        if kind == "solver_state":
            counts["solver_states"] += 1
        elif kind == "support_result":
            verdict = event.get("verdict")
            if verdict == "supported":
                counts["support_supported"] += 1
            elif verdict == "unsupported":
                counts["support_unsupported"] += 1
            elif verdict == "unknown":
                counts["support_unknown"] += 1
        elif kind == "certified_deduction":
            counts["certified_deductions"] += 1
            counts["certified_removed"] += len(event.get("removed", []))
        elif kind == "decision":
            counts["decisions"] += 1
        elif kind == "backtrack":
            counts["backtracks"] += 1
        elif kind == "nogood":
            counts["nogoods"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Replay validator
# --------------------------------------------------------------------------- #


def solver_replay_violations(
    events: list[dict],
    *,
    certificates: dict | None = None,
    certificate_mode: str = "full",
    counters: dict | None = None,
) -> list[str]:
    """Validate one solver event stream; empty list ⇒ replayable.

    Returns human-readable violation strings (never raises). Checks the ten
    VSS1-04 invariants: fingerprint lineage, live-only removals + certificate
    replay (full mode), unknown-never-removes, single-live decisions, backtrack
    lineage, nogood-not-a-deduction, solved-has-report, certified-unsat purity,
    counter agreement, and truncation honesty.
    """
    certificates = certificates or {}
    violations: list[str] = []

    active_fp: str | None = None
    live: dict[str, set[str]] = {}
    # backtrack targets: fingerprint -> (level, live snapshot)
    recorded: dict[str, tuple[int, dict[str, set[str]]]] = {}
    pending_before: str | None = None
    pending_after: str | None = None
    saw_unknown = False
    saw_budget = False
    saw_truncation = False

    def commit_pending() -> None:
        nonlocal active_fp, pending_before, pending_after
        if pending_before is not None and pending_after is not None:
            active_fp = pending_after
        pending_before = None
        pending_after = None

    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind not in SOLVER_EVENT_KINDS:
            violations.append(f"event {index}: unknown solver event kind {kind!r}")
            continue
        if event.get("trace_truncated"):
            saw_truncation = True

        if kind == "solver_state":
            commit_pending()
            active_fp = event.get("state_fingerprint")
            live = {
                hole_key: set(values)
                for hole_key, values in (event.get("domain") or {}).items()
            }
            recorded[active_fp] = (
                int(event.get("decision_level", 0)),
                {h: set(v) for h, v in live.items()},
            )

        elif kind == "support_result":
            verdict = event.get("verdict")
            if verdict == "unknown":
                saw_unknown = True
            if event.get("stop_reason", "") and str(
                event.get("stop_reason")
            ).startswith("budget"):
                saw_budget = True

        elif kind == "certified_deduction":
            before = event.get("before_fingerprint")
            after = event.get("after_fingerprint")
            # Passes share a before/after; a new before commits the prior pass.
            if before != pending_before:
                commit_pending()
                if active_fp is not None and before != active_fp:
                    violations.append(
                        f"event {index}: deduction before_fingerprint "
                        f"{before!r} != active state {active_fp!r}"
                    )
                pending_before = before
            pending_after = after
            hole_key = _hole_key(event.get("hole_id", {}))
            removed = [_value_key(v) for v in event.get("removed", [])]
            cert_ids = event.get("certificate_ids", [])
            if not cert_ids:
                violations.append(
                    f"event {index}: certified_deduction cites no certificate "
                    "(a nogood must not be relabeled a certified deduction)"
                )
            live_here = live.get(hole_key, set())
            for value_key in removed:
                if value_key not in live_here:
                    violations.append(
                        f"event {index}: deduction removes non-live value at hole"
                    )
                else:
                    live_here.discard(value_key)
            live[hole_key] = live_here
            if certificate_mode == "full":
                for cid in cert_ids:
                    if cid not in certificates:
                        violations.append(
                            f"event {index}: certificate {cid[:12]}… missing in full mode"
                        )
                    elif _digest(certificates[cid]) != cid:
                        violations.append(
                            f"event {index}: certificate {cid[:12]}… digest mismatch (tampered)"
                        )

        elif kind == "decision":
            commit_pending()
            before = event.get("before_fingerprint")
            if active_fp is not None and before != active_fp:
                violations.append(
                    f"event {index}: decision before_fingerprint "
                    f"{before!r} != active state {active_fp!r}"
                )
            hole_key = _hole_key(event.get("hole_id", {}))
            chosen = _value_key(event.get("chosen", {}))
            live_here = live.get(hole_key, set())
            if chosen not in live_here:
                violations.append(
                    f"event {index}: decision selects a non-live value"
                )
            alternatives = {_value_key(v) for v in event.get("alternatives", [])}
            expected_alts = live_here - {chosen}
            if alternatives != expected_alts:
                violations.append(
                    f"event {index}: decision alternatives do not match remaining live values"
                )
            after = event.get("after_fingerprint")
            recorded[before] = (
                int(event.get("level", 0)),
                {h: set(v) for h, v in live.items()},
            )
            live = {h: set(v) for h, v in live.items()}
            live[hole_key] = {chosen}
            active_fp = after

        elif kind == "backtrack":
            commit_pending()
            to_fp = event.get("to_fingerprint")
            to_level = int(event.get("to_level", 0))
            if to_fp not in recorded:
                violations.append(
                    f"event {index}: backtrack to unrecorded state {to_fp!r}"
                )
            else:
                level, snapshot = recorded[to_fp]
                if level != to_level:
                    violations.append(
                        f"event {index}: backtrack to_level {to_level} != recorded level {level}"
                    )
                active_fp = to_fp
                live = {h: set(v) for h, v in snapshot.items()}

        elif kind == "nogood":
            if not event.get("provenance"):
                violations.append(
                    f"event {index}: nogood missing provenance"
                )

        elif kind == "solver_terminal":
            commit_pending()
            status = event.get("status")
            if status == "solved" and event.get("verifier_report") is None:
                violations.append(
                    f"event {index}: solved terminal without a verifier report"
                )
            if status == "certified_unsat" and (
                saw_unknown or saw_budget or saw_truncation
            ):
                violations.append(
                    f"event {index}: certified_unsat with unknown/budget/truncation on the path"
                )

    if saw_truncation:
        violations.append(
            "solver trace is truncated: bounded evidence is not a replayable "
            "exhaustive proof"
        )

    if counters is not None:
        derived = solver_trace_counters(events)
        for key, value in derived.items():
            if key in counters and int(counters[key]) != int(value):
                violations.append(
                    f"counter {key} mismatch: trace {counters[key]} != events {value}"
                )

    return violations
