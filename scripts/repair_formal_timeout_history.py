#!/usr/bin/env python3
"""Reclassify historical promote false-fails (formal timeout + harness).

1. **Formal wall timeouts** (pre-#1251): ~180s wall → status ``unknown`` →
   ``promotion_failed``. Reclass → ``promotion_inconclusive`` / ``timed_out``.

2. **Harness gaps** (matrix membership rewrite, missing promote run, cert
   incomplete with no candidate metrics while formal proved): reclass →
   ``harness_failure`` (not model ``promotion_failed``).

Rewrites loop-local::

  <root>/loops/<loop_id>/champion_queue.jsonl
  <root>/loops/<loop_id>/learning_certificate_ledger.jsonl

Appends audit::

  <root>/loops/<loop_id>/historical_reclassification.jsonl

Does **not** rewrite content-addressed formal preflight digests.

Usage::

  python -m scripts.repair_formal_timeout_history \\
    --root outputs/autoresearch --loop-id continuous-openui-20260730
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


_OLD_WALL_S = 180.0
_DURATION_LO = 170.0
_DURATION_HI = 190.0
_FORMAL_UNKNOWN_REASON = "formal_preflight_unproved:status='unknown'"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _formal_duration_for_campaign(camp_dir: Path) -> float | None:
    art = camp_dir / "artifacts" / "formal_preflights"
    if not art.is_dir():
        return None
    for path in sorted(art.glob("*.json")):
        data = _read_json(path)
        dur = data.get("duration_seconds")
        if isinstance(dur, (int, float)):
            return float(dur)
    return None


def _looks_like_old_wall_timeout(
    *,
    reasons: list[str],
    formal_status: str | None,
    duration_s: float | None,
) -> bool:
    reason_hit = any(
        _FORMAL_UNKNOWN_REASON in str(r)
        or "formal_preflight_unproved:status='unknown'" in str(r)
        or "formal_preflight_timed_out" in str(r)
        for r in reasons
    )
    status_hit = formal_status in {"unknown", "timed_out"}
    dur_hit = (
        duration_s is not None and _DURATION_LO <= float(duration_s) < _DURATION_HI
    )
    # Prefer duration evidence when present; else formal_unproved unknown alone.
    if dur_hit and (status_hit or reason_hit or formal_status is None):
        return True
    if reason_hit and status_hit:
        return True
    if reason_hit and formal_status in {None, "unknown"}:
        return True
    return False


def _reclass_reasons(reasons: list[str], *, wall_s: float, duration_s: float | None) -> list[str]:
    out: list[str] = []
    for r in reasons:
        if _FORMAL_UNKNOWN_REASON in str(r) or "formal_preflight_unproved:status='unknown'" in str(
            r
        ):
            continue
        if str(r).startswith("formal_preflight_unproved:"):
            continue
        out.append(r)
    out.insert(
        0,
        f"formal_preflight_timed_out:status='timed_out':wall_s={wall_s:g}"
        + (f":duration_s={duration_s:.3f}" if duration_s is not None else ""),
    )
    out.insert(1, "measurement_incomplete:formal_timeout_not_rejection")
    out.append(
        "historical_reclassification:promotion_failed→promotion_inconclusive"
        f":old_wall_s={wall_s:g}:repair=repair_formal_timeout_history"
    )
    return out


def repair_loop(
    *,
    root: Path,
    loop_id: str,
    dry_run: bool = False,
    old_wall_s: float = _OLD_WALL_S,
) -> dict[str, Any]:
    loop_dir = root / "loops" / loop_id
    queue_path = loop_dir / "champion_queue.jsonl"
    ledger_path = loop_dir / "learning_certificate_ledger.jsonl"
    audit_path = loop_dir / "historical_reclassification.jsonl"

    queue = _load_jsonl(queue_path)
    ledger = _load_jsonl(ledger_path)
    audit_events: list[dict[str, Any]] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    queue_changed = 0
    for row in queue:
        if row.get("status") != "promotion_failed":
            continue
        reasons = [str(r) for r in (row.get("resolve_reasons") or [])]
        formal = row.get("formal_preflight_status")
        camp_id = row.get("promotion_campaign_id") or row.get("confirm_campaign_id")
        duration: float | None = None
        camp_dir: Path | None = None
        if camp_id:
            camp_dir = root / str(camp_id)
            if camp_dir.is_dir():
                duration = _formal_duration_for_campaign(camp_dir)
        if not _looks_like_old_wall_timeout(
            reasons=reasons, formal_status=str(formal) if formal else None, duration_s=duration
        ):
            continue
        before = {
            "status": row.get("status"),
            "formal_preflight_status": row.get("formal_preflight_status"),
            "resolve_reasons": list(reasons),
            "promote_attempts": row.get("promote_attempts"),
            "resolved_at": row.get("resolved_at"),
        }
        row["status"] = "promotion_inconclusive"
        row["formal_preflight_status"] = "timed_out"
        row["resolve_reasons"] = _reclass_reasons(
            reasons, wall_s=old_wall_s, duration_s=duration
        )
        row["last_inconclusive_at"] = now
        row["last_formal_timeout"] = True
        row["last_formal_timeout_wall_s"] = old_wall_s
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promotion_failed→promotion_inconclusive:formal_timeout"
        )
        # Retriable head: clear permanent resolve; refund attempts.
        row.pop("resolved_at", None)
        row["promote_attempts"] = 0
        queue_changed += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "champion_queue",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": camp_id,
                "duration_seconds": duration,
                "before": before,
                "after_status": "promotion_inconclusive",
                "reclassified_at": now,
            }
        )
        # Annotate campaign formal status file when present.
        if camp_dir is not None and (camp_dir / "formal_preflight_status.json").is_file():
            st_path = camp_dir / "formal_preflight_status.json"
            st = _read_json(st_path)
            if st.get("status") in {"unknown", "timed_out", None}:
                st["status"] = "timed_out"
                st["timed_out"] = True
                st["timeout_seconds"] = old_wall_s
                if duration is not None:
                    st["duration_seconds"] = duration
                st["reason"] = (
                    f"historical_reclassification:formal_timeout:old_wall_s={old_wall_s:g}"
                    + (f":duration_s={duration:.3f}" if duration is not None else "")
                )
                st["historical_reclassified_at"] = now
                if not dry_run:
                    st_path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")

    ledger_changed = 0
    for row in ledger:
        if row.get("outcome") != "promotion_failed":
            continue
        reasons = [str(r) for r in (row.get("reasons") or [])]
        formal = row.get("formal_preflight_status")
        camp_id = row.get("campaign_id")
        duration: float | None = None
        if camp_id:
            camp_dir = root / str(camp_id)
            if camp_dir.is_dir():
                duration = _formal_duration_for_campaign(camp_dir)
        if not _looks_like_old_wall_timeout(
            reasons=reasons, formal_status=str(formal) if formal else None, duration_s=duration
        ):
            continue
        before = {
            "outcome": row.get("outcome"),
            "formal_preflight_status": row.get("formal_preflight_status"),
            "reasons": list(reasons),
        }
        row["outcome"] = "promotion_inconclusive"
        row["formal_preflight_status"] = "timed_out"
        row["reasons"] = _reclass_reasons(reasons, wall_s=old_wall_s, duration_s=duration)
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promotion_failed→promotion_inconclusive:formal_timeout"
        )
        ledger_changed += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "learning_certificate_ledger",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": camp_id,
                "cycle_index": row.get("cycle_index"),
                "duration_seconds": duration,
                "before": before,
                "after_outcome": "promotion_inconclusive",
                "reclassified_at": now,
            }
        )

    # Pass 2: harness failures (formal proved / incomplete cert without promote run).
    harness_queue = 0
    harness_ledger = 0

    def _looks_like_harness(reasons: list[str], formal: object, camp_id: object) -> bool:
        joined = " ".join(reasons)
        if any(str(r).startswith("harness_failure:") for r in reasons):
            return True
        if "incomplete_metrics" in joined and formal == "proved":
            return True
        if "promote_requires_certificate" in joined and formal == "proved":
            # Certificate missing after proved formal was often promote-arm abort.
            if camp_id:
                camp = root / str(camp_id)
                runs = camp / "runs"
                if runs.is_dir():
                    names = [p.name for p in runs.iterdir() if p.is_dir()]
                    has_promo = any("-promote" in n for n in names)
                    has_ctrl = any(n.endswith("-control") for n in names)
                    if has_ctrl and not has_promo:
                        return True
                else:
                    return True
            return "primary_metric_unavailable" in joined
        if "exact member of the latest hypothesis matrix" in joined:
            return True
        return False

    def _harness_reasons(reasons: list[str]) -> list[str]:
        out = [r for r in reasons if not str(r).startswith("historical_reclassification:")]
        tags = [
            "harness_failure:missing_promote_run",
            "harness_failure:cert_export_no_candidate_metrics",
            "measurement_incomplete:harness_not_model_reject",
            "historical_reclassification:promotion_failed→harness_failure:matrix_or_missing_run",
        ]
        for t in tags:
            if t not in out:
                out.insert(0, t)
        return out

    for row in queue:
        if row.get("status") != "promotion_failed":
            continue
        reasons = [str(r) for r in (row.get("resolve_reasons") or [])]
        formal = row.get("formal_preflight_status")
        camp_id = row.get("promotion_campaign_id") or row.get("confirm_campaign_id")
        if not _looks_like_harness(reasons, formal, camp_id):
            continue
        before = {
            "status": row.get("status"),
            "formal_preflight_status": formal,
            "resolve_reasons": list(reasons),
            "promote_attempts": row.get("promote_attempts"),
        }
        row["status"] = "harness_failure"
        row["resolve_reasons"] = _harness_reasons(reasons)
        row["last_harness_failure_at"] = now
        row["last_harness_failure"] = True
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promotion_failed→harness_failure:missing_promote_or_cert"
        )
        row.pop("resolved_at", None)
        row["promote_attempts"] = 0
        harness_queue += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "champion_queue",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": camp_id,
                "before": before,
                "after_status": "harness_failure",
                "reclassified_at": now,
            }
        )

    for row in ledger:
        if row.get("outcome") != "promotion_failed":
            continue
        reasons = [str(r) for r in (row.get("reasons") or [])]
        formal = row.get("formal_preflight_status")
        camp_id = row.get("campaign_id")
        if not _looks_like_harness(reasons, formal, camp_id):
            continue
        before = {
            "outcome": row.get("outcome"),
            "formal_preflight_status": formal,
            "reasons": list(reasons),
        }
        row["outcome"] = "harness_failure"
        row["reasons"] = _harness_reasons(reasons)
        row["harness_failure"] = True
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promotion_failed→harness_failure:missing_promote_or_cert"
        )
        harness_ledger += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "learning_certificate_ledger",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": camp_id,
                "cycle_index": row.get("cycle_index"),
                "before": before,
                "after_outcome": "harness_failure",
                "reclassified_at": now,
            }
        )

    # Pass 3: incomplete measurement / blind primary falsely labeled rejected or
    # promotion_failed (not a quality retest with real dual-arm metrics).
    incomplete_queue = 0

    def _is_quality_retest_reject(reasons: list[str]) -> bool:
        """True when confirm/promote recorded a real quality or latency comparison."""
        for r in reasons:
            s = str(r)
            if s.startswith("primary_metric_null_or_worse:") and "control=" in s:
                # Real numeric comparison was available.
                if "control=None" in s and "candidate=None" in s:
                    continue
                return True
            if s.startswith("latency_win_rejected_low_mpr:"):
                return True
            if s.startswith("latency_win_rejected_timeout_band:"):
                return True
            if s.startswith("quality_win_rejected_latency_budget:"):
                return True
            if s.startswith("confirm_attempts_exceeded:"):
                return True
        return False

    def _is_incomplete_only(reasons: list[str]) -> bool:
        if not reasons:
            return False
        if _is_quality_retest_reject(reasons):
            return False
        incomplete_markers = (
            "empty_metrics:",
            "measurement_incomplete:",
            "primary_metric_unavailable",
            "fixture_insufficient_n",
            "fixture_insufficient_n_alone",
            "promote_requires_certificate:",
            "promote_cert_incomplete",
        )
        joined = " ".join(reasons)
        return any(m in joined for m in incomplete_markers)

    def _incomplete_reasons(reasons: list[str], *, from_status: str) -> list[str]:
        out = [r for r in reasons if not str(r).startswith("historical_reclassification:")]
        tags = [
            "harness_failure:measurement_incomplete_not_model_reject",
            "measurement_incomplete:harness_or_metric_plumbing",
            f"historical_reclassification:{from_status}→harness_failure:incomplete_only",
        ]
        for t in reversed(tags):
            if t not in out:
                out.insert(0, t)
        return out

    for row in queue:
        st = row.get("status")
        if st not in {"rejected", "promotion_failed"}:
            continue
        reasons = [str(r) for r in (row.get("resolve_reasons") or [])]
        if not _is_incomplete_only(reasons):
            continue
        before = {
            "status": st,
            "resolve_reasons": list(reasons),
            "formal_preflight_status": row.get("formal_preflight_status"),
        }
        row["status"] = "harness_failure"
        row["resolve_reasons"] = _incomplete_reasons(reasons, from_status=str(st))
        row["last_harness_failure_at"] = now
        row["last_harness_failure"] = True
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            f"{st}→harness_failure:incomplete_measurement_only"
        )
        row.pop("resolved_at", None)
        if "promote_attempts" in row:
            row["promote_attempts"] = 0
        incomplete_queue += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "champion_queue",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": row.get("promotion_campaign_id")
                or row.get("confirm_campaign_id"),
                "before": before,
                "after_status": "harness_failure",
                "reclassified_at": now,
            }
        )

    summary = {
        "loop_id": loop_id,
        "root": str(root),
        "dry_run": dry_run,
        "queue_reclassified": queue_changed,
        "ledger_reclassified": ledger_changed,
        "harness_queue_reclassified": harness_queue,
        "harness_ledger_reclassified": harness_ledger,
        "incomplete_queue_reclassified": incomplete_queue,
        "audit_events": len(audit_events),
        "old_wall_s": old_wall_s,
    }

    if dry_run:
        print(json.dumps({"summary": summary, "events": audit_events}, indent=2))
        return summary

    if queue_changed or harness_queue or incomplete_queue:
        _write_jsonl(queue_path, queue)
    if ledger_changed or harness_ledger:
        _write_jsonl(ledger_path, ledger)
    if audit_events:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as fh:
            for ev in audit_events:
                fh.write(json.dumps(ev, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Campaign bundle root (contains loops/)",
    )
    parser.add_argument("--loop-id", required=True)
    parser.add_argument(
        "--old-wall-s",
        type=float,
        default=_OLD_WALL_S,
        help="Historical formal wall used when status was mis-recorded (default 180)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned reclassifications without writing",
    )
    args = parser.parse_args(argv)
    root = args.root if args.root.is_absolute() else Path.cwd() / args.root
    repair_loop(
        root=root,
        loop_id=args.loop_id,
        dry_run=bool(args.dry_run),
        old_wall_s=float(args.old_wall_s),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
