#!/usr/bin/env python3
"""Reclassify historical vacuous cert-only promotes as promotion_failed.

Pre-effect-gate continuous dispose authorized ``promoted`` on formal proved +
LeverProof cert ``continue`` alone. Null held-out primary deltas were counted
as learning wins. This repair rewrites loop-local::

  <root>/loops/<loop_id>/champion_queue.jsonl
  <root>/loops/<loop_id>/learning_certificate_ledger.jsonl

Appends audit::

  <root>/loops/<loop_id>/historical_reclassification.jsonl

Keep rows that already show a held-out / promote primary win.

Usage::

  python -m scripts.repair_vacuous_promotes \\
    --root outputs/autoresearch --loop-id continuous-openui-20260730
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

_RECLASS_TAG = (
    "historical_reclassification:promoted→promotion_failed:vacuous_cert_null_primary"
)
_MIN_EFFECT_DEFAULT = 0.01


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


def _reasons_list(row: dict[str, Any], *, key: str) -> list[str]:
    raw = row.get(key) or []
    if not isinstance(raw, list):
        return []
    return [str(r) for r in raw]


def _has_cert_continue(row: dict[str, Any], reasons: list[str]) -> bool:
    if str(row.get("cert_policy") or "") == "continue":
        return True
    return any(
        r == "cert_policy:continue" or r.startswith("cert_policy:continue")
        for r in reasons
    )


def _has_primary_win(reasons: list[str]) -> bool:
    for r in reasons:
        if r.startswith("promote_primary_win:"):
            return True
        if r.startswith("primary_metric_win:held_out."):
            return True
        if r.startswith("primary_metric_win:structural_similarity"):
            return True
        # Phase A win string with improvement parse when present.
        if "primary_metric_win:held_out.structural_similarity" in r:
            return True
    return False


def _explicit_null_primary(reasons: list[str]) -> bool:
    return any(
        "primary_metric_null_or_worse" in r
        or "promote_primary_null_or_insufficient" in r
        or r.startswith("primary_metric_unavailable")
        for r in reasons
    )


def looks_like_vacuous_promote(
    *,
    status: str,
    reasons: list[str],
    cert_policy: object = None,
    formal_status: object = None,
) -> bool:
    """True when promoted under cert continue without held-out primary win."""
    if status != "promoted":
        return False
    row_proxy = {"cert_policy": cert_policy}
    if not _has_cert_continue(row_proxy, reasons):
        # Older rows may only say phase_a without cert_policy tag; require
        # formal proved + no primary win still is not enough without cert
        # evidence — only reclass when cert continue is present.
        return False
    if _has_primary_win(reasons):
        return False
    # Vacuous if null primary mentioned, or no primary win at all with cert continue.
    if _explicit_null_primary(reasons):
        return True
    # cert continue + no promote/held_out primary win → vacuous under old dispose.
    if formal_status in {None, "proved"} or formal_status == "proved":
        return True
    return False


def _stamp_reasons(reasons: list[str]) -> list[str]:
    out = [r for r in reasons if not str(r).startswith("historical_reclassification:")]
    if _RECLASS_TAG not in out:
        out.append(_RECLASS_TAG)
    if not any("promote_primary_null_or_insufficient" in r for r in out):
        out.append(
            "promote_primary_null_or_insufficient:historical:"
            f"min_effect={_MIN_EFFECT_DEFAULT}:no_primary_win_with_cert_continue"
        )
    return out


def repair_loop(
    *,
    root: Path,
    loop_id: str,
    dry_run: bool = False,
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
        reasons = _reasons_list(row, key="resolve_reasons")
        if not looks_like_vacuous_promote(
            status=str(row.get("status") or ""),
            reasons=reasons,
            cert_policy=row.get("cert_policy"),
            formal_status=row.get("formal_preflight_status"),
        ):
            continue
        before = {
            "status": row.get("status"),
            "cert_policy": row.get("cert_policy"),
            "formal_preflight_status": row.get("formal_preflight_status"),
            "resolve_reasons": list(reasons),
            "primary_improvement": row.get("primary_improvement"),
            "promotion_primary_met": row.get("promotion_primary_met"),
        }
        row["status"] = "promotion_failed"
        row["resolve_reasons"] = _stamp_reasons(reasons)
        row["promotion_primary_met"] = False
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promoted→promotion_failed:vacuous_cert_null_primary"
        )
        queue_changed += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "champion_queue",
                "repair": "repair_vacuous_promotes",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": row.get("promotion_campaign_id")
                or row.get("confirm_campaign_id"),
                "before": before,
                "after_status": "promotion_failed",
                "reclassified_at": now,
            }
        )

    ledger_changed = 0
    for row in ledger:
        reasons = _reasons_list(row, key="reasons")
        if not looks_like_vacuous_promote(
            status=str(row.get("outcome") or ""),
            reasons=reasons,
            cert_policy=row.get("cert_policy"),
            formal_status=row.get("formal_preflight_status"),
        ):
            continue
        before = {
            "outcome": row.get("outcome"),
            "cert_policy": row.get("cert_policy"),
            "formal_preflight_status": row.get("formal_preflight_status"),
            "reasons": list(reasons),
            "primary_improvement": row.get("primary_improvement"),
            "promotion_primary_met": row.get("promotion_primary_met"),
        }
        row["outcome"] = "promotion_failed"
        row["reasons"] = _stamp_reasons(reasons)
        row["promotion_primary_met"] = False
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            "promoted→promotion_failed:vacuous_cert_null_primary"
        )
        ledger_changed += 1
        audit_events.append(
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "learning_certificate_ledger",
                "repair": "repair_vacuous_promotes",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": row.get("campaign_id"),
                "cycle_index": row.get("cycle_index"),
                "before": before,
                "after_outcome": "promotion_failed",
                "reclassified_at": now,
            }
        )

    if not dry_run and (queue_changed or ledger_changed):
        if queue_changed:
            _write_jsonl(queue_path, queue)
        if ledger_changed:
            _write_jsonl(ledger_path, ledger)
        if audit_events:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as fh:
                for ev in audit_events:
                    fh.write(json.dumps(ev, sort_keys=True) + "\n")

    summary = {
        "schema": "repair_vacuous_promotes/v1",
        "loop_id": loop_id,
        "root": str(root),
        "dry_run": dry_run,
        "queue_reclassified": queue_changed,
        "ledger_reclassified": ledger_changed,
        "audit_events": len(audit_events),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Autoresearch root containing loops/<loop_id>/",
    )
    p.add_argument("--loop-id", required=True, help="Continuous loop id")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count reclass candidates without rewriting JSONL",
    )
    args = p.parse_args(argv)
    summary = repair_loop(
        root=args.root.resolve(),
        loop_id=args.loop_id,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
