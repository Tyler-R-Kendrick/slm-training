"""Historical formal-timeout reclassification is incomplete measurement, not rejection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "repair_formal_timeout_history.py"
)
_SPEC = importlib.util.spec_from_file_location("repair_formal_timeout_history", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_repair_reclassifies_formal_unknown_failed_as_inconclusive(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    loop_dir = root / "loops" / loop
    loop_dir.mkdir(parents=True)

    camp = root / "continuous-loop-camp-to"
    art = camp / "artifacts" / "formal_preflights"
    art.mkdir(parents=True)
    (art / "abc.json").write_text(
        json.dumps(
            {
                "status": "unknown",
                "duration_seconds": 179.3,
                "schema_version": "FormalPreflightV1",
            }
        ),
        encoding="utf-8",
    )
    (camp / "formal_preflight_status.json").write_text(
        json.dumps(
            {
                "schema": "autotrain_formal_preflight_status/v1",
                "status": "unknown",
                "reason": "preflight_status=unknown",
            }
        ),
        encoding="utf-8",
    )

    queue = [
        {
            "entry_id": "champ-1",
            "status": "promotion_failed",
            "formal_preflight_status": "unknown",
            "promotion_campaign_id": "continuous-loop-camp-to",
            "promote_attempts": 1,
            "resolved_at": "2026-08-01T00:00:00Z",
            "resolve_reasons": [
                "formal_preflight_unproved:status='unknown'",
                "primary_metric_unavailable",
            ],
        },
        {
            # Real fixture fail — leave alone
            "entry_id": "champ-2",
            "status": "promotion_failed",
            "resolve_reasons": [
                "fixture_insufficient_n:x",
                "primary_metric_unavailable",
                "fixture_insufficient_n_alone",
            ],
        },
    ]
    ledger = [
        {
            "entry_id": "champ-1",
            "campaign_id": "continuous-loop-camp-to",
            "cycle_index": 9,
            "outcome": "promotion_failed",
            "formal_preflight_status": "unknown",
            "reasons": [
                "formal_preflight_unproved:status='unknown'",
                "primary_metric_unavailable",
            ],
        }
    ]
    (loop_dir / "champion_queue.jsonl").write_text(
        "\n".join(json.dumps(r) for r in queue) + "\n", encoding="utf-8"
    )
    (loop_dir / "learning_certificate_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8"
    )

    summary = _mod.repair_loop(root=root, loop_id=loop, dry_run=False)
    assert summary["queue_reclassified"] == 1
    assert summary["ledger_reclassified"] == 1

    q = [
        json.loads(line)
        for line in (loop_dir / "champion_queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert q[0]["status"] == "promotion_inconclusive"
    assert q[0]["formal_preflight_status"] == "timed_out"
    assert q[0]["promote_attempts"] == 0
    assert "resolved_at" not in q[0]
    assert any("formal_preflight_timed_out" in r for r in q[0]["resolve_reasons"])
    assert q[1]["status"] == "promotion_failed"  # untouched

    led = [
        json.loads(l)
        for l in (loop_dir / "learning_certificate_ledger.jsonl")
        .read_text()
        .splitlines()
        if l.strip()
    ]
    assert led[0]["outcome"] == "promotion_inconclusive"
    assert led[0]["formal_preflight_status"] == "timed_out"

    st = json.loads((camp / "formal_preflight_status.json").read_text(encoding="utf-8"))
    assert st["status"] == "timed_out"
    assert st.get("timed_out") is True

    audit = (loop_dir / "historical_reclassification.jsonl").read_text(encoding="utf-8")
    assert "promotion_inconclusive" in audit
    assert "champion_queue" in audit


def test_looks_like_old_wall_timeout_duration_band() -> None:
    assert _mod._looks_like_old_wall_timeout(
        reasons=["formal_preflight_unproved:status='unknown'"],
        formal_status="unknown",
        duration_s=179.2,
    )
    assert not _mod._looks_like_old_wall_timeout(
        reasons=["fixture_insufficient_n_alone"],
        formal_status=None,
        duration_s=None,
    )
