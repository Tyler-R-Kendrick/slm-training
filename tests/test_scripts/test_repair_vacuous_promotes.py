"""Historical vacuous cert-only promotes reclass to promotion_failed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "repair_vacuous_promotes.py"
_SPEC = importlib.util.spec_from_file_location("repair_vacuous_promotes", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def test_looks_like_vacuous_promote_rules() -> None:
    assert _mod.looks_like_vacuous_promote(
        status="promoted",
        reasons=["cert_policy:continue", "phase_a_positive"],
        cert_policy="continue",
        formal_status="proved",
    )
    assert not _mod.looks_like_vacuous_promote(
        status="promoted",
        reasons=[
            "cert_policy:continue",
            "promote_primary_win:held_out.structural_similarity:0.1->0.2",
        ],
        cert_policy="continue",
        formal_status="proved",
    )
    assert not _mod.looks_like_vacuous_promote(
        status="promotion_failed",
        reasons=["cert_policy:continue"],
        cert_policy="continue",
        formal_status="proved",
    )


def test_repair_reclassifies_vacuous_promotes(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    loop_dir = root / "loops" / loop
    loop_dir.mkdir(parents=True)

    queue = [
        {
            "entry_id": "champ-vacuous",
            "status": "promoted",
            "cert_policy": "continue",
            "formal_preflight_status": "proved",
            "resolve_reasons": [
                "cert_policy:continue",
                "primary_metric_null_or_worse:held_out.structural_similarity:"
                "control=0.25 candidate=0.25 improvement=0.0",
            ],
        },
        {
            "entry_id": "champ-real",
            "status": "promoted",
            "cert_policy": "continue",
            "formal_preflight_status": "proved",
            "resolve_reasons": [
                "cert_policy:continue",
                "primary_metric_win:held_out.structural_similarity:0.1->0.2:improvement=0.1",
            ],
        },
    ]
    ledger = [
        {
            "entry_id": "champ-vacuous",
            "campaign_id": "c1",
            "cycle_index": 3,
            "outcome": "promoted",
            "cert_policy": "continue",
            "formal_preflight_status": "proved",
            "reasons": ["cert_policy:continue", "phase_a_quality_held"],
        },
        {
            "entry_id": "champ-real",
            "campaign_id": "c2",
            "outcome": "promoted",
            "cert_policy": "continue",
            "reasons": [
                "cert_policy:continue",
                "promote_primary_win:held_out.structural_similarity:0.1->0.2",
            ],
        },
    ]
    (loop_dir / "champion_queue.jsonl").write_text(
        "\n".join(json.dumps(r) for r in queue) + "\n", encoding="utf-8"
    )
    (loop_dir / "learning_certificate_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8"
    )

    dry = _mod.repair_loop(root=root, loop_id=loop, dry_run=True)
    assert dry["queue_reclassified"] == 1
    assert dry["ledger_reclassified"] == 1
    # dry-run must not rewrite
    q_dry = [
        json.loads(line)
        for line in (loop_dir / "champion_queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert q_dry[0]["status"] == "promoted"

    summary = _mod.repair_loop(root=root, loop_id=loop, dry_run=False)
    assert summary["queue_reclassified"] == 1
    assert summary["ledger_reclassified"] == 1

    q = [
        json.loads(line)
        for line in (loop_dir / "champion_queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert q[0]["status"] == "promotion_failed"
    assert q[0]["promotion_primary_met"] is False
    assert any("vacuous_cert_null_primary" in r for r in q[0]["resolve_reasons"])
    assert q[1]["status"] == "promoted"

    led = [
        json.loads(line)
        for line in (loop_dir / "learning_certificate_ledger.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert led[0]["outcome"] == "promotion_failed"
    assert led[1]["outcome"] == "promoted"

    audit = (loop_dir / "historical_reclassification.jsonl").read_text(encoding="utf-8")
    assert "repair_vacuous_promotes" in audit
    assert "champ-vacuous" in audit
