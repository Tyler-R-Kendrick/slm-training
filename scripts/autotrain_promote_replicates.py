"""Replicate evidence gating a promotion.

One responsibility: recording each promotion replicate, deciding whether the
recorded evidence is still current, and gating the promotion on it. A promotion
that cannot show current replicates is not a promotion.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from scripts.autotrain_io import read_json
from scripts.autotrain_paths import promotion_replicate_ledger_path

PROMOTION_REPLICATE_SCHEMA = "autotrain_promotion_replicate/v1"


def promotion_replicate_sha(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def promotion_replicate_evidence_is_current(root: Path, row: dict[str, Any]) -> bool:
    campaign_id = row.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        return False
    camp_dir = (root / campaign_id).resolve()
    if not camp_dir.is_relative_to(root.resolve()):
        return False
    delivery_path = camp_dir / "sdlc_delivery.json"
    delivery = read_json(delivery_path)
    evidence = row.get("evidence")
    if not delivery_path.is_file() or not isinstance(evidence, dict):
        return False
    if (
        evidence.get("delivery_sha256")
        != hashlib.sha256(delivery_path.read_bytes()).hexdigest()
    ):
        return False
    control_id = str(row.get("control_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    seed = row.get("seed")
    order = row.get("arm_order")
    if (
        type(seed) is not int
        or not control_id
        or not candidate_id
        or not isinstance(order, list)
        or len(order) != 2
        or any(not isinstance(item, str) for item in order)
        or set(order) != {control_id, candidate_id}
        or delivery.get("measurement_complete") is not True
        or delivery.get("control_id") != control_id
        or delivery.get("candidate_id") != candidate_id
        or delivery.get("arm_seed") != seed
        or delivery.get("arm_order") != order
    ):
        return False
    metrics_sha = promotion_replicate_sha(
        {
            "control": delivery.get("control_metrics") or {},
            "candidate": delivery.get("candidate_metrics") or {},
        }
    )
    if evidence.get("metrics_sha256") != metrics_sha:
        return False
    manifest_digests = evidence.get("manifests")
    if not isinstance(manifest_digests, dict) or set(manifest_digests) != {
        control_id,
        candidate_id,
    }:
        return False
    for arm_id in (control_id, candidate_id):
        manifest_path = camp_dir / "manifests" / f"{arm_id}.json"
        manifest = read_json(manifest_path)
        if (
            not manifest_path.is_file()
            or seed not in (manifest.get("seeds") or [])
            or manifest_digests.get(arm_id)
            != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ):
            return False
    certificate = camp_dir / "metric-certificate.json"
    return (
        certificate.is_file()
        and evidence.get("certificate_sha256")
        == hashlib.sha256(certificate.read_bytes()).hexdigest()
    )


def verified_promotion_replicates(
    root: Path, loop_id: str, entry: dict[str, Any]
) -> list[dict[str, Any]]:
    path = promotion_replicate_ledger_path(root, loop_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("schema") != PROMOTION_REPLICATE_SCHEMA:
            continue
        if row.get("entry_id") != entry.get("entry_id") or row.get(
            "knobs_fingerprint"
        ) != entry.get("knobs_fingerprint"):
            continue
        claimed = row.get("content_sha256")
        content = {key: value for key, value in row.items() if key != "content_sha256"}
        if claimed != promotion_replicate_sha(content):
            continue
        if not promotion_replicate_evidence_is_current(root, row):
            continue
        rows.append(row)
    return rows


def record_promotion_replicate(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
    camp_dir: Path,
    delivery: dict[str, Any],
    arm_exits: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Append one complete, content-bound paired-seed promotion result."""

    rows = verified_promotion_replicates(root, loop_id, entry)
    if any(row.get("campaign_id") == campaign_id for row in rows):
        return rows, None
    control_id = str(delivery.get("control_id") or "")
    candidate_id = str(delivery.get("candidate_id") or "")
    seed = delivery.get("arm_seed")
    order = delivery.get("arm_order")
    exits = arm_exits or {}
    if delivery.get("measurement_complete") is not True:
        return rows, "promotion_replicate_incomplete:measurement"
    if type(seed) is not int:
        return rows, "promotion_replicate_incomplete:seed"
    if (
        not isinstance(order, list)
        or len(order) != 2
        or set(order) != {control_id, candidate_id}
    ):
        return rows, "promotion_replicate_incomplete:arm_order"
    if any(exits.get(arm_id) != 0 for arm_id in (control_id, candidate_id)):
        return rows, "promotion_replicate_incomplete:arm_exit"

    manifest_digests: dict[str, str] = {}
    for arm_id in (control_id, candidate_id):
        path = camp_dir / "manifests" / f"{arm_id}.json"
        manifest = read_json(path)
        if not path.is_file() or seed not in (manifest.get("seeds") or []):
            return rows, f"promotion_replicate_incomplete:manifest:{arm_id}"
        manifest_digests[arm_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    certificate = camp_dir / "metric-certificate.json"
    if not certificate.is_file():
        return rows, "promotion_replicate_incomplete:certificate"
    delivery_path = camp_dir / "sdlc_delivery.json"
    durable_delivery = read_json(delivery_path)
    if not delivery_path.is_file() or any(
        durable_delivery.get(key) != delivery.get(key)
        for key in (
            "measurement_complete",
            "control_id",
            "candidate_id",
            "control_metrics",
            "candidate_metrics",
            "arm_seed",
            "arm_order",
        )
    ):
        return rows, "promotion_replicate_incomplete:durable_delivery"

    evidence = {
        "manifests": manifest_digests,
        "certificate_sha256": hashlib.sha256(certificate.read_bytes()).hexdigest(),
        "delivery_sha256": hashlib.sha256(delivery_path.read_bytes()).hexdigest(),
        "metrics_sha256": promotion_replicate_sha(
            {
                "control": delivery.get("control_metrics") or {},
                "candidate": delivery.get("candidate_metrics") or {},
            }
        ),
    }
    record: dict[str, Any] = {
        "schema": PROMOTION_REPLICATE_SCHEMA,
        "loop_id": loop_id,
        "entry_id": entry.get("entry_id"),
        "knobs_fingerprint": entry.get("knobs_fingerprint"),
        "campaign_id": campaign_id,
        "cycle_index": cycle_index,
        "seed": seed,
        "arm_order": order,
        "control_id": control_id,
        "candidate_id": candidate_id,
        "evidence": evidence,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record["content_sha256"] = promotion_replicate_sha(record)
    if not any(row.get("content_sha256") == record["content_sha256"] for row in rows):
        path = promotion_replicate_ledger_path(root, loop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        rows.append(record)
    return rows, None


def gate_promotion_on_replicates(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
    camp_dir: Path,
    delivery: dict[str, Any],
    arm_exits: dict[str, int] | None,
    disposition: dict[str, Any],
) -> dict[str, Any]:
    """Prevent a favorable single pair from satisfying a multi-seed claim."""

    if disposition.get("status") != "climb_accepted":
        return disposition
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        promotion_seed_floor,
    )

    min_seeds, require_multi_seed = promotion_seed_floor(load_climb_policy())
    required = int(min_seeds) if require_multi_seed else 1
    rows, error = record_promotion_replicate(
        root=root,
        loop_id=loop_id,
        entry=entry,
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        camp_dir=camp_dir,
        delivery=delivery,
        arm_exits=arm_exits,
    )
    distinct_seeds = {int(row["seed"]) for row in rows if type(row.get("seed")) is int}
    order_roles = {
        "AB" if row.get("arm_order", [None])[0] == row.get("control_id") else "BA"
        for row in rows
    }
    orders_complete = required < 2 or order_roles == {"AB", "BA"}
    if error is None and len(distinct_seeds) >= required and orders_complete:
        return {
            **disposition,
            "promotion_replicate_count": len(distinct_seeds),
            "promotion_replicate_required": required,
        }
    reason = error or (
        f"promotion_replicates_incomplete:{len(distinct_seeds)}/{required}:"
        f"orders={','.join(sorted(order_roles))}"
    )
    return {
        **disposition,
        "status": "promotion_inconclusive",
        "inconclusive": True,
        "promotion_replicate_count": len(distinct_seeds),
        "promotion_replicate_required": required,
        "reasons": [*(disposition.get("reasons") or []), reason],
    }
