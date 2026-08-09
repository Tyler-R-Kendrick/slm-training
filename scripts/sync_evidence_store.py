"""Sync durable experiment evidence into the committed local index + Supabase.

Parses the machine-readable result JSONs under ``docs/design``, the committed
autotrain-climb evidence ledger, and any local ``outputs/autoresearch``
delivery records into normalized :class:`EvidenceRecordV1` rows, merges them
(deduplicated on ``config_fingerprint + endpoint_metric``, stable sort) into
the committed mirror at
``src/slm_training/resources/evidence_store/local_index.jsonl``, and — iff
``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` are both set — upserts the
merged index to Supabase via PostgREST. With credentials absent the sync logs
one notice and still writes the local index (fail-soft by contract).

Usage:
    python -m scripts.sync_evidence_store            # sync sources -> mirror
    python -m scripts.sync_evidence_store --dry-run  # report, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from slm_training.evidence_store.client import EvidenceStoreClient
from slm_training.evidence_store.local_index import (
    DEFAULT_LOCAL_INDEX_PATH,
    dedup_records,
    load_local_index,
    write_local_index,
)
from slm_training.evidence_store.records import (
    EvidenceRecordV1,
    compute_config_fingerprint,
)

_LOGGER = logging.getLogger("scripts.sync_evidence_store")

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = REPO_ROOT / "docs" / "design"
CLIMB_LEDGER_PATH = (
    REPO_ROOT
    / "src"
    / "slm_training"
    / "resources"
    / "experiments"
    / "autotrain_climb"
    / "evidence_ledger.v1.json"
)
OUTPUTS_AUTORESEARCH = REPO_ROOT / "outputs" / "autoresearch"

#: docs/design sources this sync knows how to map (best-effort; unmappable
#: entries are skipped with a counted warning, missing files likewise).
DESIGN_SOURCES = (
    "quality-matrix-results.json",
    "grammar-matrix-results.json",
    "perf-matrix-results.json",
    "phase-abc-results.json",
    "baseline-reproduction-results.json",
)

#: Primary metric the autotrain climb screens on (evidence_ledger default).
CLIMB_PRIMARY_METRIC = "smoke.structural_similarity"


class SyncStats:
    def __init__(self) -> None:
        self.records: list[EvidenceRecordV1] = []
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path, stats: SyncStats) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        stats.warn(f"missing source: {_rel(path)}")
        return None
    except ValueError:
        stats.warn(f"unparseable JSON: {_rel(path)}")
        return None
    if not isinstance(payload, Mapping):
        stats.warn(f"non-object JSON: {_rel(path)}")
        return None
    return payload


def _stamp(payload: Mapping[str, Any]) -> dict[str, Any]:
    stamp = payload.get("version_stamp")
    return dict(stamp) if isinstance(stamp, Mapping) else {}


def _suite_metric(entry: Mapping[str, Any], suite: str, metric: str) -> float | None:
    suites = entry.get("suites")
    if not isinstance(suites, Mapping):
        return None
    board = suites.get(suite)
    if not isinstance(board, Mapping):
        return None
    value = board.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _first_failure(entry: Mapping[str, Any]) -> str | None:
    failures = entry.get("failures")
    if isinstance(failures, list) and failures:
        return str(failures[0])
    return None


# ---------------------------------------------------------------------------
# Per-source mappers (best-effort against each file's observed schema)
# ---------------------------------------------------------------------------


def _map_quality_matrix(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """quality-matrix-results.json: gate-policy screening arms with suites."""
    matrix = str(payload.get("matrix") or "quality-matrix")
    stamp = _stamp(payload)
    run_date = stamp.get("stamped_at") if isinstance(stamp.get("stamped_at"), str) else None
    results = payload.get("results")
    for entry in results if isinstance(results, list) else []:
        if not isinstance(entry, Mapping) or "id" not in entry:
            stats.warn(f"{source}: unmappable results entry {entry!r:.80}")
            continue
        arm_id = str(entry["id"])
        effect = _suite_metric(entry, "smoke", "structural_similarity")
        passed = entry.get("pass") is True
        stats.records.append(
            EvidenceRecordV1(
                experiment_id=str(entry.get("run_id") or arm_id),
                campaign_id=matrix,
                hypothesis_text=(
                    f"{matrix} arm {arm_id} passes the quality gate policy"
                ),
                lever_keys=[arm_id.lower()],
                config_fingerprint=compute_config_fingerprint(
                    {
                        "source_kind": "quality_matrix",
                        "matrix": matrix,
                        "experiment_id": arm_id,
                        "seed": payload.get("seed"),
                        "steps": payload.get("steps"),
                    }
                ),
                endpoint_metric="smoke.structural_similarity",
                effect_size=effect,
                n_seeds=1,
                steps=int(payload.get("steps") or 0),
                outcome="screen_positive" if passed else "screen_negative",
                blocker=None if passed else _first_failure(entry),
                version_stamp=stamp,
                source_path=source,
                run_date=run_date,
            )
        )


def _map_grammar_matrix(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """grammar-matrix-results.json: per-seed confirmation arms."""
    matrix = str(payload.get("matrix") or "grammar-matrix")
    stage = str(payload.get("stage") or "screening")
    results = payload.get("results")
    for entry in results if isinstance(results, list) else []:
        if not isinstance(entry, Mapping) or "id" not in entry:
            stats.warn(f"{source}: unmappable results entry {entry!r:.80}")
            continue
        arm_id = str(entry["id"])
        seed = entry.get("seed")
        passed = entry.get("pass") is True
        if stage == "confirmation":
            outcome = "confirmed" if passed else "confirm_failed"
        else:
            outcome = "screen_positive" if passed else "screen_negative"
        stats.records.append(
            EvidenceRecordV1(
                experiment_id=str(entry.get("run_id") or arm_id),
                campaign_id=matrix,
                hypothesis_text=(
                    f"{matrix} arm {arm_id}: "
                    f"{entry.get('description') or 'passes the gate policy'}"
                ),
                lever_keys=[
                    key
                    for key in (arm_id.lower(), str(entry.get("model_name") or ""))
                    if key
                ],
                config_fingerprint=compute_config_fingerprint(
                    {
                        "source_kind": "grammar_matrix",
                        "matrix": matrix,
                        "experiment_id": arm_id,
                        "seed": seed,
                        "steps": payload.get("steps"),
                        "stage": stage,
                    }
                ),
                endpoint_metric="smoke.structural_similarity",
                effect_size=_suite_metric(entry, "smoke", "structural_similarity"),
                n_seeds=1,
                steps=int(payload.get("steps") or 0),
                outcome=outcome,
                blocker=None if passed else _first_failure(entry),
                version_stamp=_stamp(payload),
                source_path=source,
                run_date=None,
            )
        )


def _map_perf_matrix(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """perf-matrix-results.json: latency/throughput arms vs the control arm."""
    stamp = _stamp(payload)
    run_date = stamp.get("stamped_at") if isinstance(stamp.get("stamped_at"), str) else None
    results = payload.get("results")
    entries = [
        entry
        for entry in (results if isinstance(results, list) else [])
        if isinstance(entry, Mapping) and "id" in entry
    ]
    if isinstance(results, list) and len(entries) != len(results):
        stats.warn(f"{source}: skipped {len(results) - len(entries)} entries")
    control_tps: float | None = None
    for entry in entries:
        if "control" in str(entry.get("run_id") or entry.get("id")).lower():
            value = entry.get("tokens_per_sec")
            if isinstance(value, (int, float)):
                control_tps = float(value)
            break
    for entry in entries:
        arm_id = str(entry["id"])
        run_id = str(entry.get("run_id") or arm_id)
        tps = entry.get("tokens_per_sec")
        tps = float(tps) if isinstance(tps, (int, float)) else None
        is_control = control_tps is not None and "control" in run_id.lower()
        if is_control or tps is None or control_tps is None:
            effect = None
            positive = False
        else:
            effect = tps - control_tps
            positive = effect > 0
        flags = entry.get("flags")
        stats.records.append(
            EvidenceRecordV1(
                experiment_id=run_id,
                campaign_id=str(payload.get("suite") or "perf"),
                hypothesis_text=(
                    f"perf arm {arm_id}: "
                    f"{entry.get('description') or 'improves decode throughput'}"
                ),
                lever_keys=sorted(flags) if isinstance(flags, Mapping) else [arm_id.lower()],
                config_fingerprint=compute_config_fingerprint(
                    {
                        "source_kind": "perf_matrix",
                        "experiment_id": arm_id,
                        "flags": dict(flags) if isinstance(flags, Mapping) else None,
                        "suite": payload.get("suite"),
                        "max_len": entry.get("max_len"),
                    }
                ),
                endpoint_metric="perf.tokens_per_sec_delta_vs_control",
                effect_size=effect,
                n_seeds=1,
                steps=0,
                outcome="screen_positive" if positive else "screen_negative",
                blocker="control_arm" if is_control else None,
                version_stamp=stamp,
                source_path=source,
                run_date=run_date,
            )
        )


def _map_phase_abc(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """phase-abc-results.json: A/B/C pipeline with a final ship-gate eval."""
    run_date = payload.get("recorded_at") or payload.get("finished_at")
    run_date = str(run_date) if isinstance(run_date, str) else None
    cleared = payload.get("ship_gates_cleared") is True
    eval_block = payload.get("eval")
    blocker = (
        _first_failure(eval_block) if isinstance(eval_block, Mapping) else None
    )
    a_block = payload.get("A")
    steps = (
        int(a_block.get("steps") or 0) if isinstance(a_block, Mapping) else 0
    )
    stats.records.append(
        EvidenceRecordV1(
            experiment_id="phase_abc_complete",
            campaign_id="phase-abc",
            hypothesis_text=(
                "Phase A (SFT) -> B (preference) -> C (RL) pipeline clears the "
                "ship gates"
            ),
            lever_keys=["phase_a_sft", "phase_b_pref", "phase_c_rl"],
            config_fingerprint=compute_config_fingerprint(
                {"source_kind": "phase_abc", "experiment_id": "phase_abc_complete"}
            ),
            endpoint_metric="ship_gates.pass",
            effect_size=1.0 if cleared else 0.0,
            n_seeds=1,
            steps=steps,
            outcome="promoted" if cleared else "ship_rejected",
            blocker=None if cleared else blocker,
            version_stamp=_stamp(payload),
            source_path=source,
            run_date=run_date,
        )
    )
    champion = payload.get("champion")
    champion_phase = (
        str(champion.get("phase")) if isinstance(champion, Mapping) else None
    )
    boards = payload.get("phase_boards")
    for phase, board in (
        sorted(boards.items()) if isinstance(boards, Mapping) else []
    ):
        if not isinstance(board, Mapping):
            stats.warn(f"{source}: unmappable phase board {phase}")
            continue
        smoke = board.get("suites", {})
        effect = None
        if isinstance(smoke, Mapping):
            smoke_board = smoke.get("smoke")
            if isinstance(smoke_board, Mapping):
                value = smoke_board.get("structural_similarity")
                effect = float(value) if isinstance(value, (int, float)) else None
        stats.records.append(
            EvidenceRecordV1(
                experiment_id=f"phase_abc_{phase}",
                campaign_id="phase-abc",
                hypothesis_text=(
                    f"Phase board {phase} is the strongest phase-ABC checkpoint"
                ),
                lever_keys=[str(phase).lower()],
                config_fingerprint=compute_config_fingerprint(
                    {"source_kind": "phase_abc", "phase": phase}
                ),
                endpoint_metric="smoke.structural_similarity",
                effect_size=effect,
                n_seeds=1,
                steps=steps,
                outcome=(
                    "screen_positive" if phase == champion_phase else "screen_negative"
                ),
                blocker=None,
                version_stamp=_stamp(payload),
                source_path=source,
                run_date=run_date,
            )
        )


def _map_climb_ledger(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """autotrain_climb/evidence_ledger.v1.json: per-arm pooled evidence."""
    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        stats.warn(f"{source}: no arms map")
        return
    for slug, arm in sorted(arms.items()):
        if not isinstance(arm, Mapping):
            stats.warn(f"{source}: unmappable arm {slug}")
            continue
        n_complete = int(arm.get("n_complete") or 0)
        n_positive = int(arm.get("n_positive") or 0)
        mean_delta = arm.get("mean_delta")
        blocker = None
        if n_positive > 0:
            outcome = "screen_positive"
        else:
            outcome = "screen_negative"
            if n_complete == 0:
                blocker = "no_complete_measurements"
        stats.records.append(
            EvidenceRecordV1(
                experiment_id=f"autotrain-climb-arm-{slug}",
                campaign_id="autotrain-climb",
                hypothesis_text=(
                    f"Screening arm '{slug}' improves {CLIMB_PRIMARY_METRIC} "
                    "over control"
                ),
                lever_keys=[str(slug)],
                config_fingerprint=compute_config_fingerprint(
                    {"source_kind": "autotrain_climb_ledger", "arm": str(slug)}
                ),
                endpoint_metric=CLIMB_PRIMARY_METRIC,
                effect_size=(
                    float(mean_delta)
                    if isinstance(mean_delta, (int, float)) and n_complete > 0
                    else None
                ),
                n_seeds=max(1, n_complete),
                steps=0,
                outcome=outcome,
                blocker=blocker,
                version_stamp={},
                source_path=source,
                run_date=None,
            )
        )


def _map_delivery(
    payload: Mapping[str, Any], source: str, stats: SyncStats
) -> None:
    """outputs/autoresearch/**/delivery records (autotrain_sdlc_delivery/v1)."""
    if payload.get("schema") != "autotrain_sdlc_delivery/v1":
        stats.warn(f"{source}: not an autotrain_sdlc_delivery/v1 record")
        return
    campaign_id = str(payload.get("campaign_id") or "")
    if not campaign_id:
        stats.warn(f"{source}: delivery record without campaign_id")
        return
    positive = payload.get("positive") is True
    arm_order = payload.get("arm_order")
    lever_keys: list[str] = []
    try:
        from slm_training.autoresearch.evidence_ledger import (
            slug_from_candidate_token,
        )

        for token in arm_order if isinstance(arm_order, list) else []:
            slug = slug_from_candidate_token(str(token))
            if slug:
                lever_keys.append(slug)
    except Exception:  # noqa: BLE001 — mapping stays best-effort
        pass
    reasons = payload.get("reasons")
    reason_text = (
        "; ".join(str(reason) for reason in reasons[:3])
        if isinstance(reasons, list) and reasons
        else "screening cycle outcome"
    )
    run_date = payload.get("finished_at")
    stats.records.append(
        EvidenceRecordV1(
            experiment_id=campaign_id,
            campaign_id=campaign_id,
            hypothesis_text=f"Autotrain cycle {campaign_id}: {reason_text}",
            lever_keys=lever_keys,
            config_fingerprint=compute_config_fingerprint(
                {
                    "source_kind": "sdlc_delivery",
                    "campaign_id": campaign_id,
                    "arm_order": list(arm_order)
                    if isinstance(arm_order, list)
                    else None,
                    "arm_seed": payload.get("arm_seed"),
                }
            ),
            endpoint_metric=CLIMB_PRIMARY_METRIC,
            effect_size=None,
            n_seeds=1,
            steps=0,
            outcome="screen_positive" if positive else "screen_negative",
            blocker=None if payload.get("measurement_complete") else "measurement_incomplete",
            version_stamp=_stamp(payload),
            source_path=source,
            run_date=str(run_date) if isinstance(run_date, str) else None,
        )
    )


_DESIGN_MAPPERS = {
    "quality-matrix-results.json": _map_quality_matrix,
    "grammar-matrix-results.json": _map_grammar_matrix,
    "perf-matrix-results.json": _map_perf_matrix,
    "phase-abc-results.json": _map_phase_abc,
    "baseline-reproduction-results.json": _map_quality_matrix,
}


def collect_records(
    *,
    design_dir: Path = DESIGN_DIR,
    climb_ledger_path: Path = CLIMB_LEDGER_PATH,
    outputs_dir: Path = OUTPUTS_AUTORESEARCH,
) -> SyncStats:
    """Parse every known source into normalized records (best-effort)."""
    stats = SyncStats()
    for name in DESIGN_SOURCES:
        path = design_dir / name
        payload = _load_json(path, stats)
        if payload is None:
            continue
        _DESIGN_MAPPERS[name](payload, _rel(path), stats)
    ledger_payload = _load_json(climb_ledger_path, stats)
    if ledger_payload is not None:
        _map_climb_ledger(ledger_payload, _rel(climb_ledger_path), stats)
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("*delivery*.json")):
            payload = _load_json(path, stats)
            if payload is not None:
                _map_delivery(payload, _rel(path), stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--climb-ledger", type=Path, default=CLIMB_LEDGER_PATH)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_AUTORESEARCH)
    parser.add_argument("--out", type=Path, default=DEFAULT_LOCAL_INDEX_PATH)
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; write nothing"
    )
    args = parser.parse_args(argv)

    stats = collect_records(
        design_dir=args.design_dir,
        climb_ledger_path=args.climb_ledger,
        outputs_dir=args.outputs_dir,
    )
    for warning in stats.warnings:
        _LOGGER.warning("sync warning: %s", warning)

    existing = load_local_index(args.out)
    merged = dedup_records(list(existing) + stats.records)
    print(
        f"parsed={len(stats.records)} warnings={len(stats.warnings)} "
        f"existing={len(existing)} merged={len(merged)}"
    )
    if args.dry_run:
        print("dry run (no writes)")
        return 0

    written = write_local_index(merged, args.out)
    print(f"wrote {len(written)} records -> {_rel(args.out)}")

    client = EvidenceStoreClient()
    if client.remote_enabled:
        pushed = client.upsert_records(written)
        print(f"supabase upsert: pushed {pushed}/{len(written)} records")
    else:
        # Emits the single fail-soft notice; local index is already written.
        client.upsert_records(written)
        print("supabase upsert skipped (credentials unset); local index written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
