"""Sync pipeline: fixture JSONs -> parse -> dedup -> local index roundtrip."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_evidence_store import (
    DESIGN_SOURCES,
    SyncStats,
    collect_records,
)
from slm_training.evidence_store.local_index import (
    dedup_records,
    load_local_index,
    write_local_index,
)
from slm_training.evidence_store.records import EvidenceRecordV1

_QUALITY_FIXTURE = {
    "matrix": "quality-experiment-matrix-v16",
    "seed": 0,
    "steps": 200,
    "results": [
        {
            "id": "E1",
            "run_id": "qx_e1_arm",
            "pass": True,
            "failures": [],
            "suites": {"smoke": {"structural_similarity": 0.41}},
        },
        {
            "id": "E2",
            "run_id": "qx_e2_arm",
            "pass": False,
            "failures": ["smoke:structural_similarity actual=0.1 need>=0.35"],
            "suites": {},
        },
        "not-a-mapping",
    ],
    "version_stamp": {"stamped_at": "2026-07-28T21:16:59+00:00"},
}

_GRAMMAR_FIXTURE = {
    "matrix": "grammar-experiment-matrix-x",
    "stage": "confirmation",
    "steps": 200,
    "results": [
        {
            "id": "X9",
            "run_id": "gx_x9",
            "seed": 0,
            "model_name": "grammar_diffusion",
            "description": "typed subtree collapse",
            "pass": False,
            "failures": ["smoke:parse_rate actual=0.0 need>=0.66"],
            "suites": {"smoke": {"structural_similarity": 0.09}},
        }
    ],
}

_PERF_FIXTURE = {
    "suite": "smoke",
    "results": [
        {
            "id": "C0",
            "run_id": "perf_c0_control",
            "description": "control",
            "tokens_per_sec": 0.34,
            "flags": {"grammar_ltr_primary": True},
        },
        {
            "id": "C9",
            "run_id": "perf_c9_arm",
            "description": "serial compiler prefill",
            "tokens_per_sec": 0.50,
            "flags": {"compiler_decode_mode": "tree"},
        },
    ],
}

_PHASE_FIXTURE = {
    "A": {"steps": 200},
    "ship_gates_cleared": False,
    "eval": {"pass": False, "failures": ["smoke:parse_rate actual=0.0 need>=0.66"]},
    "champion": {"phase": "C_rl"},
    "phase_boards": {
        "A_sft": {"suites": {"smoke": {"structural_similarity": 0.64}}},
        "C_rl": {"suites": {"smoke": {"structural_similarity": 0.42}}},
    },
    "recorded_at": "2026-07-13T14:40:07+00:00",
}

_LEDGER_FIXTURE = {
    "schema": "autotrain_evidence_ledger/v1",
    "arms": {
        "bounds": {
            "n_obs": 50,
            "n_complete": 48,
            "n_positive": 0,
            "n_null": 47,
            "n_delta": 48,
            "mean_delta": 0.0,
            "m2_delta": 0.0,
            "by_eval_key": {},
        },
        "binder-topology": {
            "n_obs": 11,
            "n_complete": 9,
            "n_positive": 1,
            "n_null": 8,
            "n_delta": 9,
            "mean_delta": 0.0121,
            "m2_delta": 0.0106,
            "by_eval_key": {},
        },
        "never-measured": {
            "n_obs": 1,
            "n_complete": 0,
            "n_positive": 0,
            "n_null": 0,
            "n_delta": 0,
            "mean_delta": 0.0,
            "m2_delta": 0.0,
            "by_eval_key": {},
        },
    },
}

_DELIVERY_FIXTURE = {
    "schema": "autotrain_sdlc_delivery/v1",
    "loop_id": "loop-1",
    "campaign_id": "loop-1-c9-bounds",
    "finished_at": "2026-08-08T00:00:00Z",
    "positive": True,
    "measurement_complete": True,
    "arm_order": ["loop-1-c9-bounds", "loop-1-c9-control"],
    "arm_seed": 3,
    "reasons": ["candidate beat control on smoke.structural_similarity"],
}


def _write_fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    design = tmp_path / "design"
    design.mkdir()
    (design / "quality-matrix-results.json").write_text(json.dumps(_QUALITY_FIXTURE))
    (design / "grammar-matrix-results.json").write_text(json.dumps(_GRAMMAR_FIXTURE))
    (design / "perf-matrix-results.json").write_text(json.dumps(_PERF_FIXTURE))
    (design / "phase-abc-results.json").write_text(json.dumps(_PHASE_FIXTURE))
    # baseline-reproduction-results.json intentionally absent -> counted warning
    ledger = tmp_path / "evidence_ledger.v1.json"
    ledger.write_text(json.dumps(_LEDGER_FIXTURE))
    outputs = tmp_path / "outputs" / "autoresearch" / "loop-1-c9-bounds"
    outputs.mkdir(parents=True)
    (outputs / "sdlc_delivery.json").write_text(json.dumps(_DELIVERY_FIXTURE))
    return design, ledger, tmp_path / "outputs" / "autoresearch"


def _collect(tmp_path: Path) -> SyncStats:
    design, ledger, outputs = _write_fixture_tree(tmp_path)
    return collect_records(
        design_dir=design, climb_ledger_path=ledger, outputs_dir=outputs
    )


def test_collect_records_maps_all_fixture_sources(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    by_source_kind: dict[str, int] = {}
    for record in stats.records:
        by_source_kind[record.source_path] = (
            by_source_kind.get(record.source_path, 0) + 1
        )
    # quality: 2, grammar: 1, perf: 2, phase: 1 aggregate + 2 boards,
    # ledger: 3 arms, delivery: 1
    assert len(stats.records) == 12
    outcomes = {record.experiment_id: record.outcome for record in stats.records}
    assert outcomes["qx_e1_arm"] == "screen_positive"
    assert outcomes["qx_e2_arm"] == "screen_negative"
    assert outcomes["gx_x9"] == "confirm_failed"
    assert outcomes["phase_abc_complete"] == "ship_rejected"
    assert outcomes["phase_abc_C_rl"] == "screen_positive"
    assert outcomes["autotrain-climb-arm-binder-topology"] == "screen_positive"
    assert outcomes["autotrain-climb-arm-bounds"] == "screen_negative"
    assert outcomes["loop-1-c9-bounds"] == "screen_positive"


def test_collect_records_counts_warnings(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    # missing baseline file + skipped non-mapping quality entry
    assert len(stats.warnings) == 2
    assert any("baseline-reproduction" in warning for warning in stats.warnings)
    assert any("unmappable" in warning for warning in stats.warnings)


def test_perf_effect_is_delta_vs_control(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    perf = {
        record.experiment_id: record
        for record in stats.records
        if record.endpoint_metric == "perf.tokens_per_sec_delta_vs_control"
    }
    assert perf["perf_c0_control"].blocker == "control_arm"
    assert perf["perf_c9_arm"].effect_size is not None
    assert abs(perf["perf_c9_arm"].effect_size - 0.16) < 1e-9
    assert perf["perf_c9_arm"].outcome == "screen_positive"


def test_delivery_arm_slugs_reuse_evidence_ledger_extraction(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    delivery = next(
        record for record in stats.records if record.campaign_id == "loop-1-c9-bounds"
    )
    assert delivery.lever_keys == ["bounds"]
    assert delivery.blocker is None


def test_unmeasured_ledger_arm_carries_blocker(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    arm = next(
        record
        for record in stats.records
        if record.experiment_id == "autotrain-climb-arm-never-measured"
    )
    assert arm.blocker == "no_complete_measurements"
    assert arm.effect_size is None


def test_dedup_and_local_index_roundtrip(tmp_path: Path) -> None:
    stats = _collect(tmp_path)
    index_path = tmp_path / "local_index.jsonl"
    written = write_local_index(stats.records, index_path)
    assert len(written) == len(stats.records)  # fixture keys are unique
    # roundtrip: load == written
    loaded = load_local_index(index_path)
    assert loaded == written
    # dedup: syncing the same records again changes nothing (stable bytes)
    first_bytes = index_path.read_bytes()
    merged = dedup_records(list(loaded) + list(stats.records))
    write_local_index(merged, index_path)
    assert index_path.read_bytes() == first_bytes
    # stable sort order by (config_fingerprint, endpoint_metric)
    keys = [record.dedup_key() for record in loaded]
    assert keys == sorted(keys)


def test_dedup_last_record_wins(tmp_path: Path) -> None:
    base = EvidenceRecordV1(
        experiment_id="old",
        config_fingerprint="a" * 64,
        endpoint_metric="smoke.structural_similarity",
        outcome="screen_negative",
    )
    refreshed = base.model_copy(update={"experiment_id": "new", "outcome": "confirmed"})
    deduped = dedup_records([base, refreshed])
    assert len(deduped) == 1
    assert deduped[0].experiment_id == "new"


def test_missing_everything_is_all_warnings_no_records(tmp_path: Path) -> None:
    stats = collect_records(
        design_dir=tmp_path / "nowhere",
        climb_ledger_path=tmp_path / "missing.json",
        outputs_dir=tmp_path / "no-outputs",
    )
    assert stats.records == []
    assert len(stats.warnings) == len(DESIGN_SOURCES) + 1
