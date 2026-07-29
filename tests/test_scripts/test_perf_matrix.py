import argparse
import json

import pytest

from scripts.run_perf_matrix import (
    _COMPLETION_REQUIRED_WORKLOADS,
    _completion_fixture_digest,
    _guardrails,
    _median_mad,
    _merge_completion_shards,
    _paired_measure,
    experiments,
)


def test_invalid_p0_cannot_promote_candidate() -> None:
    result = _guardrails(
        {"parse_rate": 0.0, "placeholder_fidelity": 0.0, "latency_ms_mean": 100.0},
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 10.0},
    )
    assert result["pass"] is False
    assert "invalid P0" in result["note"]


def test_valid_p0_applies_quality_floor() -> None:
    result = _guardrails(
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 100.0},
        {"parse_rate": 0.96, "placeholder_fidelity": 0.96, "latency_ms_mean": 50.0},
    )
    assert result["pass"] is True
    assert result["speedup_vs_p0"] == 2.0


def test_c5_c10_are_registered_without_running() -> None:
    rows = {row.eid: row for row in experiments()}
    assert set(rows) >= {"C5", "C6", "C7", "C8", "C9", "C10"}
    assert rows["C6"].grammar_active_symbol_bitsets is True
    assert rows["C8"].grammar_completion_bounds is True
    assert rows["C9"].compiler_prefill_max_states == 1
    assert rows["C10"].compiler_prefill_max_states == 0


def test_completion_pairing_alternates_and_records_stable_stats() -> None:
    measured = _paired_measure(lambda: {"v": 1}, lambda: {"v": 1}, 4, min_sample_ns=0)

    assert [pair["order"] for pair in measured["pairs"]] == [
        ["reference", "packed"],
        ["packed", "reference"],
        ["reference", "packed"],
        ["packed", "reference"],
    ]
    assert measured["bundle_size"] == 1
    assert measured["pair_count"] == 4
    assert measured["output_digest"]
    median, mad = _median_mad([1.0, 2.0, 3.0])
    assert median == 2.0
    assert mad == 1.0


def test_completion_pairing_fails_on_any_output_mismatch() -> None:
    with pytest.raises(AssertionError, match="parity mismatch"):
        _paired_measure(lambda: {"v": 1}, lambda: {"v": 2}, 1, min_sample_ns=0)


def _write_completion_shards(tmp_path, *, dirty=False, mixed_commit=False) -> argparse.Namespace:
    run_id = "completion-test"
    shard_root = tmp_path / "outputs"
    run_root = shard_root / run_id
    run_root.mkdir(parents=True)
    for index, name in enumerate(_COMPLETION_REQUIRED_WORKLOADS):
        commit = "b" * 40 if mixed_commit and index == 1 else "a" * 40
        row = {
            "schema_version": "completion-kernel-workload/v1",
            "run_id": run_id,
            "workload": name,
            "fixture_manifest_digest": _completion_fixture_digest(name),
            "recipe": {"device": "cpu", "pair_count": 7},
            "host": {"machine": "test", "python": "3.12"},
            "pass": name != "decode",
            "version_stamp": {
                "stamp_schema": "version_stamp/v1",
                "code_commit": commit,
                "code_dirty": dirty,
                "components": {"matrix.perf": "v4"},
                "stamped_at": "ignored-per-row",
            },
        }
        (run_root / f"{name}.json").write_text(json.dumps(row), encoding="utf-8")
    return argparse.Namespace(
        completion_shard_root=shard_root,
        completion_run_id=run_id,
        docs_out=tmp_path / "docs.json",
    )


def test_completion_merge_requires_homogeneous_clean_full_run(tmp_path) -> None:
    args = _write_completion_shards(tmp_path)

    assert _merge_completion_shards(args) == 1
    board = json.loads(args.docs_out.read_text(encoding="utf-8"))
    assert board["overall_status"] == "fail"
    assert board["overall_pass"] is False
    assert set(board["workloads"]) == set(_COMPLETION_REQUIRED_WORKLOADS)


@pytest.mark.parametrize(
    ("dirty", "mixed_commit", "message"),
    [
        (True, False, "code_dirty=false"),
        (False, True, "mixed run/commit"),
    ],
)
def test_completion_merge_rejects_dirty_or_mixed_shards(
    tmp_path, dirty: bool, mixed_commit: bool, message: str
) -> None:
    args = _write_completion_shards(
        tmp_path, dirty=dirty, mixed_commit=mixed_commit
    )

    with pytest.raises(ValueError, match=message):
        _merge_completion_shards(args)


def test_completion_merge_rejects_missing_or_stale_shards(tmp_path) -> None:
    args = _write_completion_shards(tmp_path)
    (args.completion_shard_root / args.completion_run_id / "decode.json").unlink()
    with pytest.raises(ValueError, match="missing workloads"):
        _merge_completion_shards(args)

    args = _write_completion_shards(tmp_path / "stale")
    path = args.completion_shard_root / args.completion_run_id / "direct.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["fixture_manifest_digest"] = "stale"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        _merge_completion_shards(args)
