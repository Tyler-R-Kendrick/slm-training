"""SLM-418 (DSH5-10): the demo replay-preference-pairs corpus builder script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_replay_preference_pairs import (
    _provenance,
    build_demo_merge_scenario,
    build_demo_trace,
    main,
)
from slm_training.dsl.operators import ReplayPreferenceRelation, extract_replay_preference_rows
from slm_training.harnesses.preference import load_pairs


def test_build_demo_trace_exercises_three_named_patterns() -> None:
    pack, library, trace = build_demo_trace()

    report = extract_replay_preference_rows(
        trace, pack=pack, library=library, provenance_for=_provenance
    )
    assert set(report.counts_by_relation) == {
        ReplayPreferenceRelation.EDIT_THEN_UNDO.value,
        ReplayPreferenceRelation.UNDO_THEN_REDO.value,
        ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE.value,
    }


def test_build_demo_merge_scenario_yields_one_real_merge_success_pair() -> None:
    rows, pairs = build_demo_merge_scenario()

    assert len(rows) == 1
    assert rows[0].semantic_relation is ReplayPreferenceRelation.MERGE_SUCCESS
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.meta["pair_corpus"] == "replay_preference"
    assert pair.meta["semantic_relation"] == "merge_success"
    assert pair.chosen != pair.rejected


def test_main_writes_a_real_pairs_file_and_reports_honest_counts(
    tmp_path: Path, capsys
) -> None:
    out_path = tmp_path / "replay_demo_pairs.jsonl"

    exit_code = main(["--out", str(out_path)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["corpus_kind"] == "fixture_or_scratch"
    assert report["rows"] == 4
    assert report["counts_by_relation"] == {
        "edit_then_undo": 1,
        "undo_then_redo": 1,
        "checkout_another_state": 1,
        "merge_success": 1,
    }
    # undo_then_redo never renders here: redo and "reapply the same
    # deterministic zero-arg operator" are the identical text by
    # construction, so the renderer's dedup guard correctly declines it.
    assert report["pairs_rendered"] == 3
    assert report["pairs_dropped"] == 1

    pairs = load_pairs(out_path)
    assert len(pairs) == report["pairs_rendered"]
    relations = {pair.meta["semantic_relation"] for pair in pairs}
    assert relations == {"edit_then_undo", "checkout_another_state", "merge_success"}
    for pair in pairs:
        assert pair.meta["pair_corpus"] == "replay_preference"
        assert pair.chosen != pair.rejected
        assert pair.prompt
