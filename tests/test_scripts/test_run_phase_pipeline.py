from __future__ import annotations

from scripts.run_phase_pipeline import _phase_b_pairs
from slm_training.harnesses.preference import PreferencePair, write_pairs
from slm_training.dsl.schema import ExampleRecord


def test_phase_b_retains_root_renderability_pairs_over_limit(tmp_path) -> None:
    write_pairs(
        tmp_path / "preference_pairs.jsonl",
        [
            PreferencePair(
                prompt="repair",
                chosen='root = Table([Col(":a", "x")])',
                rejected='root = Col(":a", "x")',
                chosen_score=1.0,
                rejected_score=0.0,
                meta={"pair_corpus": "root_renderability"},
            )
        ],
    )
    records = [
        ExampleRecord(
            id="r",
            prompt="other",
            openui='root = Stack([x])\nx = TextContent(":x")',
        )
    ]
    pairs = _phase_b_pairs(tmp_path, records, limit=1)
    assert len(pairs) == 1
    assert (pairs[0].meta or {})["pair_corpus"] == "root_renderability"
