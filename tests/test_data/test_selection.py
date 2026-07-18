"""Curation-score (quality × diversity) tests."""

from __future__ import annotations

from slm_training.data.selection import attach_curation_scores, curation_scores
from slm_training.dsl.schema import ExampleRecord


def _record(record_id: str, prompt: str, openui: str, score: float) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        placeholders=[],
        split="train",
        meta={"quality": {"score": score}},
    )


def test_large_clusters_are_discounted() -> None:
    # Three members of one semantic cluster vs a singleton.
    clones = [
        _record(f"c{i}", "A hero card with title and body text.",
                'root = Stack([x])\nx = TextContent(":hero.title")', 1.0)
        for i in range(3)
    ]
    lone = _record(
        "solo",
        "A shipping dashboard of delivery metrics.",
        'root = Stack([grid], "row")\ngrid = TextContent(":stats.total")',
        1.0,
    )
    scores = curation_scores([*clones, lone])
    assert scores["solo"] == 1.0
    assert all(scores[f"c{i}"] < scores["solo"] for i in range(3))
    assert scores["c0"] == round(1.0 / (3 ** 0.5), 4)


def test_scores_scale_with_quality_and_stamp_meta() -> None:
    good = _record("good", "Prompt one about menus.", "root = Stack([a])", 0.9)
    weak = _record("weak", "Prompt two about galleries.", "root = Stack([b])", 0.45)
    attach_curation_scores([good, weak])
    assert good.meta["curation_score"] == 0.9
    assert weak.meta["curation_score"] == 0.45
    assert good.meta["curation_score_version"] == 1
