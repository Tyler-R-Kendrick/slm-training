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
    assert good.meta["curation_score_version"] == 2


def test_difficulty_discounts_the_trivially_easy_tail(tmp_path) -> None:
    import json

    from slm_training.data.selection import difficulty_weights, load_record_nll

    topics = [
        "alpha", "bravo", "charlie", "delta", "echo",
        "foxtrot", "golf", "hotel", "india", "juliet",
    ]
    records = [
        _record(f"r{i}", f"Distinct prompt about the {topic} dashboard.",
                f"root = Stack([x{i}])", 1.0)
        for i, topic in enumerate(topics)
    ]
    # r0 is far easier (lowest NLL) than the rest.
    nll = {f"r{i}": 1.0 + i for i in range(10)}
    weights = difficulty_weights(records, nll)
    assert weights["r0"] < 1.0  # bottom decile discounted
    assert weights["r9"] == 1.0
    assert all(0.5 <= w <= 1.0 for w in weights.values())

    # Unknown ids keep full weight.
    extra = _record("unknown", "A prompt with no NLL evidence.", "root = Stack([z])", 1.0)
    assert difficulty_weights([*records, extra], nll)["unknown"] == 1.0

    # The file interface skips unscored rows and round-trips floats.
    path = tmp_path / "record_nll.jsonl"
    path.write_text(
        json.dumps({"id": "r0", "nll": 1.25}) + "\n"
        + json.dumps({"id": "bad", "nll": None, "error": "boom"}) + "\n",
        encoding="utf-8",
    )
    assert load_record_nll(path) == {"r0": 1.25}

    # End to end: difficulty multiplies into the curation score.
    attach_curation_scores(records, nll_by_id=nll)
    assert records[0].meta["curation_score"] < 1.0
    assert records[0].meta["difficulty_weight"] < 1.0
    assert records[0].meta["record_nll"] == 1.0
    assert records[9].meta["curation_score"] == 1.0
