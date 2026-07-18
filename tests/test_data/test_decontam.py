"""n-gram decontamination tests (train candidates vs eval suites)."""

from __future__ import annotations

from pathlib import Path

from slm_training.data.decontam import (
    apply_ngram_decontam,
    build_eval_ngram_index,
    load_eval_suites,
    overlap_report,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl


def _record(record_id: str, prompt: str, openui: str, split: str = "train") -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        placeholders=[],
        split=split,
    )


_EVAL = _record(
    "eval_menu",
    "Assemble a restaurant menu page listing every dish with its price and a "
    "short description under the dish name.",
    'root = Stack([menu], "column")\n'
    'dish = TextContent(":menu.dish")\n'
    'price = TextContent(":menu.price")\n'
    "menu = Card([dish, price])",
    split="held_out",
)


def test_near_copy_of_eval_record_is_flagged() -> None:
    contaminated = _record(
        "train_menu_copy",
        _EVAL.prompt,
        _EVAL.openui,
    )
    clean = _record(
        "train_gallery",
        "Photo gallery grid with captions beneath each thumbnail image cell.",
        'root = Stack([grid], "row")\ngrid = TextContent(":gallery.caption")',
    )
    kept, flagged = apply_ngram_decontam(
        [contaminated, clean], {"held_out": [_EVAL]}
    )
    assert [record.id for record in kept] == ["train_gallery"]
    assert len(flagged) == 1
    flag = flagged[0]
    assert flag["id"] == "train_menu_copy"
    assert flag["reason"] == "ngram_overlap"
    assert flag["suite"] == "held_out"
    assert flag["overlap"] >= 0.5
    assert flag["ngram_size"] == 8


def test_short_records_cannot_be_judged_and_are_kept() -> None:
    tiny = _record("tiny", "Hero.", "root = Stack([x])")
    kept, flagged = apply_ngram_decontam([tiny], {"held_out": [_EVAL]})
    assert kept and not flagged


def test_overlap_report_attributes_the_best_suite() -> None:
    index = build_eval_ngram_index({"held_out": [_EVAL], "smoke": []})
    report = overlap_report(_record("probe", _EVAL.prompt, _EVAL.openui), index)
    assert report["suite"] == "held_out"
    assert report["overlap"] == 1.0


def test_load_eval_suites_walks_versions_and_test_seeds(tmp_path: Path) -> None:
    suite_dir = tmp_path / "eval" / "v1" / "suites" / "smoke"
    suite_dir.mkdir(parents=True)
    write_jsonl(suite_dir / "records.jsonl", [_EVAL])
    seeds = tmp_path / "test_seeds.jsonl"
    write_jsonl(seeds, [_EVAL])
    suites = load_eval_suites(tmp_path / "eval", test_seed_path=seeds)
    assert sorted(suites) == ["test_seeds", "v1/smoke"]
    assert all(len(records) == 1 for records in suites.values())


def test_missing_roots_disable_the_pass() -> None:
    suites = load_eval_suites(Path("does/not/exist"), test_seed_path=None)
    assert suites == {}
    kept, flagged = apply_ngram_decontam([_EVAL], suites)
    assert kept and not flagged
