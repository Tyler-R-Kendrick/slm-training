"""Shared input fixtures for train-data pipeline tests."""

from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl


def seed_file(tmp_path: Path) -> Path:
    path = tmp_path / "seeds.jsonl"
    write_jsonl(
        path,
        [
            ExampleRecord(
                id="t1",
                prompt="Create quartz widget with a heading and supporting summary",
                openui=(
                    'root = Stack([quartz_panel], "column")\n'
                    'quartz_heading = TextContent(":slot_0")\n'
                    'quartz_summary = TextContent(":slot_1")\n'
                    "quartz_panel = Card([quartz_heading, quartz_summary])"
                ),
                placeholders=[":slot_0", ":slot_1"],
                split="train",
            ),
            ExampleRecord(
                id="t2",
                prompt="Button only",
                openui='root = Stack([cta])\ncta = Button(":slot_0")',
                placeholders=[":slot_0"],
                split="train",
            ),
        ],
    )
    return path
