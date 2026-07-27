from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm298_capacity_context_curriculum import main
from slm_training.dsl.schema import ExampleRecord


def _train_dir(tmp_path: Path) -> Path:
    path = tmp_path / "train"
    path.mkdir()
    record = ExampleRecord(id="train", prompt="x", openui='root = TextContent(":x")', placeholders=[":x"])
    (path / "records.jsonl").write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    (path / "manifest.json").write_text("{}\n", encoding="utf-8")
    return path


def test_plan_writes_locked_campaign(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    rc = main(["--mode", "plan", "--output-dir", str(output_dir), "--train-dir", str(_train_dir(tmp_path))])
    assert rc == 0
    assert (output_dir / "campaign.lock.json").is_file()
    assert (output_dir / "protocol.json").is_file()


def test_plan_records_manifest_revision_without_inventing_recipe_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    train_dir = _train_dir(tmp_path)
    assert main(["--mode", "plan", "--output-dir", str(output_dir), "--train-dir", str(train_dir)]) == 0
    output_dir.joinpath("campaign.lock.json").write_text(
        json.dumps({"manifest_sha256": "0" * 64}), encoding="utf-8"
    )
    assert main(["--mode", "plan", "--output-dir", str(output_dir), "--train-dir", str(train_dir)]) == 0
    deviation = json.loads((output_dir / "campaign.deviations.jsonl").read_text(encoding="utf-8"))
    assert deviation["changed_field"] == "campaign_manifest_sha256"
    assert deviation["old_value"] == "0" * 64
