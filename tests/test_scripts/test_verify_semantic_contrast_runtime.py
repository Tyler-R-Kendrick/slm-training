from __future__ import annotations

import json

from scripts import verify_semantic_contrast_runtime as runtime
from slm_training.data.semantic_contrast import SemanticContrastBuilder
from slm_training.data.verify import RuntimeEvidence


def test_runtime_shards_are_complete_before_finalizing(tmp_path, monkeypatch) -> None:
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id="runtime_shard_fixture",
        source_count=1,
        splits=("train",),
        split_weights=(1.0,),
    )
    builder.build()
    dataset = tmp_path / "eval" / "runtime_shard_fixture"
    monkeypatch.setattr(
        runtime,
        "run_preview_verifier_many",
        lambda sources, timeout_s: tuple(RuntimeEvidence(rendered=True) for _ in sources),
    )
    count = len((dataset / "records.jsonl").read_text().splitlines())
    runtime.verify_shard(dataset, 0, count)
    result = runtime.finalize(dataset)
    assert result["records"] == count
    summary = json.loads((dataset / "summary.json").read_text())
    assert summary["runtime_required"] is True
