"""Writer/reader metric-key contract: consumers may only require emitted keys.

The request_coverage promotion gate rotted silently because no evaluator ever
wrote that key. This test runs the real evaluator and asserts every metric a
gate or promotion path requires is actually present in its output.
"""

from __future__ import annotations

from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import evaluate
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES
from slm_training.lineage.promotion import HARD_METRICS

_GOLD = 'root = Stack([cta])\ncta = Button(":cta")'


class _EchoGoldModel:
    def generate_batch_requests(self, requests: list[object]) -> list[str]:
        return [_GOLD for _ in requests]


def _consumed_keys() -> set[str]:
    keys = set(HARD_METRICS)
    for thresholds in DEFAULT_SHIP_GATES.values():
        keys.update(thresholds)
    keys.add("fallback_count")  # certified_fallback gate input
    return keys


def test_every_consumed_metric_key_is_emitted(tmp_path: Path) -> None:
    record = ExampleRecord.from_dict(
        {
            "id": "r1",
            "prompt": "CTA",
            "openui": _GOLD,
            "placeholders": [":cta"],
            "split": "smoke",
            "meta": {"suite": "smoke"},
        }
    )
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])
    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="metric-contract",
        model_name="stub",
    )
    metrics = evaluate(config, model=_EchoGoldModel(), publish_agentv=False)
    missing = sorted(key for key in _consumed_keys() if key not in metrics)
    assert not missing, f"consumers require metrics the evaluator never emits: {missing}"
