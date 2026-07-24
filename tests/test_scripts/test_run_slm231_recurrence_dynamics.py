"""Integration checks for the committed SLM-231 evidence."""

from __future__ import annotations

import json

from scripts.run_slm231_recurrence_dynamics import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    _markdown,
    _scientific_hash,
)
from slm_training.harnesses.experiments.slm231_recurrence_dynamics import (
    RecurrenceDynamicsSnapshotV1,
)


def test_committed_report_markdown_hash_and_snapshot_are_consistent() -> None:
    report = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert report["report_hash"] == _scientific_hash(report)
    assert DEFAULT_MARKDOWN.read_text(encoding="utf-8") == _markdown(report)
    snapshot = RecurrenceDynamicsSnapshotV1(**report["snapshots"][0])
    snapshot.validate()
    assert report["agentv"]["summary"]["passed"] == 4
    assert report["agentv"]["summary"]["failed"] == 0
    assert report["training_default_changed"] is False
    assert report["generation_default_changed"] is False
    assert report["checkpoint_created"] is False
    assert report["ship_gate_claim"] is False
