from __future__ import annotations

import json

from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES
from slm_training.lineage.records import ChampionPointer, EvaluationReport
from slm_training.web.observability import Readers, gate_metric_keys


def _write_evidence(root, run_id: str) -> None:
    docs = root / "docs"
    design = docs / "design"
    design.mkdir(parents=True, exist_ok=True)
    (docs / "MODEL_CARD.md").write_text(
        f"""# Model card

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Current checkpoint | `{run_id}` | TwoTower | `outputs/runs/{run_id}/last.pt` | candidate |

## Checkpoint history

| Date (UTC) | Run id | Bucket / path | Metric headline | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-14 | `{run_id}` | `outputs/runs/{run_id}/` | pending | not evaluated |
""",
        encoding="utf-8",
    )
    (design / "quality-matrix-results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "E1",
                        "run_id": "experiment-1",
                        "pass": True,
                        "suites": {
                            "smoke": {
                                "n": 5,
                                "parse_rate": 1.0,
                                "placeholder_fidelity": 0.8,
                                "structural_similarity": 0.6,
                                "reward_score": 0.9,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_insights_persist_until_model_card_reference_changes(tmp_path) -> None:
    _write_evidence(tmp_path, "checkpoint-1")
    readers = Readers(tmp_path)

    first = readers.performance_insights()
    assert first["references"][-1]["run_id"] == "checkpoint-1"
    assert first["cache"]["persisted"] is True
    assert first["cache"]["cached"] is False
    assert first["comparisons"][0]["vs_reference"] == "reference not evaluated"

    second = readers.performance_insights()
    assert second["cache"]["cached"] is True
    assert second["insights"] == first["insights"]

    model_card = tmp_path / "docs" / "MODEL_CARD.md"
    model_card.write_text(
        model_card.read_text(encoding="utf-8")
        + "| 2026-07-15 | `checkpoint-2` | `outputs/runs/checkpoint-2/` | pending | not evaluated |\n",
        encoding="utf-8",
    )
    changed = Readers(tmp_path).performance_insights()
    assert changed["cache"]["cached"] is False
    assert changed["reference_fingerprint"] != first["reference_fingerprint"]
    assert changed["references"][-1]["run_id"] == "checkpoint-2"


def test_live_champion_becomes_comparison_baseline(tmp_path) -> None:
    _write_evidence(tmp_path, "checkpoint-1")
    readers = Readers(tmp_path)
    report = EvaluationReport(
        report_id="eval-champion",
        run_id="champion-1",
        eval_snapshot_sha="snapshot",
        created_at="2026-07-14T00:00:00Z",
        ship_gates_pass=True,
        weighted_nll=None,
        metrics={
            "parse_rate": 0.9,
            "placeholder_fidelity": 0.8,
            "structural_similarity": 0.7,
            "reward_score": 0.9,
        },
    )
    readers.lineage.write_report(report)
    readers.lineage.promote(
        ChampionPointer(
            pointer_id="champion-pointer-1",
            track="twotower",
            run_id="champion-1",
            artifact_uri="hf://bucket/champion-1",
            manifest_sha="manifest",
            evaluation_report_sha=report.sha,
            created_at="2026-07-14T00:00:00Z",
        )
    )

    performance = readers.performance_insights()
    assert performance["references"][0]["run_id"] == "champion-1"
    assert performance["references"][0]["evaluation_status"] == "evaluated"
    assert performance["stats"]["comparable"] == 1
    assert performance["comparisons"][0]["vs_reference"].endswith(" pp")

def test_metric_surfaces_track_ship_gate_policy(tmp_path) -> None:
    """Overview metrics derive from the ship-gate policy: changing a lever
    (adding/dropping a gate metric) must re-shape the dashboard automatically."""
    _write_evidence(tmp_path, "checkpoint-1")
    payload = Readers(tmp_path).performance_insights()

    lever_keys: list[str] = []
    for mins in DEFAULT_SHIP_GATES.values():
        for key in mins:
            if key not in lever_keys:
                lever_keys.append(key)
    assert lever_keys == gate_metric_keys()
    assert [column["key"] for column in payload["metric_columns"]] == lever_keys
    # The semantic-density floor must be a first-class dashboard column.
    assert "component_type_recall" in lever_keys

    row = payload["comparisons"][0]
    assert set(row["metrics"]) <= set(lever_keys)
    # Legacy parse_rate-only rows surface under the policy's meaningful lever.
    assert row["metrics"]["meaningful_program_rate"] == 1.0
