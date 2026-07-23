from pathlib import Path

from slm_training.web.observability import Readers

REPO = Path(__file__).resolve().parents[2]


def test_catalog_includes_generated_levers_and_backfilled_history() -> None:
    payload = Readers(REPO).experiment_flags()
    keys = {row["key"] for row in payload["flags"]}
    assert payload["history_runs"] > 400
    assert "slm.schema_in_context" in keys
    assert "slm.run_id" not in keys

    detail = Readers(REPO).experiment_flag("slm.schema_in_context")
    assert detail is not None
    assert detail["description"]
    assert any(
        location["path"] == "src/slm_training/harnesses/model_build/config.py"
        for location in detail["implementation"]
    )


def test_historical_run_marks_only_recorded_recipe_values() -> None:
    flags = Readers(REPO).run_feature_flags(
        "e128_judged_schema_slots_64", REPO / "outputs" / "missing"
    )
    rows = {row["key"]: row for row in flags["rows"]}

    assert flags["provenance"] == "committed"
    assert rows["slm.schema_in_context"]["cells"]["historical"] == {
        "recorded": True,
        "value": True,
        "state": "enabled",
        "source": "committed",
    }
    assert rows["slm.semantic_role_decode_weight"]["cells"]["historical"]["state"] == "not_recorded"


def test_missing_feature_history_is_empty_not_an_error(tmp_path: Path) -> None:
    flags = Readers(tmp_path).run_feature_flags("missing", tmp_path / "outputs" / "missing")

    assert flags["provenance"] == "missing"
    assert flags["history_sources"] == []


def test_feature_detail_includes_implementation_and_recorded_comparison(tmp_path: Path) -> None:
    history = tmp_path / "src/slm_training/resources/experiment_feature_flag_history.json"
    history.parent.mkdir(parents=True)
    history.write_text(
        """{
  "runs": [
    {"run_id": "before", "values": {"slm.schema_in_context": false}, "conflicts": {}, "sources": []},
    {"run_id": "after", "values": {"slm.schema_in_context": true}, "conflicts": {}, "sources": []}
  ]
}
""",
        encoding="utf-8",
    )
    readers = Readers(tmp_path)
    readers.runs = lambda: {  # type: ignore[method-assign]
        "runs": [
            {
                "run_id": "before",
                "date": "2026-07-01",
                "pass": False,
                "suites": {"smoke": {"meaningful_program_rate": 0.2}},
            },
            {
                "run_id": "after",
                "date": "2026-07-02",
                "pass": True,
                "suites": {"smoke": {"meaningful_program_rate": 0.5}},
            },
        ]
    }

    detail = readers.experiment_flag("slm.schema_in_context")

    assert detail is not None
    assert detail["key"] == "slm.schema_in_context"
    assert detail["comparisons"] == [
        {
            "baseline_run_id": "before",
            "baseline_value": False,
            "run_id": "after",
            "value": True,
            "date": "2026-07-02",
            "outcomes": ["smoke meaningful_program_rate: +0.300"],
            "outcome_summary": "smoke meaningful_program_rate: +0.300",
        }
    ]
