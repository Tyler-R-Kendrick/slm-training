from pathlib import Path

from slm_training.web.observability import Readers

REPO = Path(__file__).resolve().parents[2]


def test_catalog_includes_generated_levers_and_backfilled_history() -> None:
    payload = Readers(REPO).experiment_flags()
    keys = {row["key"] for row in payload["flags"]}
    assert payload["history_runs"] > 400
    assert "slm.schema_in_context" in keys
    assert "slm.run_id" not in keys


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
