from pathlib import Path

from slm_training.web.observability import Readers

REPO = Path(__file__).resolve().parents[2]


def test_catalog_includes_generated_levers_and_backfilled_history() -> None:
    payload = Readers(REPO).experiment_flags()
    fields = {row["field"] for row in payload["flags"]}
    assert payload["history_runs"] > 400
    assert "schema_in_context" in fields
    assert "run_id" not in fields


def test_historical_run_marks_only_recorded_recipe_values() -> None:
    flags = Readers(REPO).run_feature_flags(
        "e128_judged_schema_slots_64", REPO / "outputs" / "missing"
    )
    rows = {row["field"]: row for row in flags["rows"]}

    assert flags["provenance"] == "committed"
    assert rows["schema_in_context"]["cells"]["historical"] == {
        "recorded": True,
        "value": True,
        "state": "enabled",
        "source": "committed",
    }
    assert rows["semantic_role_decode_weight"]["cells"]["historical"]["state"] == "not_recorded"
