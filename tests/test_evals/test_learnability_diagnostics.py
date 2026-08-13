import json
from pathlib import Path

from slm_training.evals.learnability_diagnostics import (
    CLAIM_CLASS,
    diagnose_records,
    interpret_probe,
    load_suite_prompt_records,
    make_interventions,
    write_learnability_diagnostics,
)
from slm_training.harness_core.lineage.records import content_sha


def test_shuffle_does_not_carry_the_target() -> None:
    items = make_interventions(
        "Need a checkout form with a pay button",
        group_id="g1",
        target_id="gold-1",
    )
    by_name = {item.name: item for item in items}
    assert by_name["correct"].target_id == "gold-1"
    assert by_name["shuffled"].target_id is None
    assert by_name["cue_ablation"].group_id == "g1"
    assert by_name["cue_ablation"].preserves_semantics is True


def test_incomplete_is_not_a_zero() -> None:
    result = interpret_probe({"timeout": True, "train_exact": False})
    assert result["disposition"] == "incomplete"
    assert result["claim_class"] == CLAIM_CLASS


def test_fixture_run_cannot_issue_a_certificate() -> None:
    payload = diagnose_records(
        [{"id": "r1", "prompt": "Show a settings page", "group_id": "g"}],
        probe_metrics={"train_exact": True, "shuffled_output_equals_correct": True},
    )
    assert payload["capability_certificate"] is False
    assert payload["interpretation"]["disposition"] == "conditioning_or_data_alignment"


def test_diagnose_records_includes_version_stamp() -> None:
    payload = diagnose_records(
        [{"id": "r1", "prompt": "Show a settings page", "group_id": "g"}]
    )
    stamp = payload["version_stamp"]
    assert stamp["stamp_schema"] == "version_stamp/v1"
    assert stamp["components"]["evals.scoring"]
    expected = content_sha(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    assert payload["sha256"] == expected


def test_suite_prompts_feed_diagnose_records(tmp_path: Path) -> None:
    suite = tmp_path / "suites" / "smoke"
    suite.mkdir(parents=True)
    (suite / "records.jsonl").write_text(
        json.dumps({"id": "s1", "prompt": "Build a checkout form with a pay button"})
        + "\n"
        + json.dumps({"id": "s2", "prompt": "Show a settings page"})
        + "\n"
        + json.dumps({"id": "s3", "primary_metric": "parse_rate"})
        + "\n",
        encoding="utf-8",
    )
    records, reasons = load_suite_prompt_records(tmp_path, ["smoke"])
    assert reasons == []
    assert [row["prompt"] for row in records] == [
        "Build a checkout form with a pay button",
        "Show a settings page",
    ]
    payload = diagnose_records(records)
    prompts = [item["prompt"] for item in payload["interventions"]]
    assert "parse_rate" not in prompts
    assert "Build a checkout form with a pay button" in prompts


def test_missing_suite_writes_empty_interventions_with_reason(tmp_path: Path) -> None:
    path = write_learnability_diagnostics(
        tmp_path / "run",
        test_dir=tmp_path,
        suites=["missing"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["interventions"] == []
    assert payload["unavailable_reasons"]
    assert payload["capability_certificate"] is False
    assert payload["claim_class"] == CLAIM_CLASS
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
