from slm_training.evals.learnability_diagnostics import (
    CLAIM_CLASS,
    diagnose_records,
    interpret_probe,
    make_interventions,
)


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
