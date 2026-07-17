from __future__ import annotations

import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
    write_decision_events,
)
from slm_training.harnesses.preference.local_train import (
    _event_logits,
    _event_logits_many,
    _gradient_alignment,
    _minimum_norm_gradient,
    _project_conflicting_gradients,
    _guard_strata_regressions,
    evaluate_local_decisions,
    event_schedule,
    local_decision_loss,
    proposal_schedule,
    train_local_decisions,
    train_local_from_paths,
)
from slm_training.harnesses.distill.trace_store import checkpoint_sha
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _event(*, good=(2,), bad=(3,), group="group") -> DecisionEventV1:
    while split_for_group(group) != "train":
        group += "x"
    return DecisionEventV1(
        event_id=f"event-{group}-{good}-{bad}",
        group_id=group,
        context_text="Generate a card",
        canvas_ids=(1, 0, 0, 0),
        position=1,
        good_token_ids=good,
        bad_token_ids=bad,
        legal_token_ids=good,
        evidence_kind="counterfactual",
        evidence_confidence=1.0,
        decision_kind="component",
        split="train",
        policy_checkpoint_sha="policy",
        tokenizer_sha="tokenizer",
        decode_config_hash="decode",
        seed=0,
        trajectory_id="trace",
    )


def test_ftpo_gradient_moves_good_above_bad() -> None:
    logits = torch.zeros(8, requires_grad=True)
    loss, metrics = local_decision_loss(logits, _event(), objective="ftpo_single")
    loss.backward()
    assert logits.grad[2] < 0
    assert logits.grad[3] > 0
    assert metrics["chosen_win"] == 0.0
    assert metrics["active_weight"] == 1.0


def test_training_rejects_constraint_shadows_as_semantic_labels() -> None:
    model = _model()
    shadow = replace(_event(), evidence_kind="constraint_shadow")
    with pytest.raises(ValueError, match="decoder legality"):
        train_local_decisions(model, [shadow], objective="ce_margin", steps=1)


def test_reference_tether_excludes_target_tokens_within_grace() -> None:
    logits = torch.zeros(8, requires_grad=True)
    logits.data[5] = 2.0
    reference = torch.zeros(8)
    _, metrics = local_decision_loss(
        logits,
        _event(),
        objective="ftpo_single",
        reference_logits=reference,
        non_target_tether=0.4,
        target_tether=0.05,
        target_grace=1.0,
    )
    assert metrics["non_target_logit_mse"] > 0
    assert metrics["target_excess_logit_mse"] == 0.0


def test_set_objective_and_balancing_fail_closed() -> None:
    event = _event()
    logits = torch.zeros(8)
    with pytest.raises(ValueError, match="one good"):
        local_decision_loss(
            logits, _event(good=(2, 4)), objective="ftpo_single"
        )
    schedule = event_schedule([event], steps=3, seed=0, balanced=True)
    assert schedule == [event, event, event]


def test_proposal_schedule_groups_events_by_decision_kind() -> None:
    component = _event(group="component")
    component_two = replace(component, event_id="component-two")
    grammar = replace(_event(group="grammar"), decision_kind="grammar_comma")

    schedule = proposal_schedule(
        [component, component_two, grammar],
        steps=2,
        seed=0,
        balanced=False,
        block_by_decision_kind=True,
    )

    assert [[event.decision_kind for event in block] for block in schedule] == [
        ["component", "component"],
        ["grammar_comma"],
    ]


def test_project_conflicting_gradients_removes_negative_component() -> None:
    combined, report = _project_conflicting_gradients(
        [[torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])]]
    )

    assert report == {
        "task_count": 2,
        "ordered_pair_count": 2,
        "conflict_count": 2,
        "projection_count": 2,
    }
    assert combined[0] is not None
    assert torch.allclose(combined[0], torch.tensor([0.25, 0.75]))


def test_gradient_alignment_reports_cosine_and_norms() -> None:
    report = _gradient_alignment(
        [torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])]
    )

    assert report["dot"] == -1.0
    assert report["left_norm"] == 1.0
    assert report["right_norm"] == pytest.approx(2**0.5)
    assert report["cosine"] == pytest.approx(-(2**-0.5))


def test_minimum_norm_gradient_certifies_common_descent() -> None:
    combined, report = _minimum_norm_gradient(
        [[torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 2.0])]]
    )

    assert combined[0] is not None
    assert report["converged"] is True
    assert report["common_descent"] is True
    assert report["min_task_dot"] > 0
    assert torch.allclose(combined[0], torch.tensor([0.8, 0.4]), atol=1e-6)


def test_minimum_norm_gradient_ignores_inactive_tasks() -> None:
    combined, report = _minimum_norm_gradient(
        [[torch.tensor([0.0, 0.0])], [torch.tensor([1.0, 2.0])]]
    )

    assert combined[0] is not None
    assert report["active_task_count"] == 1
    assert report["inactive_task_count"] == 1
    assert report["common_descent"] is True
    assert report["weights"] == [0.0, 1.0]
    assert torch.equal(combined[0], torch.tensor([1.0, 2.0]))


def test_projection_requires_stratified_guard() -> None:
    with pytest.raises(ValueError, match="require the decision-kind guard"):
        train_local_decisions(
            _model(), [_event()], objective="ce_margin", steps=1,
            gradient_combination="pcgrad",
        )


def test_train_projects_all_decision_kinds_before_guarding() -> None:
    model = _model()
    first = _event(
        good=(model.tokenizer.eos_id,),
        bad=(model.tokenizer.mask_id,),
        group="first",
    )
    second = replace(
        first,
        event_id="second",
        group_id="second",
        decision_kind="grammar_comma",
    )
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(
        first, event_id="held", group_id=held_group, split="held_out"
    )

    summary = train_local_decisions(
        model,
        [first, second],
        objective="ce_margin",
        steps=1,
        validation_events=[held],
        guarded_updates=True,
        guard_backtrack_steps=0,
        guard_by_decision_kind=True,
        gradient_combination="pcgrad",
    )

    assert summary["gradient_combination"] == "pcgrad"
    assert summary["gradient_projection"]["task_count"] == 2
    assert summary["gradient_projection"]["ordered_pair_count"] == 2
    assert summary["decision_kind_steps"] == {
        "component": 1,
        "grammar_comma": 1,
    }


def test_uncertified_mgda_bypasses_optimizer_and_guard_trials(monkeypatch) -> None:
    model = _model()
    first = _event(
        good=(model.tokenizer.eos_id,),
        bad=(model.tokenizer.mask_id,),
        group="first",
    )
    second = replace(first, event_id="second", decision_kind="grammar_comma")
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(first, event_id="held", group_id=held_group, split="held_out")
    metrics = {
        "loss": 1.0,
        "bad_probability_mass": 0.1,
        "good_probability_mass": 0.2,
        "mean_margin": 1.0,
    }
    baseline = {
        "metrics": metrics,
        "by_decision_kind": {"component": {"metrics": metrics}},
    }
    monkeypatch.setattr(
        "slm_training.harnesses.preference.local_train._minimum_norm_gradient",
        lambda gradients: (
            [None] * len(gradients[0]),
            {"common_descent": False, "task_count": len(gradients)},
        ),
    )
    before = copy.deepcopy(model.state_dict())

    summary = train_local_decisions(
        model,
        [first, second],
        objective="ce_margin",
        steps=1,
        validation_events=[held],
        validation_baseline=baseline,
        guarded_updates=True,
        guard_by_decision_kind=True,
        gradient_combination="mgda",
    )

    history = summary["validation_selection"]["history"][1]
    assert history["rejection_reason"] == "no_common_descent_certificate"
    assert history["trials"] == []
    assert summary["validation_selection"]["accepted_steps"] == 0
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )


def test_single_objective_filters_set_valued_events() -> None:
    model = _model()
    single = _event(
        good=(model.tokenizer.eos_id,), bad=(model.tokenizer.mask_id,), group="single"
    )
    multiple = _event(
        good=(model.tokenizer.eos_id, model.tokenizer.pad_id),
        bad=(model.tokenizer.mask_id,),
        group="multiple",
    )
    summary = train_local_decisions(
        model, [single, multiple], objective="ftpo_single", steps=1
    )
    assert summary["train_events"] == 1
    assert summary["excluded_train_events"] == 1


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="a",
        prompt="Card",
        openui='root = TextContent(":card.title")',
        split="train",
        placeholders=[":card.title"],
    )
    return TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=16,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_target_len=8,
            seed=0,
        ),
        device="cpu",
    )


def test_train_local_decisions_freezes_reference() -> None:
    model = _model()
    reference = copy.deepcopy(model)
    event = _event(
        good=(model.tokenizer.eos_id,),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
        }
    )
    summary = train_local_decisions(
        model,
        [event],
        objective="ftpo_single",
        reference_model=reference,
        steps=1,
        non_target_tether=0.4,
        target_tether=0.05,
    )
    assert summary["steps"] == 1
    assert summary["reference_tethered"] is True
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_evaluate_local_decisions_reports_held_out_recurrence() -> None:
    model = _model()
    group = "held"
    while split_for_group(group) != "held_out":
        group += "x"
    event = _event(
        good=(model.tokenizer.eos_id,),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
            "event_id": f"event-{group}",
            "group_id": group,
            "split": "held_out",
        }
    )
    report = evaluate_local_decisions(
        model, [event], objective="ftpo_single"
    )
    assert report["event_count"] == 1
    assert report["by_decision_kind"]["component"]["event_count"] == 1
    assert "chosen_win" in report["metrics"]


def test_batched_event_logits_match_single_event_evaluation() -> None:
    model = _model()
    events = []
    for group in ("first", "second"):
        event = _event(
            good=(model.tokenizer.eos_id,),
            bad=(model.tokenizer.mask_id,),
            group=group,
        )
        events.append(
            DecisionEventV1.from_dict(
                {
                    **event.to_dict(),
                    "canvas_ids": [model.tokenizer.bos_id]
                    + [model.tokenizer.mask_id] * 3,
                }
            )
        )
    model.eval()

    expected = [_event_logits(model, event) for event in events]
    actual = _event_logits_many(model, events)

    assert all(torch.allclose(left, right) for left, right in zip(expected, actual))


def test_decision_kind_guard_exposes_regression_hidden_by_aggregate() -> None:
    baseline = {
        "by_decision_kind": {
            "grammar_comma": {
                "metrics": {
                    "loss": 1.0,
                    "bad_probability_mass": 0.1,
                    "good_probability_mass": 0.2,
                    "mean_margin": 1.0,
                }
            }
        }
    }
    candidate = copy.deepcopy(baseline)
    candidate["by_decision_kind"]["grammar_comma"]["metrics"]["loss"] = 2.0

    assert _guard_strata_regressions(candidate, baseline) == [
        {
            "decision_kind": "grammar_comma",
            "metric": "loss",
            "before": 1.0,
            "after": 2.0,
        }
    ]


def test_ftpo_set_requires_verified_set_event() -> None:
    with pytest.raises(ValueError, match="set-valued"):
        train_local_decisions(_model(), [_event()], objective="ftpo_set", steps=1)


def test_guarded_selection_restores_parent_when_validation_regresses(
    monkeypatch,
) -> None:
    model = _model()
    event = _event(
        good=(model.tokenizer.eos_id, model.tokenizer.pad_id),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
        }
    )
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(
        event,
        split="held_out",
        group_id=held_group,
        event_id="held",
    )
    baseline = {
        "event_count": 1,
        "metrics": {
            "loss": 1.0,
            "bad_probability_mass": 0.1,
            "good_probability_mass": 0.2,
            "mean_margin": 1.0,
        },
    }
    regressed = {
        "event_count": 1,
        "metrics": {
            "loss": 2.0,
            "bad_probability_mass": 0.2,
            "good_probability_mass": 0.1,
            "mean_margin": 0.0,
        },
    }
    monkeypatch.setattr(
        "slm_training.harnesses.preference.local_train.evaluate_local_decisions",
        lambda *args, **kwargs: regressed,
    )
    before = copy.deepcopy(model.state_dict())

    summary = train_local_decisions(
        model,
        [event],
        objective="ftpo_set",
        steps=1,
        validation_events=[held],
        validation_baseline=baseline,
        validation_every=1,
        guarded_selection=True,
    )

    assert summary["validation_selection"]["selected_step"] == 0
    assert summary["validation_selection"]["restored"] is True
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_guarded_updates_backtrack_and_accept_pareto_improvement(monkeypatch) -> None:
    model = _model()
    event = _event(
        good=(model.tokenizer.eos_id, model.tokenizer.pad_id),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
        }
    )
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(
        event, split="held_out", group_id=held_group, event_id="held"
    )
    baseline = {
        "event_count": 1,
        "metrics": {
            "loss": 1.0,
            "bad_probability_mass": 0.1,
            "good_probability_mass": 0.2,
            "mean_margin": 1.0,
        },
    }
    reports = iter(
        [
            {
                "metrics": {
                    "loss": 1.1,
                    "bad_probability_mass": 0.11,
                    "good_probability_mass": 0.21,
                    "mean_margin": 1.1,
                }
            },
            {
                "metrics": {
                    "loss": 0.9,
                    "bad_probability_mass": 0.09,
                    "good_probability_mass": 0.21,
                    "mean_margin": 1.1,
                }
            },
        ]
    )
    monkeypatch.setattr(
        "slm_training.harnesses.preference.local_train.evaluate_local_decisions",
        lambda *args, **kwargs: next(reports),
    )
    before = copy.deepcopy(model.state_dict())

    summary = train_local_decisions(
        model,
        [event],
        objective="ftpo_set",
        steps=1,
        validation_events=[held],
        validation_baseline=baseline,
        guarded_updates=True,
    )

    selection = summary["validation_selection"]
    assert selection["accepted_steps"] == 1
    assert selection["rejected_steps"] == 0
    assert selection["history"][1]["accepted_scale"] == 0.5
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )


def test_guarded_updates_restore_model_when_all_scales_regress(monkeypatch) -> None:
    model = _model()
    event = _event(
        good=(model.tokenizer.eos_id, model.tokenizer.pad_id),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
        }
    )
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(
        event, split="held_out", group_id=held_group, event_id="held"
    )
    baseline = {
        "event_count": 1,
        "metrics": {
            "loss": 1.0,
            "bad_probability_mass": 0.1,
            "good_probability_mass": 0.2,
            "mean_margin": 1.0,
        },
    }
    regressed = {
        "metrics": {
            "loss": 1.1,
            "bad_probability_mass": 0.11,
            "good_probability_mass": 0.21,
            "mean_margin": 1.1,
        }
    }
    monkeypatch.setattr(
        "slm_training.harnesses.preference.local_train.evaluate_local_decisions",
        lambda *args, **kwargs: regressed,
    )
    before = copy.deepcopy(model.state_dict())

    summary = train_local_decisions(
        model,
        [event],
        objective="ftpo_set",
        steps=1,
        validation_events=[held],
        validation_baseline=baseline,
        guarded_updates=True,
        guard_backtrack_steps=2,
    )

    selection = summary["validation_selection"]
    assert selection["accepted_steps"] == 0
    assert selection["rejected_steps"] == 1
    assert len(selection["history"][1]["trials"]) == 3
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )


def test_guarded_updates_reject_aggregate_gain_with_stratum_regression(
    monkeypatch,
) -> None:
    model = _model()
    event = _event(
        good=(model.tokenizer.eos_id, model.tokenizer.pad_id),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
        }
    )
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held = replace(
        event, split="held_out", group_id=held_group, event_id="held"
    )
    metrics = {
        "loss": 1.0,
        "bad_probability_mass": 0.1,
        "good_probability_mass": 0.2,
        "mean_margin": 1.0,
    }
    baseline = {
        "event_count": 1,
        "metrics": metrics,
        "by_decision_kind": {"component": {"metrics": metrics}},
    }
    candidate = {
        "metrics": {
            "loss": 0.9,
            "bad_probability_mass": 0.09,
            "good_probability_mass": 0.21,
            "mean_margin": 1.1,
        },
        "by_decision_kind": {
            "component": {"metrics": {**metrics, "loss": 1.1}}
        },
    }
    monkeypatch.setattr(
        "slm_training.harnesses.preference.local_train.evaluate_local_decisions",
        lambda *args, **kwargs: candidate,
    )
    before = copy.deepcopy(model.state_dict())

    summary = train_local_decisions(
        model,
        [event],
        objective="ftpo_set",
        steps=1,
        validation_events=[held],
        validation_baseline=baseline,
        guarded_updates=True,
        guard_backtrack_steps=0,
        guard_by_decision_kind=True,
    )

    trial = summary["validation_selection"]["history"][1]["trials"][0]
    assert trial["eligible"] is False
    assert trial["strata_regression_count"] == 1
    assert trial["strata_regression_kinds"] == ["component"]
    assert summary["validation_selection"]["strata_regression_counts"] == {
        "component:loss": 1
    }
    assert all(
        torch.equal(before[key], value) for key, value in model.state_dict().items()
    )


def test_train_local_from_paths_checks_identity_and_writes_checkpoint(tmp_path) -> None:
    model = _model()
    checkpoint = tmp_path / "parent.pt"
    model.save(checkpoint)
    event = _event(
        good=(model.tokenizer.eos_id,),
        bad=(model.tokenizer.mask_id,),
    )
    event = DecisionEventV1.from_dict(
        {
            **event.to_dict(),
            "canvas_ids": [model.tokenizer.bos_id]
            + [model.tokenizer.mask_id] * 3,
            "policy_checkpoint_sha": checkpoint_sha(checkpoint),
            "tokenizer_sha": model.artifact_identity()["tokenizer_sha"],
        }
    )
    events = tmp_path / "events.jsonl"
    write_decision_events(events, [event])
    summary = train_local_from_paths(
        checkpoint,
        events,
        out_dir=tmp_path / "run",
        objective="ftpo_single",
        reference_checkpoint=checkpoint,
        steps=1,
        non_target_tether=0.4,
    )
    assert (tmp_path / "run/model.pt").is_file()
    assert summary["source_checkpoint_sha"] == checkpoint_sha(checkpoint)
    assert summary["held_out_before"]["event_count"] == 0
    assert summary["held_out_after"]["event_count"] == 0
