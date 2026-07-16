from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
    write_decision_events,
)
from slm_training.harnesses.preference.local_train import (
    evaluate_local_decisions,
    event_schedule,
    local_decision_loss,
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


def test_ftpo_set_requires_verified_set_event() -> None:
    with pytest.raises(ValueError, match="set-valued"):
        train_local_decisions(_model(), [_event()], objective="ftpo_set", steps=1)


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
