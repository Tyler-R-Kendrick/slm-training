"""Mock-LM tests for the LDI1-02 causal training loop (wiring only)."""

from __future__ import annotations

import pytest
import torch

from slm_training.harnesses.preference.causal_balancing import CausalTrainingItem
from slm_training.harnesses.preference.causal_trainer import (
    CausalPolicy,
    train_causal_local,
)
from slm_training.harnesses.preference.decision_events_v2 import (
    DecisionStateV2,
    ObjectiveView,
)
from slm_training.harnesses.preference.local_decisions import split_for_group

GROUP = "ldi1-02-train"
VOCAB = 6


class MockCausalPolicy:
    """A torch-only base+adapter policy keyed on the last prefix token.

    The base table is frozen; only the adapter delta is trainable, so an
    adapter-disabled forward reproduces the frozen base (the reference parity the
    real PEFT plugin provides via ``disable_adapter``).
    """

    def __init__(self, vocab: int = VOCAB, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._base = torch.nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self._base.weight.copy_(torch.randn(vocab, vocab, generator=generator))
        self._base.weight.requires_grad_(False)
        self._adapter = torch.nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self._adapter.weight.zero_()
        self._enabled = True

    def forward_logits(self, prefix_ids):
        last = torch.tensor(int(prefix_ids[-1]), dtype=torch.long)
        logits = self._base(last)
        if self._enabled:
            logits = logits + self._adapter(last)
        return logits

    def set_adapter_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def trainable_parameters(self):
        return [self._adapter.weight]

    def adapter_state(self):
        return {name: tensor.clone() for name, tensor in self._adapter.state_dict().items()}

    def load_adapter_state(self, state) -> None:
        self._adapter.load_state_dict(state)


def _state(tag: str, *, last: int) -> DecisionStateV2:
    return DecisionStateV2(
        group_id=GROUP,
        architecture="causal",
        context_text="root=Stack([",
        context_ids=(0, last),
        decision_position=1,
        legal_action_ids=(1, 2, 3),
        decision_kind="component",
        abstract_state_role="component_slot",
        grammar_state_hash=f"gsh-{tag}",
        policy_checkpoint_sha="pcs",
        tokenizer_sha="tsha",
        decode_config_hash="dch",
        verifier_bundle_hash="vbh",
        split=split_for_group(GROUP),
    )


def _view(*, trainable: bool = True) -> ObjectiveView:
    return ObjectiveView(
        good_action_ids=(1,),
        bad_action_ids=(2,),
        ambiguous_action_ids=(),
        unobserved_action_ids=(3,),
        weights=((1, 1.0),),
        materializer_id="set_valued_v2",
        materializer_config_hash="h",
        trainable=trainable,
    )


def _items() -> list[CausalTrainingItem]:
    return [
        CausalTrainingItem(_state(f"t{last}", last=last), _view(), "suite")
        for last in range(VOCAB)
    ]


def test_reference_parity_when_adapter_disabled() -> None:
    policy = MockCausalPolicy()
    with torch.no_grad():
        policy.trainable_parameters()[0].fill_(0.5)
    policy.set_adapter_enabled(False)
    base = policy.forward_logits((0, 2))
    policy.set_adapter_enabled(True)
    enabled = policy.forward_logits((0, 2))
    assert torch.allclose(enabled - base, torch.full((VOCAB,), 0.5))


def test_training_does_not_worsen_held_out_and_updates_adapter() -> None:
    policy = MockCausalPolicy()
    summary = train_causal_local(
        _items(),
        policy,
        objective="ftpo_single",
        strata=["decision_kind"],
        seed=0,
        max_epochs=25,
        lr=0.5,
    )
    assert summary["trainable_parameters"] > 0
    # Best-state restoration guarantees the kept adapter never regresses held-out.
    assert summary["post"]["loss"] <= summary["pre"]["loss"] + 1e-9
    # And on this separable toy setup it should strictly improve.
    assert summary["post"]["loss"] < summary["pre"]["loss"]
    assert summary["claim"] == "wiring only; no quality claim"


def test_adapter_state_roundtrips() -> None:
    policy = MockCausalPolicy()
    train_causal_local(
        _items(), policy, objective="ftpo_single", strata=["decision_kind"], seed=0, max_epochs=3
    )
    saved = policy.adapter_state()
    logits_before = policy.forward_logits((0, 2)).detach().clone()
    with torch.no_grad():
        policy.trainable_parameters()[0].add_(1.0)
    policy.load_adapter_state(saved)
    assert torch.allclose(policy.forward_logits((0, 2)), logits_before)


def test_missing_context_ids_is_rejected() -> None:
    state = _state("x", last=2)
    object.__setattr__(state, "context_ids", None)
    with pytest.raises(ValueError, match="context_ids"):
        train_causal_local(
            [CausalTrainingItem(state, _view(), "s")],
            MockCausalPolicy(),
            objective="ftpo_single",
            strata=["decision_kind"],
            seed=0,
            max_epochs=1,
        )


def test_all_nontrainable_items_raise() -> None:
    items = [CausalTrainingItem(_state("a", last=1), _view(trainable=False), "s")]
    with pytest.raises(ValueError, match="no trainable items"):
        train_causal_local(
            items, MockCausalPolicy(), objective="ftpo_single", strata=["decision_kind"], seed=0
        )


def test_policy_satisfies_protocol() -> None:
    assert isinstance(MockCausalPolicy(), CausalPolicy)
