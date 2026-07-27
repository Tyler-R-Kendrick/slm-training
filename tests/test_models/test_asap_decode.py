"""A2 (SLM-38): ASAp-style distribution-aware constrained MaskGIT decode."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.parallel_decode import AsapLedger  # noqa: E402


def test_ledger_penalize_accumulates_and_floors() -> None:
    ledger = AsapLedger(floor=1e-6)
    assert ledger.keep_factor(3, 7) == 1.0
    assert not ledger.has_penalties()

    ledger.penalize(3, 7, 0.4)
    assert ledger.removed_mass(3, 7) == pytest.approx(0.4)
    assert ledger.keep_factor(3, 7) == pytest.approx(0.6)
    assert ledger.has_penalties() and ledger.has_penalties(3)
    assert not ledger.has_penalties(4)

    # Accumulation across repeated violations, clamped to [0, 1].
    ledger.penalize(3, 7, 0.8)
    assert ledger.removed_mass(3, 7) == pytest.approx(1.0)
    assert ledger.keep_factor(3, 7) == pytest.approx(1e-6)
    # Negative mass never *reduces* an observed removal.
    ledger.penalize(3, 7, -5.0)
    assert ledger.removed_mass(3, 7) == pytest.approx(1.0)
    assert ledger.penalties == 3


def test_adjust_logits_row_shifts_only_penalized_tokens() -> None:
    ledger = AsapLedger()
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    # No penalties at this position: identity (same object is fine).
    assert torch.equal(ledger.adjust_logits_row(logits, 0), logits)

    ledger.penalize(0, 0, 0.75)  # keep factor 0.25 → shift log(0.25)
    adjusted = ledger.adjust_logits_row(logits, 0)
    assert adjusted[0].item() == pytest.approx(2.0 + math.log(0.25))
    assert adjusted[1].item() == pytest.approx(1.0)
    assert adjusted[2].item() == pytest.approx(0.5)
    # Original logits untouched (clone semantics).
    assert logits[0].item() == pytest.approx(2.0)
    # Other positions unaffected.
    assert torch.equal(ledger.adjust_logits_row(logits, 1), logits)


def test_asap_removes_violating_mass_until_alternative_wins() -> None:
    """
    The core ASAp property transplanted to a canvas position: a token whose
    mass is repeatedly observed to violate loses exactly that mass, so the
    proposal argmax converges to the model's best *non-violating* token
    instead of re-proposing the same distorted winner forever.
    """
    ledger = AsapLedger()
    # Model puts 0.9 on token 0 (violates downstream), 0.1 on token 1 (legal).
    probs = torch.tensor([0.9, 0.1])
    logits = probs.log()

    assert int(ledger.adjust_logits_row(logits, 5).argmax().item()) == 0
    # Violation observed: remove the mass the model assigned to token 0.
    ledger.penalize(5, 0, float(probs[0]))
    adjusted = ledger.adjust_logits_row(logits, 5)
    assert int(adjusted.argmax().item()) == 1
    # Post-removal mass of the violator is p * (1 - p) = 0.09 < 0.1.
    assert adjusted[0].item() == pytest.approx(math.log(0.9 * 0.1), abs=1e-5)


def test_adjusted_confidence_reorders_unmask_priority() -> None:
    ledger = AsapLedger()
    # Two masked positions: position 0 has the higher raw max-prob (0.8) but
    # its argmax (token 2) was observed to violate; position 1 max-prob 0.6.
    probs = torch.tensor([[[0.1, 0.1, 0.8], [0.2, 0.6, 0.2]]])
    conf = probs.max(dim=-1).values
    unknown = torch.tensor([[True, True]])

    # No penalties: identity.
    assert torch.equal(ledger.adjusted_confidence(probs, conf, unknown), conf)

    ledger.penalize(0, 2, 0.8)
    adjusted = ledger.adjusted_confidence(probs, conf, unknown)
    # Post-removal confidence at position 0: max(0.1, 0.1, 0.8*0.2) = 0.16.
    assert adjusted[0, 0].item() == pytest.approx(0.16)
    assert adjusted[0, 1].item() == pytest.approx(0.6)
    assert adjusted[0, 1] > adjusted[0, 0]

    # Committed (not unknown) positions are never rewritten.
    known = torch.tensor([[False, True]])
    kept = ledger.adjusted_confidence(probs, conf, known)
    assert kept[0, 0].item() == pytest.approx(conf[0, 0].item())


def test_asap_decode_config_threads_from_model_build() -> None:
    from slm_training.harnesses.model_build.config import ModelBuildConfig
    from slm_training.harnesses.model_build.factory import (
        _twotower_config_from_build,
    )

    default_cfg = _twotower_config_from_build(ModelBuildConfig(train_dir="."))
    assert default_cfg.asap_decode is False
    on_cfg = _twotower_config_from_build(
        ModelBuildConfig(train_dir=".", asap_decode=True)
    )
    assert on_cfg.asap_decode is True


def test_maskgit_loop_instantiates_ledger_only_when_enabled() -> None:
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui='root = Stack([t])\nt = TextContent(":slot_0")',
            placeholders=[":slot_0"],
        )
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            seed=0,
            gen_steps=2,
            asap_decode=True,
        ),
        device="cpu",
    )
    # The lever is decode-only and opt-in: generation must run (and terminate)
    # with the ledger active, and stay inert when grammar decode is off.
    out = model.generate("Hero", grammar_constrained=True)
    assert isinstance(out, str)
    model.config.asap_decode = False
    out_off = model.generate("Hero", grammar_constrained=True)
    assert isinstance(out_off, str)
