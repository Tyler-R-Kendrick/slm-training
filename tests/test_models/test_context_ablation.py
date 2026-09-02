"""S6/N2: context-tower causal ablation on the real decode path.

``context_ablation`` is a *diagnostic control* lever (I6, "unconstrained arms
are diagnostic controls only"): it never changes what is legal -- every commit
still goes through the same grammar-constrained pick -- but it corrupts the
projected context tower output so the tower's causal contribution to
legal-set decisions can be measured instead of assumed.

The load-bearing test here is
``test_off_is_byte_identical_to_the_unablated_decode``: the default value must
leave today's decode untouched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.levers import (
    DIAGNOSTIC_ONLY_LEVERS,
    constraint_weakening_violations,
    lever_catalog,
    require_constrained_production_config,
)
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'


def _tiny_model(**overrides) -> TwoTowerModel:
    records = [
        ExampleRecord(id="a", prompt="Hero card", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="Call to action", openui=CTA, split="train"),
    ]
    cfg = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=3,
        seed=0,
        grammar_ltr_primary=False,  # route into the positionwise MaskGIT loop
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    return model


def _ctx(model: TwoTowerModel) -> tuple[torch.Tensor, torch.Tensor]:
    return model._encode_context(["Hero card", "Call to action"])


# ---------------------------------------------------------------------------
# The default is inert
# ---------------------------------------------------------------------------


def test_off_returns_the_caller_tensor_object_unchanged() -> None:
    model = _tiny_model()
    ctx, ctx_pad = _ctx(model)
    assert model.config.context_ablation == "off"
    assert model._apply_context_ablation(ctx, ctx_pad) is ctx


def test_off_is_byte_identical_to_the_unablated_decode(monkeypatch) -> None:
    """The pre-S6 decode is the ``off`` decode, token for token.

    ``_apply_context_ablation`` reads the field with ``getattr(..., "off")``,
    so deleting it reproduces a config that predates this lever. Both arms must
    produce the same text and leave every S6 counter at zero.
    """
    model = _tiny_model()
    prompts = ["Hero card", "Call to action"]
    with collect_decode_stats() as with_field:
        current = model.generate_batch(prompts)

    monkeypatch.delattr(TwoTowerConfig, "context_ablation", raising=False)
    delattr(model.config, "context_ablation")
    assert not hasattr(model.config, "context_ablation")
    with collect_decode_stats() as without_field:
        legacy = model.generate_batch(prompts)

    assert current == legacy
    for stats in (with_field, without_field):
        assert stats.context_ablation_rows == 0
        assert stats.context_ablation_degenerate_rows == 0
        assert stats.context_ablation_applications == 0
        assert stats.context_ablation_choice_changes == 0
    assert with_field.forwards_count == without_field.forwards_count
    assert with_field.tokens_emitted == without_field.tokens_emitted


# ---------------------------------------------------------------------------
# The modes do what they claim
# ---------------------------------------------------------------------------


def test_zero_mode_blanks_the_projected_context() -> None:
    model = _tiny_model(context_ablation="zero")
    ctx, ctx_pad = _ctx(model)
    with collect_decode_stats() as stats:
        out = model._apply_context_ablation(ctx, ctx_pad)
    assert torch.equal(out, torch.zeros_like(ctx))
    assert stats.context_ablation_rows == ctx.size(0)
    assert stats.context_ablation_degenerate_rows == 0


def test_shuffle_batch_gives_every_row_another_rows_prompt() -> None:
    model = _tiny_model(context_ablation="shuffle_batch")
    ctx, ctx_pad = _ctx(model)
    with collect_decode_stats() as stats:
        out = model._apply_context_ablation(ctx, ctx_pad)
    assert torch.equal(out, torch.roll(ctx, shifts=1, dims=0))
    assert stats.context_ablation_rows == ctx.size(0)


def test_shuffle_batch_reports_a_single_row_batch_as_degenerate() -> None:
    """A 1-row batch has no other row to read; that must not read as ablated."""
    model = _tiny_model(context_ablation="shuffle_batch")
    ctx, ctx_pad = model._encode_context(["Hero card"])
    with collect_decode_stats() as stats:
        out = model._apply_context_ablation(ctx, ctx_pad)
    assert out is ctx
    assert stats.context_ablation_rows == 0
    assert stats.context_ablation_degenerate_rows == 1


def test_shuffle_positions_reverses_each_rows_valid_prefix() -> None:
    model = _tiny_model(context_ablation="shuffle_positions")
    ctx, ctx_pad = _ctx(model)
    with collect_decode_stats() as stats:
        out = model._apply_context_ablation(ctx, ctx_pad)
    for row in range(ctx.size(0)):
        valid = int((~ctx_pad[row]).sum().item())
        if valid < 2:
            continue
        assert torch.equal(out[row, :valid], torch.flip(ctx[row, :valid], dims=(0,)))
    assert stats.context_ablation_rows + stats.context_ablation_degenerate_rows == (
        ctx.size(0)
    )


def test_unknown_mode_fails_closed() -> None:
    model = _tiny_model(context_ablation="nonsense")
    ctx, ctx_pad = _ctx(model)
    with pytest.raises(ValueError, match="unknown context_ablation"):
        model._apply_context_ablation(ctx, ctx_pad)


# ---------------------------------------------------------------------------
# Decision-level probe
# ---------------------------------------------------------------------------


def test_probe_counts_non_singleton_decisions_and_stays_constrained() -> None:
    model = _tiny_model(context_ablation="zero")
    with collect_decode_stats() as stats:
        texts = model.generate_batch(["Hero card", "Call to action"])
    # I6 is not weakened by a diagnostic arm: output is still certified.
    from slm_training.dsl.parser import validate

    for text in texts:
        validate(text)
    assert stats.context_ablation_rows == 2
    assert stats.context_ablation_applications > 0
    assert (
        0
        <= stats.context_ablation_choice_changes
        <= (stats.context_ablation_applications)
    )


def test_shuffle_positions_is_an_exact_null_control() -> None:
    """Cross-attention over context keys is permutation-invariant.

    Reversing the context sequence therefore cannot change a single decision.
    A non-zero flip count here means the probe is manufacturing flips, not that
    the model reads context order.
    """
    model = _tiny_model(context_ablation="shuffle_positions")
    with collect_decode_stats() as ablated:
        shuffled = model.generate_batch(["Hero card", "Call to action"])
    model.config.context_ablation = "off"
    with collect_decode_stats() as intact:
        baseline = model.generate_batch(["Hero card", "Call to action"])
    assert shuffled == baseline
    assert ablated.context_ablation_applications > 0
    assert ablated.context_ablation_choice_changes == 0
    assert intact.context_ablation_applications == 0


# ---------------------------------------------------------------------------
# Fail-closed registration
# ---------------------------------------------------------------------------


def test_registered_diagnostic_only_with_off_as_the_safe_value() -> None:
    spec = DIAGNOSTIC_ONLY_LEVERS["context_ablation"]
    assert spec["safe_value"] == "off"
    entry = lever_catalog()["context_ablation"]
    assert entry["diagnostic_only"] is True
    assert entry["default"] == "off"
    assert entry["constraint_safe_value"] == "off"


def test_production_config_guard_rejects_every_ablated_value() -> None:
    config = TwoTowerConfig()
    require_constrained_production_config(config, context="ship config")
    for mode in ("zero", "shuffle_batch", "shuffle_positions"):
        config.context_ablation = mode
        violations = constraint_weakening_violations(config)
        assert violations["context_ablation"]["value"] == mode
        with pytest.raises(ValueError, match="context_ablation"):
            require_constrained_production_config(config, context="ship config")


def test_enum_lever_guard_is_not_a_truthiness_test() -> None:
    """``bool("zero") == bool("off")``; the guard must compare values."""
    from slm_training.levers import _lever_value_is_safe

    assert _lever_value_is_safe("off", "off")
    assert not _lever_value_is_safe("zero", "off")
    # Boolean levers keep truthiness semantics (0 / None / False all mean off).
    assert _lever_value_is_safe(0, False)
    assert not _lever_value_is_safe(1, False)
