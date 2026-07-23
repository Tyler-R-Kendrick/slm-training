"""Grammar / streaming parser + HF context tower tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import bridge_available, stream_check
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.grammar import (
    StreamStatus,
    apply_structural_bias,
    pick_constrained_token,
    stream_check as grammar_stream_check,
)
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = 'root = Stack([b1], "column")\nb1 = Card([b2, b3])\nb2 = TextContent(":slot_0")\nb3 = TextContent(":slot_1")'
CTA = 'root = Stack([b1])\nb1 = Button(":slot_0")'


@pytestmark_bridge
def test_stream_check_complete_and_partial() -> None:
    raw = stream_check(HERO)
    assert raw["ok"] is True
    assert raw["has_root"] is True
    assert raw["incomplete"] is False

    partial = stream_check('root = Stack([hero], "column")\nhero = Card([')
    assert partial["incomplete"] is True
    assert partial["has_root"] is True

    bad = grammar_stream_check('root = Broken(":x")')
    assert isinstance(bad, StreamStatus)
    assert bad.hard_error


@pytestmark_bridge
def test_contract_allowed_skips_direction_literals() -> None:
    tok = OpenUITokenizer.build([HERO, CTA])
    from slm_training.models.grammar import contract_allowed_token_ids

    # Inside an ordinary "column" literal — contract must not activate.
    prefix = tok.encode('root = Stack([hero], "', add_special=False)
    allowed = contract_allowed_token_ids(
        tok, prefix, [":hero.title", ":hero.body"]
    )
    assert allowed is None


@pytestmark_bridge
def test_pick_constrained_token_avoids_unknown_component() -> None:
    torch = pytest.importorskip("torch")
    tok = OpenUITokenizer.build([HERO, CTA, "Broken"])
    # Prefix that expects a component name after '= '
    prefix = tok.encode("root = ", add_special=False)
    logits = torch.full((tok.vocab_size,), -10.0)
    # Make Broken the argmax, Stack a close second
    logits[tok.token_to_id["Broken"]] = 5.0
    logits[tok.token_to_id["Stack"]] = 4.5
    choice = pick_constrained_token(logits, tok, prefix, top_k=5)
    assert choice == tok.token_to_id["Stack"]


def test_structural_bias_boosts_keywords() -> None:
    torch = pytest.importorskip("torch")
    tok = OpenUITokenizer.build([HERO])
    logits = torch.zeros(1, 2, tok.vocab_size)
    boosted = apply_structural_bias(logits, tok, bias=2.0)
    stack_id = tok.token_to_id["Stack"]
    assert float(boosted[0, 0, stack_id]) == 2.0


@pytestmark_bridge
def test_generate_with_grammar_constrained_overfit(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64,
            n_heads=4,
            context_layers=1,
            denoiser_layers=2,
            gen_steps=6,
            grammar_constrained=True,
            grammar_ltr_primary=True,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            grammar_ltr_max_tokens=48,
            seed=0,
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        model.training_loss(records).backward()
        opt.step()

    pred = model.generate("Hero", grammar_constrained=True)
    from slm_training.dsl.parser import validate

    validate(pred)
    ckpt = tmp_path / "m.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt)
    assert loaded.config.grammar_constrained is True


def test_hf_context_tower_optional(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    _ = transformers
    # Tiny random model for CI; skip if download/cache fails.
    model_id = "hf-internal-testing/tiny-random-gpt2"
    try:
        from transformers import AutoModel, AutoTokenizer

        AutoTokenizer.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF tiny model unavailable: {exc}")

    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64,
            n_heads=4,
            context_layers=1,
            denoiser_layers=2,
            context_backend="hf",
            hf_model_name=model_id,
            freeze_context=True,
            grammar_constrained=False,
            gen_steps=4,
            seed=0,
        ),
        device="cpu",
    )
    # Backbone frozen; projection + denoiser train.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(not n.startswith("context.backbone.") for n in trainable_names)
    frozen = [
        name
        for name, p in model.named_parameters()
        if name.startswith("context.backbone.") and not p.requires_grad
    ]
    assert frozen

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        opt.step()

    ckpt = tmp_path / "hf.pt"
    model.save(ckpt)
    # Backbone weights should not be baked into the checkpoint payload.
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert not any(k.startswith("context.backbone.") for k in payload["state_dict"])
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.context_backend == "hf"
    _ = loaded.generate("Hero layout", grammar_constrained=False)
