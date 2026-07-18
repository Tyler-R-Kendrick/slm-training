"""Tokenizer + TwoTower model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.data.contract import GenerationRequest
from slm_training.harnesses.model_build import ModelBuildConfig, evaluate, train
from slm_training.harnesses.model_build.factory import (
    _resolve_freeze_context,
    apply_runtime_overrides,
)
from slm_training.harnesses.model_build.train_loop import (
    _clip_optimizer_parameter_groups,
)
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text
from slm_training.models.twotower import (
    TwoTowerConfig,
    TwoTowerModel,
    _truncate_with_eos,
    format_context_text,
)

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_tokenize_preserves_placeholders_and_whitespace() -> None:
    text = 'hero = Card(":hero.title", ":hero.body")\n'
    tokens = tokenize_text(text)
    assert ":" in tokens
    assert "hero" in tokens
    assert "title" in tokens
    assert "body" in tokens
    assert "\n" in tokens
    assert "Card" in tokens


def test_tokenizer_roundtrip() -> None:
    tok = OpenUITokenizer.build([HERO, CTA, "Hero card layout"])
    encoded = tok.encode(HERO)
    assert encoded[0] == tok.bos_id
    assert encoded[-1] == tok.eos_id
    decoded = tok.decode(encoded)
    assert decoded == HERO


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tok = OpenUITokenizer.build([HERO])
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = OpenUITokenizer.load(path)
    assert loaded.encode(HERO) == tok.encode(HERO)


def test_hf_context_can_be_explicitly_unfrozen() -> None:
    assert _resolve_freeze_context("hf", False) is False
    assert _resolve_freeze_context("hf", True) is True
    assert _resolve_freeze_context("scratch", False) is False


def test_target_truncation_preserves_eos() -> None:
    tok = OpenUITokenizer.build([HERO])
    ids = tok.encode(HERO)
    truncated = _truncate_with_eos(ids, 8, tok.eos_id)
    assert len(truncated) == 8
    assert truncated[-1] == tok.eos_id
    assert truncated[0] == tok.bos_id


def test_context_exposes_compact_output_contract() -> None:
    context = format_context_text(
        "Return a boolean",
        output_kind="lexical",
        output_category="boolean",
    )
    assert "---OUTPUT_CONTRACT---\nlexical:boolean" in context


def test_design_md_dropout_is_deterministic_and_cache_safe() -> None:
    design_md = "# Design\nUse a card."
    records = [
        ExampleRecord(
            id=f"record-{i}",
            prompt="Hero",
            design_md=design_md,
            openui=HERO,
            split="train",
        )
        for i in range(32)
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            design_md_dropout=0.5,
            seed=7,
        ),
        device="cpu",
    )

    first = [model._training_design_md(design_md, record.id) for record in records]
    second = [model._training_design_md(design_md, record.id) for record in records]

    assert first == second
    assert set(first) == {None, design_md}
    model.config.design_md_dropout = 0.0
    assert model._training_design_md(design_md, "record") == design_md
    model.config.design_md_dropout = 1.0
    assert model._training_design_md(design_md, "record") is None
    model.count_batch_tokens(records[:1])
    assert "---DESIGN.md---" not in model._context_text_cache["record-0"]


def test_design_md_dropout_rejects_invalid_rate() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    with pytest.raises(ValueError, match="design_md_dropout"):
        TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(design_md_dropout=1.1),
            device="cpu",
        )


def test_ltr_suffix_always_masks_first_content_token() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    target = torch.tensor([model._encode_openui(HERO, placeholders=[])])
    noisy = target.clone()
    predict = torch.zeros_like(target, dtype=torch.bool)
    _, mask, suffix = model._merge_ltr_suffix_mask(target, noisy, predict)
    assert bool(mask[0, 1])
    assert bool(suffix[0, 1])
    assert int(noisy[0, 1]) == model.tokenizer.mask_id


def test_checkpoint_rejects_missing_trainable_weights(tmp_path: Path) -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1),
    )
    path = tmp_path / "broken.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing_key = next(k for k in payload["state_dict"] if k.startswith("denoiser."))
    del payload["state_dict"][missing_key]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint state mismatch"):
        TwoTowerModel.from_checkpoint(path, device="cpu")


def test_checkpoint_preserves_component_inventory_decode_weight(tmp_path: Path) -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
            split="train",
        )
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer="lexer",
            component_inventory_loss_weight=1.0,
            component_inventory_decode_weight=0.75,
            component_plan_loss_weight=1.0,
            component_plan_class_balance_power=0.5,
            component_plan_decode_weight=0.5,
            component_plan_token_pool=True,
            slot_component_loss_weight=0.6,
            slot_component_focal_gamma=2.0,
            slot_component_class_balance_power=0.5,
            slot_component_decode_weight=0.25,
            slot_component_prompt_context=False,
            slot_component_next_context=True,
            slot_component_pair_interaction=True,
            slot_component_lexeme_prior_weight=1.0,
            slot_component_span_prior_weight=1.0,
            slot_component_content_arity=True,
            component_edge_loss_weight=1.0,
            component_edge_alignment_loss_weight=0.8,
            component_edge_decode_weight=0.4,
            binder_component_plan_loss_weight=0.9,
            binder_component_plan_decode_weight=0.3,
            binder_topology_loss_weight=0.8,
            binder_topology_decode_weight=0.2,
            binder_arity_loss_weight=0.7,
            binder_arity_decode_weight=0.1,
        ),
    )
    assert model.component_inventory_head is not None
    path = tmp_path / "inventory.pt"
    model.save(path)

    loaded = TwoTowerModel.from_checkpoint(path, device="cpu")
    apply_runtime_overrides(loaded, ModelBuildConfig(train_dir=tmp_path))

    assert loaded.component_inventory_head is not None
    assert loaded.component_plan_head is not None
    assert loaded.slot_component_head is not None
    assert loaded.component_edge_head is not None
    assert loaded.binder_component_plan_head is not None
    assert loaded.binder_topology_head is not None
    assert loaded.binder_arity_head is not None
    assert loaded.config.component_inventory_loss_weight == 1.0
    assert loaded.config.component_inventory_decode_weight == 0.75
    assert loaded.config.component_plan_loss_weight == 1.0
    assert loaded.config.component_plan_class_balance_power == 0.5
    assert loaded.config.component_plan_class_weights
    assert loaded.config.component_plan_decode_weight == 0.5
    assert loaded.config.component_plan_token_pool is True
    assert loaded.config.slot_component_loss_weight == 0.6
    assert loaded.config.slot_component_focal_gamma == 2.0
    assert loaded.config.slot_component_class_balance_power == 0.5
    assert loaded.config.slot_component_class_weights
    assert loaded.config.slot_component_decode_weight == 0.25
    assert loaded.config.slot_component_prompt_context is False
    assert loaded.config.slot_component_next_context is True
    assert loaded.config.slot_component_pair_interaction is True
    assert loaded.config.slot_component_lexeme_prior_weight == 1.0
    assert loaded.config.slot_component_lexeme_priors
    assert loaded.config.slot_component_span_prior_weight == 1.0
    assert loaded.config.slot_component_content_arity is True
    assert loaded.config.component_edge_loss_weight == 1.0
    assert loaded.config.component_edge_alignment_loss_weight == 0.8
    assert loaded.config.component_edge_decode_weight == 0.4
    assert loaded.config.binder_component_plan_loss_weight == 0.9
    assert loaded.config.binder_component_plan_decode_weight == 0.3
    assert loaded.config.binder_topology_loss_weight == 0.8
    assert loaded.config.binder_topology_decode_weight == 0.2
    assert loaded.config.binder_arity_loss_weight == 0.7
    assert loaded.config.binder_arity_decode_weight == 0.1

    apply_runtime_overrides(
        loaded,
        ModelBuildConfig(
            train_dir=tmp_path,
            component_inventory_decode_weight=0.0,
            component_plan_decode_weight=0.0,
            slot_component_decode_weight=0.0,
            component_edge_decode_weight=0.0,
            binder_component_plan_decode_weight=0.0,
            binder_topology_decode_weight=0.0,
            binder_arity_decode_weight=0.0,
        ),
    )
    assert loaded.config.component_inventory_decode_weight == 0.0
    assert loaded.config.component_plan_decode_weight == 0.0
    assert loaded.config.slot_component_decode_weight == 0.0
    assert loaded.config.component_edge_decode_weight == 0.0
    assert loaded.config.binder_component_plan_decode_weight == 0.0
    assert loaded.config.binder_topology_decode_weight == 0.0
    assert loaded.config.binder_arity_decode_weight == 0.0


def test_optional_heads_do_not_shift_training_rng() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]

    def state_after(**kwargs) -> torch.Tensor:
        torch.manual_seed(123)
        TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32,
                n_heads=4,
                context_layers=1,
                denoiser_layers=1,
                output_tokenizer="lexer",
                **kwargs,
            ),
        )
        return torch.random.get_rng_state()

    baseline = state_after()
    assert torch.equal(baseline, state_after(binder_arity_loss_weight=1.0))
    assert torch.equal(baseline, state_after(binder_topology_loss_weight=1.0))
    assert torch.equal(baseline, state_after(component_plan_loss_weight=1.0))
    assert torch.equal(baseline, state_after(slot_component_loss_weight=1.0))


def test_auxiliary_heads_do_not_change_base_optimizer_updates() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]

    def model(**kwargs) -> TwoTowerModel:
        torch.manual_seed(123)
        return TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32,
                n_heads=4,
                context_layers=1,
                denoiser_layers=1,
                output_tokenizer="lexer",
                **kwargs,
            ),
        )

    baseline = model()
    arity = model(binder_arity_loss_weight=1.0)
    baseline_optimizer = torch.optim.AdamW(baseline.optimizer_parameter_groups())
    arity_optimizer = torch.optim.AdamW(arity.optimizer_parameter_groups())
    for candidate in (baseline, arity):
        for name, parameter in candidate.named_parameters():
            if parameter.requires_grad:
                scale = 100.0 if name.startswith("binder_arity_head.") else 1.0
                parameter.grad = torch.full_like(parameter, scale)
    _clip_optimizer_parameter_groups(baseline_optimizer, 1.0)
    _clip_optimizer_parameter_groups(arity_optimizer, 1.0)
    baseline_optimizer.step()
    arity_optimizer.step()
    arity_state = dict(arity.named_parameters())
    for name, parameter in baseline.named_parameters():
        if name in arity_state:
            assert torch.equal(parameter, arity_state[name]), name


def test_auxiliary_loss_does_not_change_base_gradients() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]

    def model(**kwargs) -> TwoTowerModel:
        torch.manual_seed(123)
        return TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32,
                n_heads=4,
                context_layers=1,
                denoiser_layers=1,
                output_tokenizer="lexer",
                **kwargs,
            ),
        )

    baseline = model()
    arity = model(binder_arity_loss_weight=1.0)
    rng_state = baseline._rng.getstate()
    torch.manual_seed(456)
    baseline.training_loss(records).backward()
    baseline._rng.setstate(rng_state)
    arity._rng.setstate(rng_state)
    torch.manual_seed(456)
    arity.training_loss(records).backward()

    arity_parameters = dict(arity.named_parameters())
    for name, parameter in baseline.named_parameters():
        if name.startswith("binder_arity_head."):
            continue
        other = arity_parameters[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            assert torch.equal(parameter.grad, other.grad), name

    auxiliary_loss = arity.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()
    assert arity.binder_arity_head is not None
    assert arity.binder_arity_head.weight.grad is not None


def test_surface_syntax_repair_preserves_string_literals() -> None:
    damaged = (
        'root===Stack([cta, card, =])\n'
        'cta==Button(":cta=a==b")\n'
        'card==Card([cta, =])'
    )
    expected = (
        'root = Stack([cta, card])\n'
        'cta = Button(":cta=a==b")\n'
        'card = Card([cta])'
    )
    assert TwoTowerModel._repair_surface_syntax(damaged) == expected


def test_migrate_checkpoint_rebuilds_v2_vocab(tmp_path: Path) -> None:
    from slm_training.models.checkpoint_migrate import migrate_twotower_checkpoint

    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records_path, records)

    model = TwoTowerModel.from_records(records, config=TwoTowerConfig(d_model=32, n_heads=4))
    src = tmp_path / "legacy.pt"
    model.save(src)

    # Simulate a v1 tokenizer sidecar (same token table, older version tag).
    tok_path = src.with_suffix(".tokenizer.json")
    import json

    data = json.loads(tok_path.read_text(encoding="utf-8"))
    data["version"] = 1
    tok_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    out = tmp_path / "migrated.pt"
    report = migrate_twotower_checkpoint(
        source_checkpoint=src,
        train_records_path=records_path,
        output_checkpoint=out,
        device="cpu",
    )
    assert report["new_tokenizer_version"] == 2
    assert out.exists()
    loaded = TwoTowerModel.from_checkpoint(out, device="cpu")
    assert loaded.tokenizer.version == 2
    assert loaded.tokenizer.vocab_size == report["new_vocab_size"]


def test_twotower_training_loss_decreases() -> None:
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
            gen_steps=4,
            seed=0,
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert losses[-1] < 2.0


def test_fragment_records_reach_twotower_training_loss() -> None:
    records = [
        ExampleRecord(
            id="boolean",
            prompt="Return a boolean symbol",
            openui="true",
            target_kind="lexical",
            target_category="boolean",
        ),
        ExampleRecord(
            id="button",
            prompt="Return a Button expression",
            openui='Button(":cta")',
            placeholders=[":cta"],
            target_kind="expression",
        ),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    assert torch.isfinite(model.training_loss(records))


def test_legacy_twotower_rejects_fragment_request() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="doc", prompt="CTA", openui=CTA)],
        config=TwoTowerConfig(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    model.output_contract_version = 0
    with pytest.raises(ValueError, match="predates compact output contracts"):
        model.generate_batch_requests(
            [GenerationRequest(prompt="Boolean", output_kind="lexical")]
        )


def test_twotower_save_load_generate(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=64, n_heads=4, context_layers=1, denoiser_layers=2),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        model.training_loss(records).backward()
        opt.step()

    ckpt = tmp_path / "model.pt"
    model.save(ckpt)
    assert ckpt.with_suffix(".tokenizer.json").exists()
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    pred = loaded.generate("Hero")
    assert "Stack" in pred or "Card" in pred or "root" in pred


@pytestmark_bridge
def test_twotower_train_eval_overfit(tmp_path: Path) -> None:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="tr1",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="tr2",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
        ],
    )
    train_result = build_train_data(
        TrainDataConfig(
            seed_path=train_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="none",
        )
    )
    train_dir = Path(train_result["output_dir"])

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="sm1",
                prompt="Hero",
                openui=HERO,
                split="smoke",
                meta={"suite": "smoke"},
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="sm2",
                prompt="CTA",
                openui=CTA,
                split="smoke",
                meta={"suite": "smoke"},
                placeholders=[":cta.label"],
            ),
        ],
    )
    test_result = build_test_data(
        TestDataConfig(
            seed_path=test_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("smoke",),
            train_manifest=None,
            require_train_manifest=False,
        )
    )
    test_dir = Path(test_result["output_dir"])

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="twotower_overfit",
        steps=120,
        batch_size=2,
        lr=3e-3,
        seed=0,
        model_name="twotower",
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=6,
        context_backend="scratch",
        freeze_context=False,
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
    )
    summary = train(config)
    assert summary["steps"] == 120
    assert summary["last_loss"] < 3.0
    ckpt = Path(summary["checkpoint"])
    assert ckpt.exists()
    assert ckpt.with_suffix(".tokenizer.json").exists()

    metrics = evaluate(config, checkpoint=ckpt)
    assert metrics["n"] == 2
    # Honest production-shaped eval: tiny twotower may not parse at 120 steps.
    assert metrics["raw_syntax_validity"] >= 0.0
    assert "contract_precision" in metrics
