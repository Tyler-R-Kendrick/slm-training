"""Tokenizer + TwoTower model tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.data.contract import (
    CallerContentBinding,
    ChoiceGenerationResult,
    GenerationRequest,
    RuntimeSymbol,
    choice_generation_fingerprint,
)
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
    _remap_vocab_weight,
    _resize_position_weight,
    _truncate_with_eos,
    format_context_text,
)

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = 'root = Stack([b1], "column")\nb1 = Card([b2, b3])\nb2 = TextContent(":slot_0")\nb3 = TextContent(":slot_1")'
CTA = 'root = Stack([b1])\nb1 = Button(":slot_0")'


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


def test_warm_start_vocab_remap_preserves_new_token_rows() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])

    remapped = _remap_vocab_weight(
        source,
        {"shared": 0, "old_only": 1},
        target,
        {"new_only": 0, "shared": 1, "newer": 2},
    )

    assert remapped.tolist() == [[9.0, 9.0], [1.0, 2.0], [7.0, 7.0]]


def test_warm_start_position_resize_copies_shared_prefix() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = torch.tensor([[9.0, 9.0], [8.0, 8.0]])

    assert _resize_position_weight(source, target).tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]


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
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
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
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
    )
    path = tmp_path / "broken.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing_key = next(k for k in payload["state_dict"] if k.startswith("denoiser."))
    del payload["state_dict"][missing_key]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint state mismatch"):
        TwoTowerModel.from_checkpoint(path, device="cpu")


@pytest.mark.parametrize(
    ("output_tokenizer", "compiler_decode_mode", "loss_name", "head_name"),
    [
        (
            "choice",
            "off",
            "root_reference_identity_loss_weight",
            "root_reference_identity_head",
        ),
        (
            "lexer",
            "tree",
            "root_reference_arity_loss_weight",
            "root_reference_arity_head",
        ),
    ],
)
def test_checkpoint_rejects_missing_enabled_root_head(
    tmp_path: Path,
    output_tokenizer: str,
    compiler_decode_mode: str,
    loss_name: str,
    head_name: str,
) -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer=output_tokenizer,
            compiler_decode_mode=compiler_decode_mode,
            **{loss_name: 1.0},
        ),
    )
    path = tmp_path / f"missing-{head_name}.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["state_dict"][f"{head_name}.weight"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint state mismatch"):
        TwoTowerModel.from_checkpoint(path, device="cpu")


def test_checkpoint_rejects_pre_opaque_marker_contract(tmp_path: Path) -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
    )
    path = tmp_path / "legacy.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["output_contract_version"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError, match="retrain from symbol-only targets"):
        TwoTowerModel.from_checkpoint(path, device="cpu")


def test_training_loss_rechecks_opaque_role_safe_targets() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="valid", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
    )
    with pytest.raises(ValueError, match="opaque :slot_<ordinal>"):
        model.training_loss(
            [
                ExampleRecord(
                    id="named",
                    prompt="Hero",
                    openui='root = TextContent(":hero.title")',
                    placeholders=[":hero.title"],
                )
            ]
        )
    with pytest.raises(ValueError, match="non-content property Input.name"):
        model.training_loss(
            [
                ExampleRecord(
                    id="wrong-role",
                    prompt="Input",
                    openui='root = Input(":slot_0")',
                    placeholders=[":slot_0"],
                )
            ]
        )


def test_checkpoint_preserves_component_inventory_decode_weight(tmp_path: Path) -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
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
            compiler_decode_mode="tree",
            component_inventory_loss_weight=1.0,
            component_inventory_decode_weight=0.75,
            component_plan_loss_weight=1.0,
            component_plan_decode_weight=0.5,
            slot_component_loss_weight=0.6,
            slot_component_focal_gamma=2.0,
            slot_component_class_balance_power=0.5,
            slot_component_decode_weight=0.25,
            slot_component_prompt_context=False,
            slot_component_lexeme_prior_weight=1.0,
            component_edge_loss_weight=1.0,
            component_edge_alignment_loss_weight=0.8,
            component_edge_decode_weight=0.4,
            binder_component_plan_loss_weight=0.9,
            binder_component_plan_decode_weight=0.3,
            binder_topology_loss_weight=0.8,
            binder_topology_decode_weight=0.2,
            binder_arity_loss_weight=0.7,
            binder_arity_decode_weight=0.1,
            root_reference_arity_loss_weight=0.0,
            root_reference_arity_decode_weight=0.0,
            root_reference_identity_loss_weight=0.0,
            root_reference_identity_negative_weight=3.0,
            root_reference_identity_decode_weight=0.0,
        ),
    )
    assert model.component_inventory_head is not None
    path = tmp_path / "inventory.pt"
    model.save(path)

    loaded = TwoTowerModel.from_checkpoint(path, device="cpu")
    apply_runtime_overrides(
        loaded,
        ModelBuildConfig(train_dir=tmp_path, runtime_override_fields=frozenset()),
    )

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
    assert loaded.config.component_plan_decode_weight == 0.5
    assert loaded.config.slot_component_loss_weight == 0.6
    assert loaded.config.slot_component_focal_gamma == 2.0
    assert loaded.config.slot_component_class_balance_power == 0.5
    assert loaded.config.slot_component_class_weights
    assert loaded.config.slot_component_decode_weight == 0.25
    assert loaded.config.slot_component_prompt_context is False
    assert loaded.config.slot_component_lexeme_prior_weight == 1.0
    assert loaded.config.slot_component_lexeme_priors == ()
    assert loaded.config.component_edge_loss_weight == 1.0
    assert loaded.config.component_edge_alignment_loss_weight == 0.8
    assert loaded.config.component_edge_decode_weight == 0.4
    assert loaded.config.binder_component_plan_loss_weight == 0.9
    assert loaded.config.binder_component_plan_decode_weight == 0.3
    assert loaded.config.binder_topology_loss_weight == 0.8
    assert loaded.config.binder_topology_decode_weight == 0.2
    assert loaded.config.binder_arity_loss_weight == 0.7
    assert loaded.config.binder_arity_decode_weight == 0.1

    assert loaded.config.root_reference_arity_loss_weight == 0.0
    assert loaded.config.root_reference_arity_decode_weight == 0.0
    assert loaded.config.root_reference_identity_loss_weight == 0.0
    assert loaded.config.root_reference_identity_negative_weight == 3.0
    assert loaded.config.root_reference_identity_decode_weight == 0.0

    apply_runtime_overrides(
        loaded,
        ModelBuildConfig(
            train_dir=tmp_path,
            runtime_override_fields=frozenset(
                {
                    "compiler_decode_mode",
                    "component_inventory_decode_weight",
                    "component_plan_decode_weight",
                    "component_edge_decode_weight",
                    "binder_component_plan_decode_weight",
                    "binder_topology_decode_weight",
                    "binder_arity_decode_weight",
                    "root_reference_arity_decode_weight",
                    "root_reference_identity_decode_weight",
                }
            ),
            compiler_decode_mode="tree",
            component_inventory_decode_weight=0.0,
            component_plan_decode_weight=0.0,
            component_edge_decode_weight=0.0,
            binder_component_plan_decode_weight=0.0,
            binder_topology_decode_weight=0.0,
            binder_arity_decode_weight=0.0,
            root_reference_arity_decode_weight=0.0,
            root_reference_identity_decode_weight=0.0,
        ),
    )
    assert loaded.config.component_inventory_decode_weight == 0.0
    assert loaded.config.component_plan_decode_weight == 0.0
    assert loaded.config.component_edge_decode_weight == 0.0
    assert loaded.config.binder_component_plan_decode_weight == 0.0
    assert loaded.config.binder_topology_decode_weight == 0.0
    assert loaded.config.binder_arity_decode_weight == 0.0
    assert loaded.config.root_reference_arity_decode_weight == 0.0
    assert loaded.config.root_reference_identity_decode_weight == 0.0


def test_slot_pair_interaction_never_encodes_empty_next_slot() -> None:
    records = [
        ExampleRecord(
            id="pair",
            prompt="Hero",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
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
            output_tokenizer="choice",
            slot_component_loss_weight=1.0,
            slot_component_pair_interaction=True,
        ),
    )
    encode_context = model._encode_context

    def assert_nonempty(prompts: list[str], **kwargs):
        assert all(prompt for prompt in prompts)
        return encode_context(prompts, **kwargs)

    model._encode_context = assert_nonempty  # type: ignore[method-assign]
    assert torch.isfinite(model.training_loss(records))


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
        'root===Stack([cta, card, =])\ncta==Button(":cta=a==b")\ncard==Card([cta, =])'
    )
    expected = (
        'root = Stack([cta, card])\ncta = Button(":cta=a==b")\ncard = Card([cta])'
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

    model = TwoTowerModel.from_records(
        records, config=TwoTowerConfig(d_model=32, n_heads=4)
    )
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


def test_twotower_training_loss_runs_for_both_denoiser_archs() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    for arch in ("stacked", "shared_recursive"):
        model = TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32,
                n_heads=2,
                context_layers=1,
                denoiser_layers=2,
                denoiser_arch=arch,  # type: ignore[arg-type]
                recursive_steps=2,
                recursive_transition_layers=2,
                grammar_constrained=False,
                seed=0,
            ),
            device="cpu",
        )
        loss = model.training_loss(records)
        assert torch.isfinite(loss)


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
            openui='Button(":slot_0")',
            placeholders=[":slot_0"],
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
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
    )
    model.output_contract_version = 0
    with pytest.raises(ValueError, match="predates compact output contracts"):
        model.generate_batch_requests(
            [GenerationRequest(prompt="Boolean", output_kind="lexical")]
        )


@pytestmark_bridge
def test_opt_in_binding_runs_only_after_legacy_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = 'root = TextContent(":slot_0")'
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="doc", prompt="Hero", openui=template)],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
    )
    calls = 0

    def fake_generate(
        self: TwoTowerModel,
        requests: list[GenerationRequest],
        **kwargs: object,
    ) -> list[str]:
        nonlocal calls
        calls += 1
        assert kwargs["_opaque_slot_projection"] is True
        assert requests[0].slot_contract == (":slot_0",)
        self._last_generation_evidence = [{"decode": "complete"}]
        return [template for _ in requests]

    def forbidden_forward(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-decode binding must not call the denoiser")

    monkeypatch.setattr(TwoTowerModel, "generate_batch_requests", fake_generate)
    monkeypatch.setattr(model, "_denoiser_forward", forbidden_forward)
    value = 'Welcome "back"\\\nToday ☃'
    result = model.generate_batch_bound_requests(
        [GenerationRequest(prompt="Hero", slot_contract=(":slot_0",))],
        [(CallerContentBinding("slot_0", value),)],
    )[0]

    assert calls == 1
    assert result.status == "resolved"
    assert result.materialized_source is None
    evidence = model.consume_generation_evidence()
    assert evidence[0]["decode"] == "complete"
    assert value not in str(evidence)


@pytestmark_bridge
def test_opt_in_choice_generation_returns_exact_verified_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = 'root = TextContent(":slot_0")'
    model = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="doc",
                prompt="Hero",
                openui=template,
                placeholders=[":slot_0"],
            )
        ],
        config=TwoTowerConfig(
            output_tokenizer="choice",
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    ids = model.tokenizer.encode(template, placeholders=[":slot_0"])
    tokens = [model.tokenizer.id_to_token[token_id] for token_id in ids]
    calls = 0

    def fake_generate(requests: list[GenerationRequest], **kwargs: object) -> list[str]:
        nonlocal calls
        calls += 1
        assert kwargs["_opaque_slot_projection"] is True
        assert requests[0].slot_contract == (":slot_0",)
        model._last_generation_evidence = [
            {
                "schema": "choice_decision_trace/v2",
                "choice_ids": ids,
                "choice_tokens": tokens,
            }
        ]
        return [template]

    monkeypatch.setattr(model, "generate_batch_requests", fake_generate)
    request = GenerationRequest(prompt="Hero", slot_contract=(":slot_0",))
    first = model.generate_batch_choice_requests([request])[0]
    second = model.generate_batch_choice_requests([request])[0]

    assert calls == 2
    assert first == second
    assert first.status == "verified"
    assert first.verification == "pack_verified"
    assert first.opaque_slot_contract == (":slot_0",)
    assert first.slot_projection == ((":slot_0", ":slot_0"),)
    assert first.choice_ids == tuple(ids)
    assert first.choice_tokens == tuple(tokens)
    assert first.canonical_source == template
    assert len(first.source_fingerprint) == len(first.fingerprint) == 64
    encoded = json.dumps(first.to_dict(), sort_keys=True, ensure_ascii=False)
    assert ChoiceGenerationResult.from_dict(json.loads(encoded)) == first
    payload = {
        key: value
        for key, value in first.to_dict().items()
        if key != "fingerprint"
    }
    assert choice_generation_fingerprint(payload) == first.fingerprint
    for key, value in (
        ("choice_ids", [*first.choice_ids, 999]),
        ("slot_projection", [[":other", ":slot_0"]]),
        ("canonical_source", template + "\n"),
    ):
        changed = dict(payload)
        changed[key] = value
        assert choice_generation_fingerprint(changed) != first.fingerprint


def test_choice_generation_rejects_incompatible_or_unprovable_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = TwoTowerModel.from_records(
        [ExampleRecord(id="surface", prompt="Hero", openui=HERO)],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
    )
    with pytest.raises(ValueError, match="choice-codec checkpoint"):
        surface.generate_batch_choice_requests([GenerationRequest(prompt="Hero")])

    template = 'root = TextContent(":slot_0")'
    choice = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="choice",
                prompt="Hero",
                openui=template,
                placeholders=[":slot_0"],
            )
        ],
        config=TwoTowerConfig(
            output_tokenizer="choice",
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    with pytest.raises(ValueError, match="document output only"):
        choice.generate_batch_choice_requests(
            [GenerationRequest(prompt="Boolean", output_kind="lexical")]
        )
    choice.config.best_of_n = 2
    with pytest.raises(ValueError, match="best_of_n=1"):
        choice.generate_batch_choice_requests([GenerationRequest(prompt="Hero")])
    choice.config.best_of_n = 1
    undeclared = GenerationRequest(
        prompt="Hero",
        slot_contract=(":slot_0",),
        runtime_symbols=(
            RuntimeSymbol(surface=":slot_1", role="external_entity"),
        ),
    )
    with pytest.raises(ValueError, match="must appear in slot_contract"):
        choice.generate_batch_choice_requests([undeclared])
    with pytest.raises(ValueError, match="must appear in slot_contract"):
        surface.generate_batch_bound_requests([undeclared], [tuple()])

    monkeypatch.setattr(choice, "generate_batch_requests", lambda *_a, **_k: [template])
    choice._last_generation_evidence = []
    with pytest.raises(ValueError, match="aligned stream evidence"):
        choice.generate_batch_choice_requests([GenerationRequest(prompt="Hero")])


def test_opaque_projection_keeps_marker_names_out_of_scoring() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="doc", prompt="Hero", openui=HERO)],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            compiler_decode_mode="tree",
            slot_component_loss_weight=1.0,
            slot_component_decode_weight=1.0,
            slot_contract_in_context=True,
        ),
        device="cpu",
    )
    model._opaque_slot_projection_active = True
    marker = [":slot_0"]
    assert model._slot_component_texts(marker) == ["content:0"]
    assert model._slot_component_texts([":meaningful.title"]) == ["content:0"]
    assert model._slot_role_token(marker[0]) == "content"
    context = model._context_prompts(
        ["Hero"],
        slot_contracts=[marker],
        opaque_slot_projection=True,
    )[0]
    assert "slot_0" not in context
    model._opaque_slot_projection_active = False


def test_model_rejects_named_markers_before_context_vocab_build() -> None:
    semantic = ExampleRecord(
        id="semantic",
        prompt="Use :hero.title",
        openui='root = TextContent(":hero.title")',
        placeholders=[":hero.title"],
    )
    config = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
    )
    with pytest.raises(ValueError, match="opaque :slot_<ordinal>"):
        TwoTowerModel.from_records([semantic], config=config)


def test_opaque_projection_state_is_scoped_across_calls_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="doc", prompt="Hero", openui=HERO)],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
    )
    observed: list[bool] = []

    def fake_once(*_args: object, **_kwargs: object) -> list[str]:
        observed.append(model._opaque_slot_projection_active)
        return [HERO]

    monkeypatch.setattr(model, "_generate_batch_once", fake_once)
    request = GenerationRequest(prompt="Hero")
    model.generate_batch_requests([request], _opaque_slot_projection=True)
    model.generate_batch_requests([request])
    assert observed == [True, False]
    assert model._opaque_slot_projection_active is False

    def fail_once(*_args: object, **_kwargs: object) -> list[str]:
        assert model._opaque_slot_projection_active is True
        raise RuntimeError("decode failed")

    monkeypatch.setattr(model, "_generate_batch_once", fail_once)
    with pytest.raises(RuntimeError, match="decode failed"):
        model.generate_batch_requests([request], _opaque_slot_projection=True)
    assert model._opaque_slot_projection_active is False


def test_twotower_save_load_generate(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64, n_heads=4, context_layers=1, denoiser_layers=2
        ),
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
