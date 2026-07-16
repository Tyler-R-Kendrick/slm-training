"""Grammar-native block diffusion model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.block_noise import (
    BlockNoiseSchedule,
    corrupt_blocks_for_training,
    num_blocks,
    select_blocks_to_unmask,
    unmask_budget,
)
from slm_training.models.constrained_posterior import ExtendabilityChecker
from slm_training.models.grammar_diffusion import (
    GrammarDiffusionConfig,
    GrammarDiffusionModel,
    InlineProductionCodec,
    TopologyNode,
    _refresh_layout,
    _serialize_topology,
    topology_from_openui,
)
from slm_training.models.checkpoint_migrate import migrate_grammar_diffusion_checkpoint
from scripts.run_grammar_matrix import _x_experiments, main as grammar_matrix_main
from slm_training.dsl.production_codec import ProductionCodec

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_from_records_uses_production_codec() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(d_model=64),
        device="cpu",
    )
    assert isinstance(model.codec, ProductionCodec)


def test_inline_codec_roundtrip() -> None:
    codec = InlineProductionCodec.build([HERO, CTA])
    inventory = [":hero.title", ":hero.body"]
    prod, slot = codec.encode(HERO, inventory)
    decoded = codec.decode(prod, slot, inventory)
    assert "Stack" in decoded
    assert "Card" in decoded
    assert '":hero.title"' in decoded or ":hero.title" in decoded


def test_production_codec_topology_roundtrip() -> None:
    codec = ProductionCodec.build([HERO, CTA])
    inventory = [":hero.title", ":hero.body"]
    expected_prod, expected_slot = codec.encode(HERO, inventory, max_len=0)
    topology = topology_from_openui(codec, HERO, inventory)
    actual_prod, actual_slot = _serialize_topology(codec, topology)
    assert actual_prod == expected_prod
    assert actual_slot == expected_slot


def test_runtime_layout_refresh_preserves_node_ids() -> None:
    root = TopologyNode(40, "document", 1)
    child = TopologyNode(91, "statement", 5)
    grandchild = TopologyNode(123, "expression", 6)
    child.children.append(grandchild)
    root.children.append(child)
    _refresh_layout(root, preserve_ids=True)
    assert [node.node_id for node in (root, child, grandchild)] == [40, 91, 123]
    assert (child.parent_id, grandchild.parent_id) == (40, 91)
    assert (root.depth, child.depth, grandchild.depth) == (0, 1, 2)


def test_block_noise_train_infer_budget_parity() -> None:
    schedule = BlockNoiseSchedule(block_size=4, gen_steps=8)
    target = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    frozen = target.eq(0)
    noisy, mask = corrupt_blocks_for_training(
        target.size(1),
        schedule=schedule,
        mask_id=99,
        pad_id=0,
        frozen=frozen,
        target_ids=target,
    )
    assert mask.any()
    assert num_blocks(target.size(1), schedule.block_size) == 2
    remaining = 2
    assert (
        unmask_budget(remaining_blocks=remaining, step=0, steps=schedule.gen_steps) == 1
    )
    block_conf = torch.tensor([[0.2, 0.9]])
    block_unk = torch.tensor([[True, True]])
    picked = select_blocks_to_unmask(
        block_conf,
        block_unk,
        step=0,
        schedule=schedule,
        mode="topk",
    )
    assert picked


def test_training_loss_decreases() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":hero.title", ":hero.body"],
        ),
        ExampleRecord(
            id="b", prompt="CTA", openui=CTA, split="train", placeholders=[":cta.label"]
        ),
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(
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
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert losses[-1] < 4.0


def test_save_load_and_generate_batch_requests(tmp_path: Path) -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":hero.title", ":hero.body"],
        ),
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(
            d_model=64, n_heads=4, context_layers=1, denoiser_layers=2
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        model.training_loss(records).backward()
        opt.step()

    ckpt = tmp_path / "gd.pt"
    model.save(ckpt)
    loaded = GrammarDiffusionModel.from_checkpoint(ckpt, device="cpu")
    req = GenerationRequest(prompt="Hero", slot_contract=(":hero.title", ":hero.body"))
    out = loaded.generate_batch_requests([req])[0]
    assert isinstance(out, str)
    evidence = loaded.consume_generation_evidence()
    assert len(evidence) == 1
    assert evidence[0]["phases"] > 0
    # Short scratch training may remain invalid; topology decode fails closed.
    assert out or evidence[0]["candidate_productions"]


def test_factory_builds_grammar_diffusion() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    cfg = ModelBuildConfig(
        train_dir=Path("."),
        model_name="grammar_diffusion",
        d_model=64,
        context_backend="scratch",
        freeze_context=False,
    )
    model = build_model(cfg, records)
    assert isinstance(model, GrammarDiffusionModel)
    assert not hasattr(model.denoiser.core, "pos")
    loss = model.forward(records)
    assert loss >= 0.0
    evidence = model.score_topology_targets(records)
    assert len(evidence) == len(records)
    assert all(
        {
            "action_macro_f1",
            "production_head_accuracy",
            "arity_head_accuracy",
            "critic_ece",
        }
        <= row.keys()
        for row in evidence
    )


def test_topology_scoring_maps_unseen_productions_without_mutating_codec() -> None:
    records = [ExampleRecord(id="a", prompt="CTA", openui=CTA, split="train")]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    vocab_before = dict(model.codec.production_to_id)
    held_out = ExampleRecord(
        id="held-out",
        prompt="Novel component",
        openui='root = TextContent(":copy")',
        split="held_out",
        placeholders=[":copy"],
    )

    evidence = model.score_topology_targets([held_out])

    assert len(evidence) == 1
    assert evidence[0]["production_oov_rate"] > 0.0
    assert model.codec.production_to_id == vocab_before


def test_fixed_canvas_checkpoint_requires_explicit_migration(tmp_path: Path) -> None:
    records = [ExampleRecord(id="a", prompt="CTA", openui=CTA, split="train")]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    source = tmp_path / "legacy.pt"
    model.save(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["format_version"] = 1
    torch.save(payload, source)
    with pytest.raises(ValueError, match="migrate_checkpoint"):
        GrammarDiffusionModel.from_checkpoint(source)

    output = tmp_path / "topology.pt"
    report = migrate_grammar_diffusion_checkpoint(
        source_checkpoint=source,
        output_checkpoint=output,
    )
    assert report["warm_start_only"] is True
    assert report["output_format_version"] == 2
    migrated = GrammarDiffusionModel.from_checkpoint(output)
    assert migrated.CHECKPOINT_FORMAT == 2


def test_topology_matrix_rows_replace_fixed_canvas_runtime(tmp_path: Path) -> None:
    experiments = _x_experiments(tmp_path, tmp_path, design_md_in_context=False)
    ids = {experiment.xid for experiment in experiments}
    assert set(f"X{index}" for index in range(9, 16)) <= ids
    assert not {"X2", "X3", "X4", "X5", "X7", "X8"} & ids
    with pytest.raises(ValueError, match="frozen fixed-canvas"):
        grammar_matrix_main(["--only", "X2"])


def test_extendability_checker_permissive_without_bridge() -> None:
    codec = InlineProductionCodec.build([HERO])
    checker = ExtendabilityChecker(require_bridge=False)
    prod, slot = codec.encode(HERO, [":hero.title"])
    assert checker.prefix_extendable(codec, prod, slot, [":hero.title"])


def test_eval_mode_has_no_canned_fallback() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(
            d_model=64,
            eval_mode_no_fallback=True,
        ),
        device="cpu",
    )
    # Untrained model should return raw constrained decode, not a canned program.
    req = GenerationRequest(prompt="Hero", slot_contract=(":hero.title",))
    out = model.generate_batch_requests([req])[0]
    assert "Broken" not in out
    assert "stub.missing" not in out


@pytest.mark.skipif(
    __import__("slm_training.dsl", fromlist=["bridge_available"]).bridge_available()
    is False,
    reason="OpenUI bridge required for meaningful-parse eval",
)
def test_grammar_diffusion_train_eval_overfit(tmp_path: Path) -> None:
    from slm_training.harnesses.model_build import evaluate, train
    from slm_training.harnesses.test_data import TestDataConfig, build_test_data
    from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
    from slm_training.dsl.schema import write_jsonl

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
    config = ModelBuildConfig(
        train_dir=Path(train_result["output_dir"]),
        test_dir=Path(test_result["output_dir"]),
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="grammar_diffusion_overfit",
        steps=200,
        batch_size=2,
        lr=3e-3,
        seed=0,
        model_name="grammar_diffusion",
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=16,
        context_backend="scratch",
        freeze_context=False,
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
    )
    summary = train(config)
    assert summary["steps"] == 200
    assert summary["last_loss"] < 1.0
    metrics = evaluate(config, checkpoint=Path(summary["checkpoint"]))
    assert metrics["n"] == 2
    assert metrics["parse_rate"] >= 0.5
    assert metrics["placeholder_fidelity"] >= 0.5
    assert metrics["contract_precision"] >= 0.5
    assert metrics["fallback_count"] == 0
