"""Grammar-native block diffusion model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.data.contract import canonicalize_example_template_markers
from slm_training.dsl.schema import ExampleRecord
from slm_training.data.progspec import ProgramSpec, derive_scope_records
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
    production_sequence_accuracy,
    topology_arity_accuracy,
    topology_from_openui,
)
from slm_training.models.checkpoint_migrate import migrate_grammar_diffusion_checkpoint
from scripts.run_grammar_matrix import _x_experiments, main as grammar_matrix_main
from slm_training.dsl.production_codec import ProductionCodec

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":slot_0")\nhero_body = TextContent(":slot_1")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'


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


def test_context_rejects_named_markers() -> None:
    model = GrammarDiffusionModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=GrammarDiffusionConfig(d_model=64),
        device="cpu",
    )
    with pytest.raises(ValueError, match="opaque :slot_<ordinal>"):
        model._format_context(
            "Use :hero.title then :hero.cta",
            slot_contract=[":hero.title", ":hero.cta"],
        )
    canonical = model._format_context(
        "Use :slot_0 then :slot_1",
        slot_contract=[":slot_0", ":slot_1"],
    )
    assert ":slot_0" in canonical


def test_model_rejects_named_markers_before_context_vocab_build() -> None:
    semantic = ExampleRecord(
        id="semantic",
        prompt="Use :hero.title",
        openui='root = TextContent(":hero.title")',
        placeholders=[":hero.title"],
    )
    with pytest.raises(ValueError, match="opaque :slot_<ordinal>"):
        GrammarDiffusionModel.from_records(
            [semantic], config=GrammarDiffusionConfig(d_model=64)
        )


def test_inline_codec_roundtrip() -> None:
    codec = InlineProductionCodec.build([HERO, CTA])
    inventory = [":slot_0", ":slot_1"]
    prod, slot = codec.encode(HERO, inventory)
    decoded = codec.decode(prod, slot, inventory)
    assert "Stack" in decoded
    assert "Card" in decoded
    assert '":slot_0"' in decoded or ":slot_0" in decoded


def test_production_codec_topology_roundtrip() -> None:
    codec = ProductionCodec.build([HERO, CTA])
    inventory = [":slot_0", ":slot_1"]
    expected_prod, expected_slot = codec.encode(HERO, inventory, max_len=0)
    topology = topology_from_openui(codec, HERO, inventory)
    actual_prod, actual_slot = _serialize_topology(codec, topology)
    assert actual_prod == expected_prod
    assert actual_slot == expected_slot


@pytest.mark.parametrize(
    ("source", "kind"),
    [("true", "lexical"), ("x = true", "statement"), ('Button(":x")', "expression")],
)
def test_production_codec_fragment_topology_roundtrip(source: str, kind: str) -> None:
    codec = ProductionCodec.build([source], [kind])
    inventory = [":x"]
    expected_prod, expected_slot = codec.encode(
        source, inventory, max_len=0, output_kind=kind
    )
    topology = topology_from_openui(
        codec, source, inventory, output_kind=kind
    )
    actual_prod, actual_slot = _serialize_topology(codec, topology)
    assert actual_prod == expected_prod
    assert actual_slot == expected_slot
    assert codec.decode(actual_prod, actual_slot, inventory).strip() == source


def test_fragment_records_reach_grammar_diffusion_training_loss() -> None:
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
    loss = model.training_loss(records)
    assert torch.isfinite(loss)


def test_legacy_grammar_diffusion_rejects_fragment_request() -> None:
    model = GrammarDiffusionModel.from_records(
        [ExampleRecord(id="doc", prompt="CTA", openui=CTA)],
        config=GrammarDiffusionConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    model.output_contract_version = 0
    with pytest.raises(ValueError, match="predates compact output contracts"):
        model.generate_batch_requests(
            [GenerationRequest(prompt="Boolean", output_kind="lexical")]
        )


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
            placeholders=[":slot_0", ":slot_1"],
        ),
        ExampleRecord(
            id="b", prompt="CTA", openui=CTA, split="train", placeholders=[":slot_0"]
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
            placeholders=[":slot_0", ":slot_1"],
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
    req = GenerationRequest(prompt="Hero", slot_contract=(":slot_0", ":slot_1"))
    out = loaded.generate_batch_requests([req])[0]
    assert isinstance(out, str)
    evidence = loaded.consume_generation_evidence()
    assert len(evidence) == 1
    assert evidence[0]["phases"] > 0
    # Short scratch training may remain invalid; topology decode fails closed.
    assert out or evidence[0]["candidate_productions"]


def test_checkpoint_rejects_pre_opaque_marker_contract(tmp_path: Path) -> None:
    model = GrammarDiffusionModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=GrammarDiffusionConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
    )
    path = tmp_path / "legacy.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["output_contract_version"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError, match="retrain from symbol-only targets"):
        GrammarDiffusionModel.from_checkpoint(path, device="cpu")


def test_training_loss_rechecks_opaque_role_safe_targets() -> None:
    model = GrammarDiffusionModel.from_records(
        [ExampleRecord(id="valid", prompt="Hero", openui=HERO, split="train")],
        config=GrammarDiffusionConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1
        ),
        device="cpu",
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


def test_scope_contract_heads_train_only_when_enabled() -> None:
    spec = ProgramSpec.from_openui(
        id="scope",
        openui=CTA,
        facts={},
        program_family_id="scope-family",
        lineage_id="scope-lineage",
        split_group_id="scope-group",
    )
    record = canonicalize_example_template_markers(
        next(
            row
            for row in derive_scope_records(spec)
            if row.meta["scope_family"] == "local_valid_global_invalid"
        )
    )
    model = GrammarDiffusionModel.from_records(
        [record],
        config=GrammarDiffusionConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            scope_contracts=True,
            scope_independent_noise=True,
            scope_local_oracle=True,
            scope_contract_negatives=True,
        ),
        device="cpu",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    assert model.denoiser.scope_summary_head is not None
    assert model.last_training_metrics["scope_summary_loss"] >= 0.0
    assert model.last_training_metrics["scope_gate_loss"] > 0.0
    evidence = model.score_topology_targets([record])
    assert evidence[0]["scope_kind"] in {
        "component_call",
        "statement",
        "child_list",
    }
    assert evidence[0]["scope_family"] == "local_valid_global_invalid"
    assert 0.0 <= evidence[0]["scope_gate_accuracy"] <= 1.0
    assert evidence[0]["scope_summary_definitions_mae"] >= 0.0
    assert evidence[0]["failure_cone_target_size"] >= 0


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
        openui='root = TextContent(":slot_0")',
        split="held_out",
        placeholders=[":slot_0"],
    )

    evidence = model.score_topology_targets([held_out])

    assert len(evidence) == 1
    assert evidence[0]["production_oov_rate"] > 0.0
    assert model.codec.production_to_id == vocab_before

    production_sequence_accuracy(model.codec, held_out.openui, CTA)
    topology_arity_accuracy(model.codec, held_out.openui, CTA)
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


def test_topology_matrix_rows_replace_fixed_canvas_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiments = _x_experiments(tmp_path, tmp_path, design_md_in_context=False)
    ids = {experiment.xid for experiment in experiments}
    assert set(f"X{index}" for index in range(9, 22)) <= ids
    assert not {"X2", "X3", "X4", "X5", "X7", "X8"} & ids
    with pytest.raises(ValueError, match="frozen fixed-canvas"):
        grammar_matrix_main(["--only", "X2"])
    assert grammar_matrix_main(["--only", "X16,X21", "--describe"]) == 0
    described = capsys.readouterr().out
    assert '"xid": "X16"' in described
    assert '"xid": "X21"' in described
    assert not list(tmp_path.iterdir())


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
    req = GenerationRequest(prompt="Hero", slot_contract=(":slot_0",))
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
