"""End-to-end smoke: short train + eval under output_tokenizer=lexer."""

from __future__ import annotations

from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig, train
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    'hero = Card([hero_title, hero_body])'
)


def _write_mini_corpus(root: Path) -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="Create a vertical hero card with a title and body.",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
            split="train",
        ),
        ExampleRecord(
            id="t2",
            prompt="Add a primary call-to-action button under a short text line.",
            openui=(
                'root = Stack([copy, cta], "column")\n'
                'copy = TextContent(":slot_0")\n'
                'cta = Button(":slot_1")'
            ),
            placeholders=[":slot_0", ":slot_1"],
            split="train",
        ),
        ExampleRecord(
            id="t3",
            prompt="Simple text block inside a stack.",
            openui='root = Stack([blurb], "column")\nblurb = TextContent(":slot_0")',
            placeholders=[":slot_0"],
            split="train",
        ),
    ]
    write_jsonl(root / "records.jsonl", records)
    (root / "manifest.json").write_text(
        '{"version":"smoke","n":3}\n', encoding="utf-8"
    )


def test_lexer_from_records_builds_dual_tokenizers(tmp_path: Path) -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="Hero card",
            openui=HERO,
            placeholders=[":slot_0", ":slot_1"],
        )
    ]
    cfg = TwoTowerConfig(
        output_tokenizer="lexer",
        use_symbol_table=True,
        context_backend="scratch",
        grammar_constrained=False,
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        max_prompt_len=64,
        max_target_len=128,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    assert is_dsl_native_tokenizer(model.tokenizer)
    assert model.context_tokenizer is not model.tokenizer
    # Fixed corpus-independent vocabulary incl. 64 reserved <MACRO_i> rows (C3).
    assert model.tokenizer.vocab_size <= 512
    loss = model.training_loss(records)
    assert float(loss.detach()) >= 0.0


def test_surface_identifier_arm_is_prohibited() -> None:
    """Surface identifiers are not an admissible template-marker channel."""
    import pytest

    base = dict(
        output_tokenizer="lexer",
        context_backend="scratch",
        grammar_constrained=False,
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        max_prompt_len=64,
        max_target_len=160,
    )
    with pytest.raises(ValueError, match="symbol_anonymization=False is prohibited"):
        TwoTowerConfig(symbol_anonymization=False, **base)


def test_lexer_train_eval_smoke(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test" / "suites" / "smoke"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    _write_mini_corpus(train_dir)
    # Smoke suite = same records.
    write_jsonl(
        test_dir / "records.jsonl",
        [
            ExampleRecord(
                id="s1",
                prompt="Create a vertical hero card with a title and body.",
                openui=HERO,
                placeholders=[":slot_0", ":slot_1"],
                split="smoke",
            )
        ],
    )
    run_root = tmp_path / "runs"
    cfg = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=tmp_path / "test",
        run_root=run_root,
        run_id="lexer_smoke",
        steps=40,
        batch_size=2,
        device="cpu",
        model_name="twotower",
        context_backend="scratch",
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        grammar_constrained=True,
        grammar_ltr_primary=False,
        grammar_ltr_repair=True,
        grammar_ltr_max_tokens=128,
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
        fidelity_loss_weight=1.0,
        gen_steps=8,
        remask_ratio=0.1,
        design_md_in_context=False,
        output_tokenizer="lexer",
        use_symbol_table=True,
        factorized_embeddings=True,
        mask_pattern="mixed",
        remask_span="statement",
        telemetry=False,
    )
    summary = train(cfg)
    ckpt = Path(summary["checkpoint"])
    assert ckpt.is_file()
    assert ckpt.with_suffix(".tokenizer.json").is_file()
    assert ckpt.with_name(ckpt.stem + ".context.tokenizer.json").is_file()
    raw = ckpt.with_suffix(".tokenizer.json").read_text(encoding="utf-8")
    assert "dsl_native" in raw

    board = evaluate_suites(
        cfg,
        ["smoke"],
        checkpoint=ckpt,
        write_gates=False,
    )
    smoke = board["suites"]["smoke"]
    assert smoke["n"] >= 1
    # Soft gate: training should produce finite metrics (not crash).
    assert "parse_rate" in smoke
    assert "placeholder_fidelity" in smoke
    policy = smoke["evaluation_policy"]
    assert policy["context_backend"] == "scratch"
    assert policy["generate_max_attempts"] == cfg.generate_max_attempts
    assert policy["grammar_verify_chosen_only"] == cfg.grammar_verify_chosen_only
