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
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
)


def _write_mini_corpus(root: Path) -> None:
    records = [
        ExampleRecord(
            id="t1",
            prompt="Create a vertical hero card with a title and body.",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
            split="train",
        ),
        ExampleRecord(
            id="t2",
            prompt="Add a primary call-to-action button under a short text line.",
            openui=(
                'root = Stack([copy, cta], "column")\n'
                'copy = TextContent(":copy.line")\n'
                'cta = Button(":cta.label")'
            ),
            placeholders=[":copy.line", ":cta.label"],
            split="train",
        ),
        ExampleRecord(
            id="t3",
            prompt="Simple text block inside a stack.",
            openui='root = Stack([blurb], "column")\nblurb = TextContent(":page.blurb")',
            placeholders=[":page.blurb"],
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
            placeholders=[":hero.title", ":hero.body"],
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
    assert model.tokenizer.vocab_size <= 400
    loss = model.training_loss(records)
    assert float(loss.detach()) >= 0.0


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
                placeholders=[":hero.title", ":hero.body"],
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
