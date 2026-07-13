"""Metric-ceiling and vocab-coverage diagnostics."""

from __future__ import annotations

from pathlib import Path


from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.diagnostic import (
    ceiling_report,
    score_gold_as_prediction,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
)
SMOKE = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":smoke.hero.title")\n'
    'hero_body = TextContent(":smoke.hero.body")\n'
    'hero = Card([hero_title, hero_body])'
)


def test_compositional_placeholder_tokenization() -> None:
    tokens = tokenize_text('hero = TextContent(":smoke.hero.title")')
    assert ":" in tokens
    assert "smoke" in tokens
    assert "hero" in tokens
    assert "title" in tokens
    assert '":smoke.hero.title"' not in tokens


def test_gold_as_prediction_ceiling() -> None:
    record = ExampleRecord(
        id="smoke_hero",
        prompt="Hero",
        openui=SMOKE,
        placeholders=[":smoke.hero.title", ":smoke.hero.body"],
        split="smoke",
    )
    row = score_gold_as_prediction(record)
    assert row["parse_ok"] is True
    assert row["placeholder_fidelity"] == 1.0
    assert row["structural_similarity"] == 1.0


def test_vocab_coverage_atomic_vs_compositional() -> None:
    train_records = [
        ExampleRecord(id="tr1", prompt="Hero", openui=HERO, split="train"),
    ]
    test_records = [
        ExampleRecord(
            id="sm1",
            prompt="Hero smoke",
            openui=SMOKE,
            placeholders=[":smoke.hero.title", ":smoke.hero.body"],
            split="smoke",
        ),
    ]
    tokenizer = OpenUITokenizer.build(
        [r.prompt for r in train_records] + [r.openui for r in train_records]
    )
    vocab = set(tokenizer.token_to_id)
    placeholder_tokens: list[str] = []
    for record in test_records:
        for ph in record.placeholders or []:
            placeholder_tokens.extend(tokenize_text(f'"{ph}"'))
    missing = [t for t in placeholder_tokens if t not in vocab]
    # Compositional tokenization: only namespace segments (e.g. "smoke") may be OOV.
    assert set(missing) <= {"smoke"}
    assert ":" in vocab and "hero" in vocab and "title" in vocab


def test_ceiling_report_fixture_suites(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suites" / "smoke"
    suite_dir.mkdir(parents=True)
    write_jsonl(
        suite_dir / "records.jsonl",
        [
            ExampleRecord(
                id="sm1",
                prompt="Hero",
                openui=SMOKE,
                placeholders=[":smoke.hero.title", ":smoke.hero.body"],
                split="smoke",
            ),
        ],
    )
    board = ceiling_report(tmp_path, suites=("smoke",))
    assert board["smoke"]["parse_rate"] == 1.0
    assert board["smoke"]["placeholder_fidelity"] == 1.0


def test_length_budget_flags_truncation() -> None:
    from slm_training.harnesses.model_build.diagnostic import length_budget_report

    long = ExampleRecord(
        id="long",
        prompt="dashboard",
        openui=(
            'root = Stack([a, b, c, d], "column")\n'
            'a = TextContent(":smoke.a")\n'
            'b = TextContent(":smoke.b")\n'
            'c = TextContent(":smoke.c")\n'
            'd = Card([e, f])\n'
            'e = TextContent(":smoke.e")\n'
            'f = TextContent(":smoke.f")\n'
        ),
        placeholders=[
            ":smoke.a",
            ":smoke.b",
            ":smoke.c",
            ":smoke.e",
            ":smoke.f",
        ],
    )
    # Legacy 64-token budget must fail on compositional lengths.
    bad = length_budget_report(
        records=[long],
        grammar_ltr_max_tokens=64,
        grammar_ltr_stages=(32, 48, 64),
    )
    assert bad["ok"] is False
    assert bad["sections"]["records"]["p95"] > 64

    # Length-safe E18 budget must cover the same program.
    good = length_budget_report(
        records=[long],
        grammar_ltr_max_tokens=192,
        grammar_ltr_stages=(64, 128, 192, 256),
    )
    assert good["ok"] is True
    assert good["effective_budget"] >= good["sections"]["records"]["p95"]


def test_fixture_seeds_fit_e18_budget() -> None:
    from pathlib import Path

    from slm_training.dsl.schema import load_jsonl
    from slm_training.harnesses.model_build.diagnostic import length_budget_report

    root = Path(__file__).resolve().parents[2]
    train = load_jsonl(root / "fixtures" / "train_seeds.jsonl")
    test = load_jsonl(root / "fixtures" / "test_seeds.jsonl")
    report = length_budget_report(
        records=list(train) + list(test),
        grammar_ltr_max_tokens=192,
        grammar_ltr_stages=(64, 128, 192, 256),
    )
    assert report["ok"] is True, report["failures"]
    assert report["sections"]["records"]["max"] <= 192 or report[
        "sections"
    ]["records"]["max"] <= 256
