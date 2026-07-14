"""Eval-driven gates and scoreboard helpers."""

from __future__ import annotations

from pathlib import Path

from slm_training.data.leakage import (
    fingerprint_openui,
    fingerprint_openui_structure,
    normalize_openui_structure,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    component_type_recall,
    evaluate,
    evaluate_suites,
    structural_similarity,
)
from slm_training.harnesses.model_build.plugin import GenerationRequest, StubModel
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.harnesses.preference import composite_reward


def test_structural_similarity_identical() -> None:
    src = 'root = Stack([a])\na = Button(":x")'
    assert structural_similarity(src, src) == 1.0


def test_structural_similarity_partial() -> None:
    gold = 'root = Stack([a, b])\na = Button(":a")\nb = Button(":b")'
    pred = 'root = Stack([a])\na = Button(":a")'
    score = structural_similarity(pred, gold)
    assert 0.0 < score < 1.0


def test_structural_fingerprint_ignores_placeholder_namespace() -> None:
    train = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        "hero = Card([hero_title, hero_body])"
    )
    smoke = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":smoke.hero.title")\n'
        'hero_body = TextContent(":smoke.hero.body")\n'
        "hero = Card([hero_title, hero_body])"
    )
    assert fingerprint_openui(train) != fingerprint_openui(smoke)
    assert fingerprint_openui_structure(train) == fingerprint_openui_structure(smoke)
    assert normalize_openui_structure(train) == normalize_openui_structure(smoke)


def test_reward_score_on_valid_pred() -> None:
    gold = ExampleRecord(
        id="t",
        prompt="button",
        openui='root = Stack([cta])\ncta = Button(":cta")',
        placeholders=[":cta"],
    )
    score = composite_reward(gold.openui, gold=gold, design_md=None)
    assert score > 0.5


def test_reward_does_not_credit_gold_design_md_when_none() -> None:
    openui = 'root = Stack([cta])\ncta = Button(":cta")'
    gold = ExampleRecord(
        id="t",
        prompt="button",
        openui=openui,
        placeholders=[":cta"],
        design_md="# Fancy\n" + ("x" * 200),
    )
    assert composite_reward(openui, gold=gold, design_md=None) == composite_reward(
        openui, gold=gold
    )


def test_meaningful_parse_requires_component_recall() -> None:
    gold = ExampleRecord(
        id="settings",
        prompt="settings",
        openui=(
            'root = Stack([title, notify, volume], "column")\n'
            'title = TextContent(":t")\n'
            'notify = SwitchItem(":n", ":d", "x")\n'
            'volume = Slider("volume", "default", 0, 100, 1, 40, ":v")'
        ),
        placeholders=[":t", ":n", ":d", ":v"],
    )
    weak = 'root = Stack([title], "column")\ntitle = TextContent(":t")'
    ok, err, _ = _is_meaningful_program(weak, gold=gold)
    assert ok is False
    assert err and err.startswith("low_component_recall")
    assert component_type_recall(weak, gold.openui) < 0.5


def test_ship_gates_fail_when_hard_suites_miss() -> None:
    suites = {
        "smoke": {
            "n": 3,
            "parse_rate": 1.0,
            "structural_similarity": 1.0,
            "placeholder_fidelity": 0.8,
            "reward_score": 0.7,
        },
        "held_out": {
            "n": 5,
            "parse_rate": 0.2,
            "structural_similarity": 0.4,
            "placeholder_fidelity": 0.0,
        },
        # adversarial / ood / rico_held missing → fail
    }
    result = evaluate_ship_gates(suites)
    assert result["pass"] is False
    assert any("missing_suite" in f for f in result["failures"])
    assert set(DEFAULT_SHIP_GATES) >= {"smoke", "rico_held"}


def test_evaluate_suites_scoreboard(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    hero = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        "hero = Card([hero_title, hero_body])"
    )
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=hero, split="train"),
    ]
    write_jsonl(train_dir / "records.jsonl", records)
    write_jsonl(
        test_dir / "suites" / "smoke" / "records.jsonl",
        [
            ExampleRecord(
                id="s1",
                prompt="Hero",
                openui=hero,
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="gates",
        model_name="stub",
        noise_rate=0.0,
    )
    model = StubModel(noise_rate=0.0, seed=0)
    model.forward(records)
    metrics = evaluate(config, model=model)
    assert "reward_score" in metrics
    assert metrics["n"] == 1
    assert metrics["parse_rate"] == 1.0

    board = evaluate_suites(config, ["smoke"], model=model)
    assert "suites" in board
    assert board["checkpoint"] is None
    assert board["checkpoint_sha256"] is None
    assert board["checkpoint_source"] == "preloaded_model"
    assert board["suites"]["smoke"]["checkpoint_source"] == "preloaded_model"

    checkpoint = config.checkpoint_dir / "last.pt"
    model.save(checkpoint)
    loaded = evaluate_suites(config, ["smoke"], checkpoint=checkpoint)
    assert loaded["checkpoint"] == str(checkpoint)
    assert loaded["checkpoint_sha256"]
    assert loaded["checkpoint_source"] == "checkpoint"
    assert loaded["suites"]["smoke"]["checkpoint_sha256"] == loaded["checkpoint_sha256"]
    assert (tmp_path / "runs" / "gates" / "scoreboard.json").exists()


def test_evaluate_uses_production_request_not_gold_record(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    gold = 'root = Stack([cta])\ncta = Button(":prod.cta")'
    record = ExampleRecord(
        id="s1",
        prompt="CTA",
        openui=gold,
        placeholders=[":prod.cta"],
        split="smoke",
        meta={"suite": "smoke"},
    )
    write_jsonl(train_dir / "records.jsonl", [ExampleRecord.from_dict({**record.to_dict(), "split": "train"})])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class RequestOnlyModel:
        def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
            assert len(requests) == 1
            request = requests[0]
            assert request.prompt == "CTA"
            assert request.slot_contract == (":prod.cta",)
            assert not hasattr(request, "openui")
            return [gold]

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="request-only",
        model_name="stub",
    )
    metrics = evaluate(config, model=RequestOnlyModel())
    assert metrics["raw_syntax_validity"] == 1.0
    assert metrics["contract_precision"] == 1.0
    assert metrics["contract_recall"] == 1.0
    assert metrics["fallback_count"] == 0
