"""Eval-driven gates and scoreboard helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.data.leakage import (
    fingerprint_openui,
    fingerprint_openui_structure,
    normalize_openui_structure,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import (
    _effective_evaluation_policy,
    _is_meaningful_program,
    component_type_recall,
    evaluate,
    evaluate_suites,
    structural_similarity,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.plugin import GenerationRequest, StubModel
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.harnesses.preference import composite_reward
from slm_training.dsl.production_codec import ProductionCodec
from slm_training.models.decode_stats import DecodeStats


def test_evaluation_policy_reports_loaded_checkpoint_settings() -> None:
    config = ModelBuildConfig(
        train_dir=Path("train"),
        context_backend="hf",
        grammar_constrained=None,
        grammar_ltr_primary=None,
    )
    plugin = SimpleNamespace(
        config=SimpleNamespace(
            context_backend="scratch",
            grammar_constrained=True,
            grammar_ltr_primary=False,
            grammar_ltr_repair=False,
            compiler_decode_mode="off",
            decode_min_content=-1,
            local_files_only=False,
            schema_in_context=False,
            slot_contract_in_context=False,
            slot_contract_constrained_decode=False,
            honest_slot_contract=False,
            grammar_skip_exact_stream_probe=True,
            grammar_verify_chosen_only=False,
            grammar_top_k=16,
            generate_max_attempts=3,
            decode_timeout_seconds=None,
            allow_unconstrained_fallback=True,
            gen_steps=8,
            grammar_ltr_max_tokens=256,
        )
    )
    policy = _effective_evaluation_policy(config, plugin)
    assert policy["context_backend"] == "scratch"
    assert policy["grammar_constrained"] is True
    assert policy["grammar_ltr_primary"] is False
    assert policy["decode_min_content"] == -1


def test_structural_similarity_identical() -> None:
    src = 'root = Stack([a])\na = Button(":x")'
    assert structural_similarity(src, src) == 1.0


def test_suite_loader_falls_back_when_manifest_path_is_checkout_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_dir = tmp_path / "suites" / "smoke"
    suite_dir.mkdir(parents=True)
    record = ExampleRecord(id="s", prompt="p", openui="root = Stack([])")
    write_jsonl(suite_dir / "records.jsonl", [record])
    (tmp_path / "manifest.json").write_text(
        '{"suites":{"smoke":"outputs/data/eval/v1/suites/smoke/records.jsonl"}}\n'
    )
    # Resolve the checkout-relative manifest path from tmp_path so the test
    # stays hermetic even when a real outputs/data/eval/v1 has been built.
    monkeypatch.chdir(tmp_path)
    assert load_suite_records(tmp_path, "smoke") == [record]


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
            'volume = Slider("volume", "continuous", 0, 100, 1, [40], ":v")'
        ),
        placeholders=[":t", ":n", ":d", ":v"],
    )
    weak = 'root = Stack([title], "column")\ntitle = TextContent(":t")'
    ok, err, _ = _is_meaningful_program(weak, gold=gold)
    assert ok is False
    assert err and err.startswith("low_component_recall")
    assert component_type_recall(weak, gold.openui) < 0.5


def test_meaningful_parse_rejects_pathological_overgeneration() -> None:
    gold = ExampleRecord(
        id="fallback",
        prompt="x",
        openui='root = Stack([content])\ncontent = TextContent(":x.text")',
        placeholders=[":x.text"],
    )
    repeated = ", ".join(["content"] * 30)
    bloated = (
        f"root = Stack([{repeated}])\n"
        'content = TextContent(":x.text")'
    )

    ok, err, _ = _is_meaningful_program(bloated, gold=gold)

    assert ok is False
    assert err and err.startswith("excessive_output_tokens:")


def test_failed_meaningful_parse_gets_no_recall_or_reward_credit(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    gold = ExampleRecord(
        id="fallback",
        prompt="x",
        openui='root = Stack([content])\ncontent = TextContent(":x.text")',
        placeholders=[":x.text"],
        split="smoke",
    )
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [gold])
    repeated = ", ".join(["content"] * 30)
    bloated = (
        f"root = Stack([{repeated}])\n"
        'content = TextContent(":x.text")'
    )
    model = SimpleNamespace(generate=lambda _prompt: bloated)
    config = ModelBuildConfig(
        train_dir=tmp_path,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="overgeneration",
    )

    metrics = evaluate(config, model=model, publish_agentv=False)

    assert metrics["meaningful_program_rate"] == 0.0
    assert metrics["component_type_recall"] == 0.0
    assert metrics["reward_score"] == 0.0
    assert metrics["failure_breakdown"] == {"pathological_overgeneration": 1}


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


def _full_suite_metrics(**overrides: float) -> dict[str, dict[str, float]]:
    """Every policy suite passing every default bar, before overrides."""
    base = {
        suite: {
            "n": 32,
            "meaningful_program_rate": 0.9,
            "syntax_parse_rate": 1.0,
            "structural_similarity": 0.9,
            "component_type_recall": 0.9,
            "placeholder_fidelity": 0.9,
            "reward_score": 0.9,
        }
        for suite in DEFAULT_SHIP_GATES
    }
    for suite in base:
        base[suite].update(overrides)
    return base


def test_semantic_density_floor_present_for_every_suite() -> None:
    # E2: the density floor must guard every policy suite, not just smoke.
    for suite, mins in DEFAULT_SHIP_GATES.items():
        assert "component_type_recall" in mins, suite
        assert mins["component_type_recall"] <= mins["structural_similarity"]


def test_ship_gates_fail_on_shorter_but_emptier_output() -> None:
    # Valid + structural + high placeholder/reward, but component recall
    # collapses (the shorter-but-emptier failure mode) → gates must fail.
    suites = _full_suite_metrics(component_type_recall=0.0)
    result = evaluate_ship_gates(suites)
    assert result["pass"] is False
    assert any(":component_type_recall" in f for f in result["failures"])
    # The metric is surfaced in `actual` for transparency.
    assert result["actual"]["smoke"]["component_type_recall"] == 0.0


def test_ship_gates_pass_when_density_met() -> None:
    # Sanity: the density floor does not block a genuinely populated run.
    result = evaluate_ship_gates(_full_suite_metrics())
    assert result["pass"] is True
    assert not result["failures"]


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


def test_evaluate_supports_single_record_generation_with_stats(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    gold = 'root = TextContent(":copy.value")'
    record = ExampleRecord(
        id="stats-1",
        prompt="Copy value",
        openui=gold,
        split="smoke",
        meta={"suite": "smoke"},
    )
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class StatsModel:
        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            assert prompt == "Copy value"
            return gold, DecodeStats(tokens_emitted=1)

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="stats-generation",
        model_name="twotower",
    )
    metrics = evaluate(config, model=StatsModel(), publish_agentv=False)
    assert metrics["parse_rate"] == 1.0
    assert metrics["decode_stats"]["tokens_emitted_sum"] == 1.0


def test_evaluate_persists_stats_when_generation_times_out(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    record = ExampleRecord(
        id="timeout-1",
        prompt="Copy value",
        openui='root = TextContent(":copy.value")',
        split="smoke",
        meta={"suite": "smoke"},
    )
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class TimeoutModel:
        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            error = TimeoutError("decode exceeded")
            error.decode_stats = DecodeStats(tokens_emitted=7)  # type: ignore[attr-defined]
            raise error

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="timeout-generation",
        model_name="twotower",
        decode_timeout_seconds=1,
    )
    metrics = evaluate(config, model=TimeoutModel(), publish_agentv=False)
    assert metrics["decode_timeout_count"] == 1
    assert metrics["decode_stats"]["tokens_emitted_sum"] == 7.0


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
    write_jsonl(
        train_dir / "records.jsonl",
        [ExampleRecord.from_dict({**record.to_dict(), "split": "train"})],
    )
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class RequestOnlyModel:
        def generate_batch_requests(
            self, requests: list[GenerationRequest]
        ) -> list[str]:
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


def test_evaluate_keeps_production_request_when_model_also_exposes_stats(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    gold = 'root = Button(":prod.cta")'
    record = ExampleRecord(
        id="request-stats",
        prompt="CTA",
        openui=gold,
        placeholders=[":prod.cta"],
        split="smoke",
        meta={"suite": "smoke"},
    )
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class RequestAndStatsModel:
        def generate_batch_requests(
            self, requests: list[GenerationRequest]
        ) -> list[str]:
            assert requests[0].slot_contract == (":prod.cta",)
            return [gold]

        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            raise AssertionError("request-aware generation must take precedence")

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="request-and-stats",
        model_name="stub",
    )
    metrics = evaluate(config, model=RequestAndStatsModel(), publish_agentv=False)
    assert metrics["placeholder_fidelity"] == 1.0
    assert metrics["decode_stats"]["n"] == 1


def test_topology_composite_keeps_quality_structure_trace_and_efficiency(
    tmp_path: Path,
) -> None:
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
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", [record])

    class TopologyModel:
        codec = ProductionCodec.build([gold])

        def generate_batch_requests(
            self, requests: list[GenerationRequest]
        ) -> list[str]:
            return [gold for _ in requests]

        def consume_generation_evidence(self) -> list[dict[str, float]]:
            return [
                {
                    "efficiency_score": 0.75,
                    "node_passes": 8.0,
                    "active_peak": 3.0,
                    "phases": 4.0,
                }
            ]

        def score_topology_targets(
            self, records: list[ExampleRecord]
        ) -> list[dict[str, float]]:
            return [
                {
                    "action_macro_f1": 0.8,
                    "production_head_accuracy": 0.7,
                    "arity_head_accuracy": 0.6,
                    "critic_ece": 0.1,
                    "scope_kind": "statement",
                    "scope_family": "local_repair",
                    "scope_gate_accuracy": 0.75,
                    "scope_summary_definitions_mae": 0.1,
                    "scope_summary_uses_mae": 0.2,
                    "scope_summary_slots_mae": 0.3,
                    "scope_summary_realized_size_mae": 0.4,
                    "failure_cone_tp": 2,
                    "failure_cone_fp": 1,
                    "failure_cone_fn": 2,
                    "failure_cone_predicted_size": 3,
                    "failure_cone_target_size": 4,
                }
                for _ in records
            ]

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="topology-score",
        model_name="grammar_diffusion",
    )
    metrics = evaluate(config, model=TopologyModel(), publish_agentv=False)
    assert metrics["ast_node_f1"] == 1.0
    assert metrics["ast_edge_f1"] == 1.0
    assert metrics["topology_efficiency_score"] == 0.75
    assert 0.0 < metrics["topology_composite"] <= 1.0
    assert metrics["topology_telemetry"]["production_head_accuracy"] == 0.7
    scope_metrics = metrics["scope_contract_metrics"]
    assert scope_metrics["sample_count"] == 1
    assert scope_metrics["scope_gate_accuracy"] == 0.75
    assert abs(scope_metrics["failure_cone_precision"] - 2 / 3) < 1e-9
    assert scope_metrics["failure_cone_recall"] == 0.5
    assert scope_metrics["by_scope_kind"]["statement"]["sample_count"] == 1
